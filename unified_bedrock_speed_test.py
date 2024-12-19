import asyncio
import concurrent.futures
import json
import logging
import os
import random
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

import boto3
import click
import pandas as pd
import yaml
from dotenv import load_dotenv

# Langchain imports
from langchain_aws import ChatBedrockConverse
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def log_request(request, **kwargs):
    """Log the request details."""
    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        logger.info("\n" + "=" * 50)
        logger.info(f"Request intercepted:\nURL: {request.url}")
        logger.info(f"Method: {request.method}")
        logger.info(f"Headers: {request.headers}")
        logger.info(f"Body: {json.dumps(body, indent=2)}")
        logger.info("=" * 50 + "\n")
    except Exception as e:
        logger.error(f"Error logging request: {e}")


# AWS Configuration
boto3.setup_default_session(profile_name=os.getenv("AWS_PROFILE"), region_name=os.getenv("AWS_REGION"))

# Clear unsupported environment variables
for env_var in ["AWS_ENDPOINT_URL", "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID"]:
    os.environ.pop(env_var, None)


def create_bedrock_client():
    """Create and return a boto3 bedrock client."""
    return boto3.client("bedrock-runtime")


async def run_langchain_query(chat: ChatBedrockConverse, query: str, performance_config: str) -> Dict[str, Any]:
    """Execute a single query using Langchain and measure its performance."""
    start = time.perf_counter()
    response = await chat.ainvoke(query)
    elapsed = time.perf_counter() - start

    token_count = response.usage_metadata["output_tokens"]  # type: ignore
    tps = token_count / elapsed if elapsed > 0 else float("inf")

    return {
        "Query": query,
        "Method": "Langchain",
        "PerfConfig": performance_config,
        "ResponseTokens": token_count,
        "Time (s)": f"{elapsed:.4f}",
        "Tokens/s": f"{tps:.2f}",
        "Timestamp": datetime.now().isoformat(),
    }


def run_boto3_query(client, query: str, performance_config: str) -> Dict[str, Any]:
    """Execute a single query using Boto3 and measure its performance."""
    message = {
        "role": "user",
        "content": [{"text": query}],
    }

    start = time.perf_counter()

    # Make the API call
    response = client.converse(
        modelId=MODEL_ID,
        messages=[message],
        performanceConfig={"latency": performance_config},
    )

    elapsed = time.perf_counter() - start

    # Extract token count from response
    token_count = response["usage"]["outputTokens"]
    tps = token_count / elapsed if elapsed > 0 else float("inf")

    return {
        "Query": query,
        "Method": "Boto3",
        "PerfConfig": performance_config,
        "ResponseTokens": token_count,
        "Time (s)": f"{elapsed:.4f}",
        "Tokens/s": f"{tps:.2f}",
        "Timestamp": datetime.now().isoformat(),
    }


async def run_langchain_benchmark(queries: List[str], performance_config: str) -> List[Dict[str, Any]]:
    """Run Langchain benchmark tests with parallel execution."""
    chat = ChatBedrockConverse(
        model_id=MODEL_ID,  # type: ignore
        region_name="us-east-2",
        performance_config={"latency": performance_config},
    )  # type: ignore

    # Create interleaved tasks - mix different queries together
    all_tasks = []
    for iteration in range(NUM_ITERATIONS):
        for query in queries:
            all_tasks.append((query, iteration))

    # Shuffle tasks to further randomize the order
    random.shuffle(all_tasks)

    results = []
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def run_with_semaphore(query: str, iteration: int) -> Dict[str, Any]:
        async with semaphore:
            return await run_langchain_query(chat, query, performance_config)

    # Create tasks with progress bar
    tasks = [run_with_semaphore(query, iteration) for query, iteration in all_tasks]

    with tqdm(total=len(tasks), desc=f"Langchain {performance_config} configuration") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)

    return results


def run_boto3_benchmark(queries: List[str], performance_config: str) -> List[Dict[str, Any]]:
    """Run Boto3 benchmark tests with parallel execution."""
    client = create_bedrock_client()
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        # Create a list of futures for all iterations of all queries
        futures = []
        for query in queries:
            for _ in range(NUM_ITERATIONS):
                futures.append(executor.submit(run_boto3_query, client, query, performance_config))

        # Use tqdm to show progress as futures complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc=f"Boto3 {performance_config} configuration",
        ):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error in query execution: {e}")

    return results


@click.command()
@click.option("--iterations", "-i", default=10, help="Number of iterations per query")
@click.option("--concurrent", "-c", default=10, help="Maximum concurrent requests")
@click.option("--model-id", "-m", default="us.anthropic.claude-3-5-haiku-20241022-v1:0", help="Bedrock model ID")
@click.option("--queries-file", "-q", default="queries.yaml", help="YAML file containing queries")
async def main(iterations: int, concurrent: int, model_id: str, queries_file: str):
    """Main execution function."""
    print(f"\nRunning benchmarks with {iterations} iterations per query")
    print(f"Maximum concurrent requests: {concurrent}")

    # Load queries from YAML
    with open(queries_file, "r") as f:
        queries = yaml.safe_load(f)["queries"]

    # Update global variables
    global NUM_ITERATIONS, MAX_CONCURRENT, MODEL_ID
    NUM_ITERATIONS = iterations
    MAX_CONCURRENT = concurrent
    MODEL_ID = model_id

    configs = [
        "standard",
        "optimized",
    ]
    all_results = []

    for config in configs:
        print(f"\nRunning Langchain {config} configuration:")
        langchain_results = await run_langchain_benchmark(queries, config)
        all_results.extend(langchain_results)

        print(f"\nRunning Boto3 {config} configuration:")
        boto3_results = run_boto3_benchmark(queries, config)
        all_results.extend(boto3_results)

    # Save results to CSV
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"unified_benchmark_results_{timestamp}.txt"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
