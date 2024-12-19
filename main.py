import concurrent.futures
import logging
import os
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

import boto3
import click

# Plotting imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import yaml
from dotenv import load_dotenv

# Langchain imports
from langchain_aws import ChatBedrockConverse
from tqdm import tqdm

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_query(method: str, query: str, model_id: str, performance_config: str) -> Dict[str, Any]:
    start = time.perf_counter()

    if method == "Boto3":
        client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
        message = {"role": "user", "content": [{"text": query}]}
        response = client.converse(
            modelId=model_id,
            messages=[message],
            performanceConfig={"latency": performance_config},
        )
        token_count = response["usage"]["outputTokens"]

    else:  # Langchain
        chat = ChatBedrockConverse(
            model_id=model_id,
            region_name="us-east-2",
            performance_config={"latency": performance_config},
        )
        response = chat.invoke(query)
        token_count = response.usage_metadata["output_tokens"]  # type: ignore

    elapsed = time.perf_counter() - start
    tps = token_count / elapsed if elapsed > 0 else float("inf")

    return {
        "Query": query,
        "Method": method,
        "PerfConfig": performance_config,
        "ResponseTokens": token_count,
        "Time (s)": f"{elapsed:.4f}",
        "Tokens/s": tps,
        "Timestamp": datetime.now().isoformat(),
    }


def run_benchmarks(
    queries: List[str],
    model_id: str,
    iterations: int,
    max_workers: int,
    configs: List[str],
    methods: List[str],
) -> pd.DataFrame:
    # Build a single flat list of all tasks
    # Each task is a tuple of (method, query, config)
    tasks = [
        (method, query, config)
        for config in configs
        for method in methods
        for query in queries
        for _ in range(iterations)
    ]

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_query, method, query, model_id, config) for (method, query, config) in tasks]

        # Show progress as tasks complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Running all configurations",
        ):
            results.append(future.result())

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame):
    sns.set_palette("husl")
    plt.figure(figsize=(16, 8))

    grouped_data = df.groupby(["Query", "PerfConfig", "Method"])["Tokens/s"].mean().reset_index()
    bar_width = 0.2
    queries = df["Query"].unique()
    x = range(len(queries))

    aws_orange = "#FF9900"
    aws_dark_orange = "#FF6600"
    langchain_colors = ["#4CAF50", "#2E7D32"]
    methods = ["Langchain", "Boto3"]
    perf_configs = ["standard", "optimized"]
    colors = [langchain_colors[0], langchain_colors[1], aws_orange, aws_dark_orange]

    for i, method in enumerate(methods):
        for j, config in enumerate(perf_configs):
            data = grouped_data[(grouped_data["Method"] == method) & (grouped_data["PerfConfig"] == config)].set_index(
                "Query"
            )["Tokens/s"]
            plt.bar(
                [pos + (i * len(perf_configs) + j) * bar_width for pos in x],
                data.values,
                bar_width,
                label=f"{method} - {config}",
                alpha=0.8,
                color=colors[i * len(perf_configs) + j],
            )

    plt.xlabel("Queries", fontsize=12)
    plt.ylabel("Tokens per Second", fontsize=12)
    plt.title(
        "Claude 3.5 Haiku Performance Comparison\nLangchain vs Boto3 - Standard vs Optimized Configuration",
        fontsize=14,
        pad=20,
    )
    plt.xticks(x, queries, rotation=0, ha="center", fontsize=9)
    plt.tight_layout()
    plt.legend(title="Method - Configuration", loc="upper left", bbox_transform=plt.gcf().transFigure)

    plt.savefig("benchmark_comparison_methods.png", dpi=300, bbox_inches="tight")
    plt.show()


@click.command()
@click.option("--iterations", "-i", default=10, help="Iterations per query")
@click.option("--max-workers", "-c", default=10, help="Max concurrent requests")
@click.option("--model-id", "-m", default="us.anthropic.claude-3-5-haiku-20241022-v1:0", help="Bedrock model ID")
@click.option("--queries-file", "-q", default="queries.yaml", help="YAML file containing queries")
def main(iterations: int, max_workers: int, model_id: str, queries_file: str):
    with open(queries_file, "r") as f:
        queries = yaml.safe_load(f)["queries"]

    configs = ["standard", "optimized"]
    methods = ["Langchain", "Boto3"]

    df = run_benchmarks(queries, model_id, iterations, max_workers, configs, methods)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"unified_benchmark_results_{timestamp}.txt"
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    # Plot immediately after benchmarks
    plot_results(df)
    logger.info("Benchmark and plotting completed.")


if __name__ == "__main__":
    main()
