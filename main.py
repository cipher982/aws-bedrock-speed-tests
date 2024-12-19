import logging
import os
import threading
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import boto3
import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from joblib import Parallel
from joblib import delayed
from langchain_aws import ChatBedrockConverse

# Define output directory constant
OUTPUT_DIR = "outputs"

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread-local storage for clients
thread_local = threading.local()


def get_boto3_client():
    if not hasattr(thread_local, "boto3_client"):
        thread_local.boto3_client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
    return thread_local.boto3_client


def get_langchain_client(model_id: str, performance_config: str):
    if not hasattr(thread_local, "langchain_client"):
        thread_local.langchain_client = ChatBedrockConverse(
            model_id=model_id,
            region_name=os.getenv("AWS_REGION"),
            performance_config={"latency": performance_config},
        )
    return thread_local.langchain_client


def run_query(method: str, query: str, model_id: str, performance_config: str) -> Dict[str, Any]:
    start = time.perf_counter()

    if method == "Boto3":
        client = get_boto3_client()
        message = {"role": "user", "content": [{"text": query}]}
        response = client.converse(
            modelId=model_id,
            messages=[message],
            performanceConfig={"latency": performance_config},
        )
        token_count = response["usage"]["outputTokens"]

    else:  # Langchain
        client = get_langchain_client(model_id, performance_config)
        response = client.invoke(query)
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
    concurrent_calls: int,
    configs: List[str],
    methods: List[str],
) -> pd.DataFrame:
    # Build a single flat list of all tasks
    tasks = [
        (method, query, config)
        for config in configs
        for method in methods
        for query in queries
        for _ in range(iterations)
    ]

    # Run tasks in parallel with joblib
    results = Parallel(n_jobs=concurrent_calls, prefer="threads", verbose=1)(
        delayed(run_query)(method, query, model_id, config) for method, query, config in tasks
    )

    return pd.DataFrame(results)


def plot_results(df: pd.DataFrame):
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    plt.figure(figsize=(12, 6))
    x = np.arange(len(df["Query"].unique()))
    bar_width = 0.2
    methods = df["Method"].unique()
    perf_configs = df["PerfConfig"].unique()

    # Color palette for different method-config combinations
    aws_orange = "#FF9900"
    aws_dark_orange = "#FF6600"
    langchain_colors = ["#4CAF50", "#2E7D32"]
    methods = ["Langchain", "Boto3"]
    perf_configs = ["standard", "optimized"]
    colors = [langchain_colors[0], langchain_colors[1], aws_orange, aws_dark_orange]

    for i, method in enumerate(methods):
        for j, config in enumerate(perf_configs):
            data = df[(df["Method"] == method) & (df["PerfConfig"] == config)].groupby("Query")["Tokens/s"].mean()
            plt.bar(
                x + (i * len(perf_configs) + j) * bar_width,
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
    plt.xticks(x, df["Query"].unique(), rotation=0, ha="center", fontsize=9)
    plt.tight_layout()
    plt.legend(title="Method - Configuration", loc="upper left", bbox_transform=plt.gcf().transFigure)

    # Save to output directory
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_comparison_methods.png"), dpi=300, bbox_inches="tight")
    plt.show()


@click.command()
@click.option("--iterations", "-i", default=5, help="Number of iterations for each query")
@click.option("--concurrent-calls", "-c", default=4, help="Maximum number of concurrent workers")
@click.option("--model-id", "-m", default="us.anthropic.claude-3-5-haiku-20241022-v1:0", help="Bedrock model ID")
@click.option("--queries-file", "-q", default="queries.yaml", help="YAML file containing queries")
@click.option("--plot-only", "-p", is_flag=True, help="Plot existing results without running benchmarks")
@click.option("--results-file", "-r", default=None, help="Path to existing results file to plot")
def main(
    iterations: int,
    concurrent_calls: int,
    model_id: str,
    queries_file: str,
    plot_only: bool,
    results_file: Optional[str],
):
    # If plot_only is True, try to load an existing results file
    if plot_only:
        if results_file is None:
            # Find the most recent results file in the outputs directory
            try:
                results_files = [
                    f
                    for f in os.listdir(OUTPUT_DIR)
                    if f.startswith("unified_benchmark_results_") and f.endswith(".txt")
                ]
                if not results_files:
                    logger.error(
                        f"No results files found in {OUTPUT_DIR}. Use without --plot-only to generate results."
                    )
                    return

                # Sort files by name (which includes timestamp) and get the most recent
                results_file = os.path.join(OUTPUT_DIR, sorted(results_files)[-1])
            except Exception as e:
                logger.error(f"Error finding results file: {e}")
                return

        # Load the DataFrame from the specified or found results file
        try:
            df = pd.read_csv(results_file)
            logger.info(f"Loaded results from {results_file}")
        except Exception as e:
            logger.error(f"Error loading results file: {e}")
            return

        # Plot the results
        plot_results(df)
        return

    # If not plot_only, proceed with benchmarks
    with open(queries_file, "r") as f:
        queries = yaml.safe_load(f)["queries"]

    methods = ["Langchain", "Boto3"]
    configs = ["standard", "optimized"]

    df = run_benchmarks(queries, model_id, iterations, concurrent_calls, configs, methods)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"unified_benchmark_results_{timestamp}.txt"
    df.to_csv(os.path.join(OUTPUT_DIR, output_file), index=False)
    logger.info(f"Results saved to {os.path.join(OUTPUT_DIR, output_file)}")

    # Plot immediately after benchmarks
    plot_results(df)
    logger.info("Benchmark and plotting completed.")


if __name__ == "__main__":
    main()
