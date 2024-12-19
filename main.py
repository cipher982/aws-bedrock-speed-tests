import logging
import os
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


def run_query(method: str, query: str, model_id: str, performance_config: str) -> Dict[str, Any]:
    """Run a single query and return performance metrics."""
    start_time = time.time()

    try:
        if method == "boto3":
            client = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION"))
            response = client.converse(
                modelId=model_id,
                messages=[{"role": "user", "content": [{"text": query}]}],
                performanceConfig={"latency": performance_config},
            )
            token_count = response["usage"]["outputTokens"]

        elif method == "langchain":
            client = ChatBedrockConverse(
                model_id=model_id,
                region_name=os.getenv("AWS_REGION"),
                performance_config={"latency": performance_config},
            )
            response = client.invoke(query)
            token_count = response.usage_metadata["output_tokens"]

        else:
            raise ValueError(f"Unknown method: {method}")

        end_time = time.time()
        processing_time = end_time - start_time
        tokens_per_second = token_count / processing_time if processing_time > 0 else 0

        return {
            "method": method,
            "query": query,
            "performance_config": performance_config,
            "tokens_per_second": tokens_per_second,
            "token_count": token_count,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat(),
            "success": True,
            "error": None,
        }

    except Exception as e:
        raise Exception(f"Error running query: {e}")


def run_benchmarks(
    queries: List[str],
    model_id: str,
    iterations: int,
    concurrent_calls: int,
    configs: List[str],
    methods: List[str],
) -> pd.DataFrame:
    """Run benchmarks for all combinations of methods and configurations."""
    results = []

    # Create all combinations of parameters for parallel execution
    all_tasks = [
        (method, config, query, iteration)
        for method in methods
        for config in configs
        for query in queries
        for iteration in range(iterations)
    ]

    # Calculate total number of tasks
    total_tasks = len(all_tasks)
    logger.info(f"Running {total_tasks} total tasks with {concurrent_calls} concurrent calls")

    # Run all tasks in parallel batches
    all_results = Parallel(n_jobs=concurrent_calls)(
        delayed(run_query)(method, query, model_id, config) for method, config, query, iteration in all_tasks
    )

    # Add iteration information and model_id to results
    for (method, config, query, iteration), result in zip(all_tasks, all_results):
        result["iteration"] = iteration
        result["model_id"] = model_id
        results.append(result)

    # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Add method_config column for easier plotting
    df["method_config"] = df["method"] + " - " + df["performance_config"]

    # Calculate additional statistics
    stats_df = (
        df.groupby(["method", "performance_config", "query"])
        .agg(
            {
                "tokens_per_second": ["mean", "std", "min", "max"],
                "processing_time": ["mean", "std"],
                "token_count": ["mean", "sum"],
                "success": "mean",
            }
        )
        .round(2)
    )

    # Save detailed statistics
    stats_df.to_csv(os.path.join(OUTPUT_DIR, f"detailed_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"))

    return df


def plot_results(df: pd.DataFrame):
    """Create comprehensive performance visualization plots."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Color palette
    aws_orange = "#FF9900"
    aws_dark_orange = "#FF6600"
    langchain_colors = ["#4CAF50", "#2E7D32"]
    methods = ["langchain", "boto3"]
    perf_configs = ["standard", "optimized"]
    colors = [langchain_colors[0], langchain_colors[1], aws_orange, aws_dark_orange]

    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Plot 1: Bar plot with error bars
    ax1 = fig.add_subplot(gs[0, :])

    # Calculate average tokens per second for each query to determine order
    avg_tokens_per_sec = df.groupby("query")["tokens_per_second"].mean().sort_values(ascending=True)
    queries = avg_tokens_per_sec.index.tolist()  # Queries sorted by performance
    x = np.arange(len(queries))
    bar_width = 0.2

    for i, method in enumerate(methods):
        for j, config in enumerate(perf_configs):
            stats = (
                df[(df["method"] == method) & (df["performance_config"] == config)]
                .groupby("query")
                .agg({"tokens_per_second": ["mean", "std"]})
            )
            # Ensure stats are ordered by average performance
            stats = stats.reindex(queries)

            ax1.bar(
                x + (i * len(perf_configs) + j) * bar_width,
                stats["tokens_per_second"]["mean"],
                bar_width,
                yerr=stats["tokens_per_second"]["std"],
                label=f"{method} - {config}",
                color=colors[i * len(perf_configs) + j],
                alpha=0.8,
                capsize=5,
            )

    ax1.set_xlabel("Queries")
    ax1.set_ylabel("Tokens per Second")
    ax1.set_title("Performance Comparison by Query")
    ax1.set_xticks(x + bar_width * 1.5)
    ax1.set_xticklabels(queries, rotation=0, ha="center", fontsize=9)
    ax1.legend(title="Method - Configuration")

    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.02, 0.02, f"Generated: {timestamp}", fontsize=8)

    # Adjust layout and save
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "benchmark_results.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()  # DO NOT REMOVE THIS
    plt.close()

    logger.info(f"Plots saved to {output_path}")


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

    methods = [
        "langchain",
        "boto3",
    ]
    configs = [
        "standard",
        "optimized",
    ]

    df = run_benchmarks(queries, model_id, iterations, concurrent_calls, configs, methods)
    # print(df)
    # print(df[["method", "performance_config", "tokens_per_second"]])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"unified_benchmark_results_{timestamp}.txt"
    df.to_csv(os.path.join(OUTPUT_DIR, output_file), index=False)
    logger.info(f"Results saved to {os.path.join(OUTPUT_DIR, output_file)}")

    # Plot immediately after benchmarks
    plot_results(df)
    logger.info("Benchmark and plotting completed.")


if __name__ == "__main__":
    main()
