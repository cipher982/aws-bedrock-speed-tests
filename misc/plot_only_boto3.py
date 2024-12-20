import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Define output directory constant
OUTPUT_DIR = "outputs"

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def plot_results(df: pd.DataFrame):
    """Create comprehensive performance visualization plots with subplots for each model."""
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Color palette
    aws_orange = "#FF9900"
    aws_dark_orange = "#FF6600"
    # langchain_colors = ["#4CAF50", "#2E7D32"]
    methods = [
        # "langchain",
        "boto3",
    ]
    perf_configs = ["standard", "optimized"]
    # colors = [langchain_colors[0], langchain_colors[1], aws_orange, aws_dark_orange]
    colors = [aws_orange, aws_dark_orange]

    # Get unique models
    models = df["model_id"].unique()

    # Create figure with subplots - one for each model
    fig = plt.figure(figsize=(11, 3 * len(models) + 0.5))  # Add extra height for titles
    fig.suptitle("Benchmarking Optimized Endpoints for AWS Bedrock", fontsize=18)
    fig.text(
        0.5,
        0.94,
        "Measuring Tokens per Second, N=5 each query",
        ha="center",
        fontsize=12,
    )
    gs = fig.add_gridspec(len(models), 1, top=0.9)  # Adjust top margin

    for model_idx, model in enumerate(models):
        model_df = df[df["model_id"] == model]

        # Calculate average tokens per second for each query to determine order
        avg_tokens_per_sec = model_df.groupby("query")["tokens_per_second"].mean().sort_values(ascending=True)
        queries = avg_tokens_per_sec.index.tolist()  # Queries sorted by performance
        x = np.arange(len(queries))
        bar_width = 0.2

        # Create subplot for this model
        ax = fig.add_subplot(gs[model_idx])

        # Track unique labels to prevent duplicates in legend
        unique_labels = set()

        for i, method in enumerate(methods):
            for j, config in enumerate(perf_configs):
                # Commented out Langchain-specific plotting
                # if method == "langchain":
                #     continue
                stats = (
                    model_df[(model_df["method"] == method) & (model_df["performance_config"] == config)]
                    .groupby("query")
                    .agg({"tokens_per_second": ["mean", "std"]})
                )
                # Ensure stats are ordered by average performance
                stats = stats.reindex(queries)

                for query in queries:
                    # Calculate average tokens per run for this method, config, and query
                    _ = model_df[
                        (model_df["method"] == method)
                        & (model_df["performance_config"] == config)
                        & (model_df["query"] == query)
                    ]["token_count"].mean()

                    _ = ax.bar(
                        x[queries.index(query)] + (i * len(perf_configs) + j) * bar_width,
                        stats.loc[query]["tokens_per_second"]["mean"],
                        bar_width,
                        yerr=stats.loc[query]["tokens_per_second"]["std"],
                        label=config if config not in unique_labels else None,
                        color=colors[i * len(perf_configs) + j],
                        alpha=0.8,
                        capsize=5,
                    )

                    # # Add average tokens per run above each bar
                    # for bar in bars:
                    #     height = bar.get_height()
                    #     ax.text(
                    #         bar.get_x() + bar.get_width() / 2.0,
                    #         height,
                    #         f"{avg_tokens_per_run:.0f} tokens",
                    #         ha="left",
                    #         va="bottom",
                    #         fontsize=8,
                    #     )

                    unique_labels.add(config)

        ax.set_xlabel("Queries")
        ax.set_ylabel("Tokens per Second")
        ax.set_title(model)
        ax.set_xticks(x + bar_width * 1.5)
        ax.set_xticklabels(queries, rotation=0, ha="right", fontsize=9)
        ax.legend(title="Configuration", loc="upper left")

    # Add timestamp
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # fig.text(0.02, 0.02, f"Generated: {timestamp}", fontsize=8)

    # Adjust layout and save
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # Add more vertical spacing between subplots
    output_path = os.path.join(OUTPUT_DIR, "benchmark_results.png")
    plt.savefig(output_path, dpi=450, bbox_inches="tight")
    plt.show()  # DO NOT REMOVE THIS
    plt.close()

    logger.info(f"Plots saved to {output_path}")
