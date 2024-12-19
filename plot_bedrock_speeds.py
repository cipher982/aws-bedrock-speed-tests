import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set the style for a modern look
sns.set_palette("husl")

# Find the most recent benchmark results file
pattern = os.path.join(os.path.dirname(__file__), "unified_benchmark_results_*.txt")
files = glob.glob(pattern)
latest_file = max(files, key=os.path.getmtime)

# Read the benchmark results from the latest file
df = pd.read_csv(latest_file)

# Create figure and axis with a specific size
plt.figure(figsize=(16, 8))

# Prepare data
grouped_data = df.groupby(["Query", "PerfConfig", "Method"])["Tokens/s"].mean().reset_index()

# Create the grouped bar chart
bar_width = 0.2
queries = df["Query"].unique()
x = range(len(queries))

# AWS Color Palette
aws_orange = "#FF9900"  # Amazon's primary orange
aws_dark_orange = "#FF6600"
langchain_colors = ["#4CAF50", "#2E7D32"]  # Green shades for Langchain

# Plot bars for each configuration and method
methods = ["Langchain", "Boto3"]
perf_configs = ["standard", "optimized"]
colors = [
    langchain_colors[0],  # Langchain Standard
    langchain_colors[1],  # Langchain Optimized
    aws_orange,  # Boto3 Standard
    aws_dark_orange,  # Boto3 Optimized
]

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

# Customize the plot
plt.xlabel("Queries", fontsize=12)
plt.ylabel("Tokens per Second", fontsize=12)
plt.title(
    "Claude 3.5 Haiku Performance Comparison\nLangchain vs Boto3 - Standard vs Optimized Configuration",
    fontsize=14,
    pad=20,
)

# Set x-axis labels with reduced bottom margin
plt.xticks(x, queries, rotation=0, ha="center", fontsize=9)
plt.tight_layout()


# Add legend with adjusted positioning
plt.legend(title="Method - Configuration", loc="upper left", bbox_transform=plt.gcf().transFigure)

# Save the plot
plt.savefig("benchmark_comparison_methods.png", dpi=300, bbox_inches="tight")
print("Plot saved as 'benchmark_comparison_methods.png'")

plt.show()
