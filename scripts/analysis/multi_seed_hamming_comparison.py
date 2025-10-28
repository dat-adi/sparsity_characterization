#!/usr/bin/env python3
"""
Multi-Seed Hamming Distance Analysis: Wanda vs SparseGPT Comparison

For each pruning method (Wanda and SparseGPT):
1. Across 10 random seeds, select a random feature vector
2. Find the 128 most similar features by Hamming distance
3. Apply k-means clustering (k=4, 8, 16)
4. Compute within-cluster, between-cluster distances, and separation ratios
5. Aggregate results across all seeds
6. Generate side-by-side comparison visualizations

Usage:
    python multi_seed_hamming_comparison.py
"""

import sys
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from rich.console import Console
from rich.table import Table
from rich.progress import track
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import find_most_similar_features
from utils.clustering import compute_cluster_metrics, ClusterMetrics

console = Console()


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple seeds."""
    method_name: str
    matrix_name: str
    k: int
    n_seeds: int

    # Mean and std of within-cluster distances
    mean_within_mean: float
    mean_within_std: float

    # Mean and std of between-cluster distances
    mean_between_mean: float
    mean_between_std: float

    # Mean and std of separation ratios
    separation_ratio_mean: float
    separation_ratio_std: float

    # All individual measurements
    within_distances: List[float]
    between_distances: List[float]
    separation_ratios: List[float]


def run_single_seed_analysis(
    matrix: torch.Tensor,
    seed: int,
    n_features: int = 128,
    k_values: List[int] = [4, 8, 16]
) -> Dict[int, ClusterMetrics]:
    """
    Run clustering analysis for a single seed.

    Args:
        matrix: Weight matrix [D, N]
        seed: Random seed for reproducibility
        n_features: Number of similar features to find
        k_values: List of k values for clustering

    Returns:
        Dictionary mapping k -> ClusterMetrics
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Randomly select a feature
    n_total_features = matrix.shape[1]
    selected_feature_idx = random.randint(0, n_total_features - 1)

    # Find most similar features
    subset, indices, distances = find_most_similar_features(
        matrix, selected_feature_idx, n_features
    )

    # Perform clustering for each k
    results = {}
    for k in k_values:
        metrics = compute_cluster_metrics(subset, k, random_state=seed)
        results[k] = metrics

    return results


def aggregate_seed_results(
    seed_results: List[Dict[int, ClusterMetrics]],
    method_name: str,
    matrix_name: str
) -> Dict[int, AggregatedMetrics]:
    """
    Aggregate results across multiple seeds.

    Args:
        seed_results: List of results from each seed
        method_name: Name of the pruning method (Wanda/SparseGPT)
        matrix_name: Name of the matrix being analyzed

    Returns:
        Dictionary mapping k -> AggregatedMetrics
    """
    k_values = list(seed_results[0].keys())
    aggregated = {}

    for k in k_values:
        within_distances = [result[k].mean_within for result in seed_results]
        between_distances = [result[k].mean_between for result in seed_results]
        separation_ratios = [
            result[k].mean_between / result[k].mean_within
            if result[k].mean_within > 0 else 0
            for result in seed_results
        ]

        aggregated[k] = AggregatedMetrics(
            method_name=method_name,
            matrix_name=matrix_name,
            k=k,
            n_seeds=len(seed_results),
            mean_within_mean=np.mean(within_distances),
            mean_within_std=np.std(within_distances),
            mean_between_mean=np.mean(between_distances),
            mean_between_std=np.std(between_distances),
            separation_ratio_mean=np.mean(separation_ratios),
            separation_ratio_std=np.std(separation_ratios),
            within_distances=within_distances,
            between_distances=between_distances,
            separation_ratios=separation_ratios
        )

    return aggregated


def create_comparison_table(
    wanda_agg: Dict[int, AggregatedMetrics],
    sparsegpt_agg: Dict[int, AggregatedMetrics],
    matrix_name: str
) -> Table:
    """Create a rich table comparing Wanda and SparseGPT results."""
    table = Table(title=f"Clustering Metrics Comparison: {matrix_name}", show_header=True)

    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("k", justify="center")
    table.add_column("Wanda", justify="right", style="green")
    table.add_column("SparseGPT", justify="right", style="magenta")
    table.add_column("Δ (W-S)", justify="right", style="yellow")

    k_values = sorted(wanda_agg.keys())

    for k in k_values:
        w = wanda_agg[k]
        s = sparsegpt_agg[k]

        # Within-cluster distance
        table.add_row(
            "Within-cluster",
            str(k),
            f"{w.mean_within_mean:.4f} ± {w.mean_within_std:.4f}",
            f"{s.mean_within_mean:.4f} ± {s.mean_within_std:.4f}",
            f"{w.mean_within_mean - s.mean_within_mean:+.4f}"
        )

        # Between-cluster distance
        table.add_row(
            "Between-cluster",
            str(k),
            f"{w.mean_between_mean:.4f} ± {w.mean_between_std:.4f}",
            f"{s.mean_between_mean:.4f} ± {s.mean_between_std:.4f}",
            f"{w.mean_between_mean - s.mean_between_mean:+.4f}"
        )

        # Separation ratio
        table.add_row(
            "Separation ratio",
            str(k),
            f"{w.separation_ratio_mean:.4f} ± {w.separation_ratio_std:.4f}",
            f"{s.separation_ratio_mean:.4f} ± {s.separation_ratio_std:.4f}",
            f"{w.separation_ratio_mean - s.separation_ratio_mean:+.4f}",
            end_section=(k == k_values[-1])
        )

    return table


def plot_comparison_visualizations(
    wanda_results: Dict[str, Dict[int, AggregatedMetrics]],
    sparsegpt_results: Dict[str, Dict[int, AggregatedMetrics]],
    output_dir: Path
):
    """
    Create comprehensive comparison visualizations.

    Args:
        wanda_results: Dictionary mapping matrix_name -> k -> AggregatedMetrics for Wanda
        sparsegpt_results: Same for SparseGPT
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300

    matrix_names = list(wanda_results.keys())
    k_values = sorted(list(wanda_results[matrix_names[0]].keys()))

    # 1. Separation Ratio Comparison
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, matrix_name in enumerate(matrix_names):
        if idx >= len(axes):
            break

        ax = axes[idx]

        w_ratios = [wanda_results[matrix_name][k].separation_ratio_mean for k in k_values]
        w_stds = [wanda_results[matrix_name][k].separation_ratio_std for k in k_values]
        s_ratios = [sparsegpt_results[matrix_name][k].separation_ratio_mean for k in k_values]
        s_stds = [sparsegpt_results[matrix_name][k].separation_ratio_std for k in k_values]

        x = np.arange(len(k_values))
        width = 0.35

        ax.bar(x - width/2, w_ratios, width, yerr=w_stds, label='Wanda',
               alpha=0.8, color='#2E86AB', capsize=5)
        ax.bar(x + width/2, s_ratios, width, yerr=s_stds, label='SparseGPT',
               alpha=0.8, color='#A23B72', capsize=5)

        ax.set_ylabel('Separation Ratio')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_title(matrix_name.replace('layer1-', ''), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values])
        ax.legend(fontsize=8)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for idx in range(len(matrix_names), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(output_dir / 'separation_ratio_comparison.png', bbox_inches='tight')
    plt.close()
    console.print(f"[green]✓[/green] Saved: {output_dir / 'separation_ratio_comparison.png'}")

    # 2. Within vs Between Scatter Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        for matrix_name in matrix_names:
            w = wanda_results[matrix_name][k]
            s = sparsegpt_results[matrix_name][k]

            # Wanda
            ax.scatter(w.mean_within_mean, w.mean_between_mean, s=100, alpha=0.6,
                      color='#2E86AB', marker='o', edgecolors='black', linewidths=0.5)

            # SparseGPT
            ax.scatter(s.mean_within_mean, s.mean_between_mean, s=100, alpha=0.6,
                      color='#A23B72', marker='s', edgecolors='black', linewidths=0.5)

        # Diagonal line
        max_val = max(
            max([wanda_results[m][k].mean_between_mean for m in matrix_names]),
            max([sparsegpt_results[m][k].mean_between_mean for m in matrix_names])
        )
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Ratio=1')

        ax.set_xlabel('Mean Within-Cluster Distance')
        ax.set_ylabel('Mean Between-Cluster Distance')
        ax.set_title(f'k={k}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        if idx == 2:
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#2E86AB', alpha=0.6, label='Wanda'),
                Patch(facecolor='#A23B72', alpha=0.6, label='SparseGPT')
            ]
            ax.legend(handles=legend_elements, loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / 'within_vs_between_scatter.png', bbox_inches='tight')
    plt.close()
    console.print(f"[green]✓[/green] Saved: {output_dir / 'within_vs_between_scatter.png'}")

    # 3. Box plots showing variance across seeds
    fig, axes = plt.subplots(len(k_values), 3, figsize=(15, 4*len(k_values)))

    metrics_labels = ['Within-Cluster Distance', 'Between-Cluster Distance', 'Separation Ratio']

    for k_idx, k in enumerate(k_values):
        for metric_idx, metric_name in enumerate(['within_distances', 'between_distances', 'separation_ratios']):
            ax = axes[k_idx, metric_idx] if len(k_values) > 1 else axes[metric_idx]

            data_to_plot = []
            labels = []

            for matrix_name in matrix_names:
                w_data = getattr(wanda_results[matrix_name][k], metric_name)
                s_data = getattr(sparsegpt_results[matrix_name][k], metric_name)

                data_to_plot.extend([w_data, s_data])
                labels.extend([f"{matrix_name[:10]}\n(W)", f"{matrix_name[:10]}\n(S)"])

            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)

            # Color boxes
            for patch_idx, patch in enumerate(bp['boxes']):
                if patch_idx % 2 == 0:  # Wanda
                    patch.set_facecolor('#2E86AB')
                    patch.set_alpha(0.6)
                else:  # SparseGPT
                    patch.set_facecolor('#A23B72')
                    patch.set_alpha(0.6)

            ax.set_ylabel(metrics_labels[metric_idx])
            ax.set_title(f'k={k}')
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45, labelsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'variance_across_seeds.png', bbox_inches='tight')
    plt.close()
    console.print(f"[green]✓[/green] Saved: {output_dir / 'variance_across_seeds.png'}")


def save_results_json(
    wanda_results: Dict[str, Dict[int, AggregatedMetrics]],
    sparsegpt_results: Dict[str, Dict[int, AggregatedMetrics]],
    output_path: Path
):
    """Save aggregated results to JSON file."""

    def metrics_to_dict(agg: AggregatedMetrics) -> dict:
        return {
            'method_name': agg.method_name,
            'matrix_name': agg.matrix_name,
            'k': agg.k,
            'n_seeds': agg.n_seeds,
            'mean_within_mean': float(agg.mean_within_mean),
            'mean_within_std': float(agg.mean_within_std),
            'mean_between_mean': float(agg.mean_between_mean),
            'mean_between_std': float(agg.mean_between_std),
            'separation_ratio_mean': float(agg.separation_ratio_mean),
            'separation_ratio_std': float(agg.separation_ratio_std)
        }

    results = {
        'wanda': {
            matrix_name: {
                k: metrics_to_dict(agg)
                for k, agg in k_results.items()
            }
            for matrix_name, k_results in wanda_results.items()
        },
        'sparsegpt': {
            matrix_name: {
                k: metrics_to_dict(agg)
                for k, agg in k_results.items()
            }
            for matrix_name, k_results in sparsegpt_results.items()
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    console.print(f"[green]✓[/green] Saved: {output_path}")


def main():
    """Main execution function."""

    console.print("\n[bold cyan]Multi-Seed Hamming Distance Analysis: Wanda vs SparseGPT[/bold cyan]\n")

    # Configuration
    wanda_dir = Path("../../data/wanda_unstructured/layer-1")
    sparsegpt_dir = Path("../../data/sparsegpt_unstructured/layer-1")
    output_dir = Path("../../results/metrics/multi_seed_hamming_comparison")

    matrix_files = [
        "layer1-mlp.down_proj.pt",
        "layer1-mlp.up_proj.pt",
        "layer1-mlp.gate_proj.pt",
        "layer1-self_attn.q_proj.pt",
        "layer1-self_attn.k_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt",
    ]

    n_seeds = 10
    n_features = 128
    k_values = [4, 8, 16]

    console.print(f"Configuration:")
    console.print(f"  • Seeds: {n_seeds}")
    console.print(f"  • Features per seed: {n_features}")
    console.print(f"  • k values: {k_values}")
    console.print(f"  • Matrices: {len(matrix_files)}\n")

    # Storage for results
    wanda_results = {}
    sparsegpt_results = {}

    # Analyze each matrix
    for matrix_file in track(matrix_files, description="Analyzing matrices..."):
        matrix_name = matrix_file.replace('.pt', '')

        console.print(f"\n[bold]Processing: {matrix_name}[/bold]")

        # Load matrices
        wanda_matrix = torch.load(wanda_dir / matrix_file, weights_only=True)
        sparsegpt_matrix = torch.load(sparsegpt_dir / matrix_file, weights_only=True)

        # Run analysis across seeds for Wanda
        console.print("  Running Wanda analysis across seeds...")
        wanda_seed_results = []
        for seed in range(n_seeds):
            result = run_single_seed_analysis(wanda_matrix, seed, n_features, k_values)
            wanda_seed_results.append(result)

        # Aggregate Wanda results
        wanda_agg = aggregate_seed_results(wanda_seed_results, "Wanda", matrix_name)
        wanda_results[matrix_name] = wanda_agg

        # Run analysis across seeds for SparseGPT
        console.print("  Running SparseGPT analysis across seeds...")
        sparsegpt_seed_results = []
        for seed in range(n_seeds):
            result = run_single_seed_analysis(sparsegpt_matrix, seed, n_features, k_values)
            sparsegpt_seed_results.append(result)

        # Aggregate SparseGPT results
        sparsegpt_agg = aggregate_seed_results(sparsegpt_seed_results, "SparseGPT", matrix_name)
        sparsegpt_results[matrix_name] = sparsegpt_agg

        # Display comparison table
        table = create_comparison_table(wanda_agg, sparsegpt_agg, matrix_name)
        console.print(table)

    # Generate visualizations
    console.print("\n[bold cyan]Generating Comparison Visualizations[/bold cyan]\n")
    plot_comparison_visualizations(wanda_results, sparsegpt_results, output_dir)

    # Save results
    console.print("\n[bold cyan]Saving Results[/bold cyan]\n")
    save_results_json(
        wanda_results,
        sparsegpt_results,
        output_dir / 'aggregated_metrics.json'
    )

    # Summary statistics
    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print(f"\nResults saved to: [cyan]{output_dir}[/cyan]")
    console.print("\nGenerated files:")
    console.print("  • separation_ratio_comparison.png")
    console.print("  • within_vs_between_scatter.png")
    console.print("  • variance_across_seeds.png")
    console.print("  • aggregated_metrics.json")


if __name__ == "__main__":
    main()
