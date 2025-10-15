"""
Multi-Execution Hamming Distance and Cluster Analysis Script

Runs clustering analysis over multiple random feature selections and aggregates statistics.
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from rich import print
from rich.table import Table
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
import argparse
from pathlib import Path
import json
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict


@dataclass
class ClusterMetrics:
    """Stores metrics for a single clustering execution"""
    main_feature_idx: int
    hamming_min: int
    hamming_max: int
    hamming_mean: float
    hamming_std: float
    mean_within_cluster: float
    mean_between_cluster: float
    separation_ratio: float
    total_inertia: float
    cluster_sizes: List[int]
    cluster_balance_std: float  # Standard deviation of cluster sizes


@dataclass
class AggregatedMetrics:
    """Stores aggregated metrics across multiple executions"""
    n_executions: int
    hamming_min_mean: float
    hamming_min_std: float
    hamming_max_mean: float
    hamming_max_std: float
    hamming_range_mean: float
    hamming_range_std: float
    mean_within_cluster_mean: float
    mean_within_cluster_std: float
    mean_between_cluster_mean: float
    mean_between_cluster_std: float
    separation_ratio_mean: float
    separation_ratio_std: float
    total_inertia_mean: float
    total_inertia_std: float
    cluster_balance_mean: float
    cluster_balance_std: float


def compute_hamming_distance(vec1, vec2):
    """Compute hamming distance between two binary vectors."""
    return (vec1 != vec2).sum().item()


def find_top_similar_features(matrix, main_feature_idx, n_features=128):
    """
    Find the top N most similar features to a selected feature based on hamming distance.

    Returns:
        tuple: (selected_features matrix, sorted_indices, hamming_distances)
    """
    main_feature = matrix[:, main_feature_idx]
    n_total_features = matrix.shape[1]

    # Compute hamming distances to all other features
    hamming_distances = []
    for i in range(n_total_features):
        if i == main_feature_idx:
            continue
        dist = compute_hamming_distance(main_feature, matrix[:, i])
        hamming_distances.append((i, dist))

    # Sort by hamming distance (ascending)
    hamming_distances.sort(key=lambda x: x[1])

    # Take top N most similar
    top_indices = [idx for idx, _ in hamming_distances[:n_features]]
    top_distances = [dist for _, dist in hamming_distances[:n_features]]

    # Build feature subset
    selected_features = torch.cat([
        main_feature.unsqueeze(1),
        matrix[:, top_indices]
    ], dim=1)

    return selected_features, [main_feature_idx] + top_indices, [0] + top_distances


def compute_absolute_cluster_distances(data, labels, cluster_centers):
    """Compute absolute mean within-cluster and between-cluster distances."""
    n_clusters = len(np.unique(labels))

    # Within-cluster distances
    within_cluster_distances = []
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_points = data[cluster_mask]
        cluster_center = cluster_centers[cluster_id]
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        within_cluster_distances.extend(distances)

    mean_within_cluster = np.mean(within_cluster_distances)

    # Between-cluster distances
    if n_clusters > 1:
        between_distances = pairwise_distances(cluster_centers, metric='euclidean')
        between_cluster_distances = between_distances[np.triu_indices(n_clusters, k=1)]
        mean_between_cluster = np.mean(between_cluster_distances)
    else:
        mean_between_cluster = 0.0

    return {
        'mean_within_cluster': mean_within_cluster,
        'mean_between_cluster': mean_between_cluster,
    }


def run_single_analysis(binary_matrix, main_feature_idx, n_features, n_clusters, random_seed):
    """Run a single clustering analysis on a random feature."""

    # Find top similar features by hamming distance
    selected_features, feature_indices, hamming_distances = find_top_similar_features(
        binary_matrix, main_feature_idx, n_features
    )

    # Perform k-means clustering
    features_as_points = selected_features.T.cpu().numpy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(features_as_points)

    # Compute cluster distances
    distance_metrics = compute_absolute_cluster_distances(
        features_as_points, labels, kmeans.cluster_centers_
    )

    # Compute cluster size statistics
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_balance_std = np.std(counts)

    # Calculate separation ratio
    separation_ratio = 0.0
    if distance_metrics['mean_within_cluster'] > 0:
        separation_ratio = (distance_metrics['mean_between_cluster'] /
                          distance_metrics['mean_within_cluster'])

    # Hamming distance statistics (excluding the reference feature at index 0)
    hamming_array = np.array(hamming_distances[1:])  # Skip the 0 distance to self

    return ClusterMetrics(
        main_feature_idx=main_feature_idx,
        hamming_min=int(hamming_array.min()),
        hamming_max=int(hamming_array.max()),
        hamming_mean=float(hamming_array.mean()),
        hamming_std=float(hamming_array.std()),
        mean_within_cluster=distance_metrics['mean_within_cluster'],
        mean_between_cluster=distance_metrics['mean_between_cluster'],
        separation_ratio=separation_ratio,
        total_inertia=kmeans.inertia_,
        cluster_sizes=counts.tolist(),
        cluster_balance_std=cluster_balance_std
    )


def aggregate_metrics(metrics_list: List[ClusterMetrics]) -> AggregatedMetrics:
    """Aggregate metrics across multiple executions."""
    n = len(metrics_list)

    hamming_mins = [m.hamming_min for m in metrics_list]
    hamming_maxs = [m.hamming_max for m in metrics_list]
    hamming_ranges = [m.hamming_max - m.hamming_min for m in metrics_list]

    return AggregatedMetrics(
        n_executions=n,
        hamming_min_mean=np.mean(hamming_mins),
        hamming_min_std=np.std(hamming_mins),
        hamming_max_mean=np.mean(hamming_maxs),
        hamming_max_std=np.std(hamming_maxs),
        hamming_range_mean=np.mean(hamming_ranges),
        hamming_range_std=np.std(hamming_ranges),
        mean_within_cluster_mean=np.mean([m.mean_within_cluster for m in metrics_list]),
        mean_within_cluster_std=np.std([m.mean_within_cluster for m in metrics_list]),
        mean_between_cluster_mean=np.mean([m.mean_between_cluster for m in metrics_list]),
        mean_between_cluster_std=np.std([m.mean_between_cluster for m in metrics_list]),
        separation_ratio_mean=np.mean([m.separation_ratio for m in metrics_list]),
        separation_ratio_std=np.std([m.separation_ratio for m in metrics_list]),
        total_inertia_mean=np.mean([m.total_inertia for m in metrics_list]),
        total_inertia_std=np.std([m.total_inertia for m in metrics_list]),
        cluster_balance_mean=np.mean([m.cluster_balance_std for m in metrics_list]),
        cluster_balance_std=np.std([m.cluster_balance_std for m in metrics_list])
    )


def display_aggregated_results(agg_metrics: AggregatedMetrics, matrix_name: str, console: Console):
    """Display aggregated results in a formatted table."""

    console.print(f"\n{'='*80}")
    console.print(f"[bold cyan]AGGREGATED METRICS: {matrix_name}[/bold cyan]")
    console.print(f"[dim]Based on {agg_metrics.n_executions} random feature selections[/dim]")
    console.print(f"{'='*80}\n")

    # Create metrics table
    table = Table(show_header=True, header_style="bold blue", title="Clustering Statistics")
    table.add_column("Metric", style="cyan", width=40)
    table.add_column("Mean", justify="right", width=15)
    table.add_column("Std Dev", justify="right", width=15)

    # Hamming distances
    table.add_section()
    table.add_row("[bold]Hamming Distance Metrics[/bold]", "", "")
    table.add_row(
        "  Min Hamming Distance",
        f"{agg_metrics.hamming_min_mean:.2f}",
        f"± {agg_metrics.hamming_min_std:.2f}"
    )
    table.add_row(
        "  Max Hamming Distance",
        f"{agg_metrics.hamming_max_mean:.2f}",
        f"± {agg_metrics.hamming_max_std:.2f}"
    )
    table.add_row(
        "  Hamming Range",
        f"{agg_metrics.hamming_range_mean:.2f}",
        f"± {agg_metrics.hamming_range_std:.2f}"
    )

    # Cluster distances
    table.add_section()
    table.add_row("[bold]Cluster Distance Metrics[/bold]", "", "")
    table.add_row(
        "  Mean Within-Cluster Distance",
        f"{agg_metrics.mean_within_cluster_mean:.4f}",
        f"± {agg_metrics.mean_within_cluster_std:.4f}"
    )
    table.add_row(
        "  Mean Between-Cluster Distance",
        f"{agg_metrics.mean_between_cluster_mean:.4f}",
        f"± {agg_metrics.mean_between_cluster_std:.4f}"
    )
    table.add_row(
        "  Separation Ratio",
        f"{agg_metrics.separation_ratio_mean:.4f}",
        f"± {agg_metrics.separation_ratio_std:.4f}",
        style="yellow" if agg_metrics.separation_ratio_mean > 1.0 else "red"
    )

    # Other metrics
    table.add_section()
    table.add_row("[bold]Other Metrics[/bold]", "", "")
    table.add_row(
        "  Total Inertia",
        f"{agg_metrics.total_inertia_mean:.2f}",
        f"± {agg_metrics.total_inertia_std:.2f}"
    )
    table.add_row(
        "  Cluster Balance (std of sizes)",
        f"{agg_metrics.cluster_balance_mean:.2f}",
        f"± {agg_metrics.cluster_balance_std:.2f}"
    )

    console.print(table)

    # Quality assessment
    console.print(f"\n[bold]Quality Assessment:[/bold]")
    if agg_metrics.separation_ratio_mean > 2.0:
        console.print("  [green]✓ Excellent cluster separation[/green]")
    elif agg_metrics.separation_ratio_mean > 1.0:
        console.print("  [yellow]○ Good cluster separation[/yellow]")
    elif agg_metrics.separation_ratio_mean > 0.5:
        console.print("  [yellow]○ Moderate cluster separation[/yellow]")
    else:
        console.print("  [red]✗ Poor cluster separation[/red]")

    if agg_metrics.cluster_balance_mean < 10:
        console.print("  [green]✓ Well-balanced clusters[/green]")
    elif agg_metrics.cluster_balance_mean < 20:
        console.print("  [yellow]○ Moderately balanced clusters[/yellow]")
    else:
        console.print("  [red]✗ Imbalanced clusters[/red]")

    console.print(f"{'='*80}\n")


def run_multi_execution_analysis(
    weight_matrix_path: str,
    n_executions: int = 10,
    n_features: int = 128,
    n_clusters: int = 8,
    random_seed: int = 42,
    output_json: str = None
):
    """
    Run clustering analysis over multiple random feature selections.
    """
    console = Console()

    # Load weight matrix
    console.print(f"\n[bold cyan]Loading weight matrix:[/bold cyan] {weight_matrix_path}")
    matrix = torch.load(weight_matrix_path)
    console.print(f"[green]Matrix shape:[/green] {matrix.shape}")

    # Convert to binary
    binary_matrix = (matrix.abs() > 0).int()

    # Run multiple executions
    metrics_list = []
    np.random.seed(random_seed)

    console.print(f"\n[bold cyan]Running {n_executions} executions with random feature selections...[/bold cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]Analyzing...", total=n_executions)

        for i in range(n_executions):
            # Select random feature
            main_feature_idx = np.random.randint(0, binary_matrix.shape[1])

            # Run analysis
            metrics = run_single_analysis(
                binary_matrix,
                main_feature_idx,
                n_features,
                n_clusters,
                random_seed + i  # Different seed for each k-means
            )
            metrics_list.append(metrics)

            progress.update(task, advance=1)

    # Aggregate metrics
    agg_metrics = aggregate_metrics(metrics_list)

    # Display results
    matrix_name = Path(weight_matrix_path).stem
    display_aggregated_results(agg_metrics, matrix_name, console)

    # Save to JSON if requested
    if output_json:
        output_data = {
            'matrix_path': weight_matrix_path,
            'matrix_name': matrix_name,
            'matrix_shape': list(matrix.shape),
            'n_executions': n_executions,
            'n_features': n_features,
            'n_clusters': n_clusters,
            'aggregated_metrics': asdict(agg_metrics),
            'individual_metrics': [asdict(m) for m in metrics_list]
        }

        with open(output_json, 'w') as f:
            json.dump(output_data, f, indent=2)

        console.print(f"[green]✓ Results saved to:[/green] {output_json}")

    return agg_metrics, metrics_list


def main():
    parser = argparse.ArgumentParser(
        description="Multi-execution hamming distance and clustering analysis"
    )
    parser.add_argument(
        'matrix_path',
        type=str,
        help='Path to .pt file containing weight matrix'
    )
    parser.add_argument(
        '--n-executions',
        type=int,
        default=10,
        help='Number of random feature selections to analyze (default: 10)'
    )
    parser.add_argument(
        '--n-features',
        type=int,
        default=128,
        help='Number of similar features to analyze per execution (default: 128)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=8,
        help='Number of k-means clusters (default: 8)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--output-json',
        type=str,
        default=None,
        help='Path to save JSON output (optional)'
    )

    args = parser.parse_args()

    # Validate file path
    if not Path(args.matrix_path).exists():
        print(f"[bold red]Error:[/bold red] File not found: {args.matrix_path}")
        return

    # Run analysis
    run_multi_execution_analysis(
        weight_matrix_path=args.matrix_path,
        n_executions=args.n_executions,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        random_seed=args.seed,
        output_json=args.output_json
    )


if __name__ == "__main__":
    main()
