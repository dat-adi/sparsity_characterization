"""
Hamming Distance and Cluster Analysis Script

This script:
1. Selects a feature vector from a weight matrix
2. Finds the top 128 most similar features by hamming distance
3. Prints all hamming distances between the selected vector and top 128 similar ones
4. Performs k-means clustering
5. Computes and displays absolute mean within-cluster and between-cluster distances
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from rich import print
from rich.table import Table
from rich.console import Console
import argparse
from pathlib import Path


def compute_hamming_distance(vec1, vec2):
    """
    Compute hamming distance between two binary vectors.

    Args:
        vec1: Binary tensor
        vec2: Binary tensor

    Returns:
        Integer count of differing positions
    """
    return (vec1 != vec2).sum().item()


def find_top_similar_features(matrix, main_feature_idx, n_features=128):
    """
    Find the top N most similar features to a selected feature based on hamming distance.

    Args:
        matrix: Binary weight matrix (samples x features)
        main_feature_idx: Index of the feature to compare against
        n_features: Number of similar features to return (default: 128)

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

    # Sort by hamming distance (ascending - most similar first)
    hamming_distances.sort(key=lambda x: x[1])

    # Take top N most similar
    top_indices = [idx for idx, _ in hamming_distances[:n_features]]
    top_distances = [dist for _, dist in hamming_distances[:n_features]]

    # Build feature subset: main feature + top N similar features
    selected_features = torch.cat([
        main_feature.unsqueeze(1),
        matrix[:, top_indices]
    ], dim=1)

    return selected_features, [main_feature_idx] + top_indices, [0] + top_distances


def compute_absolute_cluster_distances(data, labels, cluster_centers):
    """
    Compute absolute mean within-cluster and between-cluster distances.

    Args:
        data: Feature matrix (n_features x n_samples)
        labels: Cluster assignments for each feature
        cluster_centers: K-means cluster centers

    Returns:
        dict: Contains mean_within_cluster and mean_between_cluster distances
    """
    n_clusters = len(np.unique(labels))

    # Compute within-cluster distances (mean distance from each point to its cluster center)
    within_cluster_distances = []
    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_points = data[cluster_mask]
        cluster_center = cluster_centers[cluster_id]

        # Euclidean distance from each point to its cluster center
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        within_cluster_distances.extend(distances)

    mean_within_cluster = np.mean(within_cluster_distances)

    # Compute between-cluster distances (pairwise distances between cluster centers)
    if n_clusters > 1:
        between_distances = pairwise_distances(cluster_centers, metric='euclidean')
        # Take upper triangle (excluding diagonal) to avoid counting each pair twice
        between_cluster_distances = between_distances[np.triu_indices(n_clusters, k=1)]
        mean_between_cluster = np.mean(between_cluster_distances)
    else:
        mean_between_cluster = 0.0

    return {
        'mean_within_cluster': mean_within_cluster,
        'mean_between_cluster': mean_between_cluster,
        'within_cluster_distances': within_cluster_distances,
        'between_cluster_distances': between_cluster_distances if n_clusters > 1 else []
    }


def analyze_feature_clustering(weight_matrix_path, main_feature_idx=None, n_features=128, n_clusters=8, random_seed=42):
    """
    Main analysis function that loads data, computes hamming distances, and performs clustering.

    Args:
        weight_matrix_path: Path to .pt file containing weight matrix
        main_feature_idx: Index of feature to analyze (if None, randomly selected)
        n_features: Number of similar features to analyze (default: 128)
        n_clusters: Number of k-means clusters (default: 8)
        random_seed: Random seed for reproducibility
    """
    console = Console()

    # Load weight matrix
    console.print(f"\n[bold cyan]Loading weight matrix from:[/bold cyan] {weight_matrix_path}")
    matrix = torch.load(weight_matrix_path)
    console.print(f"[green]Matrix shape:[/green] {matrix.shape}")

    # Convert to binary (non-zero = 1)
    binary_matrix = (matrix.abs() > 0).int()

    # Select main feature
    if main_feature_idx is None:
        np.random.seed(random_seed)
        main_feature_idx = np.random.randint(0, binary_matrix.shape[1])

    console.print(f"[bold yellow]Selected main feature index:[/bold yellow] {main_feature_idx}")

    # Find top similar features by hamming distance
    console.print(f"\n[bold cyan]Finding top {n_features} most similar features by Hamming distance...[/bold cyan]")
    selected_features, feature_indices, hamming_distances = find_top_similar_features(
        binary_matrix, main_feature_idx, n_features
    )

    # Print hamming distances
    console.print(f"\n[bold magenta]Hamming Distances (Selected Feature vs Top {n_features} Similar):[/bold magenta]")
    console.print("=" * 80)

    # Create table for hamming distances
    table = Table(show_header=True, header_style="bold blue")
    table.add_column("Rank", style="dim", width=6)
    table.add_column("Feature Index", justify="right", width=15)
    table.add_column("Hamming Distance", justify="right", width=18)

    for rank, (feat_idx, dist) in enumerate(zip(feature_indices, hamming_distances)):
        if rank == 0:
            table.add_row(str(rank), str(feat_idx), str(dist), style="bold green")
        else:
            table.add_row(str(rank), str(feat_idx), str(dist))

    console.print(table)

    # Perform k-means clustering on features (transpose so features are rows)
    console.print(f"\n[bold cyan]Performing k-means clustering with k={n_clusters}...[/bold cyan]")
    features_as_points = selected_features.T.cpu().numpy()  # Shape: (n_features, n_samples)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = kmeans.fit_predict(features_as_points)

    # Compute absolute cluster distances
    console.print(f"\n[bold cyan]Computing absolute cluster distance metrics...[/bold cyan]")
    distance_metrics = compute_absolute_cluster_distances(
        features_as_points, labels, kmeans.cluster_centers_
    )

    # Display results
    console.print("\n" + "=" * 80)
    console.print("[bold green]CLUSTER DISTANCE METRICS (ABSOLUTE VALUES)[/bold green]")
    console.print("=" * 80)

    console.print(f"\n[bold]Number of clusters (k):[/bold] {n_clusters}")
    console.print(f"[bold]Number of features analyzed:[/bold] {len(feature_indices)}")
    console.print(f"[bold]Total inertia:[/bold] {kmeans.inertia_:.2f}")

    console.print(f"\n[bold yellow]Mean Within-Cluster Distance:[/bold yellow] {distance_metrics['mean_within_cluster']:.4f}")
    console.print(f"  → Average distance from each feature to its cluster center")

    console.print(f"\n[bold yellow]Mean Between-Cluster Distance:[/bold yellow] {distance_metrics['mean_between_cluster']:.4f}")
    console.print(f"  → Average pairwise distance between cluster centers")

    # Cluster separation metric
    if distance_metrics['mean_within_cluster'] > 0:
        separation_ratio = distance_metrics['mean_between_cluster'] / distance_metrics['mean_within_cluster']
        console.print(f"\n[bold cyan]Cluster Separation Ratio:[/bold cyan] {separation_ratio:.4f}")
        console.print(f"  → Between-cluster / Within-cluster distance")
        if separation_ratio > 2.0:
            console.print("  [green]✓ Good cluster separation[/green]")
        elif separation_ratio > 1.0:
            console.print("  [yellow]○ Moderate cluster separation[/yellow]")
        else:
            console.print("  [red]✗ Poor cluster separation[/red]")

    # Display cluster size distribution
    unique_labels, counts = np.unique(labels, return_counts=True)
    console.print(f"\n[bold]Cluster Size Distribution:[/bold]")
    for cluster_id, count in zip(unique_labels, counts):
        console.print(f"  Cluster {cluster_id}: {count} features")

    console.print("\n" + "=" * 80 + "\n")

    return {
        'selected_features': selected_features,
        'feature_indices': feature_indices,
        'hamming_distances': hamming_distances,
        'labels': labels,
        'kmeans': kmeans,
        'distance_metrics': distance_metrics
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze hamming distances and clustering for sparse weight matrices"
    )
    parser.add_argument(
        'matrix_path',
        type=str,
        help='Path to .pt file containing weight matrix'
    )
    parser.add_argument(
        '--feature-idx',
        type=int,
        default=None,
        help='Index of main feature to analyze (default: randomly selected)'
    )
    parser.add_argument(
        '--n-features',
        type=int,
        default=128,
        help='Number of similar features to analyze (default: 128)'
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

    args = parser.parse_args()

    # Validate file path
    if not Path(args.matrix_path).exists():
        print(f"[bold red]Error:[/bold red] File not found: {args.matrix_path}")
        return

    # Run analysis
    results = analyze_feature_clustering(
        weight_matrix_path=args.matrix_path,
        main_feature_idx=args.feature_idx,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        random_seed=args.seed
    )


if __name__ == "__main__":
    main()
