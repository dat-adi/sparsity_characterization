#!/usr/bin/env python3
"""
Nearest-Neighbor Clustering Analysis

For each feature vector in a weight matrix:
1. Find the 8 most similar features based on Hamming distance
2. Group them as a nearest-neighbor cluster
3. Compute within-cluster and between-cluster distances
4. Generate heatmap visualizations

Processes all matrices in data/clustering/{wanda,sparsegpt}/
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.nearest_neighbor_clustering import (
    create_nearest_neighbor_clusters,
    compute_cluster_distance_matrix
)
from utils.visualization import plot_cluster_distance_heatmap


def analyze_matrix(
    matrix_path: Path,
    k_neighbors: int = 8,
    max_clusters_to_compute: int = None,
    max_clusters_to_display: int = 100
) -> dict:
    """
    Perform nearest-neighbor clustering analysis on a single matrix.

    Args:
        matrix_path: Path to .pt file
        k_neighbors: Number of neighbors per cluster
        max_clusters_to_compute: Max clusters to compute (None = all)
        max_clusters_to_display: Max clusters to show in heatmap

    Returns:
        Dictionary with analysis results and full distance distributions
    """
    print(f"\nAnalyzing: {matrix_path.name}")

    # 1. Load matrix
    try:
        matrix = torch.load(matrix_path, weights_only=True)
    except Exception as e:
        print(f"  Error loading matrix: {e}")
        return None

    print(f"  Shape: {matrix.shape}")

    # Check if matrix is valid
    if matrix.numel() == 0:
        print(f"  Warning: Matrix is empty, skipping")
        return None

    binary_matrix = (matrix > 0).int()
    sparsity = 1.0 - (binary_matrix.sum().item() / binary_matrix.numel())
    print(f"  Sparsity: {sparsity:.2%}")

    if binary_matrix.sum() == 0:
        print(f"  Warning: Matrix is entirely zero after binarization, skipping")
        return None

    # 2. Create clusters
    print(f"  Creating nearest-neighbor clusters (k={k_neighbors})...")
    clusters = create_nearest_neighbor_clusters(
        matrix,
        k_neighbors=k_neighbors,
        max_clusters=max_clusters_to_compute
    )
    print(f"  Created {len(clusters)} nearest-neighbor clusters")

    # 3. Compute distance matrices
    print(f"  Computing within/between cluster distances...")
    within_distances, between_distances, between_distances_sampled = compute_cluster_distance_matrix(
        binary_matrix,
        clusters
    )

    # 4. Compute statistics
    mean_within = np.mean(within_distances)

    # Use sampled between-cluster distances for accurate mean
    # (the matrix is sparse - most values are zeros, only sampled pairs have real values)
    mean_between = np.mean(between_distances_sampled) if between_distances_sampled else 0.0

    separation_ratio = mean_between / mean_within if mean_within > 0 else 0

    print(f"  Mean within-cluster distance: {mean_within:.4f}")
    print(f"  Mean between-cluster distance: {mean_between:.4f}")
    print(f"  Separation ratio: {separation_ratio:.4f}")

    # Use sampled between-cluster distances for distribution
    between_distance_distribution = between_distances_sampled

    return {
        'matrix_name': matrix_path.name,
        'shape': list(matrix.shape),
        'sparsity': float(sparsity),
        'n_clusters': len(clusters),
        'k_neighbors': k_neighbors,
        'mean_within_distance': float(mean_within),
        'mean_between_distance': float(mean_between),
        'separation_ratio': float(separation_ratio),
        'within_distances_stats': {
            'min': float(np.min(within_distances)),
            'max': float(np.max(within_distances)),
            'std': float(np.std(within_distances)),
            'median': float(np.median(within_distances))
        },
        'between_distances_stats': {
            'min': float(np.min(between_distance_distribution)) if between_distance_distribution else 0.0,
            'max': float(np.max(between_distance_distribution)) if between_distance_distribution else 0.0,
            'std': float(np.std(between_distance_distribution)) if between_distance_distribution else 0.0,
            'median': float(np.median(between_distance_distribution)) if between_distance_distribution else 0.0
        },
        'distributions': {
            'within_distances': within_distances.tolist(),
            'between_distances': between_distance_distribution
        }
    }


def main():
    """Process all matrices in clustering directories."""

    # Define paths using absolute paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    data_dir = base_dir / "data" / "clustering"
    metrics_dir = base_dir / "results" / "metrics" / "nearest_neighbor_clustering"
    viz_dir = base_dir / "results" / "visualizations" / "nearest_neighbor_clustering"

    # Create output directories
    metrics_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directories:")
    print(f"  Metrics: {metrics_dir}")
    print(f"  Visualizations: {viz_dir}")

    # Find all .pt files
    wanda_files = sorted((data_dir / "wanda").glob("*.pt")) if (data_dir / "wanda").exists() else []
    sparsegpt_files = sorted((data_dir / "sparsegpt").glob("*.pt")) if (data_dir / "sparsegpt").exists() else []

    print(f"\nFound {len(wanda_files)} Wanda matrices")
    print(f"Found {len(sparsegpt_files)} SparseGPT matrices")

    all_results = {
        'timestamp': datetime.now().isoformat(),
        'analysis_type': 'nearest_neighbor_clustering',
        'k_neighbors': 8,
        'wanda': {},
        'sparsegpt': {}
    }

    # Process Wanda matrices
    print("\n" + "="*80)
    print("Processing Wanda matrices")
    print("="*80)

    for matrix_path in wanda_files:
        result = analyze_matrix(
            matrix_path,
            k_neighbors=8,
            max_clusters_to_compute=None,  # Process all clusters
            max_clusters_to_display=100
        )

        if result is not None:
            # Save individual metrics
            metrics_file = metrics_dir / f"wanda_{matrix_path.stem}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved metrics: {metrics_file.name}")

            # Generate and save heatmap
            # Reload for visualization (we don't keep full distance matrices in result)
            matrix = torch.load(matrix_path, weights_only=True)
            binary_matrix = (matrix > 0).int()
            clusters = create_nearest_neighbor_clusters(matrix, k_neighbors=8)
            within_distances, between_distances, _ = compute_cluster_distance_matrix(
                binary_matrix, clusters
            )

            viz_file = viz_dir / f"wanda_{matrix_path.stem}_heatmap.png"
            plot_cluster_distance_heatmap(
                within_distances,
                between_distances,
                title=f"Wanda - {matrix_path.stem}",
                output_path=viz_file,
                max_clusters_display=100
            )

            # Store in summary
            all_results['wanda'][matrix_path.stem] = result

    # Process SparseGPT matrices
    print("\n" + "="*80)
    print("Processing SparseGPT matrices")
    print("="*80)

    for matrix_path in sparsegpt_files:
        result = analyze_matrix(
            matrix_path,
            k_neighbors=8,
            max_clusters_to_compute=None,
            max_clusters_to_display=100
        )

        if result is not None:
            # Save individual metrics
            metrics_file = metrics_dir / f"sparsegpt_{matrix_path.stem}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved metrics: {metrics_file.name}")

            # Generate and save heatmap
            matrix = torch.load(matrix_path, weights_only=True)
            binary_matrix = (matrix > 0).int()
            clusters = create_nearest_neighbor_clusters(matrix, k_neighbors=8)
            within_distances, between_distances, _ = compute_cluster_distance_matrix(
                binary_matrix, clusters
            )

            viz_file = viz_dir / f"sparsegpt_{matrix_path.stem}_heatmap.png"
            plot_cluster_distance_heatmap(
                within_distances,
                between_distances,
                title=f"SparseGPT - {matrix_path.stem}",
                output_path=viz_file,
                max_clusters_display=100
            )

            # Store in summary
            all_results['sparsegpt'][matrix_path.stem] = result

    # Save summary JSON with all results
    summary_file = metrics_dir / "summary_all_matrices.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print(f"Summary saved to: {summary_file}")
    print(f"Total matrices processed: {len(all_results['wanda']) + len(all_results['sparsegpt'])}")


if __name__ == "__main__":
    main()
