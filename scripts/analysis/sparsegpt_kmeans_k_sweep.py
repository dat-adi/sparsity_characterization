#!/usr/bin/env python3
"""
SparseGPT K-means Separation Ratio Experiment

Run k-means clustering on SparseGPT matrices with k ∈ {4, 8, 16, 32, 64}
to measure how separation ratio changes with the number of clusters.

Saves results to JSON for further analysis.
"""

import sys
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import find_most_similar_features
from utils.clustering import compute_cluster_metrics, ClusterMetrics


def analyze_matrix(
    matrix_path: Path,
    n_features: int = 128,
    k_values: List[int] = [4, 8, 16, 32, 64],
    random_seed: int = 42
) -> Dict[int, Dict]:
    """
    Analyze a single sparse matrix with multiple k values.

    Args:
        matrix_path: Path to .pt file
        n_features: Number of similar features to select
        k_values: List of k values for KMeans
        random_seed: Random seed

    Returns:
        Dictionary mapping k -> metrics dict
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"\n{'='*80}")
    print(f"Analyzing: {matrix_path.name}")
    print(f"{'='*80}")

    # Load matrix (force CPU since CUDA may not be available)
    matrix = torch.load(matrix_path, weights_only=True, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")
    print(f"Sparsity: {(matrix == 0).float().mean():.2%}")

    # Randomly select a feature
    n_total_features = matrix.shape[1]
    selected_feature_idx = random.randint(0, n_total_features - 1)
    print(f"Selected feature index: {selected_feature_idx}")

    # Find most similar features
    print(f"Finding {n_features} most similar features by hamming distance...")
    subset, indices, distances = find_most_similar_features(
        matrix, selected_feature_idx, n_features
    )
    print(f"Subset shape: {subset.shape}")
    print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")
    print(f"Mean distance to selected feature: {distances.mean():.4f}")

    # Perform clustering for each k
    results = {}
    for k in k_values:
        print(f"\n--- KMeans with k={k} ---")
        metrics = compute_cluster_metrics(subset, k, random_state=random_seed)

        separation_ratio = metrics.mean_between / metrics.mean_within if metrics.mean_within > 0 else 0

        print(f"Cluster sizes: {metrics.cluster_sizes}")
        print(f"Mean within-cluster distance: {metrics.mean_within:.4f}")
        print(f"Mean between-cluster distance: {metrics.mean_between:.4f}")
        print(f"Separation ratio (between/within): {separation_ratio:.4f}")

        results[k] = {
            'k': k,
            'cluster_sizes': metrics.cluster_sizes,
            'within_cluster_distances': metrics.within_cluster_distances,
            'mean_within': metrics.mean_within,
            'mean_between': metrics.mean_between,
            'separation_ratio': separation_ratio,
            'std_within': float(np.std(metrics.within_cluster_distances)),
            'min_cluster_size': min(metrics.cluster_sizes),
            'max_cluster_size': max(metrics.cluster_sizes),
        }

    return results


def main():
    """Run clustering analysis on SparseGPT matrices with extended k values."""

    # Define paths using absolute paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    sparsegpt_dir = base_dir / "data" / "clustering" / "sparsegpt"
    output_dir = base_dir / "results" / "metrics" / "sparsegpt_kmeans_k_sweep"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Matrix files to analyze
    matrix_files = [
        "layer1-mlp.down_proj.pt",
        "layer1-mlp.up_proj.pt",
        "layer1-mlp.gate_proj.pt",
        "layer1-self_attn.q_proj.pt",
        "layer1-self_attn.k_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt",
    ]

    k_values = [4, 8, 16, 32, 64]
    random_seed = 42  # Fixed seed for reproducibility

    # Analyze SparseGPT matrices
    print("\n" + "="*80)
    print("SPARSEGPT K-MEANS K-VALUE SWEEP")
    print(f"K values: {k_values}")
    print(f"Random seed: {random_seed}")
    print("="*80)

    all_results = {}
    for matrix_file in matrix_files:
        matrix_path = sparsegpt_dir / matrix_file
        if matrix_path.exists():
            results = analyze_matrix(matrix_path, n_features=128, k_values=k_values, random_seed=random_seed)
            all_results[matrix_file] = results
        else:
            print(f"\nWarning: {matrix_path} not found, skipping...")

    # Save results to JSON
    output_file = output_dir / "sparsegpt_kmeans_k_sweep_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: SEPARATION RATIOS BY K VALUE")
    print("="*80)

    # Header
    print(f"\n{'Projection':30s}", end='')
    for k in k_values:
        print(f" k={k:2d}", end='   ')
    print()

    # Rows
    for matrix_file in matrix_files:
        if matrix_file in all_results:
            proj_name = matrix_file.replace('layer1-', '').replace('.pt', '')
            print(f"{proj_name:30s}", end='')

            for k in k_values:
                ratio = all_results[matrix_file][k]['separation_ratio']
                print(f" {ratio:5.2f}", end='  ')
            print()

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
