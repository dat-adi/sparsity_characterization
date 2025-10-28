#!/usr/bin/env python3
"""
Feature Vector Clustering Analysis

For each sparse matrix (down_proj, up_proj, gate_proj, etc.):
1. Randomly pick one feature vector (column)
2. Find the 128 most similar feature vectors based on hamming distance
3. Apply KMeans clustering with k=4, 8, 16
4. Measure mean hamming distance within and between clusters
"""

import sys
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
    k_values: List[int] = [4, 8, 16],
    random_seed: int = 42
) -> Dict[int, ClusterMetrics]:
    """
    Analyze a single sparse matrix.

    Args:
        matrix_path: Path to .pt file
        n_features: Number of similar features to select
        k_values: List of k values for KMeans
        random_seed: Random seed

    Returns:
        Dictionary mapping k -> ClusterMetrics
    """
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    print(f"\n{'='*80}")
    print(f"Analyzing: {matrix_path.name}")
    print(f"{'='*80}")

    # Load matrix
    matrix = torch.load(matrix_path, weights_only=True)
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
        metrics = compute_cluster_metrics(subset, k, random_seed)

        print(f"Cluster sizes: {metrics.cluster_sizes}")
        print(f"Within-cluster distances: {[f'{d:.4f}' for d in metrics.within_cluster_distances]}")
        print(f"Mean within-cluster distance: {metrics.mean_within:.4f}")
        print(f"Mean between-cluster distance: {metrics.mean_between:.4f}")
        print(f"Separation ratio (between/within): {metrics.mean_between / metrics.mean_within:.4f}")

        results[k] = metrics

    return results


def main():
    """Run clustering analysis on all matrices in both directories."""

    # Define paths
    wanda_dir = Path("../../data/wanda_unstructured/layer-1")
    sparsegpt_dir = Path("../../data/sparsegpt_unstructured/layer-1")

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

    k_values = [4, 8, 16]
    random_seed = random.randint(1, 1000000)

    # Analyze Wanda matrices
    print("\n" + "="*80)
    print("WANDA UNSTRUCTURED PRUNING")
    print("="*80)

    wanda_results = {}
    for matrix_file in matrix_files:
        matrix_path = wanda_dir / matrix_file
        if matrix_path.exists():
            results = analyze_matrix(matrix_path, n_features=128, k_values=k_values, random_seed=random_seed)
            wanda_results[matrix_file] = results

    # Analyze SparseGPT matrices
    print("\n" + "="*80)
    print("SPARSEGPT UNSTRUCTURED PRUNING")
    print("="*80)

    sparsegpt_results = {}
    for matrix_file in matrix_files:
        matrix_path = sparsegpt_dir / matrix_file
        if matrix_path.exists():
            results = analyze_matrix(matrix_path, n_features=128, k_values=k_values, random_seed=random_seed)
            sparsegpt_results[matrix_file] = results

    # Print summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)

    for matrix_file in matrix_files:
        if matrix_file in wanda_results and matrix_file in sparsegpt_results:
            print(f"\n{matrix_file}:")
            print(f"{'':20s} {'Wanda':>30s} {'SparseGPT':>30s}")
            print(f"{'':20s} {'Within':>10s} {'Between':>10s} {'Ratio':>8s} {'Within':>10s} {'Between':>10s} {'Ratio':>8s}")

            for k in k_values:
                w_metrics = wanda_results[matrix_file][k]
                s_metrics = sparsegpt_results[matrix_file][k]

                w_ratio = w_metrics.mean_between / w_metrics.mean_within if w_metrics.mean_within > 0 else 0
                s_ratio = s_metrics.mean_between / s_metrics.mean_within if s_metrics.mean_within > 0 else 0

                print(f"  k={k:2d}: {w_metrics.mean_within:10.4f} {w_metrics.mean_between:10.4f} {w_ratio:8.4f} "
                      f"{s_metrics.mean_within:10.4f} {s_metrics.mean_between:10.4f} {s_ratio:8.4f}")


if __name__ == "__main__":
    main()
