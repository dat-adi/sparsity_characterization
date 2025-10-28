#!/usr/bin/env python3
"""
SparseGPT K-means Separation Ratio Experiment - FULL MATRIX

Run k-means clustering on ENTIRE SparseGPT matrices (all features, not just 128)
with k ∈ {4, 8, 16, 32, 64} to measure clustering structure across the full matrix.

WARNING: This is computationally intensive and may take significant time/memory.
"""

import sys
import json
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List
from sklearn.cluster import KMeans

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.clustering import compute_cluster_metrics, ClusterMetrics


def compute_full_matrix_cluster_metrics(
    features: torch.Tensor,
    k: int,
    random_state: int = 42
) -> ClusterMetrics:
    """
    Perform k-means clustering on ALL features and compute metrics.

    Args:
        features: Feature matrix [D, N] where N is number of features
        k: Number of clusters
        random_state: Random seed for k-means

    Returns:
        ClusterMetrics object with all computed metrics
    """
    # Hamming distance function
    def hamming_distance(a, b):
        binary_a = (a != 0).int() if not a.dtype == torch.bool else a.int()
        binary_b = (b != 0).int() if not b.dtype == torch.bool else b.int()
        return (binary_a != binary_b).float().mean().item()

    # Transpose so each row is a feature vector for k-means
    features_T = features.T.cpu().numpy()  # [N, D]
    n_features = features_T.shape[0]

    print(f"  Running k-means on {n_features} features...")

    # Apply k-means
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10, verbose=0)
    labels = kmeans.fit_predict(features_T)

    print(f"  K-means completed. Computing distances...")

    # Convert back to torch for distance computation
    features_tensor = features.cpu()

    # Compute within-cluster distances
    within_cluster_distances = []
    cluster_sizes = []

    for cluster_id in range(k):
        cluster_mask = (labels == cluster_id)
        cluster_features = features_tensor[:, cluster_mask]  # [D, n_cluster]
        cluster_size = cluster_features.shape[1]
        cluster_sizes.append(cluster_size)

        if cluster_size <= 1:
            within_cluster_distances.append(0.0)
            continue

        # Sample pairs to avoid O(n^2) computation for large clusters
        n_samples = min(100, cluster_size * (cluster_size - 1) // 2)
        distances = []

        for _ in range(n_samples):
            i = random.randint(0, cluster_size - 1)
            j = random.randint(0, cluster_size - 1)
            if i != j:
                dist = hamming_distance(
                    cluster_features[:, i],
                    cluster_features[:, j]
                )
                distances.append(dist)

        mean_dist = np.mean(distances) if distances else 0.0
        within_cluster_distances.append(mean_dist)

    # Compute between-cluster distances (sampling)
    between_cluster_distances = np.zeros((k, k))

    for i in range(k):
        for j in range(i + 1, k):
            cluster_i_mask = (labels == i)
            cluster_j_mask = (labels == j)

            cluster_i_features = features_tensor[:, cluster_i_mask]
            cluster_j_features = features_tensor[:, cluster_j_mask]

            # Sample pairs to avoid O(n^2) computation
            n_samples = min(100, cluster_i_features.shape[1] * cluster_j_features.shape[1])
            distances = []

            for _ in range(n_samples):
                idx_i = random.randint(0, cluster_i_features.shape[1] - 1)
                idx_j = random.randint(0, cluster_j_features.shape[1] - 1)

                dist = hamming_distance(
                    cluster_i_features[:, idx_i],
                    cluster_j_features[:, idx_j]
                )
                distances.append(dist)

            mean_dist = np.mean(distances)
            between_cluster_distances[i, j] = mean_dist
            between_cluster_distances[j, i] = mean_dist

    return ClusterMetrics(
        k=k,
        within_cluster_distances=within_cluster_distances,
        between_cluster_distances=between_cluster_distances,
        mean_within=np.mean(within_cluster_distances),
        mean_between=np.mean(between_cluster_distances[np.triu_indices(k, k=1)]),
        cluster_sizes=cluster_sizes,
        inertia=kmeans.inertia_,
        n_iter=kmeans.n_iter_
    )


def analyze_matrix(
    matrix_path: Path,
    k_values: List[int] = [4, 8, 16, 32, 64],
    random_seed: int = 42
) -> Dict[int, Dict]:
    """
    Analyze entire sparse matrix with multiple k values.

    Args:
        matrix_path: Path to .pt file
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

    # Load matrix (force CPU)
    matrix = torch.load(matrix_path, weights_only=True, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")
    print(f"Sparsity: {(matrix == 0).float().mean():.2%}")
    print(f"Total features (columns): {matrix.shape[1]}")

    # Perform clustering for each k on FULL matrix
    results = {}
    for k in k_values:
        print(f"\n--- KMeans with k={k} on FULL MATRIX ---")
        metrics = compute_full_matrix_cluster_metrics(matrix, k, random_state=random_seed)

        separation_ratio = metrics.mean_between / metrics.mean_within if metrics.mean_within > 0 else 0

        print(f"Cluster sizes (min/max/mean): {min(metrics.cluster_sizes)}/{max(metrics.cluster_sizes)}/{np.mean(metrics.cluster_sizes):.1f}")
        print(f"Mean within-cluster distance: {metrics.mean_within:.4f}")
        print(f"Mean between-cluster distance: {metrics.mean_between:.4f}")
        print(f"Separation ratio (between/within): {separation_ratio:.4f}")
        print(f"K-means inertia: {metrics.inertia:.2f}")
        print(f"K-means iterations: {metrics.n_iter}")

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
            'mean_cluster_size': float(np.mean(metrics.cluster_sizes)),
            'inertia': float(metrics.inertia),
            'n_iter': int(metrics.n_iter),
        }

    return results


def main():
    """Run clustering analysis on FULL SparseGPT matrices with extended k values."""

    # Define paths using absolute paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    sparsegpt_dir = base_dir / "data" / "clustering" / "sparsegpt"
    output_dir = base_dir / "results" / "metrics" / "sparsegpt_kmeans_k_sweep_full"

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
    print("SPARSEGPT K-MEANS K-VALUE SWEEP - FULL MATRIX ANALYSIS")
    print(f"K values: {k_values}")
    print(f"Random seed: {random_seed}")
    print("WARNING: This will analyze ALL features in each matrix")
    print("Expected runtime: Several minutes to hours depending on hardware")
    print("="*80)

    all_results = {}
    for matrix_file in matrix_files:
        matrix_path = sparsegpt_dir / matrix_file
        if matrix_path.exists():
            results = analyze_matrix(matrix_path, k_values=k_values, random_seed=random_seed)
            all_results[matrix_file] = results
        else:
            print(f"\nWarning: {matrix_path} not found, skipping...")

    # Save results to JSON
    output_file = output_dir / "sparsegpt_kmeans_k_sweep_full_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")

    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: SEPARATION RATIOS BY K VALUE (FULL MATRIX)")
    print("="*80)

    # Header
    print(f"\n{'Projection':30s}", end='')
    for k in k_values:
        print(f" k={k:2d}", end='   ')
    print()
    print("-" * 80)

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

    # Print cluster size summary
    print("\n" + "="*80)
    print("CLUSTER SIZE STATISTICS (k=16)")
    print("="*80)
    print(f"\n{'Projection':30s} {'Min':>8s} {'Max':>8s} {'Mean':>8s} {'Std':>8s}")
    print("-" * 80)

    for matrix_file in matrix_files:
        if matrix_file in all_results and 16 in [int(k) for k in all_results[matrix_file].keys()]:
            proj_name = matrix_file.replace('layer1-', '').replace('.pt', '')
            k16_data = all_results[matrix_file][16]
            sizes = k16_data['cluster_sizes']
            print(f"{proj_name:30s} {min(sizes):8d} {max(sizes):8d} {np.mean(sizes):8.1f} {np.std(sizes):8.1f}")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
