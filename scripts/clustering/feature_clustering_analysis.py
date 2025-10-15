#!/usr/bin/env python3
"""
Feature Vector Clustering Analysis

For each sparse matrix (down_proj, up_proj, gate_proj, etc.):
1. Randomly pick one feature vector (column)
2. Find the 128 most similar feature vectors based on hamming distance
3. Apply KMeans clustering with k=4, 8, 16
4. Measure mean hamming distance within and between clusters
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random


@dataclass
class ClusterMetrics:
    """Stores clustering metrics for a given k value"""
    k: int
    within_cluster_distances: List[float]  # Mean hamming distance within each cluster
    between_cluster_distances: np.ndarray  # Pairwise mean hamming distances between clusters
    mean_within: float
    mean_between: float
    cluster_sizes: List[int]


def compute_hamming_distance(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute hamming distance between two binary vectors.

    Args:
        vec1, vec2: Binary tensors (values are 0 or non-zero)

    Returns:
        Hamming distance as fraction of differing positions
    """
    binary1 = (vec1 != 0).int()
    binary2 = (vec2 != 0).int()
    return (binary1 != binary2).float().mean().item()


def compute_hamming_distance_batch(vec: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute hamming distances between one vector and all columns of a matrix.

    Args:
        vec: Single feature vector [D]
        matrix: Matrix with features as columns [D, N]

    Returns:
        Tensor of hamming distances [N]
    """
    binary_vec = (vec != 0).int().unsqueeze(1)  # [D, 1]
    binary_matrix = (matrix != 0).int()  # [D, N]
    distances = (binary_vec != binary_matrix).float().mean(dim=0)  # [N]
    return distances


def find_most_similar_features(
    matrix: torch.Tensor,
    feature_idx: int,
    n_similar: int = 128
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the n_similar most similar feature vectors to a given feature.

    Args:
        matrix: Weight matrix [D, N] where N is number of features
        feature_idx: Index of the reference feature
        n_similar: Number of similar features to find (including the reference)

    Returns:
        subset: Matrix of selected features [D, n_similar]
        indices: Indices of selected features [n_similar]
        distances: Hamming distances of selected features [n_similar]
    """
    reference_vec = matrix[:, feature_idx]

    # Compute hamming distances to all features
    distances = compute_hamming_distance_batch(reference_vec, matrix)

    # Get indices of n_similar closest features
    # Note: index 0 will be the reference itself (distance = 0)
    sorted_indices = torch.argsort(distances)
    top_indices = sorted_indices[:n_similar]

    subset = matrix[:, top_indices]
    selected_distances = distances[top_indices]

    return subset, top_indices, selected_distances


def compute_cluster_metrics(
    features: torch.Tensor,
    k: int,
    random_state: int = 42
) -> ClusterMetrics:
    """
    Perform KMeans clustering and compute within/between cluster hamming distances.

    Args:
        features: Feature matrix [D, N] where N is number of features
        k: Number of clusters
        random_state: Random seed for KMeans

    Returns:
        ClusterMetrics object with all computed metrics
    """
    # Transpose so each row is a feature vector for KMeans
    features_T = features.T.cpu().numpy()  # [N, D]

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_T)

    # Convert back to torch for hamming distance computation
    features_tensor = features.cpu()

    # Compute within-cluster distances
    within_cluster_distances = []
    cluster_sizes = []

    for cluster_id in range(k):
        cluster_mask = (labels == cluster_id)
        cluster_features = features_tensor[:, cluster_mask]  # [D, n_cluster]
        cluster_sizes.append(cluster_features.shape[1])

        if cluster_features.shape[1] <= 1:
            within_cluster_distances.append(0.0)
            continue

        # Compute pairwise hamming distances within cluster
        n_features = cluster_features.shape[1]
        distances = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                dist = compute_hamming_distance(
                    cluster_features[:, i],
                    cluster_features[:, j]
                )
                distances.append(dist)

        mean_dist = np.mean(distances) if distances else 0.0
        within_cluster_distances.append(mean_dist)

    # Compute between-cluster distances
    between_cluster_distances = np.zeros((k, k))

    for i in range(k):
        for j in range(i + 1, k):
            cluster_i_mask = (labels == i)
            cluster_j_mask = (labels == j)

            cluster_i_features = features_tensor[:, cluster_i_mask]
            cluster_j_features = features_tensor[:, cluster_j_mask]

            # Sample pairs to avoid O(n^2) computation for large clusters
            n_samples = min(50, cluster_i_features.shape[1] * cluster_j_features.shape[1])
            distances = []

            for _ in range(n_samples):
                idx_i = random.randint(0, cluster_i_features.shape[1] - 1)
                idx_j = random.randint(0, cluster_j_features.shape[1] - 1)

                dist = compute_hamming_distance(
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
        cluster_sizes=cluster_sizes
    )


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
