"""
Clustering utilities for feature co-activation analysis.

Provides k-means clustering functionality and metrics computation:
- Cluster assignment and analysis
- Within/between cluster distance metrics
- Cluster quality metrics
"""

import torch
import numpy as np
from sklearn.cluster import KMeans
from dataclasses import dataclass
from typing import List, Tuple
import random


@dataclass
class ClusterMetrics:
    """Stores clustering metrics for a given k value."""
    k: int
    within_cluster_distances: List[float]  # Mean Hamming distance within each cluster
    between_cluster_distances: np.ndarray  # Pairwise mean Hamming distances between clusters
    mean_within: float
    mean_between: float
    cluster_sizes: List[int]
    inertia: float = None  # Optional: k-means inertia
    n_iter: int = None  # Optional: k-means iterations


def compute_cluster_metrics(
    features: torch.Tensor,
    k: int,
    distance_fn=None,
    random_state: int = 42
) -> ClusterMetrics:
    """
    Perform k-means clustering and compute within/between cluster distances.

    Args:
        features: Feature matrix [D, N] where N is number of features
        k: Number of clusters
        distance_fn: Distance function to use (default: Hamming distance)
        random_state: Random seed for k-means

    Returns:
        ClusterMetrics object with all computed metrics
    """
    # Default to Hamming distance if not provided
    if distance_fn is None:
        def hamming_distance(a, b):
            binary_a = (a != 0).int() if not a.dtype == torch.bool else a.int()
            binary_b = (b != 0).int() if not b.dtype == torch.bool else b.int()
            return (binary_a != binary_b).float().mean().item()
        distance_fn = hamming_distance

    # Transpose so each row is a feature vector for k-means
    features_T = features.T.cpu().numpy()  # [N, D]

    # Apply k-means
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_T)

    # Convert back to torch for distance computation
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

        # Compute pairwise distances within cluster
        n_features = cluster_features.shape[1]
        distances = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                dist = distance_fn(
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

                dist = distance_fn(
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


def compute_absolute_cluster_distances(
    data: np.ndarray,
    labels: np.ndarray,
    cluster_centers: np.ndarray
) -> Tuple[List[float], float]:
    """
    Compute absolute distances from each point to its cluster center.

    Args:
        data: Data points [N, D]
        labels: Cluster assignments [N]
        cluster_centers: Cluster centers [k, D]

    Returns:
        within_distances: Mean distance to center for each cluster
        overall_mean: Overall mean distance to assigned cluster centers
    """
    n_clusters = len(np.unique(labels))
    within_distances = []
    all_distances = []

    for cluster_id in range(n_clusters):
        cluster_mask = (labels == cluster_id)
        cluster_points = data[cluster_mask]
        cluster_center = cluster_centers[cluster_id]

        # Compute distances to center
        distances = np.linalg.norm(cluster_points - cluster_center, axis=1)
        within_distances.append(np.mean(distances))
        all_distances.extend(distances)

    return within_distances, np.mean(all_distances)


def compute_block_coherence(
    data: np.ndarray,
    labels: np.ndarray,
    block_size: int = 8
) -> Tuple[List[float], float]:
    """
    Calculate block coherence for MMM (Multiply-Mask-Multiply) optimization analysis.

    Block coherence measures the proportion of samples where all features
    in a block are zero, which is exploitable by MMM optimization.

    Args:
        data: Binary activation data [N_samples, N_features]
        labels: Cluster labels [N_samples]
        block_size: Size of feature blocks (default: 8 for 8-vector operations)

    Returns:
        coherence_scores: Coherence score for each block
        mean_coherence: Mean coherence across all blocks
    """
    # Sort data by cluster assignment
    cluster_sort_idx = np.argsort(labels)
    sorted_data = data[:, cluster_sort_idx]  # Shape: (N_samples, N_features)

    n_blocks = sorted_data.shape[1] // block_size
    coherence_scores = []

    for block_idx in range(n_blocks):
        start = block_idx * block_size
        end = start + block_size
        block_data = sorted_data[:, start:end]  # Shape: (N_samples, block_size)

        # Count samples where all features in block are zero (exploitable by MMM)
        all_zero = (block_data.sum(axis=1) == 0).sum()
        coherence = all_zero / block_data.shape[0]
        coherence_scores.append(coherence)

    mean_coherence = np.mean(coherence_scores) if coherence_scores else 0.0

    return coherence_scores, mean_coherence
