"""
Nearest-neighbor clustering utilities for feature similarity analysis.

Creates clusters by grouping each feature with its k nearest neighbors
based on Hamming distance, then computes within/between cluster distances.
"""

import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from .hamming_analysis import compute_hamming_distance_batch


@dataclass
class NearestNeighborCluster:
    """Represents a single nearest-neighbor cluster."""
    center_idx: int  # Index of the central feature
    neighbor_indices: List[int]  # Indices of the k nearest neighbors
    hamming_distances: List[float]  # Hamming distances to each neighbor

    @property
    def all_indices(self) -> List[int]:
        """Get all indices (center + neighbors)."""
        return [self.center_idx] + self.neighbor_indices


def create_nearest_neighbor_clusters(
    matrix: torch.Tensor,
    k_neighbors: int = 8,
    max_clusters: int = None
) -> List[NearestNeighborCluster]:
    """
    Create nearest-neighbor clusters for all features in the matrix.

    For each feature vector (column), find its k most similar neighbors
    based on Hamming distance and group them as a cluster.

    Args:
        matrix: Weight matrix [D, N] where N is number of features
        k_neighbors: Number of nearest neighbors per cluster (default: 8)
        max_clusters: Optional limit on number of clusters to create

    Returns:
        List of NearestNeighborCluster objects, one per feature
    """
    # Convert to binary matrix (values > 0 become 1)
    binary_matrix = (matrix > 0).int()

    n_features = binary_matrix.shape[1]

    # Limit number of clusters if specified
    if max_clusters is not None:
        n_features = min(n_features, max_clusters)

    clusters = []

    for feature_idx in tqdm(range(n_features), desc="Creating clusters", unit="cluster"):
        # Get reference vector
        reference_vec = binary_matrix[:, feature_idx]

        # Compute Hamming distances to all features
        distances = compute_hamming_distance_batch(reference_vec, binary_matrix)

        # Sort by distance and get top k+1 indices (k neighbors + self)
        sorted_indices = torch.argsort(distances)

        # Exclude self from neighbors (it will be at index 0 with distance 0)
        # Get the next k nearest neighbors
        neighbor_indices = sorted_indices[1:k_neighbors+1].tolist()
        neighbor_distances = distances[sorted_indices[1:k_neighbors+1]].tolist()

        # Create cluster
        cluster = NearestNeighborCluster(
            center_idx=feature_idx,
            neighbor_indices=neighbor_indices,
            hamming_distances=neighbor_distances
        )
        clusters.append(cluster)

    return clusters


def compute_cluster_distance_matrix(
    binary_matrix: torch.Tensor,
    clusters: List[NearestNeighborCluster],
    n_samples_between: int = 50,
    max_cluster_pairs: int = 10000
) -> Tuple[np.ndarray, np.ndarray, List[float]]:
    """
    Compute within-cluster and between-cluster distance matrices.

    Args:
        binary_matrix: Binary weight matrix [D, N]
        clusters: List of NearestNeighborCluster objects
        n_samples_between: Number of sample pairs for between-cluster distances
        max_cluster_pairs: Maximum number of cluster pairs to sample for between-cluster distances

    Returns:
        within_distances: Array of mean within-cluster distances [num_clusters]
        between_distances: Matrix of mean between-cluster distances [num_clusters, num_clusters]
        between_distances_sampled: List of sampled between-cluster distances (for accurate mean)
    """
    n_clusters = len(clusters)

    within_distances = np.zeros(n_clusters)
    between_distances = np.zeros((n_clusters, n_clusters))
    between_distances_sampled = []  # Store all sampled between-cluster distances

    # Compute within-cluster distances
    for i, cluster in tqdm(enumerate(clusters), total=n_clusters, desc="Computing within-cluster distances", unit="cluster"):
        all_indices = cluster.all_indices
        cluster_features = binary_matrix[:, all_indices]  # [D, k+1]

        # Compute all pairwise Hamming distances within cluster
        distances = []
        n_features_in_cluster = len(all_indices)

        for idx1 in range(n_features_in_cluster):
            for idx2 in range(idx1 + 1, n_features_in_cluster):
                dist = (cluster_features[:, idx1] != cluster_features[:, idx2]).float().mean().item()
                distances.append(dist)

        within_distances[i] = np.mean(distances) if distances else 0.0

    # Compute between-cluster distances (sample cluster pairs to avoid O(n^2) explosion)
    total_possible_pairs = (n_clusters * (n_clusters - 1)) // 2
    n_pairs_to_sample = min(max_cluster_pairs, total_possible_pairs)

    # Generate random cluster pairs
    sampled_pairs = set()
    while len(sampled_pairs) < n_pairs_to_sample:
        i = np.random.randint(0, n_clusters)
        j = np.random.randint(0, n_clusters)
        if i < j:
            sampled_pairs.add((i, j))
        elif j < i:
            sampled_pairs.add((j, i))

    with tqdm(total=n_pairs_to_sample, desc="Computing between-cluster distances", unit="pair") as pbar:
        for i, j in sampled_pairs:
            cluster_i = clusters[i]
            cluster_j = clusters[j]

            cluster_i_features = binary_matrix[:, cluster_i.all_indices]
            cluster_j_features = binary_matrix[:, cluster_j.all_indices]

            # Sample pairs to avoid O(n^2) explosion
            n_i = cluster_i_features.shape[1]
            n_j = cluster_j_features.shape[1]
            max_possible_pairs = n_i * n_j
            n_samples = min(n_samples_between, max_possible_pairs)

            distances = []
            for _ in range(n_samples):
                idx_i = np.random.randint(0, n_i)
                idx_j = np.random.randint(0, n_j)
                dist = (cluster_i_features[:, idx_i] != cluster_j_features[:, idx_j]).float().mean().item()
                distances.append(dist)

            mean_dist = np.mean(distances)
            between_distances[i, j] = mean_dist
            between_distances[j, i] = mean_dist  # Symmetric

            # Store all sampled distances for accurate mean computation
            between_distances_sampled.extend(distances)

            pbar.update(1)

    # Fill diagonal with zeros (distance from cluster to itself)
    np.fill_diagonal(between_distances, 0.0)

    return within_distances, between_distances, between_distances_sampled
