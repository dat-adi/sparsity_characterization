"""
Hamming distance analysis utilities for feature similarity.

Provides functions for:
- Computing Hamming distances between binary vectors
- Finding most similar features based on Hamming distance
- Creating feature subsets for clustering analysis
"""

import torch
import numpy as np
import random
from typing import Tuple, List


def compute_hamming_distance(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """
    Compute Hamming distance between two binary vectors.

    Args:
        vec1: First binary tensor (values are 0 or non-zero)
        vec2: Second binary tensor (same shape as vec1)

    Returns:
        Hamming distance as fraction of differing positions [0, 1]
    """
    binary1 = (vec1 != 0).int()
    binary2 = (vec2 != 0).int()
    return (binary1 != binary2).float().mean().item()


def compute_hamming_distance_batch(vec: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute Hamming distances between one vector and all columns of a matrix.

    Args:
        vec: Single feature vector [D]
        matrix: Matrix with features as columns [D, N]

    Returns:
        Tensor of Hamming distances [N]
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
    Find the n most similar feature vectors to a given feature by Hamming distance.

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

    # Compute Hamming distances to all features
    distances = compute_hamming_distance_batch(reference_vec, matrix)

    # Get indices of n_similar closest features
    # Note: index 0 will be the reference itself (distance = 0)
    sorted_indices = torch.argsort(distances)
    top_indices = sorted_indices[:n_similar]

    subset = matrix[:, top_indices]
    selected_distances = distances[top_indices]

    return subset, top_indices, selected_distances


def get_most_similar_features_with_distances(
    main_feature_idx: int,
    n_features: int,
    matrix: torch.Tensor
) -> Tuple[torch.Tensor, List[int], List[float]]:
    """
    Get the n most similar features to the main feature based on Hamming distance.

    Alternative implementation that returns lists for compatibility with older code.

    Args:
        main_feature_idx: Index of the main/reference feature
        n_features: Number of similar features to return
        matrix: Weight matrix (will be converted to binary internally)

    Returns:
        subset: Binary feature subset [D, n_features+1] with main feature first
        indices: List of feature indices (main feature + n most similar)
        distances: List of Hamming distances for the n most similar features
    """
    # Convert to binary mask
    binary_matrix = (matrix.abs() > 0).int()
    main_feature = binary_matrix[:, main_feature_idx]

    # Calculate Hamming distance to all other features
    n_total_features = matrix.shape[1]
    hamming_distances = []

    for i in range(n_total_features):
        if i != main_feature_idx:
            hamming_dist = (main_feature != binary_matrix[:, i]).sum().item()
            hamming_distances.append((i, hamming_dist))

    # Sort by Hamming distance (ascending - most similar first)
    hamming_distances.sort(key=lambda x: x[1])

    # Get the n most similar feature indices
    most_similar_indices = [idx for idx, _ in hamming_distances[:n_features]]

    # Return subset with main feature first, then most similar features
    subset = torch.cat([
        binary_matrix[:, main_feature_idx].unsqueeze(1),
        binary_matrix[:, most_similar_indices]
    ], dim=1)

    # Return distances for the selected features
    selected_distances = [dist for _, dist in hamming_distances[:n_features]]

    return subset, [main_feature_idx] + most_similar_indices, selected_distances


def create_feature_subset(
    main_feature_idx: int,
    n_comparative_features: int,
    matrix: torch.Tensor
) -> Tuple[torch.Tensor, List[int]]:
    """
    Create a subset from a weights matrix with randomly selected features.

    Args:
        main_feature_idx: Index of the main feature to include
        n_comparative_features: Number of random features to select
        matrix: Weight matrix [D, N]

    Returns:
        subset: Feature subset [D, n_comparative_features+1] with main feature first
        indices: List of selected feature indices
    """
    n_features = matrix.shape[1]
    feature_range = list(range(0, n_features))
    random_feature_range = feature_range[:main_feature_idx] + feature_range[main_feature_idx+1:]
    random_feature_indices = random.sample(
        random_feature_range,
        min(n_comparative_features, len(random_feature_range))
    )

    # Always have the main feature at the start of the subset
    subset = torch.cat([
        matrix[:, main_feature_idx].unsqueeze(1),
        matrix[:, random_feature_indices]
    ], dim=1)

    return subset, [main_feature_idx] + random_feature_indices


def get_coactivation_gradient(matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Arrange features by Hamming distance to the main feature (at index 0).

    Orders features from most relevant (lowest Hamming distance) to least relevant.

    Args:
        matrix: Binary feature matrix [D, N] where feature 0 is the reference

    Returns:
        sorted_matrix: Matrix with features reordered by Hamming distance [D, N]
        sorted_indices: Indices showing the new order [N]
        hamming_distances: Hamming distances for features 1 to N-1 [N-1]
    """
    hamming_distances = torch.tensor([
        (matrix[:, 0] != matrix[:, i]).sum().item()
        for i in range(1, matrix.shape[1])
    ])
    sorted_indices = torch.argsort(hamming_distances, descending=False)

    # Add the 0th index to pad the coactivation gradient with the main feature
    sorted_indices = torch.cat([torch.tensor([0]), sorted_indices + 1])

    return matrix[:, sorted_indices], sorted_indices, hamming_distances
