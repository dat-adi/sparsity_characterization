"""
Hamming distance utilities for feature similarity analysis.

Functions for computing Hamming distances and finding similar features.
"""

import torch


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
    n_similar: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def compute_pairwise_hamming_distances_efficient(
     matrix: torch.Tensor,
     return_numpy: bool = False
):
    """
    Memory-efficient version using XOR operations (for large matrices).

    Args:
        matrix: Weight matrix [D, N] where N is number of features
        return_numpy: If True, return numpy array instead of torch tensor

    Returns:
        distance_matrix: Pairwise Hamming distance matrix [N, N]
    """
    N = matrix.shape[1]
    D = matrix.shape[0]

    # Convert to binary if not already
    binary_matrix = (matrix != 0).int()

    # Create distance matrix
    distance_matrix = torch.zeros((N, N), dtype=torch.float32, device=matrix.device)

    # Compute using matrix multiplication trick
    # For binary vectors: hamming_distance = sum(a XOR b) / D
    # XOR can be computed as: a XOR b = a + b - 2 * (a AND b)
    # For binary: a AND b = a * b

    # Convert to float for matrix operations
    binary_float = binary_matrix.float()  # [D, N]

    # Compute dot products (counts matching 1s)
    dot_products = binary_float.T @ binary_float  # [N, N]

    # Count of 1s in each column
    ones_count = binary_matrix.sum(dim=0)  # [N]

    # Hamming distance formula:
    # hamming(i, j) = (ones_i + ones_j - 2 * matches) / D
    # where matches = number of positions where both are 1
    ones_count_i = ones_count.unsqueeze(1)  # [N, 1]
    ones_count_j = ones_count.unsqueeze(0)  # [1, N]

    distance_matrix = (ones_count_i + ones_count_j - 2 * dot_products).float() / D

    if return_numpy:
        return distance_matrix.cpu().numpy()
    return distance_matrix


def compute_hamming_distance_cdf(distance_matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the cumulative distribution function (CDF) of pairwise Hamming distances.

    Args:
        distance_matrix: Pairwise Hamming distance matrix [N, N]

    Returns:
        values: Sorted unique distance values
        cdf: Cumulative probabilities for each value
    """
    # Get upper triangle to avoid counting pairs twice and exclude diagonal (self-distances)
    N = distance_matrix.shape[0]
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool, device=distance_matrix.device), diagonal=1)
    distances = distance_matrix[mask]

    # Sort distances
    sorted_distances, _ = torch.sort(distances)

    # Compute CDF
    n_pairs = len(sorted_distances)
    cdf_values = torch.arange(1, n_pairs + 1, dtype=torch.float32, device=distance_matrix.device) / n_pairs

    # Get unique values and their corresponding CDF values
    unique_distances, inverse_indices = torch.unique(sorted_distances, return_inverse=True)

    # For each unique distance, get the maximum CDF value (rightmost occurrence)
    unique_cdf = torch.zeros(len(unique_distances), dtype=torch.float32, device=distance_matrix.device)
    for i in range(len(unique_distances)):
        mask_unique = (inverse_indices == i)
        unique_cdf[i] = cdf_values[mask_unique].max()

    return unique_distances, unique_cdf
