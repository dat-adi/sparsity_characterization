"""
Similarity metrics for comparing sparse weight matrices.

Provides standard metrics for measuring similarity between pruned weight matrices:
- Jaccard similarity: Overlap of non-zero positions
- Cosine similarity: Directional similarity of weight values
- Hamming distance: Proportion of differing sparsity patterns
"""

import torch
import numpy as np
from typing import Dict, Union


def jaccard_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute Jaccard similarity between two tensors based on non-zero positions.

    Jaccard similarity measures the overlap of non-zero elements:
    J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        a: First tensor (any shape)
        b: Second tensor (same shape as a)

    Returns:
        Jaccard similarity coefficient [0, 1], where 1 means perfect overlap
    """
    a_nz = (a != 0).flatten()
    b_nz = (b != 0).flatten()
    intersection = torch.sum(a_nz & b_nz).item()
    union = torch.sum(a_nz | b_nz).item()
    return intersection / union if union > 0 else 0.0


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute cosine similarity between two tensors.

    Cosine similarity measures directional similarity:
    cos(θ) = (A · B) / (||A|| ||B||)

    Args:
        a: First tensor (any shape)
        b: Second tensor (same shape as a)

    Returns:
        Cosine similarity [-1, 1], where 1 means same direction
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    return (dot / (norm_a * norm_b)).item() if norm_a > 0 and norm_b > 0 else 0.0


def hamming_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """
    Compute Hamming distance between two tensors based on sparsity patterns.

    Hamming distance measures the proportion of positions where the
    sparsity patterns differ (one is zero, the other is non-zero).

    Args:
        a: First tensor (any shape)
        b: Second tensor (same shape as a)

    Returns:
        Hamming distance [0, 1], where 0 means identical sparsity patterns
    """
    a_nz = (a != 0).flatten()
    b_nz = (b != 0).flatten()
    return (torch.sum(a_nz != b_nz).item() / len(a_nz))


def compute_metrics_by_feature(
    mat1: torch.Tensor,
    mat2: torch.Tensor,
    matrix_name: str
) -> Dict[str, Dict[str, float]]:
    """
    Compute similarity metrics feature-by-feature (row-wise or column-wise).

    For down_proj matrices, computes metrics row-wise (output features).
    For other projections, computes metrics column-wise (input features).

    Args:
        mat1: First weight matrix
        mat2: Second weight matrix (same shape as mat1)
        matrix_name: Name of the matrix (used to determine axis)

    Returns:
        Dictionary with metrics for each similarity measure:
        {
            'jaccard': {'mean': float, 'std': float},
            'cosine': {'mean': float, 'std': float},
            'hamming': {'mean': float, 'std': float}
        }
    """
    is_down_proj = "down_proj" in matrix_name
    axis = 0 if is_down_proj else 1  # row-wise for down_proj, column-wise for others

    jaccard_scores = []
    cosine_scores = []
    hamming_scores = []

    if axis == 0:  # row-wise
        for i in range(mat1.shape[0]):
            jaccard_scores.append(jaccard_similarity(mat1[i], mat2[i]))
            cosine_scores.append(cosine_similarity(mat1[i], mat2[i]))
            hamming_scores.append(hamming_distance(mat1[i], mat2[i]))
    else:  # column-wise
        for i in range(mat1.shape[1]):
            jaccard_scores.append(jaccard_similarity(mat1[:, i], mat2[:, i]))
            cosine_scores.append(cosine_similarity(mat1[:, i], mat2[:, i]))
            hamming_scores.append(hamming_distance(mat1[:, i], mat2[:, i]))

    return {
        'jaccard': {'mean': np.mean(jaccard_scores), 'std': np.std(jaccard_scores)},
        'cosine': {'mean': np.mean(cosine_scores), 'std': np.std(cosine_scores)},
        'hamming': {'mean': np.mean(hamming_scores), 'std': np.std(hamming_scores)}
    }
