"""
Matrix analysis utilities for computing sparsity metrics.

Functions for analyzing binary matrices including row statistics
and identifying patterns.
"""

import torch


def count_inactive_rows(matrix: torch.Tensor) -> int:
    """
    Count rows that are all zeros.

    Args:
        matrix: Binary matrix [D, N]

    Returns:
        Number of rows that are all zeros
    """
    return (matrix == 0).all(dim=1).sum().item()


def count_fully_active_rows(matrix: torch.Tensor) -> int:
    """
    Count rows that are all ones.

    Args:
        matrix: Binary matrix [D, N]

    Returns:
        Number of rows that are all ones
    """
    return (matrix == 1).all(dim=1).sum().item()


def count_identical_rows(matrix: torch.Tensor) -> tuple[int, int]:
    """
    Count duplicate rows in the matrix.

    Args:
        matrix: Binary matrix [D, N]

    Returns:
        num_duplicate_rows: Number of unique rows that appear more than once
        total_duplicate_instances: Total number of redundant row instances
    """
    unique_rows, counts = torch.unique(matrix, dim=0, return_counts=True)
    num_duplicate_rows = (counts > 1).sum().item()
    total_duplicate_instances = (counts - 1).sum().item()

    return num_duplicate_rows, total_duplicate_instances


def compute_subset_metrics(subset: torch.Tensor) -> dict:
    """
    Compute all metrics for a feature subset.

    Args:
        subset: Binary feature subset matrix [D, N]

    Returns:
        Dictionary containing:
            - zeros: Number of all-zero rows
            - ones: Number of all-one rows
            - duplicates: Number of unique rows that appear more than once
            - total_dups: Total number of redundant row instances
            - unique_rows: Number of unique rows
            - density: Proportion of non-zero elements
    """
    zeros = count_inactive_rows(subset)
    ones = count_fully_active_rows(subset)
    duplicates, total_dups = count_identical_rows(subset)
    unique_rows = subset.shape[0] - total_dups
    density = subset.float().mean().item()

    return {
        'zeros': zeros,
        'ones': ones,
        'duplicates': duplicates,
        'total_dups': total_dups,
        'unique_rows': unique_rows,
        'density': density
    }
