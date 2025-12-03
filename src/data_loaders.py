"""
Data loading utilities for weight matrices.

Functions for loading Wanda and SparseGPT matrices from the data directory.
"""

from pathlib import Path
import random
import numpy as np
import torch


def set_seed(random_seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)


def get_wanda_matrices(files: list[str]) -> list[Path]:
    """
    Get paths to Wanda unstructured matrices.

    Args:
        files: List of matrix filenames

    Returns:
        List of Path objects to Wanda matrices
    """
    WANDA_DIR = "/home/datadi/burns/aws/workloads/data/wanda_unstructured/"
    layer_1 = Path(WANDA_DIR) / "layer-1"
    return [layer_1 / f for f in files]


def get_sparsegpt_matrices(files: list[str]) -> list[Path]:
    """
    Get paths to SparseGPT unstructured matrices.

    Args:
        files: List of matrix filenames

    Returns:
        List of Path objects to SparseGPT matrices
    """
    SPARSEGPT_DIR = "/home/datadi/burns/aws/workloads/data/sparsegpt_unstructured/"
    layer_1 = Path(SPARSEGPT_DIR) / "layer-1"
    return [layer_1 / f for f in files]


def get_unstructured_matrices_layer_1() -> tuple[list[Path], list[Path]]:
    """
    Get all layer 1 matrices for both Wanda and SparseGPT.

    Returns:
        Tuple of (wanda_matrices, sparsegpt_matrices) as lists of Paths
    """
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

    wanda_matrices = get_wanda_matrices(matrix_files)
    sparsegpt_matrices = get_sparsegpt_matrices(matrix_files)

    return wanda_matrices, sparsegpt_matrices


def select_feature_columns(matrix: torch.Tensor, n: int) -> list[int]:
    """
    Randomly select n columns from the matrix.

    Args:
        matrix: Weight matrix [D, N] where N is number of features
        n: Number of columns to select

    Returns:
        List of selected column indices
    """
    return random.sample(range(matrix.shape[1]), min(n, matrix.shape[1]))
