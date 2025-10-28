#!/usr/bin/env python3
"""
Quick test to verify the between-cluster distance fix.
Runs on a single matrix to check the new implementation.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.nearest_neighbor_clustering import (
    create_nearest_neighbor_clusters,
    compute_cluster_distance_matrix
)

def test_fix():
    """Test the between-cluster distance fix on layer1-mlp.down_proj."""

    # Load the matrix
    matrix_path = Path(__file__).parent.parent.parent / "data/clustering/wanda/layer1-mlp.down_proj.pt"

    if not matrix_path.exists():
        print(f"Matrix file not found: {matrix_path}")
        return

    print(f"Loading: {matrix_path.name}")
    matrix = torch.load(matrix_path, weights_only=True)
    print(f"Shape: {matrix.shape}")

    # Convert to binary
    binary_matrix = (matrix > 0).int()
    sparsity = 1.0 - (binary_matrix.sum().item() / binary_matrix.numel())
    print(f"Sparsity: {sparsity:.2%}")

    # Create clusters (limit to 1000 for speed)
    print("\nCreating 1000 nearest-neighbor clusters...")
    clusters = create_nearest_neighbor_clusters(matrix, k_neighbors=8, max_clusters=1000)
    print(f"Created {len(clusters)} clusters")

    # Compute distances
    print("\nComputing within/between cluster distances...")
    within_distances, between_distances, between_distances_sampled = compute_cluster_distance_matrix(
        binary_matrix,
        clusters,
        max_cluster_pairs=5000  # Sample 5000 pairs
    )

    # Print results
    print("\nResults:")
    print(f"  Number of within-cluster distances: {len(within_distances)}")
    print(f"  Number of sampled between-cluster distances: {len(between_distances_sampled)}")
    print(f"  Mean within-cluster distance: {within_distances.mean():.4f}")
    print(f"  Mean between-cluster distance (from sampled): {sum(between_distances_sampled) / len(between_distances_sampled):.4f}")
    print(f"  Separation ratio: {(sum(between_distances_sampled) / len(between_distances_sampled)) / within_distances.mean():.4f}")

    # Show distribution stats
    import numpy as np
    print("\nBetween-cluster distance distribution:")
    print(f"  Min: {min(between_distances_sampled):.4f}")
    print(f"  Max: {max(between_distances_sampled):.4f}")
    print(f"  Median: {np.median(between_distances_sampled):.4f}")
    print(f"  Std: {np.std(between_distances_sampled):.4f}")

    # Verify the fix
    print("\n" + "="*60)
    if sum(between_distances_sampled) / len(between_distances_sampled) > 0.001:
        print("✓ FIX VERIFIED: Between-cluster distance is non-zero!")
    else:
        print("✗ ISSUE PERSISTS: Between-cluster distance is still near zero")
    print("="*60)

if __name__ == "__main__":
    test_fix()
