"""
Main analysis script for nearest neighbor feature analysis.

This script:
1. Loads weight matrices from Wanda and SparseGPT
2. For each matrix, selects random features
3. For each feature, finds the k nearest neighbors by Hamming distance
4. Computes pairwise distances and various metrics
5. Stores results in DuckDB database
"""

import torch
from pathlib import Path
import argparse

from data_loaders import set_seed, get_unstructured_matrices_layer_1, select_feature_columns
from hamming_utils import find_most_similar_features, compute_pairwise_hamming_distances_efficient
from matrix_analysis import compute_subset_metrics
from visualization import viz_group_metrics
from database import save_batch_results


def analyze_matrix(
    matrix_path: Path,
    method: str,
    n_samples: int = 64,
    n_neighbors: int = 8,
    save_viz: bool = False,
    viz_dir: Path | None = None
) -> list[dict]:
    """
    Analyze a single weight matrix.

    Args:
        matrix_path: Path to the .pt matrix file
        method: Pruning method ('wanda' or 'sparsegpt')
        n_samples: Number of random features to sample
        n_neighbors: Number of nearest neighbors to find for each feature
        save_viz: Whether to save visualizations
        viz_dir: Directory to save visualizations

    Returns:
        List of result dictionaries for database storage
    """
    matrix_name = matrix_path.name
    print(f"\n{'='*60}")
    print(f"Analyzing: {matrix_name} ({method})")
    print(f"{'='*60}")

    # Load and binarize matrix
    matrix = torch.load(matrix_path, weights_only=True)
    matrix = (matrix != 0).int()  # Convert to binary matrix

    matrix_shape = matrix.shape
    print(f"Matrix shape: {matrix_shape}")
    print(f"Density: {matrix.float().mean():.2%}")

    # Select random feature columns
    cols = select_feature_columns(matrix=matrix, n=n_samples)
    print(f"Selected {len(cols)} random features for analysis")

    results = []

    for i, feature_idx in enumerate(cols):
        print(f"\nFeature {i+1}/{len(cols)} (index {feature_idx}):")

        # Find nearest neighbors
        subset, indices, distances = find_most_similar_features(
            matrix,
            feature_idx=feature_idx,
            n_similar=n_neighbors
        )

        # Compute pairwise distances
        pairwise_dist = compute_pairwise_hamming_distances_efficient(subset)
        mean_dist = pairwise_dist.mean().item()

        # Compute metrics
        metrics = compute_subset_metrics(subset)

        print(f"  Mean pairwise Hamming distance: {mean_dist:.4f}")
        print(f"  All-zero rows: {metrics['zeros']}")
        print(f"  All-one rows: {metrics['ones']}")
        print(f"  Rows appearing >1 time: {metrics['duplicates']}")
        print(f"  Total duplicate instances: {metrics['total_dups']}")
        print(f"  Unique rows: {metrics['unique_rows']}")
        print(f"  Density: {metrics['density']:.2%}")

        # Store result
        results.append({
            'method': method,
            'matrix_name': matrix_name,
            'feature_idx': feature_idx,
            'n_neighbors': n_neighbors,
            'mean_distance': mean_dist,
            'metrics': metrics
        })

    # Save comprehensive visualization with all groups for this matrix
    if save_viz and viz_dir:
        viz_path = viz_dir / f"{method}_{matrix_name.replace('.pt', '')}_group_metrics.png"
        viz_group_metrics(results, method, matrix_name, viz_path, matrix_shape=matrix_shape)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Grouped hamming distance analysis for pruned weight matrices"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=64,
        help="Number of random features to sample per matrix"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=8,
        help="Number of nearest neighbors to find"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./characterization.db",
        help="Path to DuckDB database"
    )
    parser.add_argument(
        "--save-viz",
        action="store_true",
        help="Save visualizations of feature subsets"
    )
    parser.add_argument(
        "--viz-dir",
        type=str,
        default="./results/hamming_dist/",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["wanda", "sparsegpt", "both"],
        default="both",
        help="Which pruning method to analyze"
    )

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Create visualization directory if needed
    if args.save_viz:
        viz_dir = Path(args.viz_dir)
        viz_dir.mkdir(parents=True, exist_ok=True)
        print(f"Visualizations will be saved to: {viz_dir}")
    else:
        viz_dir = None

    # Load matrices
    wanda_matrices, sparsegpt_matrices = get_unstructured_matrices_layer_1()

    all_results = []

    # Analyze Wanda matrices
    if args.method in ["wanda", "both"]:
        print(f"\n{'#'*60}")
        print("ANALYZING WANDA MATRICES")
        print(f"{'#'*60}")

        for matrix_path in wanda_matrices:
            results = analyze_matrix(
                matrix_path,
                method="wanda",
                n_samples=args.n_samples,
                n_neighbors=args.n_neighbors,
                save_viz=args.save_viz,
                viz_dir=viz_dir
            )
            all_results.extend(results)

    # Analyze SparseGPT matrices
    if args.method in ["sparsegpt", "both"]:
        print(f"\n{'#'*60}")
        print("ANALYZING SPARSEGPT MATRICES")
        print(f"{'#'*60}")

        for matrix_path in sparsegpt_matrices:
            results = analyze_matrix(
                matrix_path,
                method="sparsegpt",
                n_samples=args.n_samples,
                n_neighbors=args.n_neighbors,
                save_viz=args.save_viz,
                viz_dir=viz_dir
            )
            all_results.extend(results)

    # Save all results to database
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS TO DATABASE")
    print(f"{'='*60}")
    save_batch_results(args.db_path, all_results)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total results: {len(all_results)}")
    print(f"Database: {args.db_path}")


if __name__ == "__main__":
    main()
