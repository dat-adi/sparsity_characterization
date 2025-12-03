"""
Exhaustive sampling analysis without replacement.

This script performs clustering analysis on the entire matrix by:
1. Sampling features without replacement
2. Processing all 4096 features into groups of 8 (512 groups)
3. For each group, finding the 7 nearest neighbors to the selected feature
4. Computing pairwise distances and metrics for each group

The last group may have fewer than 8 features if the matrix width
is not perfectly divisible by 8.
"""

import torch
from pathlib import Path
import argparse
from typing import List, Dict
from tqdm import tqdm

from data_loaders import set_seed, get_unstructured_matrices_layer_1
from hamming_utils import compute_hamming_distance_batch, compute_pairwise_hamming_distances_efficient
from matrix_analysis import compute_subset_metrics
from visualization import viz_group_metrics
from database import save_batch_results


def find_nearest_neighbors_excluding(
    matrix: torch.Tensor,
    feature_idx: int,
    n_neighbors: int,
    excluded_indices: set[int]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Find the n nearest neighbors to a feature, excluding already-used features.

    Args:
        matrix: Weight matrix [D, N] where N is number of features
        feature_idx: Index of the reference feature
        n_neighbors: Number of neighbors to find (excluding the reference itself)
        excluded_indices: Set of feature indices to exclude from selection

    Returns:
        subset: Matrix of selected features [D, n_neighbors+1] (includes reference)
        indices: Indices of selected features [n_neighbors+1]
        distances: Hamming distances of selected features [n_neighbors+1]
    """
    reference_vec = matrix[:, feature_idx]

    # Compute Hamming distances to all features
    distances = compute_hamming_distance_batch(reference_vec, matrix)

    # Set distance of excluded indices to infinity so they won't be selected
    for idx in excluded_indices:
        distances[idx] = float('inf')

    # Get indices of n_neighbors+1 closest features (including the reference itself)
    sorted_indices = torch.argsort(distances)
    # Filter out any remaining infinity values
    valid_indices = sorted_indices[distances[sorted_indices] != float('inf')]
    top_indices = valid_indices[:n_neighbors + 1]

    subset = matrix[:, top_indices]
    selected_distances = distances[top_indices]

    return subset, top_indices, selected_distances


def analyze_matrix_exhaustive(
    matrix_path: Path,
    method: str,
    group_size: int = 8,
    save_viz: bool = False,
    viz_dir: Path | None = None
) -> List[Dict]:
    """
    Analyze entire matrix using exhaustive sampling without replacement.

    Args:
        matrix_path: Path to the .pt matrix file
        method: Pruning method ('wanda' or 'sparsegpt')
        group_size: Size of each group (default: 8)
        save_viz: Whether to save visualizations
        viz_dir: Directory to save visualizations

    Returns:
        List of result dictionaries for database storage
    """
    matrix_name = matrix_path.name
    print(f"\n{'='*60}")
    print(f"Analyzing: {matrix_name} ({method})")
    print(f"Exhaustive sampling without replacement")
    print(f"{'='*60}")

    # Load and binarize matrix
    matrix = torch.load(matrix_path, weights_only=True)
    matrix = (matrix != 0).int()  # Convert to binary matrix

    matrix_shape = matrix.shape
    n_features = matrix_shape[1]
    print(f"Matrix shape: {matrix_shape}")
    print(f"Density: {matrix.float().mean():.2%}")

    # Calculate number of complete groups
    n_complete_groups = n_features // group_size
    n_remaining = n_features % group_size

    print(f"Total features: {n_features}")
    print(f"Group size: {group_size}")
    print(f"Complete groups: {n_complete_groups}")
    if n_remaining > 0:
        print(f"Remaining features (partial group): {n_remaining}")
    print(f"Total groups: {n_complete_groups + (1 if n_remaining > 0 else 0)}")

    # Track which features have been used
    excluded_indices = set()
    results = []

    # Process complete groups with progress bar
    total_groups = n_complete_groups + (1 if n_remaining > 0 else 0)
    pbar = tqdm(total=total_groups, desc="Processing groups", unit="group")

    import random

    for group_idx in range(n_complete_groups):
        # Select a random feature from those not yet used
        available_features = [i for i in range(n_features) if i not in excluded_indices]

        if len(available_features) < group_size:
            pbar.write(f"Warning: Only {len(available_features)} features available, cannot form complete group")
            break

        # Pick a random seed feature for this group
        seed_feature_idx = random.choice(available_features)

        # Find nearest neighbors, excluding already-used features
        subset, indices, distances = find_nearest_neighbors_excluding(
            matrix,
            feature_idx=seed_feature_idx,
            n_neighbors=group_size - 1,  # -1 because we include the seed
            excluded_indices=excluded_indices
        )

        # Check if we got the full group
        actual_group_size = indices.shape[0]
        if actual_group_size < group_size:
            pbar.write(f"Warning: Only found {actual_group_size} features (expected {group_size})")

        # Mark these features as used
        for idx in indices.tolist():
            excluded_indices.add(idx)

        # Compute pairwise distances
        pairwise_dist = compute_pairwise_hamming_distances_efficient(subset)
        mean_dist = pairwise_dist.mean().item()

        # Compute metrics
        metrics = compute_subset_metrics(subset)

        # Show sample metrics every 50 groups
        if (group_idx + 1) % 50 == 0 or group_idx == 0:
            pbar.write(f"\n  Group {group_idx+1} Sample:")
            pbar.write(f"    Seed: {seed_feature_idx}, Mean dist: {mean_dist:.4f}, Density: {metrics['density']:.2%}")
            pbar.write(f"    Zeros: {metrics['zeros']}, Ones: {metrics['ones']}, Unique: {metrics['unique_rows']}")

        # Store result
        results.append({
            'method': method,
            'matrix_name': matrix_name,
            'group_idx': group_idx,
            'seed_feature_idx': seed_feature_idx,
            'feature_indices': indices.tolist(),
            'n_neighbors': actual_group_size - 1,
            'mean_distance': mean_dist,
            'metrics': metrics,
            'is_complete_group': actual_group_size == group_size
        })

        pbar.update(1)

    # Handle remaining features (partial group) if any
    if n_remaining > 0 and len(excluded_indices) < n_features:
        remaining_features = [i for i in range(n_features) if i not in excluded_indices]
        pbar.write(f"\nProcessing partial group ({len(remaining_features)} remaining features)")

        if len(remaining_features) >= 2:
            # Pick first remaining feature as seed
            seed_feature_idx = remaining_features[0]

            # Find neighbors from remaining features only
            subset, indices, distances = find_nearest_neighbors_excluding(
                matrix,
                feature_idx=seed_feature_idx,
                n_neighbors=len(remaining_features) - 1,
                excluded_indices=excluded_indices
            )

            actual_group_size = indices.shape[0]

            # Compute pairwise distances
            pairwise_dist = compute_pairwise_hamming_distances_efficient(subset)
            mean_dist = pairwise_dist.mean().item()

            # Compute metrics
            metrics = compute_subset_metrics(subset)

            pbar.write(f"  Partial group - Seed: {seed_feature_idx}, Size: {actual_group_size}, Mean dist: {mean_dist:.4f}")

            # Store result
            results.append({
                'method': method,
                'matrix_name': matrix_name,
                'group_idx': n_complete_groups,
                'seed_feature_idx': seed_feature_idx,
                'feature_indices': indices.tolist(),
                'n_neighbors': actual_group_size - 1,
                'mean_distance': mean_dist,
                'metrics': metrics,
                'is_complete_group': False
            })
        else:
            pbar.write(f"  Skipping partial group: Need at least 2 features")

        pbar.update(1)

    pbar.close()

    # Save visualization if requested
    if save_viz and viz_dir:
        viz_path = viz_dir / f"{method}_{matrix_name.replace('.pt', '')}_exhaustive_groups.png"
        # Convert to format expected by viz_group_metrics
        viz_results = []
        for r in results:
            viz_results.append({
                'method': r['method'],
                'matrix_name': r['matrix_name'],
                'feature_idx': r['seed_feature_idx'],
                'n_neighbors': r['n_neighbors'],
                'mean_distance': r['mean_distance'],
                'metrics': r['metrics']
            })
        viz_group_metrics(viz_results, method, matrix_name, viz_path, matrix_shape=matrix_shape)

    print(f"\n{'='*60}")
    print(f"Summary for {matrix_name}:")
    print(f"  Total groups created: {len(results)}")
    print(f"  Complete groups: {sum(1 for r in results if r['is_complete_group'])}")
    print(f"  Partial groups: {sum(1 for r in results if not r['is_complete_group'])}")
    print(f"  Features utilized: {len(excluded_indices)}/{n_features}")
    print(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Exhaustive hamming distance analysis without replacement"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=8,
        help="Size of each feature group"
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
        default="./results/hamming_dist/exhaustive/",
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
        print("ANALYZING WANDA MATRICES - EXHAUSTIVE SAMPLING")
        print(f"{'#'*60}")

        for matrix_path in wanda_matrices:
            results = analyze_matrix_exhaustive(
                matrix_path,
                method="wanda",
                group_size=args.group_size,
                save_viz=args.save_viz,
                viz_dir=viz_dir
            )
            all_results.extend(results)

    # Analyze SparseGPT matrices
    if args.method in ["sparsegpt", "both"]:
        print(f"\n{'#'*60}")
        print("ANALYZING SPARSEGPT MATRICES - EXHAUSTIVE SAMPLING")
        print(f"{'#'*60}")

        for matrix_path in sparsegpt_matrices:
            results = analyze_matrix_exhaustive(
                matrix_path,
                method="sparsegpt",
                group_size=args.group_size,
                save_viz=args.save_viz,
                viz_dir=viz_dir
            )
            all_results.extend(results)

    # Save all results to database
    print(f"\n{'='*60}")
    print(f"SAVING RESULTS TO DATABASE")
    print(f"{'='*60}")

    # Convert results to format expected by save_batch_results
    db_results = []
    for r in all_results:
        db_results.append({
            'method': r['method'],
            'matrix_name': r['matrix_name'],
            'feature_idx': r['seed_feature_idx'],
            'n_neighbors': r['n_neighbors'],
            'mean_distance': r['mean_distance'],
            'metrics': r['metrics']
        })

    save_batch_results(args.db_path, db_results)

    print(f"\n{'='*60}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Total groups: {len(all_results)}")
    print(f"Complete groups: {sum(1 for r in all_results if r['is_complete_group'])}")
    print(f"Partial groups: {sum(1 for r in all_results if not r['is_complete_group'])}")
    print(f"Database: {args.db_path}")


if __name__ == "__main__":
    main()
