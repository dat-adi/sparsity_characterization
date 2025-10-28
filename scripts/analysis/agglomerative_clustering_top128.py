"""
Agglomerative clustering on top 128 most similar features to a randomly selected feature.

For a randomly chosen reference feature, find its 128 nearest neighbors by Hamming
distance, then perform hierarchical clustering on this subset.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import compute_hamming_distance_batch


def find_top_k_neighbors(
    binary_matrix: torch.Tensor,
    reference_idx: int,
    k: int = 128
) -> Tuple[List[int], List[float]]:
    """
    Find k nearest neighbors to reference feature by Hamming distance.

    Args:
        binary_matrix: Binary weight matrix [D, N]
        reference_idx: Index of reference feature
        k: Number of neighbors to find

    Returns:
        neighbor_indices: List of k nearest neighbor indices (excluding self)
        distances: Hamming distances to each neighbor
    """
    reference_vec = binary_matrix[:, reference_idx]

    # Compute Hamming distances to all features
    distances = compute_hamming_distance_batch(reference_vec, binary_matrix)

    # Sort by distance and get top k+1 (k neighbors + self)
    sorted_indices = torch.argsort(distances)

    # Exclude self (will be at index 0 with distance 0)
    neighbor_indices = sorted_indices[1:k+1].tolist()
    neighbor_distances = distances[sorted_indices[1:k+1]].tolist()

    return neighbor_indices, neighbor_distances


def compute_cluster_metrics(
    binary_matrix: torch.Tensor,
    cluster_labels: np.ndarray,
    n_samples_per_cluster: int = 50,
    n_samples_between: int = 30
) -> Dict:
    """Compute within and between cluster metrics."""
    n_clusters = len(np.unique(cluster_labels))

    within_distances = []
    cluster_sizes = []

    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = cluster_labels == cluster_id
        cluster_features = binary_matrix[:, cluster_mask]

        n_features = cluster_features.shape[1]
        cluster_sizes.append(n_features)

        if n_features < 2:
            within_distances.append(0.0)
            continue

        n_possible_pairs = (n_features * (n_features - 1)) // 2
        n_samples = min(n_samples_per_cluster, n_possible_pairs)

        distances = []
        for _ in range(n_samples):
            i = np.random.randint(0, n_features)
            j = np.random.randint(0, n_features)
            if i != j:
                dist = (cluster_features[:, i] != cluster_features[:, j]).float().mean().item()
                distances.append(dist)

        within_distances.append(np.mean(distances) if distances else 0.0)

    # Between-cluster distances
    between_distances = []
    n_cluster_pairs = min(100, (n_clusters * (n_clusters - 1)) // 2)

    sampled_pairs = set()
    attempts = 0
    while len(sampled_pairs) < n_cluster_pairs and attempts < n_cluster_pairs * 3:
        i = np.random.randint(1, n_clusters + 1)
        j = np.random.randint(1, n_clusters + 1)
        if i < j:
            sampled_pairs.add((i, j))
        elif j < i:
            sampled_pairs.add((j, i))
        attempts += 1

    for i, j in sampled_pairs:
        cluster_i_mask = cluster_labels == i
        cluster_j_mask = cluster_labels == j

        cluster_i_features = binary_matrix[:, cluster_i_mask]
        cluster_j_features = binary_matrix[:, cluster_j_mask]

        n_i = cluster_i_features.shape[1]
        n_j = cluster_j_features.shape[1]

        n_samples = min(n_samples_between, n_i * n_j)
        distances = []

        for _ in range(n_samples):
            idx_i = np.random.randint(0, n_i)
            idx_j = np.random.randint(0, n_j)
            dist = (cluster_i_features[:, idx_i] != cluster_j_features[:, idx_j]).float().mean().item()
            distances.append(dist)

        between_distances.extend(distances)

    mean_within = np.mean(within_distances)
    mean_between = np.mean(between_distances) if between_distances else 0.0
    separation_ratio = mean_between / mean_within if mean_within > 0 else 0.0

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "mean_within": float(mean_within),
        "std_within": float(np.std(within_distances)),
        "mean_between": float(mean_between),
        "std_between": float(np.std(between_distances)) if between_distances else 0.0,
        "separation_ratio": float(separation_ratio)
    }


def analyze_top_k_clustering(
    matrix_path: str,
    k_neighbors: int = 128,
    reference_idx: int = None,
    n_clusters_list: List[int] = [3, 5, 10, 20, 32]
) -> Dict:
    """
    Perform agglomerative clustering on top k neighbors of a reference feature.

    Args:
        matrix_path: Path to weight matrix
        k_neighbors: Number of neighbors to analyze
        reference_idx: Reference feature index (random if None)
        n_clusters_list: Cluster counts to evaluate

    Returns:
        Dictionary with analysis results
    """
    print(f"\n{'='*60}")
    print(f"TOP-{k_neighbors} NEIGHBOR CLUSTERING: {Path(matrix_path).name}")
    print(f"{'='*60}")

    start_time = time.time()

    # Load and binarize
    matrix = torch.load(matrix_path, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")

    binary_matrix = (matrix > 0).int()
    n_features = binary_matrix.shape[1]

    # Select reference feature
    if reference_idx is None:
        reference_idx = np.random.randint(0, n_features)

    print(f"Reference feature: {reference_idx} (out of {n_features})")

    # Find top k neighbors
    print(f"Finding top {k_neighbors} nearest neighbors by Hamming distance...")
    neighbor_indices, neighbor_distances = find_top_k_neighbors(
        binary_matrix, reference_idx, k_neighbors
    )

    # Include reference in the subset
    all_indices = [reference_idx] + neighbor_indices
    subset_matrix = binary_matrix[:, all_indices]

    print(f"Subset shape: {subset_matrix.shape}")
    print(f"Hamming distance range: {min(neighbor_distances):.4f} - {max(neighbor_distances):.4f}")
    print(f"Mean Hamming distance: {np.mean(neighbor_distances):.4f}")

    # Compute pairwise distances
    print(f"Computing pairwise Hamming distances for {len(all_indices)} features...")
    subset_np = subset_matrix.cpu().numpy().T
    distances = pdist(subset_np, metric='hamming')

    # Compute linkage
    print("Computing hierarchical clustering linkage...")
    linkage_matrix = linkage(distances, method='ward')

    results = {
        "matrix_path": str(matrix_path),
        "matrix_shape": list(matrix.shape),
        "reference_feature_idx": reference_idx,
        "k_neighbors": k_neighbors,
        "neighbor_indices": neighbor_indices,
        "neighbor_hamming_distances": neighbor_distances,
        "mean_neighbor_distance": float(np.mean(neighbor_distances)),
        "clusterings": []
    }

    # Try different cluster counts
    for n_clusters in n_clusters_list:
        if n_clusters > len(all_indices):
            print(f"Skipping k={n_clusters} (more than {len(all_indices)} features)")
            continue

        print(f"\n--- Analyzing with {n_clusters} clusters ---")

        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        metrics = compute_cluster_metrics(
            binary_matrix=subset_matrix,
            cluster_labels=cluster_labels,
            n_samples_per_cluster=50,
            n_samples_between=30
        )

        print(f"  Mean within-cluster distance: {metrics['mean_within']:.4f}")
        print(f"  Mean between-cluster distance: {metrics['mean_between']:.4f}")
        print(f"  Separation ratio: {metrics['separation_ratio']:.4f}")

        results["clusterings"].append({
            "n_clusters": n_clusters,
            "metrics": metrics,
            "cluster_labels": cluster_labels.tolist()
        })

    elapsed = time.time() - start_time
    results["computation_time_seconds"] = elapsed

    print(f"\n{'='*60}")
    print(f"Computation time: {elapsed:.2f} seconds")
    print(f"{'='*60}")

    return results


def create_dendrogram_visualization(
    matrix_path: str,
    k_neighbors: int = 128,
    reference_idx: int = None,
    output_dir: Path = None
):
    """Create dendrogram for top k neighbors."""

    # Load and process
    matrix = torch.load(matrix_path, map_location=torch.device('cpu'))
    binary_matrix = (matrix > 0).int()
    n_features = binary_matrix.shape[1]

    if reference_idx is None:
        reference_idx = np.random.randint(0, n_features)

    # Find neighbors
    neighbor_indices, neighbor_distances = find_top_k_neighbors(
        binary_matrix, reference_idx, k_neighbors
    )

    all_indices = [reference_idx] + neighbor_indices
    subset_matrix = binary_matrix[:, all_indices]

    # Compute linkage
    subset_np = subset_matrix.cpu().numpy().T
    distances = pdist(subset_np, metric='hamming')
    linkage_matrix = linkage(distances, method='ward')

    # Create dendrogram
    fig, ax = plt.subplots(figsize=(16, 8))

    dendrogram(
        linkage_matrix,
        ax=ax,
        no_labels=True,
        color_threshold=0.3 * max(linkage_matrix[:, 2]),
        above_threshold_color='gray'
    )

    matrix_name = Path(matrix_path).stem
    ax.set_title(f'Top-{k_neighbors} Neighbor Clustering: {matrix_name}\n'
                 f'Reference Feature: {reference_idx}',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index (in subset)', fontsize=12)
    ax.set_ylabel('Hamming Distance', fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_dir:
        output_path = output_dir / f"{matrix_name}_top{k_neighbors}_dendrogram_ref{reference_idx}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved dendrogram: {output_path}")

    plt.close()


def main():
    """Run top-k neighbor clustering analysis."""

    # Select example matrices
    data_dirs = {
        'sparsegpt': Path(__file__).parent.parent.parent / "data/sparsegpt_unstructured",
        'wanda': Path(__file__).parent.parent.parent / "data/wanda_unstructured"
    }

    results_dir = Path(__file__).parent.parent.parent / "results/metrics/agglomerative_clustering/top128"
    results_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = Path(__file__).parent.parent.parent / "results/visualizations/agglomerative_clustering/top128"
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Analyze one matrix per method and projection type
    test_cases = [
        ('sparsegpt', 'down_proj'),
        ('sparsegpt', 'up_proj'),
        ('sparsegpt', 'gate_proj'),
        ('wanda', 'down_proj'),
        ('wanda', 'up_proj'),
        ('wanda', 'gate_proj'),
    ]

    for method, proj_type in test_cases:
        data_dir = data_dirs[method]

        # Find layer 0 matrix
        pattern = f"layer0-mlp.{proj_type}.pt"
        matrix_paths = list(data_dir.glob(pattern))

        if not matrix_paths:
            print(f"Warning: No {pattern} found for {method}")
            continue

        matrix_path = matrix_paths[0]

        # Set random seed for reproducibility
        np.random.seed(42)

        # Run analysis
        results = analyze_top_k_clustering(
            matrix_path=str(matrix_path),
            k_neighbors=128,
            reference_idx=None,  # Random
            n_clusters_list=[3, 5, 10, 20, 32, 64]
        )
        all_results.append(results)

        # Create dendrogram
        create_dendrogram_visualization(
            matrix_path=str(matrix_path),
            k_neighbors=128,
            reference_idx=results['reference_feature_idx'],
            output_dir=viz_dir
        )

    # Save results
    output_path = results_dir / "top128_clustering_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ All results saved to: {output_path}")
    print(f"✓ Visualizations in: {viz_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
