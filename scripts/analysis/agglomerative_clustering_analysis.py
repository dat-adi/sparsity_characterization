"""
Agglomerative (hierarchical) clustering analysis for SparseGPT weight matrices.

This script performs bottom-up hierarchical clustering based on Hamming distance,
starting with each feature as its own cluster and iteratively merging similar clusters.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import compute_hamming_distance_batch


def compute_hamming_distance_matrix(binary_matrix: torch.Tensor, max_features: int = None) -> np.ndarray:
    """
    Compute pairwise Hamming distance matrix for all features.

    Args:
        binary_matrix: Binary weight matrix [D, N]
        max_features: Optional limit on number of features to analyze

    Returns:
        Distance matrix [N, N] where entry (i,j) is Hamming distance between features i and j
    """
    n_features = binary_matrix.shape[1]

    if max_features is not None:
        n_features = min(n_features, max_features)
        binary_matrix = binary_matrix[:, :n_features]

    print(f"Computing pairwise Hamming distances for {n_features} features...")

    # Convert to numpy for scipy compatibility
    binary_np = binary_matrix.cpu().numpy().T  # [N, D]

    # Compute condensed distance matrix using scipy
    # This is more memory efficient than computing full matrix
    distances = pdist(binary_np, metric='hamming')

    return distances


def perform_agglomerative_clustering(
    distance_matrix: np.ndarray,
    method: str = 'ward',
    n_clusters: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform agglomerative clustering using scipy's hierarchical clustering.

    Args:
        distance_matrix: Condensed distance matrix from pdist
        method: Linkage method ('ward', 'complete', 'average', 'single')
        n_clusters: Number of clusters to form

    Returns:
        linkage_matrix: Hierarchical clustering encoded as linkage matrix
        cluster_labels: Cluster assignment for each feature
    """
    print(f"Performing agglomerative clustering with method='{method}', n_clusters={n_clusters}...")

    # Perform hierarchical clustering
    linkage_matrix = linkage(distance_matrix, method=method)

    # Cut dendrogram to get desired number of clusters
    cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    return linkage_matrix, cluster_labels


def compute_cluster_metrics(
    binary_matrix: torch.Tensor,
    cluster_labels: np.ndarray
) -> Dict:
    """
    Compute within-cluster and between-cluster distance metrics.

    Args:
        binary_matrix: Binary weight matrix [D, N]
        cluster_labels: Cluster assignment for each feature [N]

    Returns:
        Dictionary with clustering quality metrics
    """
    n_clusters = len(np.unique(cluster_labels))
    print(f"Computing metrics for {n_clusters} clusters...")

    within_distances = []
    cluster_sizes = []

    # Compute within-cluster distances
    for cluster_id in tqdm(range(1, n_clusters + 1), desc="Computing within-cluster distances"):
        cluster_mask = cluster_labels == cluster_id
        cluster_features = binary_matrix[:, cluster_mask]

        n_features = cluster_features.shape[1]
        cluster_sizes.append(n_features)

        if n_features < 2:
            within_distances.append(0.0)
            continue

        # Compute all pairwise distances within cluster
        distances = []
        for i in range(n_features):
            for j in range(i + 1, n_features):
                dist = (cluster_features[:, i] != cluster_features[:, j]).float().mean().item()
                distances.append(dist)

        within_distances.append(np.mean(distances) if distances else 0.0)

    # Compute between-cluster distances (sample to avoid O(n^2) explosion)
    between_distances = []
    n_samples_per_pair = 50

    print("Computing between-cluster distances...")
    for i in tqdm(range(1, n_clusters + 1)):
        for j in range(i + 1, n_clusters + 1):
            cluster_i_mask = cluster_labels == i
            cluster_j_mask = cluster_labels == j

            cluster_i_features = binary_matrix[:, cluster_i_mask]
            cluster_j_features = binary_matrix[:, cluster_j_mask]

            n_i = cluster_i_features.shape[1]
            n_j = cluster_j_features.shape[1]

            # Sample pairs
            n_samples = min(n_samples_per_pair, n_i * n_j)
            distances = []

            for _ in range(n_samples):
                idx_i = np.random.randint(0, n_i)
                idx_j = np.random.randint(0, n_j)
                dist = (cluster_i_features[:, idx_i] != cluster_j_features[:, idx_j]).float().mean().item()
                distances.append(dist)

            between_distances.extend(distances)

    # Compute summary statistics
    mean_within = np.mean(within_distances)
    mean_between = np.mean(between_distances)
    separation_ratio = mean_between / mean_within if mean_within > 0 else 0.0

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "within_distances": within_distances,
        "mean_within": float(mean_within),
        "std_within": float(np.std(within_distances)),
        "mean_between": float(mean_between),
        "std_between": float(np.std(between_distances)),
        "separation_ratio": float(separation_ratio)
    }


def analyze_matrix(
    matrix_path: str,
    max_features: int = 500,
    linkage_method: str = 'ward',
    n_clusters_list: List[int] = [5, 10, 20, 50]
) -> Dict:
    """
    Perform complete agglomerative clustering analysis on a weight matrix.

    Args:
        matrix_path: Path to .pt file containing weight matrix
        max_features: Maximum number of features to analyze (for computational efficiency)
        linkage_method: Linkage method for hierarchical clustering
        n_clusters_list: List of cluster counts to evaluate

    Returns:
        Dictionary containing all analysis results
    """
    print(f"\n{'='*60}")
    print(f"Analyzing: {matrix_path}")
    print(f"{'='*60}")

    # Load matrix (map to CPU if needed)
    matrix = torch.load(matrix_path, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")

    # Convert to binary
    binary_matrix = (matrix > 0).int()

    # Compute pairwise distance matrix
    distance_matrix = compute_hamming_distance_matrix(binary_matrix, max_features=max_features)

    results = {
        "matrix_path": str(matrix_path),
        "matrix_shape": list(matrix.shape),
        "max_features_analyzed": min(max_features, matrix.shape[1]) if max_features else matrix.shape[1],
        "linkage_method": linkage_method,
        "clusterings": []
    }

    # Try different numbers of clusters
    for n_clusters in n_clusters_list:
        print(f"\n--- Analyzing with {n_clusters} clusters ---")

        # Perform clustering
        linkage_matrix, cluster_labels = perform_agglomerative_clustering(
            distance_matrix=distance_matrix,
            method=linkage_method,
            n_clusters=n_clusters
        )

        # Compute metrics
        metrics = compute_cluster_metrics(
            binary_matrix=binary_matrix[:, :len(cluster_labels)],
            cluster_labels=cluster_labels
        )

        # Print summary
        print(f"\nClustering Summary (n_clusters={n_clusters}):")
        print(f"  Mean within-cluster distance: {metrics['mean_within']:.4f}")
        print(f"  Mean between-cluster distance: {metrics['mean_between']:.4f}")
        print(f"  Separation ratio: {metrics['separation_ratio']:.4f}")
        print(f"  Cluster size distribution: min={min(metrics['cluster_sizes'])}, "
              f"max={max(metrics['cluster_sizes'])}, mean={np.mean(metrics['cluster_sizes']):.1f}")

        results["clusterings"].append({
            "n_clusters": n_clusters,
            "metrics": metrics,
            "cluster_labels": cluster_labels.tolist()
        })

    return results


def main():
    """Run agglomerative clustering analysis on SparseGPT matrices."""

    data_dir = Path(__file__).parent.parent.parent / "data/sparsegpt_unstructured"
    results_dir = Path(__file__).parent.parent.parent / "results/metrics/agglomerative_clustering"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find all SparseGPT down_proj matrices (sorted by layer)
    matrix_paths = sorted(data_dir.glob("layer*-mlp.down_proj.pt"))

    if not matrix_paths:
        print("No matrices found! Please check the path.")
        return

    # Sort by layer number
    matrix_paths = sorted(matrix_paths, key=lambda p: int(p.stem.split('-')[0].replace('layer', '')))

    print(f"Found {len(matrix_paths)} matrices to analyze")
    print("Matrices:", [p.name for p in matrix_paths[:5]], "...")

    all_results = []

    # Analyze each matrix
    for matrix_path in matrix_paths[:3]:  # Start with first 3 layers
        results = analyze_matrix(
            matrix_path=str(matrix_path),
            max_features=500,  # Limit for computational efficiency
            linkage_method='ward',
            n_clusters_list=[5, 10, 20, 50, 100]
        )
        all_results.append(results)

    # Save results
    output_path = results_dir / "agglomerative_clustering_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
