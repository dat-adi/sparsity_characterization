"""
Agglomerative clustering analysis on FULL feature width for SparseGPT gate_proj matrices.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from typing import Dict, List
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm
import time

sys.path.insert(0, str(Path(__file__).parent.parent))


def compute_cluster_metrics(
    binary_matrix: torch.Tensor,
    cluster_labels: np.ndarray,
    n_samples_per_cluster: int = 100,
    n_samples_between: int = 50
) -> Dict:
    n_clusters = len(np.unique(cluster_labels))
    print(f"Computing metrics for {n_clusters} clusters...")

    within_distances = []
    cluster_sizes = []

    for cluster_id in tqdm(range(1, n_clusters + 1), desc="Computing within-cluster distances"):
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

    between_distances = []
    n_cluster_pairs = min(1000, (n_clusters * (n_clusters - 1)) // 2)

    sampled_pairs = set()
    while len(sampled_pairs) < n_cluster_pairs:
        i = np.random.randint(1, n_clusters + 1)
        j = np.random.randint(1, n_clusters + 1)
        if i < j:
            sampled_pairs.add((i, j))
        elif j < i:
            sampled_pairs.add((j, i))

    print(f"Computing between-cluster distances for {len(sampled_pairs)} cluster pairs...")
    for i, j in tqdm(sampled_pairs):
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
    mean_between = np.mean(between_distances)
    separation_ratio = mean_between / mean_within if mean_within > 0 else 0.0

    return {
        "n_clusters": n_clusters,
        "cluster_sizes": cluster_sizes,
        "mean_within": float(mean_within),
        "std_within": float(np.std(within_distances)),
        "mean_between": float(mean_between),
        "std_between": float(np.std(between_distances)),
        "separation_ratio": float(separation_ratio)
    }


def analyze_matrix_full(matrix_path: str, linkage_method: str = 'ward', n_clusters_list: List[int] = [10, 50, 100, 500, 1000, 2000]) -> Dict:
    print(f"\n{'='*60}")
    print(f"FULL ANALYSIS: {matrix_path}")
    print(f"{'='*60}")

    start_time = time.time()

    matrix = torch.load(matrix_path, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")

    binary_matrix = (matrix > 0).int()
    n_features = binary_matrix.shape[1]

    print(f"Computing pairwise Hamming distances for ALL {n_features} features...")
    binary_np = binary_matrix.cpu().numpy().T
    distances = pdist(binary_np, metric='hamming')
    print(f"Distance matrix computed (shape: {distances.shape})")

    print(f"Computing hierarchical clustering linkage with method='{linkage_method}'...")
    linkage_matrix = linkage(distances, method=linkage_method)
    print("Linkage matrix computed")

    results = {
        "matrix_path": str(matrix_path),
        "matrix_shape": list(matrix.shape),
        "linkage_method": linkage_method,
        "clusterings": []
    }

    for n_clusters in n_clusters_list:
        print(f"\n--- Analyzing with {n_clusters} clusters ---")
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        metrics = compute_cluster_metrics(binary_matrix=binary_matrix, cluster_labels=cluster_labels, n_samples_per_cluster=100, n_samples_between=50)

        print(f"\nClustering Summary (n_clusters={n_clusters}):")
        print(f"  Mean within-cluster distance: {metrics['mean_within']:.4f}")
        print(f"  Mean between-cluster distance: {metrics['mean_between']:.4f}")
        print(f"  Separation ratio: {metrics['separation_ratio']:.4f}")
        print(f"  Cluster size distribution: min={min(metrics['cluster_sizes'])}, max={max(metrics['cluster_sizes'])}, mean={np.mean(metrics['cluster_sizes']):.1f}")

        results["clusterings"].append({"n_clusters": n_clusters, "metrics": metrics})

    elapsed = time.time() - start_time
    results["computation_time_seconds"] = elapsed
    print(f"\n{'='*60}")
    print(f"Total computation time: {elapsed/60:.2f} minutes")
    print(f"{'='*60}")

    return results


def main():
    data_dir = Path(__file__).parent.parent.parent / "data/sparsegpt_unstructured"
    results_dir = Path(__file__).parent.parent.parent / "results/metrics/agglomerative_clustering/sparsegpt"
    results_dir.mkdir(parents=True, exist_ok=True)

    matrix_paths = sorted(data_dir.glob("layer*-mlp.gate_proj.pt"))
    matrix_paths = sorted(matrix_paths, key=lambda p: int(p.stem.split('-')[0].replace('layer', '')))

    print(f"Found {len(matrix_paths)} gate_proj matrices to analyze")

    all_results = []
    for matrix_path in matrix_paths[:3]:
        results = analyze_matrix_full(matrix_path=str(matrix_path), linkage_method='ward', n_clusters_list=[10, 50, 100, 500, 1000, 2000])
        all_results.append(results)

        output_path = results_dir / "agglomerative_clustering_gate_proj_full_results.json"
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"✓ Intermediate results saved to: {output_path}")

    print(f"\n{'='*60}")
    print(f"✓ FULL SparseGPT gate_proj analysis complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
