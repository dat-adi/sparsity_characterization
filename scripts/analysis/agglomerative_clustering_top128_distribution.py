"""
Sample 10% of features and perform top-128 neighbor clustering to get distribution of stats.

For each sampled reference feature, find its 128 nearest neighbors and perform
agglomerative clustering, collecting statistics across all samples.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from typing import Dict, List, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import compute_hamming_distance_batch


def find_top_k_neighbors(
    binary_matrix: torch.Tensor,
    reference_idx: int,
    k: int = 128
) -> Tuple[List[int], List[float]]:
    """Find k nearest neighbors to reference feature by Hamming distance."""
    reference_vec = binary_matrix[:, reference_idx]
    distances = compute_hamming_distance_batch(reference_vec, binary_matrix)
    sorted_indices = torch.argsort(distances)
    neighbor_indices = sorted_indices[1:k+1].tolist()
    neighbor_distances = distances[sorted_indices[1:k+1]].tolist()
    return neighbor_indices, neighbor_distances


def compute_cluster_metrics_fast(
    binary_matrix: torch.Tensor,
    cluster_labels: np.ndarray,
    n_samples: int = 30
) -> Dict:
    """Fast approximation of cluster metrics using sampling."""
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

        distances = []
        for _ in range(min(n_samples, (n_features * (n_features - 1)) // 2)):
            i = np.random.randint(0, n_features)
            j = np.random.randint(0, n_features)
            if i != j:
                dist = (cluster_features[:, i] != cluster_features[:, j]).float().mean().item()
                distances.append(dist)

        within_distances.append(np.mean(distances) if distances else 0.0)

    # Between-cluster (sample fewer pairs for speed)
    between_distances = []
    n_cluster_pairs = min(50, (n_clusters * (n_clusters - 1)) // 2)

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

        for _ in range(min(20, n_i * n_j)):
            idx_i = np.random.randint(0, n_i)
            idx_j = np.random.randint(0, n_j)
            dist = (cluster_i_features[:, idx_i] != cluster_j_features[:, idx_j]).float().mean().item()
            between_distances.append(dist)

    mean_within = np.mean(within_distances)
    mean_between = np.mean(between_distances) if between_distances else 0.0
    separation_ratio = mean_between / mean_within if mean_within > 0 else 0.0

    return {
        "n_clusters": n_clusters,
        "mean_within": float(mean_within),
        "mean_between": float(mean_between),
        "separation_ratio": float(separation_ratio)
    }


def analyze_distribution(
    matrix_path: str,
    k_neighbors: int = 128,
    sample_ratio: float = 0.1,
    n_clusters: int = 32,
    seed: int = 42
) -> Dict:
    """
    Sample features and collect clustering statistics distribution.

    Args:
        matrix_path: Path to weight matrix
        k_neighbors: Number of neighbors per sample
        sample_ratio: Fraction of features to sample (0.1 = 10%)
        n_clusters: Number of clusters for each sample
        seed: Random seed

    Returns:
        Dictionary with distribution statistics
    """
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"DISTRIBUTION ANALYSIS: {Path(matrix_path).name}")
    print(f"{'='*60}")

    # Load matrix
    matrix = torch.load(matrix_path, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")

    binary_matrix = (matrix > 0).int()
    n_features = binary_matrix.shape[1]

    # Sample features
    n_samples = int(n_features * sample_ratio)
    sample_indices = np.random.choice(n_features, size=n_samples, replace=False)

    print(f"Sampling {n_samples} features ({sample_ratio*100:.0f}% of {n_features})")
    print(f"Performing top-{k_neighbors} neighbor clustering with k={n_clusters} clusters")

    # Collect statistics
    separation_ratios = []
    within_distances = []
    between_distances = []
    mean_neighbor_distances = []

    for ref_idx in tqdm(sample_indices, desc="Processing samples"):
        # Find neighbors
        neighbor_indices, neighbor_dists = find_top_k_neighbors(
            binary_matrix, int(ref_idx), k_neighbors
        )

        mean_neighbor_distances.append(np.mean(neighbor_dists))

        # Create subset
        all_indices = [int(ref_idx)] + neighbor_indices
        subset_matrix = binary_matrix[:, all_indices]

        # Compute linkage
        subset_np = subset_matrix.cpu().numpy().T
        distances = pdist(subset_np, metric='hamming')
        linkage_matrix = linkage(distances, method='ward')

        # Cluster
        cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Compute metrics
        metrics = compute_cluster_metrics_fast(subset_matrix, cluster_labels, n_samples=30)

        separation_ratios.append(metrics['separation_ratio'])
        within_distances.append(metrics['mean_within'])
        between_distances.append(metrics['mean_between'])

    # Compute distribution statistics
    results = {
        "matrix_path": str(matrix_path),
        "matrix_shape": list(matrix.shape),
        "parameters": {
            "k_neighbors": k_neighbors,
            "n_clusters": n_clusters,
            "sample_ratio": sample_ratio,
            "n_samples": n_samples,
            "seed": seed
        },
        "distributions": {
            "separation_ratio": {
                "mean": float(np.mean(separation_ratios)),
                "std": float(np.std(separation_ratios)),
                "median": float(np.median(separation_ratios)),
                "min": float(np.min(separation_ratios)),
                "max": float(np.max(separation_ratios)),
                "q25": float(np.percentile(separation_ratios, 25)),
                "q75": float(np.percentile(separation_ratios, 75)),
                "values": separation_ratios
            },
            "within_distance": {
                "mean": float(np.mean(within_distances)),
                "std": float(np.std(within_distances)),
                "median": float(np.median(within_distances)),
                "values": within_distances
            },
            "between_distance": {
                "mean": float(np.mean(between_distances)),
                "std": float(np.std(between_distances)),
                "median": float(np.median(between_distances)),
                "values": between_distances
            },
            "mean_neighbor_distance": {
                "mean": float(np.mean(mean_neighbor_distances)),
                "std": float(np.std(mean_neighbor_distances)),
                "median": float(np.median(mean_neighbor_distances)),
                "values": mean_neighbor_distances
            }
        }
    }

    # Print summary
    print(f"\n{'='*60}")
    print("DISTRIBUTION SUMMARY")
    print(f"{'='*60}")
    print(f"Separation Ratio:")
    print(f"  Mean:   {results['distributions']['separation_ratio']['mean']:.4f}")
    print(f"  Median: {results['distributions']['separation_ratio']['median']:.4f}")
    print(f"  Std:    {results['distributions']['separation_ratio']['std']:.4f}")
    print(f"  Range:  [{results['distributions']['separation_ratio']['min']:.4f}, "
          f"{results['distributions']['separation_ratio']['max']:.4f}]")
    print(f"\nWithin-cluster distance:")
    print(f"  Mean:   {results['distributions']['within_distance']['mean']:.4f}")
    print(f"  Median: {results['distributions']['within_distance']['median']:.4f}")
    print(f"\nBetween-cluster distance:")
    print(f"  Mean:   {results['distributions']['between_distance']['mean']:.4f}")
    print(f"  Median: {results['distributions']['between_distance']['median']:.4f}")
    print(f"{'='*60}")

    return results


def create_distribution_plots(all_results: List[Dict], output_dir: Path):
    """Create distribution visualization plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    methods_projs = []
    for r in all_results:
        matrix_name = Path(r['matrix_path']).name
        method = 'SparseGPT' if 'sparsegpt' in r['matrix_path'] else 'Wanda'
        proj = matrix_name.split('-')[1].split('.')[1]
        methods_projs.append((method, proj, r))

    colors = {'down_proj': '#2E86AB', 'up_proj': '#A23B72', 'gate_proj': '#F18F01'}

    # Plot 1: Separation ratio distributions
    ax = axes[0, 0]
    for method, proj, r in methods_projs:
        if method == 'SparseGPT':
            data = r['distributions']['separation_ratio']['values']
            ax.hist(data, bins=30, alpha=0.5, label=proj, color=colors[proj], edgecolor='black')
    ax.set_xlabel('Separation Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('SparseGPT: Separation Ratio Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Wanda separation ratio distributions
    ax = axes[0, 1]
    for method, proj, r in methods_projs:
        if method == 'Wanda':
            data = r['distributions']['separation_ratio']['values']
            ax.hist(data, bins=30, alpha=0.5, label=proj, color=colors[proj], edgecolor='black')
    ax.set_xlabel('Separation Ratio', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Wanda: Separation Ratio Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 3: Box plots comparison
    ax = axes[0, 2]
    sparsegpt_data = []
    wanda_data = []
    labels = []

    for proj in ['down_proj', 'up_proj', 'gate_proj']:
        for method, p, r in methods_projs:
            if p == proj:
                if method == 'SparseGPT':
                    sparsegpt_data.append(r['distributions']['separation_ratio']['values'])
                else:
                    wanda_data.append(r['distributions']['separation_ratio']['values'])
        labels.append(proj.replace('_', '\n'))

    positions_sg = np.arange(len(labels)) * 2
    positions_w = positions_sg + 0.7

    bp1 = ax.boxplot(sparsegpt_data, positions=positions_sg, widths=0.6,
                     patch_artist=True, label='SparseGPT')
    bp2 = ax.boxplot(wanda_data, positions=positions_w, widths=0.6,
                     patch_artist=True, label='Wanda')

    for patch in bp1['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    for patch in bp2['boxes']:
        patch.set_facecolor('#e74c3c')
        patch.set_alpha(0.7)

    ax.set_xticks(positions_sg + 0.35)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Separation Ratio', fontsize=12, fontweight='bold')
    ax.set_title('Comparison Across Projections', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Plot 4-6: Mean vs Std scatter plots
    for idx, (metric_name, ax_pos) in enumerate([
        ('separation_ratio', (1, 0)),
        ('within_distance', (1, 1)),
        ('between_distance', (1, 2))
    ]):
        ax = axes[ax_pos]

        for method, proj, r in methods_projs:
            dist = r['distributions'][metric_name]
            marker = 'o' if method == 'SparseGPT' else 's'
            ax.scatter(dist['mean'], dist['std'],
                      s=150, alpha=0.7, label=f"{method} {proj}",
                      color=colors[proj], marker=marker, edgecolor='black', linewidth=1.5)

        ax.set_xlabel('Mean', fontsize=11, fontweight='bold')
        ax.set_ylabel('Std Dev', fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Top-128 Neighbor Clustering: Distribution Analysis\n(10% sample, k=32 clusters)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / "top128_distribution_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved visualization: {output_path}")
    plt.close()


def main():
    """Run distribution analysis on all projection types."""

    data_dirs = {
        'sparsegpt': Path(__file__).parent.parent.parent / "data/sparsegpt_unstructured",
        'wanda': Path(__file__).parent.parent.parent / "data/wanda_unstructured"
    }

    results_dir = Path(__file__).parent.parent.parent / "results/metrics/agglomerative_clustering/top128_dist"
    results_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = Path(__file__).parent.parent.parent / "results/visualizations/agglomerative_clustering/top128_dist"
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

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
        pattern = f"layer0-mlp.{proj_type}.pt"
        matrix_paths = list(data_dir.glob(pattern))

        if not matrix_paths:
            print(f"Warning: No {pattern} found for {method}")
            continue

        matrix_path = matrix_paths[0]

        results = analyze_distribution(
            matrix_path=str(matrix_path),
            k_neighbors=128,
            sample_ratio=0.1,  # 10% sample
            n_clusters=32,
            seed=42
        )
        all_results.append(results)

    # Save results
    output_path = results_dir / "top128_distribution_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Create visualizations
    create_distribution_plots(all_results, viz_dir)

    print(f"\n{'='*60}")
    print(f"✓ Results saved to: {output_path}")
    print(f"✓ Visualizations in: {viz_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
