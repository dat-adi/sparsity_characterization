"""
Multi-seed clustering analysis for feature co-activation patterns.

This script performs clustering analysis across multiple random seeds to understand
the stability and structure of feature co-activation in sparse weight matrices.

For each seed, it:
1. Randomly selects a feature
2. Finds the top 128 most similar features (by Hamming distance)
3. Performs k-means clustering with k=[4, 8, 16]
4. Computes and visualizes cluster metrics

Usage:
    python multi_seed_clustering_analysis.py --method wanda --layer 1 --component mlp.down_proj
    python multi_seed_clustering_analysis.py --method sparsegpt --layer 1 --component self_attn.q_proj --seeds 20
"""

import sys
from pathlib import Path
import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import find_most_similar_features
from utils.clustering import compute_cluster_metrics, ClusterMetrics
from utils.visualization import (
    create_clustered_spy_plot,
    plot_cluster_statistics,
    plot_separation_ratio
)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def identify_feature(matrix: torch.Tensor, seed: int) -> int:
    """
    Randomly select a feature index from the matrix.

    Args:
        matrix: Weight matrix [D, N] where N is number of features
        seed: Random seed for reproducibility

    Returns:
        Selected feature index
    """
    set_seed(seed)
    n_features = matrix.shape[1]
    feature_idx = np.random.randint(0, n_features)
    return feature_idx


def get_closest_features(
    selected_feature: int,
    matrix: torch.Tensor,
    n_features: int,
    distance_type: str = "hamming"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get the closest features to the selected feature.

    Args:
        selected_feature: Index of the reference feature
        matrix: Weight matrix [D, N]
        n_features: Number of similar features to retrieve
        distance_type: Type of distance metric (currently only "hamming")

    Returns:
        subset: Matrix of selected features [D, n_features]
        indices: Indices of selected features
        distances: Distances of selected features
    """
    if distance_type != "hamming":
        raise ValueError(f"Unsupported distance type: {distance_type}")

    return find_most_similar_features(matrix, selected_feature, n_features)


def get_clusters(
    feature_subset: torch.Tensor,
    method: str,
    k: int,
    random_state: int = 42
) -> tuple[torch.Tensor, np.ndarray]:
    """
    Perform clustering on feature subset.

    Args:
        feature_subset: Feature matrix [D, N]
        method: Clustering method (currently only "kmeans")
        k: Number of clusters
        random_state: Random seed for clustering

    Returns:
        feature_subset: Original feature matrix
        labels: Cluster assignments for each feature
    """
    if method != "kmeans":
        raise ValueError(f"Unsupported clustering method: {method}")

    # Compute cluster metrics (includes k-means clustering)
    metrics = compute_cluster_metrics(feature_subset, k, random_state=random_state)

    # Re-run k-means to get labels (metrics function doesn't return them)
    from sklearn.cluster import KMeans
    features_T = feature_subset.T.cpu().numpy()
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(features_T)

    return feature_subset, labels


def get_cluster_metrics(
    feature_subset: torch.Tensor,
    labels: np.ndarray,
    k: int,
    random_state: int = 42
) -> ClusterMetrics:
    """
    Compute clustering metrics.

    Args:
        feature_subset: Feature matrix [D, N]
        labels: Cluster assignments (not used, recomputed internally)
        k: Number of clusters
        random_state: Random seed

    Returns:
        ClusterMetrics object with all computed metrics
    """
    return compute_cluster_metrics(feature_subset, k, random_state=random_state)


def visualize_cluster(
    feature_subset: torch.Tensor,
    labels: np.ndarray,
    metrics: ClusterMetrics,
    output_dir: Path,
    seed: int,
    k: int,
    feature_idx: int,
    component: str
):
    """
    Create comprehensive visualizations for cluster analysis.

    Generates:
    1. Clustered spy plot showing sparsity patterns
    2. Cluster statistics (sizes and cohesion)
    3. t-SNE visualization of feature embeddings
    4. Between-cluster distance heatmap

    Args:
        feature_subset: Feature matrix [D, N]
        labels: Cluster assignments
        metrics: Computed cluster metrics
        output_dir: Directory to save visualizations
        seed: Random seed used
        k: Number of clusters
        feature_idx: Selected feature index
        component: Component name (e.g., "mlp.down_proj")
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"seed{seed}_k{k}_feat{feature_idx}"

    # Convert to numpy for visualization
    data = (feature_subset != 0).int().cpu().numpy().T  # [N, D]

    # 1. Clustered spy plot
    spy_path = output_dir / f"{prefix}_spy_plot.png"
    create_clustered_spy_plot(
        data=data,
        labels=labels,
        title=f"{component} - Seed {seed}, k={k}, Feature {feature_idx}\n"
              f"Separation: {metrics.mean_between/metrics.mean_within:.2f}x",
        output_path=spy_path,
        figsize=(16, 10)
    )

    # 2. Cluster statistics
    stats_path = output_dir / f"{prefix}_statistics.png"
    plot_cluster_statistics(
        cluster_sizes=metrics.cluster_sizes,
        within_distances=metrics.within_cluster_distances,
        k=k,
        output_path=stats_path
    )

    # 3. t-SNE visualization
    if data.shape[0] >= 30:  # Need enough samples for t-SNE
        tsne_path = output_dir / f"{prefix}_tsne.png"
        plot_tsne_clusters(
            data=data,
            labels=labels,
            k=k,
            output_path=tsne_path,
            title=f"t-SNE: {component} - Seed {seed}, k={k}"
        )

    # 4. Between-cluster distance heatmap
    heatmap_path = output_dir / f"{prefix}_distance_heatmap.png"
    plot_distance_heatmap(
        between_distances=metrics.between_cluster_distances,
        within_distances=metrics.within_cluster_distances,
        k=k,
        output_path=heatmap_path,
        title=f"Cluster Distances - Seed {seed}, k={k}"
    )


def plot_tsne_clusters(
    data: np.ndarray,
    labels: np.ndarray,
    k: int,
    output_path: Path,
    title: str,
    perplexity: int = 30
):
    """
    Create t-SNE visualization of clustered features.

    Args:
        data: Feature data [N, D]
        labels: Cluster assignments [N]
        k: Number of clusters
        output_path: Path to save figure
        title: Plot title
        perplexity: t-SNE perplexity parameter
    """
    # Adjust perplexity if we have fewer samples
    perplexity = min(perplexity, data.shape[0] - 1, 30)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedded = tsne.fit_transform(data)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each cluster with different color
    colors = plt.cm.tab10(np.linspace(0, 1, k))
    for cluster_id in range(k):
        mask = (labels == cluster_id)
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=[colors[cluster_id]],
            label=f'Cluster {cluster_id}',
            alpha=0.6,
            edgecolors='black',
            linewidth=0.5,
            s=50
        )

    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_distance_heatmap(
    between_distances: np.ndarray,
    within_distances: list,
    k: int,
    output_path: Path,
    title: str
):
    """
    Create heatmap of between-cluster and within-cluster distances.

    Args:
        between_distances: Pairwise between-cluster distances [k, k]
        within_distances: Within-cluster distances [k]
        k: Number of clusters
        output_path: Path to save figure
        title: Plot title
    """
    # Create a combined distance matrix with within-cluster on diagonal
    distance_matrix = between_distances.copy()
    np.fill_diagonal(distance_matrix, within_distances)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(distance_matrix, cmap='YlOrRd', aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean Hamming Distance', rotation=270, labelpad=20, fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(k))
    ax.set_yticks(np.arange(k))
    ax.set_xticklabels([f'C{i}' for i in range(k)])
    ax.set_yticklabels([f'C{i}' for i in range(k)])

    # Add text annotations
    for i in range(k):
        for j in range(k):
            text = ax.text(j, i, f'{distance_matrix[i, j]:.3f}',
                          ha="center", va="center", color="black", fontsize=9)

    ax.set_xlabel('Cluster ID', fontsize=12)
    ax.set_ylabel('Cluster ID', fontsize=12)
    ax.set_title(f'{title}\n(Diagonal = Within-Cluster, Off-Diagonal = Between-Cluster)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def summarize_multi_seed_results(
    all_metrics: Dict[int, Dict[int, ClusterMetrics]],
    output_dir: Path,
    component: str
):
    """
    Create summary visualizations across all seeds and k values.

    Args:
        all_metrics: Nested dict {seed: {k: ClusterMetrics}}
        output_dir: Directory to save summary plots
        component: Component name
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract data for summary plots
    seeds = sorted(all_metrics.keys())
    k_values = sorted(list(all_metrics[seeds[0]].keys()))

    # 1. Separation ratio across k values (averaged over seeds)
    separation_ratios = {k: [] for k in k_values}

    for seed in seeds:
        for k in k_values:
            metrics = all_metrics[seed][k]
            ratio = metrics.mean_between / metrics.mean_within if metrics.mean_within > 0 else 0
            separation_ratios[k].append(ratio)

    # Plot separation ratios
    mean_ratios = [np.mean(separation_ratios[k]) for k in k_values]
    std_ratios = [np.std(separation_ratios[k]) for k in k_values]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(k_values, mean_ratios, yerr=std_ratios, marker='o',
                linewidth=2, markersize=8, capsize=5, capthick=2,
                color='#2E86AB', label='Mean ± Std')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1,
               alpha=0.5, label='Ratio = 1')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Separation Ratio (Between/Within)', fontsize=12)
    ax.set_title(f'{component} - Multi-Seed Separation Ratios ({len(seeds)} seeds)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    sep_path = output_dir / f"summary_separation_ratios.png"
    plt.savefig(sep_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {sep_path}")

    # 2. Within and between distances by k
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Within-cluster distances
    for k in k_values:
        within_vals = [all_metrics[seed][k].mean_within for seed in seeds]
        ax1.scatter([k] * len(within_vals), within_vals, alpha=0.5, s=50)

    within_means = [np.mean([all_metrics[seed][k].mean_within for seed in seeds])
                    for k in k_values]
    ax1.plot(k_values, within_means, 'r-', linewidth=2, marker='o',
             markersize=10, label='Mean')

    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Mean Within-Cluster Distance', fontsize=12)
    ax1.set_title('Within-Cluster Cohesion', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Between-cluster distances
    for k in k_values:
        between_vals = [all_metrics[seed][k].mean_between for seed in seeds]
        ax2.scatter([k] * len(between_vals), between_vals, alpha=0.5, s=50)

    between_means = [np.mean([all_metrics[seed][k].mean_between for seed in seeds])
                     for k in k_values]
    ax2.plot(k_values, between_means, 'b-', linewidth=2, marker='o',
             markersize=10, label='Mean')

    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Mean Between-Cluster Distance', fontsize=12)
    ax2.set_title('Between-Cluster Separation', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'{component} - Clustering Metrics Across Seeds ({len(seeds)} seeds)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    dist_path = output_dir / f"summary_distances.png"
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {dist_path}")

    # 3. Save metrics as JSON
    metrics_dict = {}
    for seed in seeds:
        metrics_dict[f"seed_{seed}"] = {}
        for k in k_values:
            m = all_metrics[seed][k]
            metrics_dict[f"seed_{seed}"][f"k_{k}"] = {
                "mean_within": float(m.mean_within),
                "mean_between": float(m.mean_between),
                "separation_ratio": float(m.mean_between / m.mean_within) if m.mean_within > 0 else 0,
                "cluster_sizes": [int(s) for s in m.cluster_sizes],
                "inertia": float(m.inertia) if m.inertia is not None else None
            }

    # Add summary statistics
    metrics_dict["summary"] = {
        "separation_ratios_by_k": {
            f"k_{k}": {
                "mean": float(np.mean(separation_ratios[k])),
                "std": float(np.std(separation_ratios[k])),
                "min": float(np.min(separation_ratios[k])),
                "max": float(np.max(separation_ratios[k]))
            }
            for k in k_values
        }
    }

    json_path = output_dir / "metrics_summary.json"
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"Saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed clustering analysis for feature co-activation patterns"
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["wanda", "sparsegpt"],
        help="Pruning method to analyze"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=1,
        help="Layer number to analyze (default: 1)"
    )
    parser.add_argument(
        "--component",
        type=str,
        required=True,
        help="Component name (e.g., mlp.down_proj, self_attn.q_proj)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of random seeds to test (default: 10)"
    )
    parser.add_argument(
        "--n-similar",
        type=int,
        default=128,
        help="Number of similar features to analyze (default: 128)"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs='+',
        default=[4, 8, 16],
        help="K values for clustering (default: 4 8 16)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/metrics/multi_seed_clustering/{method}/{component})"
    )

    args = parser.parse_args()

    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / "data" / "clustering" / args.method / f"layer{args.layer}-{args.component}.pt"

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "results" / "metrics" / "multi_seed_clustering" / args.method / args.component

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Multi-Seed Clustering Analysis")
    print(f"{'='*80}")
    print(f"Method: {args.method}")
    print(f"Layer: {args.layer}")
    print(f"Component: {args.component}")
    print(f"Data file: {data_path}")
    print(f"Seeds: {args.seeds}")
    print(f"Similar features: {args.n_similar}")
    print(f"K values: {args.k_values}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")

    # Load weight matrix
    print(f"Loading matrix from {data_path}...")
    matrix = torch.load(data_path)
    print(f"Matrix shape: {matrix.shape}")
    print(f"Matrix sparsity: {(matrix == 0).float().mean():.2%}\n")

    # Store all metrics for summary
    all_metrics: Dict[int, Dict[int, ClusterMetrics]] = {}

    # Run analysis for each seed
    for seed in range(args.seeds):
        print(f"\n{'-'*80}")
        print(f"Seed {seed}/{args.seeds-1}")
        print(f"{'-'*80}")

        set_seed(seed)

        # 1. Select a random feature
        selected_feature = identify_feature(matrix, seed)
        print(f"Selected feature: {selected_feature}")

        # 2. Get top N most similar features by Hamming distance
        top_128_similar, indices, distances = get_closest_features(
            selected_feature,
            matrix,
            args.n_similar,
            "hamming"
        )
        print(f"Found {len(indices)} similar features")
        print(f"Distance range: [{distances.min():.4f}, {distances.max():.4f}]")

        # Store metrics for this seed
        all_metrics[seed] = {}

        # 3. Cluster with different k values
        for k in args.k_values:
            print(f"\n  Clustering with k={k}...")

            # Get clusters
            clustered_features, labels = get_clusters(
                top_128_similar,
                "kmeans",
                k=k,
                random_state=seed
            )

            # Compute metrics
            metrics = get_cluster_metrics(
                clustered_features,
                labels,
                k=k,
                random_state=seed
            )

            all_metrics[seed][k] = metrics

            # Print metrics
            print(f"    Cluster sizes: {metrics.cluster_sizes}")
            print(f"    Mean within-cluster distance: {metrics.mean_within:.4f}")
            print(f"    Mean between-cluster distance: {metrics.mean_between:.4f}")
            print(f"    Separation ratio: {metrics.mean_between/metrics.mean_within:.2f}x")
            print(f"    Inertia: {metrics.inertia:.2f}")

            # Visualize
            seed_output_dir = output_dir / f"seed_{seed}" / f"k_{k}"
            visualize_cluster(
                clustered_features,
                labels,
                metrics,
                seed_output_dir,
                seed=seed,
                k=k,
                feature_idx=selected_feature,
                component=args.component
            )

    # Create summary visualizations
    print(f"\n{'='*80}")
    print("Creating summary visualizations...")
    print(f"{'='*80}\n")

    summarize_multi_seed_results(
        all_metrics,
        output_dir / "summary",
        component=args.component
    )

    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
