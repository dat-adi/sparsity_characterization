"""
Visualize agglomerative clustering results with dendrograms and cluster quality plots.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from typing import Dict, List
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_dendrogram_visualization(
    matrix_path: str,
    max_features: int = 1000,
    output_dir: Path = None
) -> None:
    """
    Create dendrogram visualization for hierarchical clustering.

    Args:
        matrix_path: Path to .pt file containing weight matrix
        max_features: Maximum number of features to visualize
        output_dir: Directory to save visualizations
    """
    print(f"\nCreating dendrogram for: {matrix_path}")

    # Load matrix
    matrix = torch.load(matrix_path, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")

    # Convert to binary
    binary_matrix = (matrix > 0).int()

    # Limit features for visualization
    n_features = min(max_features, binary_matrix.shape[1])
    binary_matrix = binary_matrix[:, :n_features]

    print(f"Computing pairwise Hamming distances for {n_features} features...")
    binary_np = binary_matrix.cpu().numpy().T  # [N, D]
    distances = pdist(binary_np, metric='hamming')

    # Compute linkage
    print("Computing hierarchical clustering linkage...")
    Z = linkage(distances, method='ward')

    # Create dendrogram
    fig, ax = plt.subplots(figsize=(20, 10))

    dendrogram(
        Z,
        ax=ax,
        no_labels=True,  # Don't show feature labels (too many)
        color_threshold=0.3 * max(Z[:, 2]),  # Color clusters
        above_threshold_color='gray'
    )

    matrix_name = Path(matrix_path).stem
    ax.set_title(f'Hierarchical Clustering Dendrogram: {matrix_name}\n({n_features} features)',
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('Feature Index', fontsize=14)
    ax.set_ylabel('Hamming Distance', fontsize=14)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_dir:
        output_path = output_dir / f"{matrix_name}_dendrogram.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved dendrogram to: {output_path}")

    plt.close()


def create_separation_ratio_plot(
    results_json_path: str,
    output_dir: Path = None
) -> None:
    """
    Create plot showing separation ratio vs number of clusters.

    Args:
        results_json_path: Path to JSON file with clustering results
        output_dir: Directory to save visualizations
    """
    print(f"\nCreating separation ratio plot from: {results_json_path}")

    # Load results
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)

    # Create plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, results in enumerate(all_results):
        matrix_name = Path(results['matrix_path']).stem

        n_clusters_list = []
        separation_ratios = []
        mean_within = []
        mean_between = []

        for clustering in results['clusterings']:
            n_clusters_list.append(clustering['n_clusters'])
            separation_ratios.append(clustering['metrics']['separation_ratio'])
            mean_within.append(clustering['metrics']['mean_within'])
            mean_between.append(clustering['metrics']['mean_between'])

        # Plot 1: Separation ratio
        axes[0].plot(n_clusters_list, separation_ratios, marker='o', linewidth=2, label=matrix_name)
        axes[0].set_xlabel('Number of Clusters', fontsize=12)
        axes[0].set_ylabel('Separation Ratio', fontsize=12)
        axes[0].set_title('Cluster Separation vs Number of Clusters', fontsize=14, fontweight='bold')
        axes[0].grid(alpha=0.3)
        axes[0].axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No separation')
        axes[0].legend()

        # Plot 2: Within-cluster distance
        axes[1].plot(n_clusters_list, mean_within, marker='s', linewidth=2, label=matrix_name)
        axes[1].set_xlabel('Number of Clusters', fontsize=12)
        axes[1].set_ylabel('Mean Within-Cluster Distance', fontsize=12)
        axes[1].set_title('Within-Cluster Cohesion', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        # Plot 3: Between-cluster distance
        axes[2].plot(n_clusters_list, mean_between, marker='^', linewidth=2, label=matrix_name)
        axes[2].set_xlabel('Number of Clusters', fontsize=12)
        axes[2].set_ylabel('Mean Between-Cluster Distance', fontsize=12)
        axes[2].set_title('Between-Cluster Separation', fontsize=14, fontweight='bold')
        axes[2].grid(alpha=0.3)
        axes[2].legend()

    plt.tight_layout()

    # Save figure
    if output_dir:
        output_path = output_dir / "separation_ratio_comparison.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved separation ratio plot to: {output_path}")

    plt.close()


def create_cluster_size_distribution_plot(
    results_json_path: str,
    output_dir: Path = None
) -> None:
    """
    Create plot showing cluster size distributions.

    Args:
        results_json_path: Path to JSON file with clustering results
        output_dir: Directory to save visualizations
    """
    print(f"\nCreating cluster size distribution plot...")

    # Load results
    with open(results_json_path, 'r') as f:
        all_results = json.load(f)

    # Create subplots for each matrix
    n_matrices = len(all_results)
    fig, axes = plt.subplots(n_matrices, 1, figsize=(14, 5 * n_matrices))

    if n_matrices == 1:
        axes = [axes]

    for idx, results in enumerate(all_results):
        matrix_name = Path(results['matrix_path']).stem
        ax = axes[idx]

        # Get cluster sizes for different n_clusters values
        for clustering in results['clusterings']:
            n_clusters = clustering['n_clusters']
            cluster_sizes = clustering['metrics']['cluster_sizes']

            # Create histogram
            ax.hist(cluster_sizes, bins=30, alpha=0.5, label=f'{n_clusters} clusters', edgecolor='black')

        ax.set_xlabel('Cluster Size (number of features)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Cluster Size Distribution: {matrix_name}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    if output_dir:
        output_path = output_dir / "cluster_size_distributions.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved cluster size distribution plot to: {output_path}")

    plt.close()


def main():
    """Create visualizations for agglomerative clustering results."""

    data_dir = Path(__file__).parent.parent.parent / "data/sparsegpt_unstructured"
    results_dir = Path(__file__).parent.parent.parent / "results/metrics/agglomerative_clustering"
    viz_dir = Path(__file__).parent.parent.parent / "results/visualizations/agglomerative_clustering"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Find matrices
    matrix_paths = sorted(data_dir.glob("layer*-mlp.down_proj.pt"))
    matrix_paths = sorted(matrix_paths, key=lambda p: int(p.stem.split('-')[0].replace('layer', '')))

    print(f"Found {len(matrix_paths)} matrices")

    # Create dendrograms for first 3 layers
    for matrix_path in matrix_paths[:3]:
        create_dendrogram_visualization(
            matrix_path=str(matrix_path),
            max_features=1000,  # Visualize up to 1000 features
            output_dir=viz_dir
        )

    # Create comparison plots from JSON results
    results_json = results_dir / "agglomerative_clustering_results.json"
    if results_json.exists():
        create_separation_ratio_plot(
            results_json_path=str(results_json),
            output_dir=viz_dir
        )

        create_cluster_size_distribution_plot(
            results_json_path=str(results_json),
            output_dir=viz_dir
        )

    print(f"\n{'='*60}")
    print(f"✓ All visualizations saved to: {viz_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
