"""
Common visualization utilities for sparsity analysis.

Provides reusable plotting functions:
- Spy plots for sparsity visualization
- Heatmaps for correlation analysis
- Cluster visualization
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple


def create_spy_visualization(
    matrix: torch.Tensor,
    title: str,
    output_path: Path,
    figsize: Tuple[int, int] = (12, 10)
) -> float:
    """
    Create a spy visualization showing binary sparsity patterns.

    Args:
        matrix: Weight matrix to visualize
        title: Plot title
        output_path: Path to save the figure
        figsize: Figure size (width, height)

    Returns:
        sparsity: Computed sparsity level [0, 1]
    """
    # Convert to binary sparsity mask (1 = non-zero, 0 = zero)
    binary_mask = (matrix.abs() > 0).cpu().numpy()

    # Calculate sparsity
    total_elements = binary_mask.size
    nonzero_elements = np.count_nonzero(binary_mask)
    sparsity = 1 - (nonzero_elements / total_elements)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create spy plot with proper black/white coloring
    ax.imshow(binary_mask, cmap='binary', aspect='auto', interpolation='nearest')

    ax.set_xlabel('Input Features', fontsize=12)
    ax.set_ylabel('Output Features', fontsize=12)
    ax.set_title(f'{title}\nShape: {matrix.shape}, Sparsity: {sparsity:.2%}',
                 fontsize=14, fontweight='bold')

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  Shape: {matrix.shape}")
    print(f"  Non-zero elements: {nonzero_elements:,} / {total_elements:,}")
    print(f"  Sparsity: {sparsity:.2%}\n")

    return sparsity


def create_clustered_spy_plot(
    data: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 10),
    show_boundaries: bool = True
) -> plt.Figure:
    """
    Create a spy plot with data reordered by cluster assignment.

    Args:
        data: Binary activation data [N_features, N_samples]
        labels: Cluster assignments [N_features]
        title: Plot title
        output_path: Optional path to save figure
        figsize: Figure size (width, height)
        show_boundaries: Whether to show cluster boundaries

    Returns:
        matplotlib Figure object
    """
    # Sort features by cluster assignment
    cluster_sort_idx = np.argsort(labels)
    sorted_labels = labels[cluster_sort_idx]
    sorted_data = data[cluster_sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create spy plot
    ax.imshow(sorted_data, cmap='binary', aspect='auto', interpolation='nearest')

    # Add cluster boundaries
    if show_boundaries:
        cluster_boundaries = np.where(np.diff(sorted_labels))[0] + 1
        for boundary in cluster_boundaries:
            ax.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)

    ax.set_xlabel('Samples', fontsize=12)
    ax.set_ylabel('Features (sorted by cluster)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    return fig


def plot_cluster_statistics(
    cluster_sizes: list,
    within_distances: list,
    k: int,
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Create a bar plot showing cluster sizes and within-cluster distances.

    Args:
        cluster_sizes: Number of features in each cluster
        within_distances: Mean within-cluster distance for each cluster
        k: Number of clusters
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    cluster_ids = list(range(k))

    # Plot cluster sizes
    bars1 = ax1.bar(cluster_ids, cluster_sizes, color='steelblue',
                    edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Cluster ID', fontsize=12)
    ax1.set_ylabel('Number of Features', fontsize=12)
    ax1.set_title('Cluster Size Distribution', fontsize=14)
    ax1.set_xticks(cluster_ids)

    # Add count labels
    for i, count in enumerate(cluster_sizes):
        ax1.text(i, count + 0.5, str(count), ha='center', fontsize=10)

    # Plot within-cluster distances
    bars2 = ax2.bar(cluster_ids, within_distances, color='coral',
                    edgecolor='black', alpha=0.7)
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('Mean Within-Cluster Distance', fontsize=12)
    ax2.set_title('Cluster Cohesion', fontsize=14)
    ax2.set_xticks(cluster_ids)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    return fig


def plot_separation_ratio(
    k_values: list,
    ratios: list,
    title: str = "Cluster Separation Ratios",
    output_path: Optional[Path] = None
) -> plt.Figure:
    """
    Plot separation ratios (between/within distances) across different k values.

    Args:
        k_values: List of k values
        ratios: List of separation ratios for each k
        title: Plot title
        output_path: Optional path to save figure

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(k_values, ratios, marker='o', linewidth=2, markersize=8,
            color='#2E86AB', label='Separation Ratio')
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1,
               alpha=0.5, label='Ratio = 1')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Separation Ratio (Between/Within)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

    return fig


def plot_cluster_distance_heatmap(
    within_distances: np.ndarray,
    between_distances: np.ndarray,
    title: str,
    output_path: Path,
    max_clusters_display: int = 100
) -> None:
    """
    Create a heatmap visualization of cluster distance metrics.

    Args:
        within_distances: Array of mean within-cluster distances [num_clusters]
        between_distances: Matrix of between-cluster distances [num_clusters, num_clusters]
        title: Plot title
        output_path: Path to save the figure
        max_clusters_display: Maximum number of clusters to show in heatmap
    """
    import seaborn as sns

    # Limit display size for readability
    n_clusters = min(len(within_distances), max_clusters_display)

    # Truncate data if needed
    within_display = within_distances[:n_clusters]
    between_display = between_distances[:n_clusters, :n_clusters]

    # Calculate statistics
    mean_within = np.mean(within_distances)
    mean_between_upper = np.mean(between_distances[np.triu_indices_from(between_distances, k=1)])
    separation_ratio = mean_between_upper / mean_within if mean_within > 0 else 0

    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left panel: Between-cluster distance heatmap
    ax1 = axes[0]
    im = ax1.imshow(between_display, cmap='coolwarm', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Cluster Index', fontsize=12)
    ax1.set_ylabel('Cluster Index', fontsize=12)
    ax1.set_title('Between-Cluster Distances', fontsize=14, fontweight='bold')

    # Add colorbar
    cbar1 = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Hamming Distance', fontsize=11)

    # Right panel: Within-cluster distance bar plot
    ax2 = axes[1]
    sorted_within = np.sort(within_display)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_within)))

    bars = ax2.bar(range(len(sorted_within)), sorted_within, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Cluster (sorted by within-distance)', fontsize=12)
    ax2.set_ylabel('Mean Within-Cluster Distance', fontsize=12)
    ax2.set_title('Within-Cluster Distance Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics to right panel
    stats_text = f'Mean Within: {mean_within:.4f}\nMean Between: {mean_between_upper:.4f}\nSeparation Ratio: {separation_ratio:.4f}'
    ax2.text(0.97, 0.97, stats_text,
             transform=ax2.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=11,
             family='monospace')

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    # Add note if clusters were truncated
    if n_clusters < len(within_distances):
        fig.text(0.5, 0.02,
                f'Note: Showing first {n_clusters} of {len(within_distances)} clusters for readability',
                ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
