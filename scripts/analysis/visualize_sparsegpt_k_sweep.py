#!/usr/bin/env python3
"""
Visualize SparseGPT K-means K-Value Sweep Results

Creates a heatmap showing separation ratios across different k values
and projection types for SparseGPT pruning method.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_results(results_file: Path) -> dict:
    """Load results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def create_separation_ratio_heatmap(results: dict, output_dir: Path):
    """
    Create heatmap of separation ratios across k values and projections.

    Args:
        results: Dictionary of results from k-sweep experiment
        output_dir: Directory to save visualization
    """
    # Define projection order for consistent display
    projection_order = [
        'layer1-mlp.down_proj.pt',
        'layer1-mlp.up_proj.pt',
        'layer1-mlp.gate_proj.pt',
        'layer1-self_attn.q_proj.pt',
        'layer1-self_attn.k_proj.pt',
        'layer1-self_attn.v_proj.pt',
        'layer1-self_attn.o_proj.pt',
    ]

    # Pretty names for display
    projection_names = [
        'MLP Down',
        'MLP Up',
        'MLP Gate',
        'Attn Q',
        'Attn K',
        'Attn V',
        'Attn O',
    ]

    # K values
    k_values = [4, 8, 16, 32, 64]

    # Build matrix of separation ratios
    separation_matrix = []
    for proj_file in projection_order:
        if proj_file in results:
            row = []
            for k in k_values:
                ratio = results[proj_file][str(k)]['separation_ratio']
                row.append(ratio)
            separation_matrix.append(row)
        else:
            separation_matrix.append([0] * len(k_values))

    separation_matrix = np.array(separation_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap with custom colormap
    # Use log scale for better visualization since values range from ~1 to 70
    im = ax.imshow(separation_matrix, cmap='YlOrRd', aspect='auto',
                   norm=plt.matplotlib.colors.LogNorm(vmin=0.9, vmax=100))

    # Set ticks and labels
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(projection_names)))
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_yticklabels(projection_names)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Separation Ratio (log scale)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(projection_names)):
        for j in range(len(k_values)):
            value = separation_matrix[i, j]
            # Choose text color based on background
            text_color = 'white' if value > 5 else 'black'
            text = ax.text(j, i, f'{value:.2f}',
                          ha="center", va="center", color=text_color,
                          fontsize=9, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Projection Type', fontsize=12)
    ax.set_title('SparseGPT: Separation Ratios Across K Values\n(Between-cluster / Within-cluster Hamming Distance)',
                fontsize=14, pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(k_values)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(projection_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'sparsegpt_k_sweep_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {output_file}")

    plt.close()


def create_cluster_size_heatmap(results: dict, output_dir: Path):
    """
    Create heatmap showing maximum cluster sizes across k values.

    Args:
        results: Dictionary of results from k-sweep experiment
        output_dir: Directory to save visualization
    """
    projection_order = [
        'layer1-mlp.down_proj.pt',
        'layer1-mlp.up_proj.pt',
        'layer1-mlp.gate_proj.pt',
        'layer1-self_attn.q_proj.pt',
        'layer1-self_attn.k_proj.pt',
        'layer1-self_attn.v_proj.pt',
        'layer1-self_attn.o_proj.pt',
    ]

    projection_names = [
        'MLP Down',
        'MLP Up',
        'MLP Gate',
        'Attn Q',
        'Attn K',
        'Attn V',
        'Attn O',
    ]

    k_values = [4, 8, 16, 32, 64]

    # Build matrix of max cluster sizes
    max_cluster_matrix = []
    for proj_file in projection_order:
        if proj_file in results:
            row = []
            for k in k_values:
                max_size = results[proj_file][str(k)]['max_cluster_size']
                row.append(max_size)
            max_cluster_matrix.append(row)
        else:
            max_cluster_matrix.append([0] * len(k_values))

    max_cluster_matrix = np.array(max_cluster_matrix)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    im = ax.imshow(max_cluster_matrix, cmap='viridis', aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(projection_names)))
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_yticklabels(projection_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Max Cluster Size (out of 128 features)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(projection_names)):
        for j in range(len(k_values)):
            value = max_cluster_matrix[i, j]
            text_color = 'white' if value > 64 else 'black'
            text = ax.text(j, i, f'{value}',
                          ha="center", va="center", color=text_color,
                          fontsize=9, fontweight='bold')

    # Labels and title
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Projection Type', fontsize=12)
    ax.set_title('SparseGPT: Maximum Cluster Size Across K Values\n(Largest cluster out of 128 features)',
                fontsize=14, pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(k_values)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(projection_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'sparsegpt_k_sweep_cluster_sizes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved cluster size heatmap to: {output_file}")

    plt.close()


def create_line_plot(results: dict, output_dir: Path):
    """
    Create line plot showing how separation ratios change with k.

    Args:
        results: Dictionary of results from k-sweep experiment
        output_dir: Directory to save visualization
    """
    projection_order = [
        'layer1-mlp.down_proj.pt',
        'layer1-mlp.up_proj.pt',
        'layer1-mlp.gate_proj.pt',
        'layer1-self_attn.q_proj.pt',
        'layer1-self_attn.k_proj.pt',
        'layer1-self_attn.v_proj.pt',
        'layer1-self_attn.o_proj.pt',
    ]

    projection_names = [
        'MLP Down',
        'MLP Up',
        'MLP Gate',
        'Attn Q',
        'Attn K',
        'Attn V',
        'Attn O',
    ]

    k_values = [4, 8, 16, 32, 64]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Define colors for different projection types
    colors = plt.cm.tab10(np.linspace(0, 1, len(projection_names)))

    # Plot lines for each projection
    for idx, (proj_file, proj_name) in enumerate(zip(projection_order, projection_names)):
        if proj_file in results:
            ratios = [results[proj_file][str(k)]['separation_ratio'] for k in k_values]
            ax.plot(k_values, ratios, marker='o', linewidth=2,
                   label=proj_name, color=colors[idx], markersize=8)

    # Add reference line at y=1 (no separation)
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No separation')

    # Labels and title
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Separation Ratio', fontsize=12)
    ax.set_title('SparseGPT: Separation Ratio vs K Value\n(Between-cluster / Within-cluster Distance)',
                fontsize=14, pad=20)

    # Set log scale for y-axis to handle wide range
    ax.set_yscale('log')
    ax.set_xscale('log', base=2)

    # Set x-ticks to exact k values
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])

    # Grid
    ax.grid(True, alpha=0.3, which='both')

    # Legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()

    # Save figure
    output_file = output_dir / 'sparsegpt_k_sweep_line_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved line plot to: {output_file}")

    plt.close()


def main():
    """Generate all visualizations for SparseGPT k-sweep results."""
    # Define paths
    base_dir = Path(__file__).resolve().parent.parent.parent
    results_file = base_dir / "results" / "metrics" / "sparsegpt_kmeans_k_sweep" / "sparsegpt_kmeans_k_sweep_results.json"
    output_dir = base_dir / "results" / "visualizations" / "sparsegpt_kmeans_k_sweep"

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from: {results_file}")
    results = load_results(results_file)

    print("\nGenerating visualizations...")

    # Create visualizations
    create_separation_ratio_heatmap(results, output_dir)
    create_cluster_size_heatmap(results, output_dir)
    create_line_plot(results, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
