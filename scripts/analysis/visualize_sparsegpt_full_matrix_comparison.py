#!/usr/bin/env python3
"""
Visualize SparseGPT Full Matrix K-sweep Results and Compare with 128-feature Subset

Creates visualizations comparing:
1. Full matrix clustering results
2. Comparison between 128-feature subset vs full matrix
3. Analysis of how clustering structure differs at different scales
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


def create_comparison_heatmap(subset_results: dict, full_results: dict, output_dir: Path):
    """
    Create side-by-side heatmaps comparing 128-feature vs full matrix.
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

    # Build matrices
    subset_matrix = []
    full_matrix = []

    for proj_file in projection_order:
        if proj_file in subset_results and proj_file in full_results:
            subset_row = [subset_results[proj_file][str(k)]['separation_ratio'] for k in k_values]
            full_row = [full_results[proj_file][str(k)]['separation_ratio'] for k in k_values]
            subset_matrix.append(subset_row)
            full_matrix.append(full_row)
        else:
            subset_matrix.append([0] * len(k_values))
            full_matrix.append([0] * len(k_values))

    subset_matrix = np.array(subset_matrix)
    full_matrix = np.array(full_matrix)

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Use same scale for both
    vmin = 0.9
    vmax = max(subset_matrix.max(), full_matrix.max())

    # Subset heatmap
    im1 = axes[0].imshow(subset_matrix, cmap='YlOrRd', aspect='auto',
                         norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    axes[0].set_xticks(np.arange(len(k_values)))
    axes[0].set_yticks(np.arange(len(projection_names)))
    axes[0].set_xticklabels([f'k={k}' for k in k_values])
    axes[0].set_yticklabels(projection_names)
    axes[0].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[0].set_ylabel('Projection Type', fontsize=11)
    axes[0].set_title('128-Feature Subset\n(Local Clustering)', fontsize=12, pad=10)

    # Add annotations
    for i in range(len(projection_names)):
        for j in range(len(k_values)):
            value = subset_matrix[i, j]
            text_color = 'white' if value > 5 else 'black'
            axes[0].text(j, i, f'{value:.1f}', ha="center", va="center",
                        color=text_color, fontsize=8, fontweight='bold')

    # Full matrix heatmap
    im2 = axes[1].imshow(full_matrix, cmap='YlOrRd', aspect='auto',
                         norm=plt.matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax))
    axes[1].set_xticks(np.arange(len(k_values)))
    axes[1].set_yticks(np.arange(len(projection_names)))
    axes[1].set_xticklabels([f'k={k}' for k in k_values])
    axes[1].set_yticklabels(projection_names)
    axes[1].set_xlabel('Number of Clusters (k)', fontsize=11)
    axes[1].set_ylabel('Projection Type', fontsize=11)
    axes[1].set_title('Full Matrix\n(Global Clustering)', fontsize=12, pad=10)

    # Add annotations
    for i in range(len(projection_names)):
        for j in range(len(k_values)):
            value = full_matrix[i, j]
            text_color = 'white' if value > 5 else 'black'
            axes[1].text(j, i, f'{value:.1f}', ha="center", va="center",
                        color=text_color, fontsize=8, fontweight='bold')

    # Add shared colorbar
    fig.colorbar(im2, ax=axes, label='Separation Ratio (log scale)',
                 orientation='vertical', pad=0.02)

    plt.suptitle('SparseGPT: Local vs Global Clustering Structure', fontsize=14, y=0.98)
    plt.tight_layout()

    output_file = output_dir / 'sparsegpt_subset_vs_full_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison heatmap to: {output_file}")
    plt.close()


def create_ratio_difference_heatmap(subset_results: dict, full_results: dict, output_dir: Path):
    """
    Create heatmap showing the ratio of (subset / full) separation ratios.
    Values > 1 mean subset has stronger clustering.
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

    # Build ratio matrix (subset / full)
    ratio_matrix = []

    for proj_file in projection_order:
        if proj_file in subset_results and proj_file in full_results:
            row = []
            for k in k_values:
                subset_val = subset_results[proj_file][str(k)]['separation_ratio']
                full_val = full_results[proj_file][str(k)]['separation_ratio']
                ratio = subset_val / full_val if full_val > 0 else 0
                row.append(ratio)
            ratio_matrix.append(row)
        else:
            ratio_matrix.append([1] * len(k_values))

    ratio_matrix = np.array(ratio_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use diverging colormap centered at 1.0 (no difference)
    im = ax.imshow(ratio_matrix, cmap='RdBu_r', aspect='auto',
                   norm=plt.matplotlib.colors.LogNorm(vmin=0.1, vmax=10))

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(projection_names)))
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_yticklabels(projection_names)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Subset / Full Ratio (log scale)', rotation=270, labelpad=20)

    # Add annotations
    for i in range(len(projection_names)):
        for j in range(len(k_values)):
            value = ratio_matrix[i, j]
            text_color = 'white' if (value > 2 or value < 0.5) else 'black'
            ax.text(j, i, f'{value:.1f}x', ha="center", va="center",
                   color=text_color, fontsize=9, fontweight='bold')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Projection Type', fontsize=12)
    ax.set_title('SparseGPT: Local vs Global Clustering Strength\n(Ratio > 1: Subset stronger | Ratio < 1: Full matrix stronger)',
                fontsize=13, pad=20)

    # Add grid
    ax.set_xticks(np.arange(len(k_values)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(projection_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    plt.tight_layout()

    output_file = output_dir / 'sparsegpt_subset_vs_full_ratio.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved ratio heatmap to: {output_file}")
    plt.close()


def create_full_matrix_line_plot(full_results: dict, output_dir: Path):
    """
    Create line plot for full matrix results.
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

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(projection_names)))

    for idx, (proj_file, proj_name) in enumerate(zip(projection_order, projection_names)):
        if proj_file in full_results:
            ratios = [full_results[proj_file][str(k)]['separation_ratio'] for k in k_values]
            ax.plot(k_values, ratios, marker='o', linewidth=2,
                   label=proj_name, color=colors[idx], markersize=8)

    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='No separation')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Separation Ratio', fontsize=12)
    ax.set_title('SparseGPT Full Matrix: Separation Ratio vs K Value\n(Global Clustering Structure)',
                fontsize=14, pad=20)

    ax.set_yscale('log')
    ax.set_xscale('log', base=2)
    ax.set_xticks(k_values)
    ax.set_xticklabels([str(k) for k in k_values])

    ax.grid(True, alpha=0.3, which='both')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    plt.tight_layout()

    output_file = output_dir / 'sparsegpt_full_matrix_line_plot.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved full matrix line plot to: {output_file}")
    plt.close()


def create_cluster_size_distribution(full_results: dict, output_dir: Path):
    """
    Create visualization of cluster size distributions for full matrix.
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

    # Get coefficient of variation (std/mean) for cluster sizes
    cv_matrix = []

    for proj_file in projection_order:
        if proj_file in full_results:
            row = []
            for k in k_values:
                sizes = full_results[proj_file][str(k)]['cluster_sizes']
                cv = np.std(sizes) / np.mean(sizes) if np.mean(sizes) > 0 else 0
                row.append(cv)
            cv_matrix.append(row)
        else:
            cv_matrix.append([0] * len(k_values))

    cv_matrix = np.array(cv_matrix)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(cv_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1.5)

    ax.set_xticks(np.arange(len(k_values)))
    ax.set_yticks(np.arange(len(projection_names)))
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_yticklabels(projection_names)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Coefficient of Variation (std/mean)', rotation=270, labelpad=20)

    # Add annotations
    for i in range(len(projection_names)):
        for j in range(len(k_values)):
            value = cv_matrix[i, j]
            text_color = 'white' if value > 0.75 else 'black'
            ax.text(j, i, f'{value:.2f}', ha="center", va="center",
                   color=text_color, fontsize=9, fontweight='bold')

    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Projection Type', fontsize=12)
    ax.set_title('SparseGPT Full Matrix: Cluster Size Uniformity\n(Lower = more uniform, Higher = more skewed)',
                fontsize=13, pad=20)

    ax.set_xticks(np.arange(len(k_values)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(projection_names)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)

    plt.tight_layout()

    output_file = output_dir / 'sparsegpt_full_matrix_cluster_uniformity.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved cluster uniformity heatmap to: {output_file}")
    plt.close()


def main():
    """Generate all visualizations for SparseGPT full matrix results."""
    base_dir = Path(__file__).resolve().parent.parent.parent

    # Load both datasets
    subset_file = base_dir / "results" / "metrics" / "sparsegpt_kmeans_k_sweep" / "sparsegpt_kmeans_k_sweep_results.json"
    full_file = base_dir / "results" / "metrics" / "sparsegpt_kmeans_k_sweep_full" / "sparsegpt_kmeans_k_sweep_full_results.json"
    output_dir = base_dir / "results" / "visualizations" / "sparsegpt_kmeans_k_sweep_full"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading 128-feature subset results from: {subset_file}")
    subset_results = load_results(subset_file)

    print(f"Loading full matrix results from: {full_file}")
    full_results = load_results(full_file)

    print("\nGenerating visualizations...")

    # Create visualizations
    create_comparison_heatmap(subset_results, full_results, output_dir)
    create_ratio_difference_heatmap(subset_results, full_results, output_dir)
    create_full_matrix_line_plot(full_results, output_dir)
    create_cluster_size_distribution(full_results, output_dir)

    print(f"\nAll visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()
