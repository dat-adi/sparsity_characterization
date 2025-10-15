#!/usr/bin/env python3
"""
Visualization Script for Feature Clustering Analysis

Creates PNG plots showing:
1. Separation ratios (between/within cluster distances) across k values
2. Within vs between cluster distances comparison
3. Cluster size distributions
4. Hamming distance distributions for selected features
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import json

# Import functions from the analysis script
from feature_clustering_analysis import (
    analyze_matrix,
    find_most_similar_features,
    compute_cluster_metrics
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


def plot_separation_ratios(
    wanda_results: Dict,
    sparsegpt_results: Dict,
    output_dir: Path
):
    """
    Plot separation ratios (between/within) for all matrices and k values.
    """
    matrix_files = list(wanda_results.keys())
    k_values = [4, 8, 16]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, matrix_file in enumerate(matrix_files):
        ax = axes[idx]

        # Extract ratios
        wanda_ratios = []
        sparsegpt_ratios = []

        for k in k_values:
            w_metrics = wanda_results[matrix_file][k]
            s_metrics = sparsegpt_results[matrix_file][k]

            w_ratio = w_metrics.mean_between / w_metrics.mean_within if w_metrics.mean_within > 0 else 0
            s_ratio = s_metrics.mean_between / s_metrics.mean_within if s_metrics.mean_within > 0 else 0

            wanda_ratios.append(w_ratio)
            sparsegpt_ratios.append(s_ratio)

        # Plot
        x = np.arange(len(k_values))
        width = 0.35

        ax.bar(x - width/2, wanda_ratios, width, label='Wanda', alpha=0.8, color='#2E86AB')
        ax.bar(x + width/2, sparsegpt_ratios, width, label='SparseGPT', alpha=0.8, color='#A23B72')

        ax.set_ylabel('Separation Ratio (Between/Within)')
        ax.set_xlabel('Number of Clusters (k)')
        ax.set_title(matrix_file.replace('.pt', '').replace('layer1-', ''), fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'k={k}' for k in k_values])
        ax.legend(loc='upper left', fontsize=8)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    if len(matrix_files) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_dir / 'separation_ratios.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'separation_ratios.png'}")


def plot_within_between_comparison(
    wanda_results: Dict,
    sparsegpt_results: Dict,
    output_dir: Path
):
    """
    Scatter plot comparing within vs between cluster distances.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    k_values = [4, 8, 16]

    for idx, k in enumerate(k_values):
        ax = axes[idx]

        wanda_within = []
        wanda_between = []
        sparsegpt_within = []
        sparsegpt_between = []
        labels = []

        for matrix_file in wanda_results.keys():
            w_metrics = wanda_results[matrix_file][k]
            s_metrics = sparsegpt_results[matrix_file][k]

            wanda_within.append(w_metrics.mean_within)
            wanda_between.append(w_metrics.mean_between)
            sparsegpt_within.append(s_metrics.mean_within)
            sparsegpt_between.append(s_metrics.mean_between)

            # Short label
            label = matrix_file.replace('layer1-', '').replace('.pt', '')
            label = label.replace('self_attn.', '').replace('mlp.', '')
            labels.append(label)

        # Plot Wanda
        ax.scatter(wanda_within, wanda_between, s=100, alpha=0.6,
                  color='#2E86AB', marker='o', label='Wanda', edgecolors='black', linewidths=0.5)

        # Plot SparseGPT
        ax.scatter(sparsegpt_within, sparsegpt_between, s=100, alpha=0.6,
                  color='#A23B72', marker='s', label='SparseGPT', edgecolors='black', linewidths=0.5)

        # Add diagonal line (where within=between, ratio=1)
        max_val = max(max(wanda_between), max(sparsegpt_between),
                     max(wanda_within), max(sparsegpt_within))
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1, label='Ratio=1')

        # Annotate points
        for i, label in enumerate(labels):
            # Wanda annotation
            ax.annotate(label, (wanda_within[i], wanda_between[i]),
                       fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

        ax.set_xlabel('Mean Within-Cluster Distance')
        ax.set_ylabel('Mean Between-Cluster Distance')
        ax.set_title(f'k={k}')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_dir / 'within_vs_between.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'within_vs_between.png'}")


def plot_cluster_size_distributions(
    wanda_results: Dict,
    sparsegpt_results: Dict,
    output_dir: Path
):
    """
    Visualize cluster size distributions as histograms.
    """
    matrix_files = list(wanda_results.keys())
    k = 16  # Focus on k=16 for most detail

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, matrix_file in enumerate(matrix_files):
        ax = axes[idx]

        w_metrics = wanda_results[matrix_file][k]
        s_metrics = sparsegpt_results[matrix_file][k]

        # Create histogram data
        bins = [1, 2, 5, 10, 20, 50, 150]

        ax.hist([w_metrics.cluster_sizes], bins=bins, alpha=0.6, label='Wanda',
               color='#2E86AB', edgecolor='black', linewidth=0.5)
        ax.hist([s_metrics.cluster_sizes], bins=bins, alpha=0.6, label='SparseGPT',
               color='#A23B72', edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Cluster Size')
        ax.set_ylabel('Frequency')
        ax.set_title(matrix_file.replace('.pt', '').replace('layer1-', ''), fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    if len(matrix_files) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_dir / 'cluster_sizes_k16.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'cluster_sizes_k16.png'}")


def plot_hamming_distance_distributions(
    wanda_dir: Path,
    sparsegpt_dir: Path,
    output_dir: Path
):
    """
    Plot distributions of hamming distances to randomly selected features.
    """
    matrix_files = [
        "layer1-mlp.down_proj.pt",
        "layer1-mlp.up_proj.pt",
        "layer1-mlp.gate_proj.pt",
        "layer1-self_attn.q_proj.pt",
        "layer1-self_attn.k_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    for idx, matrix_file in enumerate(matrix_files):
        ax = axes[idx]

        # Load Wanda matrix
        wanda_matrix = torch.load(wanda_dir / matrix_file, weights_only=True)
        n_total = wanda_matrix.shape[1]
        feature_idx = np.random.randint(0, n_total)

        # Find similar features for Wanda
        _, _, wanda_distances = find_most_similar_features(
            wanda_matrix, feature_idx, n_similar=128
        )

        # Load SparseGPT matrix
        sparsegpt_matrix = torch.load(sparsegpt_dir / matrix_file, weights_only=True)

        # Find similar features for SparseGPT
        _, _, sparsegpt_distances = find_most_similar_features(
            sparsegpt_matrix, feature_idx, n_similar=128
        )

        # Plot distributions
        ax.hist(wanda_distances.cpu().numpy(), bins=30, alpha=0.6,
               label='Wanda', color='#2E86AB', edgecolor='black', linewidth=0.5)
        ax.hist(sparsegpt_distances.cpu().numpy(), bins=30, alpha=0.6,
               label='SparseGPT', color='#A23B72', edgecolor='black', linewidth=0.5)

        ax.set_xlabel('Hamming Distance')
        ax.set_ylabel('Frequency')
        ax.set_title(matrix_file.replace('.pt', '').replace('layer1-', ''), fontsize=10)
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Remove extra subplot
    if len(matrix_files) < len(axes):
        fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(output_dir / 'hamming_distance_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'hamming_distance_distributions.png'}")


def plot_method_comparison_heatmap(
    wanda_results: Dict,
    sparsegpt_results: Dict,
    output_dir: Path
):
    """
    Create heatmap showing separation ratios across all matrices and k values.
    """
    matrix_files = list(wanda_results.keys())
    k_values = [4, 8, 16]

    # Prepare data
    wanda_data = np.zeros((len(matrix_files), len(k_values)))
    sparsegpt_data = np.zeros((len(matrix_files), len(k_values)))

    row_labels = []
    for i, matrix_file in enumerate(matrix_files):
        label = matrix_file.replace('layer1-', '').replace('.pt', '')
        label = label.replace('self_attn.', 'attn.').replace('mlp.', '')
        row_labels.append(label)

        for j, k in enumerate(k_values):
            w_metrics = wanda_results[matrix_file][k]
            s_metrics = sparsegpt_results[matrix_file][k]

            w_ratio = w_metrics.mean_between / w_metrics.mean_within if w_metrics.mean_within > 0 else 0
            s_ratio = s_metrics.mean_between / s_metrics.mean_within if s_metrics.mean_within > 0 else 0

            wanda_data[i, j] = w_ratio
            sparsegpt_data[i, j] = s_ratio

    # Create side-by-side heatmaps
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # Wanda heatmap
    sns.heatmap(wanda_data, annot=True, fmt='.2f', cmap='YlOrRd',
               xticklabels=[f'k={k}' for k in k_values],
               yticklabels=row_labels, ax=ax1, cbar_kws={'label': 'Separation Ratio'},
               vmin=0, vmax=max(wanda_data.max(), sparsegpt_data.max()))
    ax1.set_title('Wanda: Separation Ratios', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Clusters')

    # SparseGPT heatmap
    sns.heatmap(sparsegpt_data, annot=True, fmt='.2f', cmap='YlOrRd',
               xticklabels=[f'k={k}' for k in k_values],
               yticklabels=row_labels, ax=ax2, cbar_kws={'label': 'Separation Ratio'},
               vmin=0, vmax=max(wanda_data.max(), sparsegpt_data.max()))
    ax2.set_title('SparseGPT: Separation Ratios', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Clusters')

    plt.tight_layout()
    plt.savefig(output_dir / 'separation_ratios_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'separation_ratios_heatmap.png'}")


def plot_sparsity_patterns_sample(
    wanda_dir: Path,
    sparsegpt_dir: Path,
    output_dir: Path
):
    """
    Visualize actual sparsity patterns for selected feature subsets.
    """
    # Select a few representative matrices
    selected_matrices = [
        "layer1-mlp.down_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt"
    ]

    fig, axes = plt.subplots(len(selected_matrices), 2, figsize=(12, 12))

    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    for row_idx, matrix_file in enumerate(selected_matrices):
        # Load matrices
        wanda_matrix = torch.load(wanda_dir / matrix_file, weights_only=True)
        sparsegpt_matrix = torch.load(sparsegpt_dir / matrix_file, weights_only=True)

        # Randomly select feature and find similar ones
        n_total = wanda_matrix.shape[1]
        feature_idx = np.random.randint(0, n_total)

        wanda_subset, _, _ = find_most_similar_features(wanda_matrix, feature_idx, n_similar=64)
        sparsegpt_subset, _, _ = find_most_similar_features(sparsegpt_matrix, feature_idx, n_similar=64)

        # Convert to binary for visualization
        wanda_binary = (wanda_subset != 0).float().cpu().numpy()
        sparsegpt_binary = (sparsegpt_subset != 0).float().cpu().numpy()

        # Plot Wanda
        ax_w = axes[row_idx, 0]
        im1 = ax_w.imshow(wanda_binary, aspect='auto', cmap='Greys', interpolation='nearest')
        ax_w.set_title(f"Wanda - {matrix_file.replace('layer1-', '').replace('.pt', '')}", fontsize=10)
        ax_w.set_ylabel('Input Dimension')
        if row_idx == len(selected_matrices) - 1:
            ax_w.set_xlabel('64 Similar Features')

        # Plot SparseGPT
        ax_s = axes[row_idx, 1]
        im2 = ax_s.imshow(sparsegpt_binary, aspect='auto', cmap='Greys', interpolation='nearest')
        ax_s.set_title(f"SparseGPT - {matrix_file.replace('layer1-', '').replace('.pt', '')}", fontsize=10)
        if row_idx == len(selected_matrices) - 1:
            ax_s.set_xlabel('64 Similar Features')

    plt.tight_layout()
    plt.savefig(output_dir / 'sparsity_patterns_sample.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'sparsity_patterns_sample.png'}")


def main():
    """Generate all visualizations."""

    # Define paths
    wanda_dir = Path("../../data/wanda_unstructured/layer-1")
    sparsegpt_dir = Path("../../data/sparsegpt_unstructured/layer-1")
    output_dir = Path("../../results/visualizations/clustering_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_files = [
        "layer1-mlp.down_proj.pt",
        "layer1-mlp.up_proj.pt",
        "layer1-mlp.gate_proj.pt",
        "layer1-self_attn.q_proj.pt",
        "layer1-self_attn.k_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt",
    ]

    k_values = [4, 8, 16]
    random_seed = 42

    print("Running clustering analysis for visualizations...")

    # Analyze Wanda matrices
    print("\nAnalyzing Wanda matrices...")
    wanda_results = {}
    for matrix_file in matrix_files:
        matrix_path = wanda_dir / matrix_file
        if matrix_path.exists():
            print(f"  Processing {matrix_file}...")
            results = analyze_matrix(matrix_path, n_features=128, k_values=k_values, random_seed=random_seed)
            wanda_results[matrix_file] = results

    # Analyze SparseGPT matrices
    print("\nAnalyzing SparseGPT matrices...")
    sparsegpt_results = {}
    for matrix_file in matrix_files:
        matrix_path = sparsegpt_dir / matrix_file
        if matrix_path.exists():
            print(f"  Processing {matrix_file}...")
            results = analyze_matrix(matrix_path, n_features=128, k_values=k_values, random_seed=random_seed)
            sparsegpt_results[matrix_file] = results

    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Generate all plots
    print("\n1. Separation Ratios Bar Charts...")
    plot_separation_ratios(wanda_results, sparsegpt_results, output_dir)

    print("\n2. Within vs Between Scatter Plots...")
    plot_within_between_comparison(wanda_results, sparsegpt_results, output_dir)

    print("\n3. Cluster Size Distributions...")
    plot_cluster_size_distributions(wanda_results, sparsegpt_results, output_dir)

    print("\n4. Hamming Distance Distributions...")
    plot_hamming_distance_distributions(wanda_dir, sparsegpt_dir, output_dir)

    print("\n5. Separation Ratios Heatmap...")
    plot_method_comparison_heatmap(wanda_results, sparsegpt_results, output_dir)

    print("\n6. Sparsity Pattern Samples...")
    plot_sparsity_patterns_sample(wanda_dir, sparsegpt_dir, output_dir)

    print("\n" + "="*80)
    print(f"All visualizations saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
