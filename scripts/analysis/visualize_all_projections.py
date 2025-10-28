"""
Create comprehensive visualizations for all projection types (down, up, gate)
for both SparseGPT and Wanda clustering results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_projection_comparison(method: str, viz_dir: Path):
    """Create comparison visualization for all projection types of a method."""

    results_dir = Path(__file__).parent.parent.parent / f"results/metrics/agglomerative_clustering/{method}"

    # Load results for each projection type
    projections = {}
    for proj_type in ['down_proj', 'up_proj', 'gate_proj']:
        if proj_type == 'down_proj':
            json_file = results_dir / "agglomerative_clustering_full_results.json"
        else:
            json_file = results_dir / f"agglomerative_clustering_{proj_type}_full_results.json"

        if json_file.exists():
            with open(json_file, 'r') as f:
                projections[proj_type] = json.load(f)
            print(f"Loaded {method} {proj_type}: {len(projections[proj_type])} layers")

    if not projections:
        print(f"No results found for {method}")
        return

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    colors = {
        'down_proj': '#2E86AB',
        'up_proj': '#A23B72',
        'gate_proj': '#F18F01'
    }

    # Plot 1: Separation Ratio Comparison (all projections, layer 0)
    ax1 = fig.add_subplot(gs[0, 0])
    for proj_type, results in projections.items():
        if results:
            layer0 = results[0]
            n_clusters = [c['n_clusters'] for c in layer0['clusterings']]
            sep_ratios = [c['metrics']['separation_ratio'] for c in layer0['clusterings']]
            ax1.plot(n_clusters, sep_ratios, marker='o', linewidth=2.5,
                    color=colors[proj_type], label=proj_type.replace('_', ' '), markersize=8)

    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    ax1.set_xlabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Separation Ratio', fontsize=12, fontweight='bold')
    ax1.set_title(f'{method.upper()} Layer 0: Projection Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_xscale('log')

    # Plot 2: Within-Cluster Distance
    ax2 = fig.add_subplot(gs[0, 1])
    for proj_type, results in projections.items():
        if results:
            layer0 = results[0]
            n_clusters = [c['n_clusters'] for c in layer0['clusterings']]
            within = [c['metrics']['mean_within'] for c in layer0['clusterings']]
            ax2.plot(n_clusters, within, marker='s', linewidth=2.5,
                    color=colors[proj_type], label=proj_type.replace('_', ' '), markersize=8)

    ax2.set_xlabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Within Distance', fontsize=12, fontweight='bold')
    ax2.set_title('Within-Cluster Cohesion', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_xscale('log')

    # Plot 3: Between-Cluster Distance
    ax3 = fig.add_subplot(gs[0, 2])
    for proj_type, results in projections.items():
        if results:
            layer0 = results[0]
            n_clusters = [c['n_clusters'] for c in layer0['clusterings']]
            between = [c['metrics']['mean_between'] for c in layer0['clusterings']]
            ax3.plot(n_clusters, between, marker='^', linewidth=2.5,
                    color=colors[proj_type], label=proj_type.replace('_', ' '), markersize=8)

    ax3.set_xlabel('Number of Clusters', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Mean Between Distance', fontsize=12, fontweight='bold')
    ax3.set_title('Between-Cluster Separation', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3)
    ax3.set_xscale('log')

    # Plots 4-6: Individual projection type heatmaps (separation ratio vs k and layer)
    proj_list = list(projections.keys())
    for idx, proj_type in enumerate(proj_list):
        ax = fig.add_subplot(gs[1, idx])
        results = projections[proj_type]

        # Create heatmap data
        n_layers = len(results)
        n_k_values = len(results[0]['clusterings'])
        k_values = [c['n_clusters'] for c in results[0]['clusterings']]

        heatmap_data = np.zeros((n_layers, n_k_values))
        for i, layer_result in enumerate(results):
            for j, clustering in enumerate(layer_result['clusterings']):
                heatmap_data[i, j] = clustering['metrics']['separation_ratio']

        im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=1.0, vmax=2.0)
        ax.set_xlabel('Cluster Count', fontsize=11, fontweight='bold')
        ax.set_ylabel('Layer', fontsize=11, fontweight='bold')
        ax.set_title(f'{proj_type.replace("_", " ").title()}', fontsize=12, fontweight='bold')
        ax.set_xticks(range(n_k_values))
        ax.set_xticklabels(k_values, rotation=45)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels([f"L{i}" for i in range(n_layers)])

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Sep Ratio', fontsize=10)

    # Plots 7-9: Best k analysis per projection
    for idx, proj_type in enumerate(proj_list):
        ax = fig.add_subplot(gs[2, idx])
        results = projections[proj_type]

        layers = []
        best_k = []
        best_sep = []

        for i, layer_result in enumerate(results):
            layer_name = f"Layer {i}"
            best_clustering = max(layer_result['clusterings'],
                                key=lambda x: x['metrics']['separation_ratio'])
            layers.append(i)
            best_k.append(best_clustering['n_clusters'])
            best_sep.append(best_clustering['metrics']['separation_ratio'])

        ax.bar(layers, best_sep, color=colors[proj_type], alpha=0.7, edgecolor='black')

        # Add k values on top of bars
        for i, (k, sep) in enumerate(zip(best_k, best_sep)):
            ax.text(i, sep + 0.02, f'k={k}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Layer', fontsize=11, fontweight='bold')
        ax.set_ylabel('Best Separation Ratio', fontsize=11, fontweight='bold')
        ax.set_title(f'{proj_type.replace("_", " ").title()}: Best k per Layer', fontsize=12, fontweight='bold')
        ax.set_xticks(layers)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)

    plt.suptitle(f'{method.upper()} Agglomerative Clustering Analysis\nFull Width (11,008 features)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save
    output_path = viz_dir / f"{method}_all_projections_analysis.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


def create_summary_table(method: str, viz_dir: Path):
    """Create summary table for all projections."""

    results_dir = Path(__file__).parent.parent.parent / f"results/metrics/agglomerative_clustering/{method}"

    print(f"\n{'='*80}")
    print(f"{method.upper()} AGGLOMERATIVE CLUSTERING SUMMARY")
    print(f"{'='*80}")

    for proj_type in ['down_proj', 'up_proj', 'gate_proj']:
        if proj_type == 'down_proj':
            json_file = results_dir / "agglomerative_clustering_full_results.json"
        else:
            json_file = results_dir / f"agglomerative_clustering_{proj_type}_full_results.json"

        if not json_file.exists():
            continue

        with open(json_file, 'r') as f:
            results = json.load(f)

        print(f"\n{proj_type.upper().replace('_', ' ')}:")
        print(f"  Layers analyzed: {len(results)}")

        for layer_idx, layer_result in enumerate(results):
            print(f"\n  Layer {layer_idx}:")
            print(f"    Matrix shape: {layer_result['matrix_shape']}")
            print(f"    Computation time: {layer_result.get('computation_time_seconds', 0)/60:.1f} min")

            # Find best clustering
            best = max(layer_result['clusterings'], key=lambda x: x['metrics']['separation_ratio'])
            print(f"    Best separation: {best['metrics']['separation_ratio']:.4f} @ k={best['n_clusters']}")

            # Show all k values
            print(f"\n    k   | Sep Ratio | Within  | Between")
            print(f"    " + "-"*40)
            for c in layer_result['clusterings']:
                m = c['metrics']
                print(f"    {c['n_clusters']:>4} | {m['separation_ratio']:>9.4f} | {m['mean_within']:>7.4f} | {m['mean_between']:>7.4f}")


def main():
    """Create all visualizations."""

    base_viz_dir = Path(__file__).parent.parent.parent / "results/visualizations/agglomerative_clustering"

    # Create visualizations for both methods
    for method in ['sparsegpt', 'wanda']:
        method_viz_dir = base_viz_dir / method
        method_viz_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Creating visualizations for {method.upper()}")
        print(f"{'='*60}")

        create_projection_comparison(method, method_viz_dir)
        create_summary_table(method, method_viz_dir)

    print(f"\n{'='*80}")
    print(f"✓ All visualizations complete!")
    print(f"✓ Results in: {base_viz_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
