"""
Visualize the full-width agglomerative clustering results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load results
results_path = Path(__file__).parent.parent.parent / "results/metrics/agglomerative_clustering/agglomerative_clustering_full_results.json"
viz_dir = Path(__file__).parent.parent.parent / "results/visualizations/agglomerative_clustering"
viz_dir.mkdir(parents=True, exist_ok=True)

with open(results_path, 'r') as f:
    results = json.load(f)

# Extract data for plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = ['#2E86AB', '#A23B72', '#F18F01']
layer_names = [f"Layer {Path(r['matrix_path']).stem.split('-')[0].replace('layer', '')}" for r in results]

# Plot 1: Separation Ratio vs Number of Clusters
ax = axes[0, 0]
for idx, (result, color, name) in enumerate(zip(results, colors, layer_names)):
    n_clusters = [c['n_clusters'] for c in result['clusterings']]
    sep_ratios = [c['metrics']['separation_ratio'] for c in result['clusterings']]
    ax.plot(n_clusters, sep_ratios, marker='o', linewidth=2.5, color=color, label=name, markersize=8)

ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, linewidth=2, label='No separation')
ax.set_xlabel('Number of Clusters', fontsize=14, fontweight='bold')
ax.set_ylabel('Separation Ratio (Between/Within)', fontsize=14, fontweight='bold')
ax.set_title('Cluster Quality: Full Width Analysis\n(All 11,008 Features)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Plot 2: Within-Cluster Distance
ax = axes[0, 1]
for idx, (result, color, name) in enumerate(zip(results, colors, layer_names)):
    n_clusters = [c['n_clusters'] for c in result['clusterings']]
    within_dist = [c['metrics']['mean_within'] for c in result['clusterings']]
    ax.plot(n_clusters, within_dist, marker='s', linewidth=2.5, color=color, label=name, markersize=8)

ax.set_xlabel('Number of Clusters', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Within-Cluster Distance', fontsize=14, fontweight='bold')
ax.set_title('Cluster Cohesion', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Plot 3: Between-Cluster Distance
ax = axes[1, 0]
for idx, (result, color, name) in enumerate(zip(results, colors, layer_names)):
    n_clusters = [c['n_clusters'] for c in result['clusterings']]
    between_dist = [c['metrics']['mean_between'] for c in result['clusterings']]
    ax.plot(n_clusters, between_dist, marker='^', linewidth=2.5, color=color, label=name, markersize=8)

ax.set_xlabel('Number of Clusters', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Between-Cluster Distance', fontsize=14, fontweight='bold')
ax.set_title('Cluster Separation', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
ax.set_xscale('log')

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

# Create summary text
summary_text = "FULL-WIDTH CLUSTERING SUMMARY\n"
summary_text += "="*50 + "\n\n"

for result, name in zip(results, layer_names):
    summary_text += f"{name}:\n"
    summary_text += f"  Matrix shape: {result['matrix_shape']}\n"

    # Get best separation ratio
    best_clustering = max(result['clusterings'], key=lambda x: x['metrics']['separation_ratio'])
    n_clust = best_clustering['n_clusters']
    sep_ratio = best_clustering['metrics']['separation_ratio']

    summary_text += f"  Best separation: {sep_ratio:.4f} @ {n_clust} clusters\n"
    summary_text += f"  Computation time: {result.get('computation_time_seconds', 0)/60:.1f} min\n\n"

# Add key findings
summary_text += "\nKEY FINDINGS:\n"
summary_text += "-" * 50 + "\n"
summary_text += "• Layer 0 shows strongest hierarchical structure\n"
summary_text += "  (separation improves 1.03→2.01 with more clusters)\n\n"
summary_text += "• Layer 2 shows weakest clustering\n"
summary_text += "  (flat separation ~1.04-1.14 across all k)\n\n"
summary_text += "• Layer 3 is intermediate\n"
summary_text += "  (moderate improvement 1.04→1.64)\n\n"
summary_text += "• Higher cluster counts (1000-2000) provide\n"
summary_text += "  best separation in all layers\n"

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

# Save
output_path = viz_dir / "full_width_clustering_analysis.png"
plt.savefig(output_path, dpi=200, bbox_inches='tight')
print(f"Saved visualization to: {output_path}")

# Print summary to console
print("\n" + "="*60)
print("FULL-WIDTH AGGLOMERATIVE CLUSTERING RESULTS")
print("="*60)
for result, name in zip(results, layer_names):
    print(f"\n{name}:")
    print(f"  Features analyzed: {result['matrix_shape'][1]}")
    print(f"  Computation time: {result.get('computation_time_seconds', 0)/60:.2f} minutes")
    print(f"\n  Cluster Count | Sep Ratio | Within Dist | Between Dist")
    print(f"  " + "-"*55)
    for clustering in result['clusterings']:
        n = clustering['n_clusters']
        m = clustering['metrics']
        print(f"  {n:>12} | {m['separation_ratio']:>9.4f} | {m['mean_within']:>11.4f} | {m['mean_between']:>12.4f}")

print("\n" + "="*60)
