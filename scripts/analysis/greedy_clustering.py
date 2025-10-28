"""
Greedy bottom-up clustering algorithm.

Start with 8 random feature vectors, iteratively merge the most similar pairs
based on Hamming distance, building up to 64 clusters. Compare two strategies:
1. Replace: Remove merged features from pool
2. Keep: Put merged features back into pool

Analyze distribution of cluster quality (good vs bad clusters).
"""

import sys
from pathlib import Path
import torch
import numpy as np
import json
from typing import Dict, List, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import compute_hamming_distance_batch


@dataclass
class Cluster:
    """Represents a cluster of feature vectors."""
    feature_indices: List[int]  # Original feature indices in this cluster
    centroid_idx: int = None  # Representative feature (if using centroids)

    def size(self) -> int:
        return len(self.feature_indices)

    def __repr__(self):
        return f"Cluster(size={self.size()}, features={self.feature_indices[:5]}{'...' if self.size() > 5 else ''})"


def compute_cluster_hamming_distance(
    cluster1: Cluster,
    cluster2: Cluster,
    binary_matrix: torch.Tensor
) -> float:
    """
    Compute average Hamming distance between two clusters.

    Uses average linkage: mean distance between all pairs.
    """
    features1 = binary_matrix[:, cluster1.feature_indices]
    features2 = binary_matrix[:, cluster2.feature_indices]

    total_dist = 0.0
    count = 0

    for i in range(features1.shape[1]):
        for j in range(features2.shape[1]):
            dist = (features1[:, i] != features2[:, j]).float().mean().item()
            total_dist += dist
            count += 1

    return total_dist / count if count > 0 else 0.0


def find_closest_cluster_pair(
    clusters: List[Cluster],
    binary_matrix: torch.Tensor
) -> Tuple[int, int, float]:
    """
    Find the two closest clusters.

    Returns:
        idx1, idx2: Indices of closest cluster pair
        distance: Hamming distance between them
    """
    min_dist = float('inf')
    best_i, best_j = 0, 1

    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            dist = compute_cluster_hamming_distance(clusters[i], clusters[j], binary_matrix)
            if dist < min_dist:
                min_dist = dist
                best_i, best_j = i, j

    return best_i, best_j, min_dist


def merge_clusters(cluster1: Cluster, cluster2: Cluster) -> Cluster:
    """Merge two clusters."""
    return Cluster(
        feature_indices=cluster1.feature_indices + cluster2.feature_indices
    )


def greedy_clustering_replace(
    initial_features: List[int],
    target_n_clusters: int,
    binary_matrix: torch.Tensor,
    n_total_features: int,
    verbose: bool = False
) -> Tuple[List[Cluster], List[Dict]]:
    """
    Greedy clustering with REPLACE strategy.

    Start with 8 features, grow to 64 clusters by:
    1. Add a new random feature as a singleton cluster
    2. Find closest pair of clusters and merge them
    This keeps cluster count growing: n -> n+1 -> n (merge) -> n+1 -> ...
    Net effect: +1 cluster every 2 steps, so 8 → 64 requires 56*2 = 112 steps

    Args:
        initial_features: Initial feature indices
        target_n_clusters: Target number of clusters to create
        binary_matrix: Binary weight matrix
        n_total_features: Total features available for sampling
        verbose: Print progress

    Returns:
        final_clusters: List of final clusters
        history: Operation history with statistics
    """
    # Initialize: each feature is its own cluster
    clusters = [Cluster(feature_indices=[f]) for f in initial_features]
    used_features = set(initial_features)
    available_features = set(range(n_total_features)) - used_features
    history = []

    n_initial = len(initial_features)
    n_additions_needed = target_n_clusters - n_initial

    if verbose:
        print(f"Replace strategy: {n_initial} initial → {target_n_clusters} clusters ({n_additions_needed} net additions)")

    step = 0
    pbar = tqdm(total=n_additions_needed, desc="Replace strategy", disable=not verbose)

    # We want to reach target_n_clusters total features involved
    # Start with n_initial (8), add features one at a time, merge after each add
    # This keeps cluster count manageable while involving more features
    while len(used_features) < target_n_clusters:
        # Step 1: Add new random feature
        if not available_features:
            break
        new_feature = np.random.choice(list(available_features))
        available_features.remove(new_feature)
        used_features.add(new_feature)
        clusters.append(Cluster(feature_indices=[new_feature]))

        history.append({
            "step": step,
            "action": "add",
            "feature": new_feature,
            "n_clusters": len(clusters)
        })
        step += 1

        # Step 2: Always merge closest pair after adding
        if len(clusters) >= 2:
            i, j, dist = find_closest_cluster_pair(clusters, binary_matrix)
            merged = merge_clusters(clusters[i], clusters[j])

            history.append({
                "step": step,
                "action": "merge",
                "cluster1_size": clusters[i].size(),
                "cluster2_size": clusters[j].size(),
                "merged_size": merged.size(),
                "distance": float(dist),
                "n_clusters": len(clusters) - 1
            })

            # Remove old clusters and add merged cluster
            if i < j:
                clusters.pop(j)
                clusters.pop(i)
            else:
                clusters.pop(i)
                clusters.pop(j)
            clusters.append(merged)
            step += 1

        pbar.update(1)

    pbar.close()
    return clusters, history


def greedy_clustering_keep(
    initial_features: List[int],
    target_n_clusters: int,
    binary_matrix: torch.Tensor,
    n_total_features: int,
    verbose: bool = False
) -> Tuple[List[Cluster], List[Dict]]:
    """
    Greedy clustering with KEEP strategy.

    Merged clusters are put back into the feature pool (sampling with replacement).
    After each merge, randomly sample a new feature from the full matrix.

    Args:
        initial_features: Initial feature indices
        target_n_clusters: Target number of clusters to create
        binary_matrix: Binary weight matrix
        n_total_features: Total number of features in matrix
        verbose: Print progress

    Returns:
        final_clusters: List of final clusters
        history: Merging history with statistics
    """
    # Initialize: each feature is its own cluster
    clusters = [Cluster(feature_indices=[f]) for f in initial_features]
    history = []

    n_initial = len(initial_features)
    n_additions_needed = target_n_clusters - n_initial

    if verbose:
        print(f"Keep strategy: {n_initial} initial → {target_n_clusters} clusters ({n_additions_needed} net additions)")

    step = 0
    pbar = tqdm(total=n_additions_needed, desc="Keep strategy", disable=not verbose)

    # Track how many features we've added (not unique due to replacement)
    features_added = 0

    while features_added < n_additions_needed:
        # Step 1: Add new random feature (with replacement - can reuse)
        new_feature = np.random.randint(0, n_total_features)
        clusters.append(Cluster(feature_indices=[new_feature]))
        features_added += 1

        history.append({
            "step": step,
            "action": "add",
            "feature": new_feature,
            "n_clusters": len(clusters)
        })
        step += 1

        # Step 2: Always merge closest pair after adding
        if len(clusters) >= 2:
            i, j, dist = find_closest_cluster_pair(clusters, binary_matrix)
            merged = merge_clusters(clusters[i], clusters[j])

            history.append({
                "step": step,
                "action": "merge",
                "cluster1_size": clusters[i].size(),
                "cluster2_size": clusters[j].size(),
                "merged_size": merged.size(),
                "distance": float(dist),
                "n_clusters": len(clusters) - 1
            })

            # Remove old clusters and add merged cluster
            if i < j:
                clusters.pop(j)
                clusters.pop(i)
            else:
                clusters.pop(i)
                clusters.pop(j)
            clusters.append(merged)
            step += 1

        pbar.update(1)

    pbar.close()
    return clusters, history


def evaluate_cluster_quality(
    cluster: Cluster,
    binary_matrix: torch.Tensor,
    n_samples: int = 50
) -> Dict:
    """
    Evaluate quality of a single cluster.

    Quality metrics:
    - Within-cluster Hamming distance (cohesion)
    - Size
    """
    if cluster.size() < 2:
        return {
            "size": cluster.size(),
            "within_distance": 0.0,
            "cohesion_score": 1.0  # Perfect cohesion for singleton
        }

    features = binary_matrix[:, cluster.feature_indices]

    # Sample pairs for within-cluster distance
    distances = []
    n_possible_pairs = (cluster.size() * (cluster.size() - 1)) // 2
    n_sample = min(n_samples, n_possible_pairs)

    for _ in range(n_sample):
        i = np.random.randint(0, cluster.size())
        j = np.random.randint(0, cluster.size())
        if i != j:
            dist = (features[:, i] != features[:, j]).float().mean().item()
            distances.append(dist)

    within_dist = np.mean(distances) if distances else 0.0

    # Cohesion score: inverse of within-distance (higher is better)
    # Range: [0, 1] where 1 = perfect cohesion
    cohesion_score = 1.0 - within_dist

    return {
        "size": cluster.size(),
        "within_distance": float(within_dist),
        "cohesion_score": float(cohesion_score)
    }


def classify_cluster_quality(cohesion_score: float, threshold: float = 0.7) -> str:
    """
    Classify cluster as good or bad based on cohesion score.

    Good cluster: high cohesion (low within-cluster distance)
    Bad cluster: low cohesion (high within-cluster distance)
    """
    return "good" if cohesion_score >= threshold else "bad"


def analyze_clustering_distribution(
    matrix_path: str,
    n_initial_features: int = 8,
    target_n_clusters: int = 64,
    n_trials: int = 100,
    cohesion_threshold: float = 0.7,
    seed: int = 42
) -> Dict:
    """
    Run multiple trials of greedy clustering and analyze distribution.

    Args:
        matrix_path: Path to weight matrix
        n_initial_features: Number of initial features (default: 8)
        target_n_clusters: Target number of clusters (default: 64)
        n_trials: Number of trials to run
        cohesion_threshold: Threshold for good vs bad clusters
        seed: Random seed

    Returns:
        Dictionary with results for both strategies
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"GREEDY CLUSTERING ANALYSIS: {Path(matrix_path).name}")
    print(f"{'='*60}")
    print(f"Initial features: {n_initial_features}")
    print(f"Target clusters: {target_n_clusters}")
    print(f"Trials: {n_trials}")
    print(f"Cohesion threshold: {cohesion_threshold}")

    # Load matrix
    matrix = torch.load(matrix_path, map_location=torch.device('cpu'))
    print(f"Matrix shape: {matrix.shape}")

    binary_matrix = (matrix > 0).int()
    n_total_features = binary_matrix.shape[1]

    results = {
        "matrix_path": str(matrix_path),
        "matrix_shape": list(matrix.shape),
        "parameters": {
            "n_initial_features": n_initial_features,
            "target_n_clusters": target_n_clusters,
            "n_trials": n_trials,
            "cohesion_threshold": cohesion_threshold,
            "seed": seed
        },
        "replace_strategy": {
            "cohesion_scores": [],
            "cluster_sizes": [],
            "n_good_clusters": [],
            "n_bad_clusters": [],
            "trials": []
        },
        "keep_strategy": {
            "cohesion_scores": [],
            "cluster_sizes": [],
            "n_good_clusters": [],
            "n_bad_clusters": [],
            "trials": []
        }
    }

    print(f"\n{'='*60}")
    print("REPLACE STRATEGY")
    print(f"{'='*60}")

    for trial in tqdm(range(n_trials), desc="Replace trials"):
        # Sample initial features
        initial_features = np.random.choice(n_total_features, size=n_initial_features, replace=False).tolist()

        # Run replace strategy
        clusters, history = greedy_clustering_replace(
            initial_features, target_n_clusters, binary_matrix, n_total_features, verbose=False
        )

        # Evaluate clusters
        trial_cohesion = []
        trial_sizes = []
        n_good = 0
        n_bad = 0

        for cluster in clusters:
            quality = evaluate_cluster_quality(cluster, binary_matrix)
            trial_cohesion.append(quality['cohesion_score'])
            trial_sizes.append(quality['size'])

            if classify_cluster_quality(quality['cohesion_score'], cohesion_threshold) == "good":
                n_good += 1
            else:
                n_bad += 1

        results["replace_strategy"]["cohesion_scores"].extend(trial_cohesion)
        results["replace_strategy"]["cluster_sizes"].extend(trial_sizes)
        results["replace_strategy"]["n_good_clusters"].append(n_good)
        results["replace_strategy"]["n_bad_clusters"].append(n_bad)
        results["replace_strategy"]["trials"].append({
            "trial": trial,
            "n_good": n_good,
            "n_bad": n_bad,
            "mean_cohesion": float(np.mean(trial_cohesion)),
            "mean_size": float(np.mean(trial_sizes))
        })

    print(f"\n{'='*60}")
    print("KEEP STRATEGY")
    print(f"{'='*60}")

    for trial in tqdm(range(n_trials), desc="Keep trials"):
        # Sample initial features
        initial_features = np.random.choice(n_total_features, size=n_initial_features, replace=False).tolist()

        # Run keep strategy
        clusters, history = greedy_clustering_keep(
            initial_features, target_n_clusters, binary_matrix, n_total_features, verbose=False
        )

        # Evaluate clusters
        trial_cohesion = []
        trial_sizes = []
        n_good = 0
        n_bad = 0

        for cluster in clusters:
            quality = evaluate_cluster_quality(cluster, binary_matrix)
            trial_cohesion.append(quality['cohesion_score'])
            trial_sizes.append(quality['size'])

            if classify_cluster_quality(quality['cohesion_score'], cohesion_threshold) == "good":
                n_good += 1
            else:
                n_bad += 1

        results["keep_strategy"]["cohesion_scores"].extend(trial_cohesion)
        results["keep_strategy"]["cluster_sizes"].extend(trial_sizes)
        results["keep_strategy"]["n_good_clusters"].append(n_good)
        results["keep_strategy"]["n_bad_clusters"].append(n_bad)
        results["keep_strategy"]["trials"].append({
            "trial": trial,
            "n_good": n_good,
            "n_bad": n_bad,
            "mean_cohesion": float(np.mean(trial_cohesion)),
            "mean_size": float(np.mean(trial_sizes))
        })

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for strategy_name in ["replace_strategy", "keep_strategy"]:
        strategy = results[strategy_name]
        print(f"\n{strategy_name.replace('_', ' ').title()}:")
        print(f"  Mean good clusters per trial: {np.mean(strategy['n_good_clusters']):.1f}")
        print(f"  Mean bad clusters per trial: {np.mean(strategy['n_bad_clusters']):.1f}")
        print(f"  Mean cohesion score: {np.mean(strategy['cohesion_scores']):.4f}")
        print(f"  Median cohesion score: {np.median(strategy['cohesion_scores']):.4f}")
        print(f"  Mean cluster size: {np.mean(strategy['cluster_sizes']):.2f}")

    return results


def create_visualization(results: Dict, output_dir: Path):
    """Create comprehensive visualization of greedy clustering results."""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    replace = results["replace_strategy"]
    keep = results["keep_strategy"]

    # Plot 1: Cohesion score distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(replace['cohesion_scores'], bins=30, alpha=0.6, label='Replace', color='#2E86AB', edgecolor='black')
    ax1.hist(keep['cohesion_scores'], bins=30, alpha=0.6, label='Keep', color='#A23B72', edgecolor='black')
    ax1.axvline(results['parameters']['cohesion_threshold'], color='red', linestyle='--', linewidth=2, label='Threshold')
    ax1.set_xlabel('Cohesion Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('Cohesion Score Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Plot 2: Good vs Bad clusters per trial
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(2)
    width = 0.35
    replace_means = [np.mean(replace['n_good_clusters']), np.mean(replace['n_bad_clusters'])]
    keep_means = [np.mean(keep['n_good_clusters']), np.mean(keep['n_bad_clusters'])]

    ax2.bar(x - width/2, replace_means, width, label='Replace', color='#2E86AB', alpha=0.7, edgecolor='black')
    ax2.bar(x + width/2, keep_means, width, label='Keep', color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Mean # Clusters', fontsize=12, fontweight='bold')
    ax2.set_title('Good vs Bad Clusters', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Good', 'Bad'])
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Cluster size distributions
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(replace['cluster_sizes'], bins=30, alpha=0.6, label='Replace', color='#2E86AB', edgecolor='black')
    ax3.hist(keep['cluster_sizes'], bins=30, alpha=0.6, label='Keep', color='#A23B72', edgecolor='black')
    ax3.set_xlabel('Cluster Size', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Cluster Size Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(alpha=0.3)
    ax3.set_yscale('log')

    # Plot 4: Box plot of good clusters per trial
    ax4 = fig.add_subplot(gs[1, 0])
    box_data = [replace['n_good_clusters'], keep['n_good_clusters']]
    bp = ax4.boxplot(box_data, labels=['Replace', 'Keep'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#2E86AB')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#A23B72')
    bp['boxes'][1].set_alpha(0.7)
    ax4.set_ylabel('# Good Clusters', fontsize=12, fontweight='bold')
    ax4.set_title('Good Clusters Distribution', fontsize=14, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # Plot 5: Scatter plot - good vs bad per trial
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(replace['n_good_clusters'], replace['n_bad_clusters'],
               alpha=0.5, s=50, c='#2E86AB', label='Replace', edgecolor='black')
    ax5.scatter(keep['n_good_clusters'], keep['n_bad_clusters'],
               alpha=0.5, s=50, c='#A23B72', label='Keep', edgecolor='black')
    ax5.set_xlabel('# Good Clusters', fontsize=12, fontweight='bold')
    ax5.set_ylabel('# Bad Clusters', fontsize=12, fontweight='bold')
    ax5.set_title('Good vs Bad per Trial', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # Plot 6: Mean cohesion over trials
    ax6 = fig.add_subplot(gs[1, 2])
    replace_cohesion_per_trial = [t['mean_cohesion'] for t in replace['trials']]
    keep_cohesion_per_trial = [t['mean_cohesion'] for t in keep['trials']]
    ax6.plot(replace_cohesion_per_trial, alpha=0.7, linewidth=2, color='#2E86AB', label='Replace')
    ax6.plot(keep_cohesion_per_trial, alpha=0.7, linewidth=2, color='#A23B72', label='Keep')
    ax6.axhline(results['parameters']['cohesion_threshold'], color='red', linestyle='--', alpha=0.5)
    ax6.set_xlabel('Trial', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Mean Cohesion Score', fontsize=12, fontweight='bold')
    ax6.set_title('Cohesion Over Trials', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(alpha=0.3)

    # Plot 7-8: Violin plots
    ax7 = fig.add_subplot(gs[2, 0])
    parts = ax7.violinplot([replace['cohesion_scores']], positions=[0], widths=0.7, showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#2E86AB')
        pc.set_alpha(0.7)
    ax7.set_xticks([0])
    ax7.set_xticklabels(['Replace'])
    ax7.set_ylabel('Cohesion Score', fontsize=12, fontweight='bold')
    ax7.set_title('Replace Strategy Detail', fontsize=14, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    parts = ax8.violinplot([keep['cohesion_scores']], positions=[0], widths=0.7, showmeans=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#A23B72')
        pc.set_alpha(0.7)
    ax8.set_xticks([0])
    ax8.set_xticklabels(['Keep'])
    ax8.set_ylabel('Cohesion Score', fontsize=12, fontweight='bold')
    ax8.set_title('Keep Strategy Detail', fontsize=14, fontweight='bold')
    ax8.grid(axis='y', alpha=0.3)

    # Plot 9: Summary statistics table
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')

    summary_text = "SUMMARY STATISTICS\n"
    summary_text += "="*40 + "\n\n"
    summary_text += f"Initial features: {results['parameters']['n_initial_features']}\n"
    summary_text += f"Target clusters: {results['parameters']['target_n_clusters']}\n"
    summary_text += f"Trials: {results['parameters']['n_trials']}\n\n"

    summary_text += "REPLACE STRATEGY:\n"
    summary_text += f"  Good clusters: {np.mean(replace['n_good_clusters']):.1f} ± {np.std(replace['n_good_clusters']):.1f}\n"
    summary_text += f"  Bad clusters: {np.mean(replace['n_bad_clusters']):.1f} ± {np.std(replace['n_bad_clusters']):.1f}\n"
    summary_text += f"  Mean cohesion: {np.mean(replace['cohesion_scores']):.4f}\n\n"

    summary_text += "KEEP STRATEGY:\n"
    summary_text += f"  Good clusters: {np.mean(keep['n_good_clusters']):.1f} ± {np.std(keep['n_good_clusters']):.1f}\n"
    summary_text += f"  Bad clusters: {np.mean(keep['n_bad_clusters']):.1f} ± {np.std(keep['n_bad_clusters']):.1f}\n"
    summary_text += f"  Mean cohesion: {np.mean(keep['cohesion_scores']):.4f}\n"

    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    matrix_name = Path(results['matrix_path']).stem
    plt.suptitle(f'Greedy Clustering Analysis: {matrix_name}\n8 initial features → 64 clusters',
                 fontsize=16, fontweight='bold')

    output_path = output_dir / f"{matrix_name}_greedy_clustering.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()


def main():
    """Run greedy clustering analysis."""

    # Test on example matrices
    data_dirs = {
        'sparsegpt': Path(__file__).parent.parent.parent / "data/sparsegpt_unstructured",
        'wanda': Path(__file__).parent.parent.parent / "data/wanda_unstructured"
    }

    results_dir = Path(__file__).parent.parent.parent / "results/metrics/greedy_clustering"
    results_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = Path(__file__).parent.parent.parent / "results/visualizations/greedy_clustering"
    viz_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    # Analyze layer 0 for each method and projection
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

        # Run analysis
        results = analyze_clustering_distribution(
            matrix_path=str(matrix_path),
            n_initial_features=8,
            target_n_clusters=64,
            n_trials=100,
            cohesion_threshold=0.7,
            seed=42
        )
        all_results.append(results)

        # Create visualization
        create_visualization(results, viz_dir)

    # Save all results
    output_path = results_dir / "greedy_clustering_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✓ All results saved to: {output_path}")
    print(f"✓ Visualizations in: {viz_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
