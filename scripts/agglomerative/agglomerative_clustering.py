"""
Hierarchical Agglomerative Clustering with Random Sampling

This script performs hierarchical agglomerative clustering on neural network weight matrices
using the following algorithm:

Algorithm Steps:
1. Sample N seed features: Randomly select N non-zero feature vectors (columns) from the matrix
2. Create initial groups: For each seed, find G most similar features → N groups of G features
3. Random sampling within groups: From each group of G, randomly sample 1 representative
4. Recursive clustering: Repeat grouping and sampling until convergence:
   - Number of levels is automatically calculated based on N and G
   - Each level reduces features by a factor of G
5. Visualize: Create visualization showing the hierarchical clustering structure

Configuration:
    N_SEED_FEATURES: Number of seed features to sample (default: 512)
    GROUP_SIZE: Size of each group at every level (default: 8)

    This creates: N_SEED_FEATURES × GROUP_SIZE total initial features
    Number of levels: ceil(log_GROUP_SIZE(N_SEED_FEATURES × GROUP_SIZE))

Usage:
    python agglomerative_clustering.py

    To change configuration, modify N_SEED_FEATURES and GROUP_SIZE constants below.
"""

import sys
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.cluster.hierarchy import dendrogram, linkage
import math

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.custom import get_unstructured_matrices, set_seed
from utils.hamming_analysis import find_most_similar_features

# ============================================================================
# CONFIGURATION - Modify these values to change clustering parameters
# ============================================================================
N_SEED_FEATURES = 128  # Number of seed features to sample
GROUP_SIZE = 8         # Size of groups at each level (features per group)
RANDOM_SEED = 42       # Random seed for reproducibility

# Calculate derived parameters
INITIAL_FEATURES = N_SEED_FEATURES * GROUP_SIZE  # Total features at level 0
MAX_LEVELS = math.ceil(math.log(INITIAL_FEATURES, GROUP_SIZE))  # Auto-calculate levels needed

print(f"\n{'='*80}")
print(f"CONFIGURATION")
print(f"{'='*80}")
print(f"Seed features: {N_SEED_FEATURES}")
print(f"Group size: {GROUP_SIZE}")
print(f"Initial features (Level 0): {INITIAL_FEATURES} ({N_SEED_FEATURES} × {GROUP_SIZE})")
print(f"Maximum levels: {MAX_LEVELS}")
print(f"Random seed: {RANDOM_SEED}")
print(f"{'='*80}\n")

def sample_nonzero_features(matrix, n_sample=128, samples_per_feature=8):
    """
    Sample n_sample seed features, then for each seed find samples_per_feature most similar features.
    This creates n_sample groups of samples_per_feature features each.

    Args:
        matrix: Feature matrix (neurons x features)
        n_sample: Number of seed features to sample (default: 128)
        samples_per_feature: Number of similar features per seed (default: 8)

    Returns:
        Tuple of (all_sampled_indices, seed_groups) where:
            - all_sampled_indices: Flat list of all sampled feature indices
            - seed_groups: List of lists, where each inner list has samples_per_feature indices
    """
    n_features = matrix.shape[1]

    # Get all feature indices where sum is not 0
    valid_indices = []
    for i in range(n_features):
        if matrix[:, i].sum() != 0:
            valid_indices.append(i)

    if len(valid_indices) < n_sample:
        print(f"Warning: Only {len(valid_indices)} non-zero features available, using all of them")
        n_sample = len(valid_indices)

    # Sample n_sample seed features
    seed_indices = random.sample(valid_indices, n_sample)

    # For each seed, find its most similar features
    seed_groups = []
    used_features = set()

    for seed_idx in seed_indices:
        # Find samples_per_feature most similar features
        _, similar_indices, _ = find_most_similar_features(matrix, seed_idx, n_similar=samples_per_feature)
        similar_indices = similar_indices.cpu().numpy().tolist()

        # Filter out already used features to avoid overlap
        available = [idx for idx in similar_indices if idx not in used_features][:samples_per_feature]

        # If we don't have enough available, just take what we can
        if len(available) < samples_per_feature:
            # Try to fill with unused valid features
            remaining = [idx for idx in valid_indices if idx not in used_features and idx not in available]
            available.extend(remaining[:samples_per_feature - len(available)])

        seed_groups.append(available)
        used_features.update(available)

    # Flatten to get all sampled indices
    all_sampled_indices = [idx for group in seed_groups for idx in group]

    return all_sampled_indices, seed_groups


def agglomerative_cluster_step(matrix, current_indices, group_size=8):
    """
    Perform one step of agglomerative clustering:
    - Partition current_indices into approximately len(current_indices)/group_size groups
    - Each group has exactly group_size members (or fewer for the last group)
    - Use similarity-based greedy assignment to form groups
    - From each group, randomly sample 1 representative

    Args:
        matrix: Full feature matrix (neurons x features)
        current_indices: Current set of feature indices to cluster
        group_size: Size of each group (default: 8)

    Returns:
        next_indices: Sampled representatives for next level
        clusters: List of cluster information
    """
    n_features = len(current_indices)

    # If we have fewer features than group_size, just return them all
    if n_features <= group_size:
        # Single group, sample one representative
        rep = random.choice(current_indices)
        cluster = {
            'members': current_indices,
            'mean_distance': 0.0,
            'size': len(current_indices)
        }
        return [rep], [cluster]

    # Greedy clustering: start with random seeds, assign nearest features to each seed
    n_groups = max(1, n_features // group_size)

    # Randomly select seed features for each group
    random.shuffle(current_indices)
    seeds = current_indices[:n_groups]
    remaining = set(current_indices[n_groups:])

    # Initialize groups with seeds
    groups = {seed: [seed] for seed in seeds}

    # Assign remaining features to nearest seed (by Hamming distance)
    for feat_idx in remaining:
        feat_vec = matrix[:, feat_idx]

        # Find closest seed
        min_dist = float('inf')
        closest_seed = seeds[0]

        for seed in seeds:
            # Only assign to groups that aren't full yet
            if len(groups[seed]) >= group_size:
                continue

            seed_vec = matrix[:, seed]
            dist = (feat_vec != seed_vec).float().mean().item()

            if dist < min_dist:
                min_dist = dist
                closest_seed = seed

        groups[closest_seed].append(feat_idx)

    # Build cluster information
    clusters = []
    next_indices = []

    for seed, members in groups.items():
        # Compute intra-group distance
        if len(members) > 1:
            submatrix = matrix[:, members]
            distances = []
            for j in range(len(members)):
                for k in range(j+1, len(members)):
                    feat_j = submatrix[:, j]
                    feat_k = submatrix[:, k]
                    hamming_dist = (feat_j != feat_k).float().mean().item()
                    distances.append(hamming_dist)
            mean_distance = np.mean(distances) if distances else 0.0
        else:
            mean_distance = 0.0

        clusters.append({
            'members': members,
            'mean_distance': mean_distance,
            'size': len(members),
            'seed': seed
        })

        # Sample one representative
        rep = random.choice(members)
        next_indices.append(rep)

    return next_indices, clusters


def recursive_agglomerative_clustering(matrix, initial_groups, group_size=8, max_levels=4):
    """
    Recursively perform agglomerative clustering until we reach a single cluster or max_levels.

    Expected progression (starting with 128 groups of 8):
    - Level 0: 1024 features (128 groups of 8) → 128 representatives
    - Level 1: 128 features (16 groups of 8) → 16 representatives
    - Level 2: 16 features (2 groups of 8) → 2 representatives
    - Level 3: 2 features (1 group of 2) → 1 representative

    Args:
        matrix: Feature matrix (neurons x features)
        initial_groups: Initial groups (list of lists of feature indices)
        group_size: Size of groups at each level
        max_levels: Maximum number of levels (default: 4)

    Returns:
        hierarchy: List of levels with cluster information
    """
    hierarchy = []
    level = 0

    # Level 0: Sample one representative from each initial group
    print(f"Level 0: {len(initial_groups)} groups of {group_size} features each")
    current_representatives = []
    level0_clusters = []

    for group in initial_groups:
        if len(group) <= 0:

        # Compute intra-group distance
        mean_distance = 0.0
        if len(group) > 1:
            submatrix = matrix[:, group]
            distances = []
            for j in range(len(group)):
                for k in range(j+1, len(group)):
                    feat_j = submatrix[:, j]
                    feat_k = submatrix[:, k]
                    hamming_dist = (feat_j != feat_k).float().mean().item()
                    distances.append(hamming_dist)
            mean_distance = np.mean(distances) if distances else 0.0

        # Random sample one representative
        rep = random.choice(group)
        current_representatives.append(rep)

        level0_clusters.append({
            'members': group,
            'mean_distance': mean_distance,
            'size': len(group),
            'representative': rep
        })

    print(f"  → {len(current_representatives)} representatives sampled")

    hierarchy.append({
        'level': 0,
        'n_features': sum(len(g) for g in initial_groups),
        'n_clusters': len(level0_clusters),
        'current_indices': [idx for group in initial_groups for idx in group],
        'next_indices': current_representatives.copy(),
        'clusters': level0_clusters
    })

    # Continue with subsequent levels
    level = 1
    current_indices = current_representatives

    while len(current_indices) > 1 and level < max_levels:
        # Perform one clustering step
        next_indices, clusters = agglomerative_cluster_step(matrix, current_indices, group_size)

        print(f"Level {level}: {len(current_indices)} features → {len(clusters)} groups → {len(next_indices)} representatives")

        hierarchy.append({
            'level': level,
            'n_features': len(current_indices),
            'n_clusters': len(clusters),
            'current_indices': current_indices.copy(),
            'next_indices': next_indices.copy(),
            'clusters': clusters
        })

        # Safety check: stop if we can't reduce further
        if len(next_indices) >= len(current_indices):
            print(f"Cannot reduce further at level {level}: {len(next_indices)} >= {len(current_indices)}")
            break

        # Move to next level
        current_indices = next_indices
        level += 1

    print(f"Clustering complete: {level} levels, final size: {len(current_indices)}")
    return hierarchy


def compute_distance_matrix(matrix, indices):
    """
    Compute pairwise Hamming distances between features.

    Args:
        matrix: Feature matrix (neurons x features)
        indices: Feature indices to compute distances for

    Returns:
        Distance matrix (n_indices x n_indices)
    """
    n = len(indices)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            # Hamming distance: proportion of differing elements
            feat_i = matrix[:, indices[i]]
            feat_j = matrix[:, indices[j]]
            hamming_dist = (feat_i != feat_j).float().mean().item()
            dist_matrix[i, j] = hamming_dist
            dist_matrix[j, i] = hamming_dist

    return dist_matrix


def build_linkage_from_hierarchy(hierarchy, initial_indices):
    """
    Build a linkage matrix from our custom hierarchical clustering.

    Scipy linkage format: [cluster_i, cluster_j, distance, n_samples]
    Each element can only appear once in a merge operation.

    Args:
        hierarchy: List of level data from recursive_agglomerative_clustering
        initial_indices: All initial feature indices (flat list of 1024 features)

    Returns:
        linkage_matrix: numpy array for scipy dendrogram [n-1, 4]
    """
    n_leaves = len(initial_indices)

    # Create mapping from feature index to leaf node ID
    feature_to_leaf = {feat_idx: i for i, feat_idx in enumerate(initial_indices)}

    # Track which leaf/cluster ID each feature currently belongs to
    # This gets updated as we merge clusters
    feature_to_current_cluster = feature_to_leaf.copy()

    linkage_rows = []
    next_cluster_id = n_leaves

    # Process each level in the hierarchy
    for level_idx, level_data in enumerate(hierarchy):
        clusters = level_data['clusters']
        distance_height = float(level_idx)  # Use level index as merge height

        # Track which clusters have been merged at this level
        merged_at_this_level = {}  # feature_idx -> new_cluster_id

        # For each cluster/group at this level
        for cluster_info in clusters:
            members = cluster_info['members']

            if len(members) < 2:
                # Single member - no merging needed, just track it
                if len(members) == 1:
                    feat_idx = members[0]
                    # Keep the same cluster ID for single-member groups
                    merged_at_this_level[feat_idx] = feature_to_current_cluster[feat_idx]
                continue

            # Get the current cluster IDs for all members in this group
            member_cluster_ids = []
            for feat_idx in members:
                if feat_idx in feature_to_current_cluster:
                    cid = feature_to_current_cluster[feat_idx]
                    member_cluster_ids.append(cid)

            if len(member_cluster_ids) < 2:
                continue

            # Remove duplicates (shouldn't happen, but be safe)
            member_cluster_ids = sorted(list(set(member_cluster_ids)))

            # Merge all member clusters into one by building a binary tree
            remaining = member_cluster_ids

            while len(remaining) > 1:
                new_remaining = []

                # Process pairs
                for i in range(0, len(remaining), 2):
                    if i + 1 < len(remaining):
                        # Merge a pair
                        left_id = remaining[i]
                        right_id = remaining[i + 1]

                        # Calculate sizes
                        left_size = 1 if left_id < n_leaves else int(linkage_rows[left_id - n_leaves][3])
                        right_size = 1 if right_id < n_leaves else int(linkage_rows[right_id - n_leaves][3])
                        total_size = left_size + right_size

                        # Add linkage row
                        linkage_rows.append([float(left_id), float(right_id), distance_height, float(total_size)])

                        # Create new cluster
                        new_remaining.append(next_cluster_id)
                        next_cluster_id += 1
                    else:
                        # Odd one out
                        new_remaining.append(remaining[i])

                remaining = new_remaining

            # The final cluster ID after all merges
            final_cluster_id = remaining[0]

            # Determine which feature is the representative
            if 'representative' in cluster_info:
                rep_feat = cluster_info['representative']
            elif 'seed' in cluster_info:
                rep_feat = cluster_info['seed']
            else:
                # Default to first member
                rep_feat = members[0]

            # Map the representative to the final cluster
            merged_at_this_level[rep_feat] = final_cluster_id

        # Update the feature_to_current_cluster mapping for the next level
        # Only update features that are representatives at this level
        for feat_idx, cluster_id in merged_at_this_level.items():
            feature_to_current_cluster[feat_idx] = cluster_id

    return np.array(linkage_rows) if linkage_rows else None


def visualize_dendrogram(hierarchy, initial_indices, matrix_name, save_path):
    """
    Create custom dendrogram visualization showing the hierarchical clustering levels.

    Args:
        hierarchy: Hierarchy from recursive_agglomerative_clustering
        initial_indices: All sampled feature indices
        matrix_name: Name for the plot
        save_path: Where to save the visualization
    """
    print(f"Creating custom dendrogram visualization...")

    fig, ax = plt.subplots(figsize=(20, 12))

    # Plot hierarchy levels as horizontal bars showing group structure
    n_levels = len(hierarchy)
    level_heights = list(range(n_levels))

    # Collect statistics for annotation
    stats_lines = []

    for level_idx, level_data in enumerate(hierarchy):
        clusters = level_data['clusters']
        n_features = level_data['n_features']
        n_clusters = level_data['n_clusters']
        mean_dist = sum(c['mean_distance'] for c in clusters) / len(clusters) if clusters else 0

        # Create visualization showing cluster groupings
        y_pos = level_idx
        cluster_sizes = [c['size'] for c in clusters]

        # Plot as stacked horizontal bars to show cluster structure
        x_offset = 0
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))

        for cluster_idx, (cluster, color) in enumerate(zip(clusters, colors)):
            width = cluster['size']
            ax.barh(y_pos, width, left=x_offset, height=0.6, color=color,
                   edgecolor='black', linewidth=0.5, alpha=0.7)
            x_offset += width

        # Add level annotation
        stats_line = f"Level {level_idx}: {n_features} features → {n_clusters} groups (mean dist: {mean_dist:.3f})"
        stats_lines.append(stats_line)
        ax.text(x_offset + 20, y_pos, f"{n_clusters} groups", fontsize=10, va='center')

    # Configure axes
    ax.set_yticks(level_heights)
    ax.set_yticklabels([f"Level {i}" for i in range(n_levels)])
    ax.set_xlabel('Number of Features', fontsize=14)
    ax.set_ylabel('Clustering Level', fontsize=14)
    ax.set_title(f'Hierarchical Agglomerative Clustering - {matrix_name}\n' +
                 f'{hierarchy[0]["n_features"]} initial features → {len(hierarchy)} levels → 1 final cluster',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Level 0 at top

    # Add statistics box
    stats_text = '\n'.join(stats_lines)
    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def visualize_feature_flow(hierarchy, initial_indices, matrix_name, save_path):
    """
    Create a Sankey-style flow diagram showing how features contribute to the next level.
    This tracks which features from each cluster become representatives at the next level.

    Args:
        hierarchy: Hierarchy from recursive_agglomerative_clustering
        initial_indices: All sampled feature indices
        matrix_name: Name for the plot
        save_path: Where to save the visualization
    """
    print(f"Creating feature flow visualization...")

    fig, ax = plt.subplots(figsize=(24, 14))

    n_levels = len(hierarchy)

    # Build feature-to-representative mapping across levels
    # This tells us: which features at level L became which representative at level L+1
    feature_lineage = {}  # feature_idx -> list of (level, cluster_idx, is_representative)

    # Track all features and their positions
    level_positions = {}  # (level, feature_idx) -> x_position

    # Process each level to build lineage
    for level_idx, level_data in enumerate(hierarchy):
        clusters = level_data['clusters']
        representatives = level_data.get('next_indices', [])

        # Assign x-positions to features at this level
        x_offset = 0
        for cluster_idx, cluster in enumerate(clusters):
            members = cluster['members']
            cluster_width = len(members)

            for local_idx, feat_idx in enumerate(members):
                x_pos = x_offset + local_idx
                level_positions[(level_idx, feat_idx)] = x_pos

                # Track lineage
                if feat_idx not in feature_lineage:
                    feature_lineage[feat_idx] = []

                # Determine if this feature is a representative
                is_rep = feat_idx in representatives
                feature_lineage[feat_idx].append({
                    'level': level_idx,
                    'cluster_idx': cluster_idx,
                    'is_representative': is_rep,
                    'x_pos': x_pos
                })

            x_offset += cluster_width

    # Draw flow lines connecting features to their representatives
    # For each level, draw lines from all members to their representative
    for level_idx in range(n_levels - 1):
        level_data = hierarchy[level_idx]
        next_level_data = hierarchy[level_idx + 1]

        clusters = level_data['clusters']
        representatives = level_data.get('next_indices', [])

        # For each cluster at this level
        for cluster_idx, cluster in enumerate(clusters):
            members = cluster['members']

            # Find which member became the representative
            rep_feature = None
            if 'representative' in cluster:
                rep_feature = cluster['representative']
            elif 'seed' in cluster:
                rep_feature = cluster['seed']

            if rep_feature is None or rep_feature not in representatives:
                continue

            # Get representative's position at next level
            rep_next_pos = level_positions.get((level_idx + 1, rep_feature))
            if rep_next_pos is None:
                continue

            # Draw lines from all members to the representative
            y_curr = n_levels - level_idx - 1
            y_next = n_levels - level_idx - 2

            # Use cluster color
            color = plt.cm.Set3(cluster_idx / max(1, len(clusters) - 1))

            for member in members:
                x_curr = level_positions.get((level_idx, member))
                if x_curr is None:
                    continue

                # Draw line
                is_rep = (member == rep_feature)
                alpha = 0.8 if is_rep else 0.2
                linewidth = 2 if is_rep else 0.5

                ax.plot([x_curr, rep_next_pos], [y_curr, y_next],
                       color=color, alpha=alpha, linewidth=linewidth, zorder=1)

    # Draw feature points at each level
    for level_idx, level_data in enumerate(hierarchy):
        y_pos = n_levels - level_idx - 1
        clusters = level_data['clusters']
        representatives = level_data.get('next_indices', [])

        for cluster_idx, cluster in enumerate(clusters):
            members = cluster['members']
            color = plt.cm.Set3(cluster_idx / max(1, len(clusters) - 1))

            for feat_idx in members:
                x_pos = level_positions.get((level_idx, feat_idx))
                if x_pos is None:
                    continue

                # Highlight representatives
                is_rep = feat_idx in representatives
                marker_size = 40 if is_rep else 10
                marker = 'o' if is_rep else '.'
                edge_color = 'black' if is_rep else 'none'
                edge_width = 1.5 if is_rep else 0

                ax.scatter(x_pos, y_pos, s=marker_size, c=[color],
                          marker=marker, alpha=0.8, edgecolors=edge_color,
                          linewidths=edge_width, zorder=2)

    # Configure axes
    ax.set_yticks(range(n_levels))
    ax.set_yticklabels([f"Level {n_levels - i - 1}" for i in range(n_levels)])
    ax.set_xlabel('Feature Position', fontsize=14)
    ax.set_ylabel('Clustering Level', fontsize=14)
    ax.set_title(f'Feature Flow Through Hierarchical Clustering - {matrix_name}\n' +
                 f'Large circles = selected representatives, lines show cluster membership',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='both')

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray',
               markersize=10, label='Representative feature'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='gray',
               markersize=5, label='Cluster member'),
        Line2D([0], [0], color='gray', linewidth=2, alpha=0.8,
               label='Representative lineage'),
        Line2D([0], [0], color='gray', linewidth=0.5, alpha=0.3,
               label='Member connection')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved feature flow visualization to {save_path}")


def calculate_expected_levels(n_seed_features, group_size):
    """
    Calculate the expected number of hierarchical levels.

    Args:
        n_seed_features: Number of seed features
        group_size: Size of each group

    Returns:
        Expected number of levels needed to reduce to 1 cluster
    """
    initial_features = n_seed_features * group_size
    return math.ceil(math.log(initial_features, group_size))


def process_matrix(matrix_path, n_sample=128, group_size=8, max_levels=None):
    """
    Process a single matrix: sample features, cluster, and visualize.

    Args:
        matrix_path: Path to matrix file
        n_sample: Number of seed features to sample
        group_size: Size of groups at each level
        max_levels: Maximum number of hierarchical levels (None = auto-calculate)
    """
    # Auto-calculate max_levels if not provided
    if max_levels is None:
        max_levels = calculate_expected_levels(n_sample, group_size)
    matrix = torch.load(matrix_path)
    matrix_name = Path(matrix_path).stem

    print(f"\n{'='*60}")
    print(f"Processing: {matrix_path}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"{'='*60}")

    # Step 1: Sample non-zero features and create initial groups
    all_indices, initial_groups = sample_nonzero_features(matrix, n_sample, group_size)
    print(f"Sampled {len(initial_groups)} groups with {len(all_indices)} total features")

    # Step 2-5: Recursive agglomerative clustering
    hierarchy = recursive_agglomerative_clustering(matrix, initial_groups, group_size, max_levels)

    # Print summary
    print("\n=== Clustering Hierarchy Summary ===")
    for level_data in hierarchy:
        n_next = len(level_data['next_indices'])
        n_clusters = level_data['n_clusters']
        mean_dist = sum(c['mean_distance'] for c in level_data['clusters']) / len(level_data['clusters']) if level_data['clusters'] else 0
        print(f"Level {level_data['level']}: {level_data['n_features']} features → {n_clusters} groups → {n_next} representatives")
        print(f"  Mean intra-group Hamming distance: {mean_dist:.4f}")

    # Visualize the hierarchy we just built
    dendrogram_path = output_dir / f"{matrix_name}_dendrogram.png"
    visualize_dendrogram(hierarchy, all_indices, matrix_name, dendrogram_path)

    # Create feature flow visualization
    flow_path = output_dir / f"{matrix_name}_feature_flow.png"
    visualize_feature_flow(hierarchy, all_indices, matrix_name, flow_path)

    return hierarchy

# ============================================================================

set_seed(RANDOM_SEED)
wanda_matrices, sparsegpt_matrices = get_unstructured_matrices()

# Create output directory for visualizations
output_dir = Path(__file__).parent.parent.parent / "results" / "visualizations" / "agglomerative_clustering"
output_dir.mkdir(parents=True, exist_ok=True)

# Process all Wanda matrices using configuration constants
print("\n" + "="*80)
print("Processing Wanda Matrices")
print("="*80)

for matrix_path in wanda_matrices:
    try:
        # max_levels is auto-calculated based on n_sample and group_size
        hierarchy = process_matrix(
            matrix_path,
            n_sample=N_SEED_FEATURES,
            group_size=GROUP_SIZE
        )
    except Exception as e:
        print(f"ERROR processing {matrix_path}: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*80)
print("All matrices processed successfully!")
print(f"Results saved to: {output_dir}")
print("="*80)
