import sys
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.custom import get_unstructured_matrices, set_seed
from utils.hamming_analysis import find_most_similar_features

set_seed(42)
wanda_matrices, sparsegpt_matrices = get_unstructured_matrices()

# Create output directory for visualizations
output_dir = Path(__file__).parent.parent.parent / "results" / "visualizations" / "agglomerative_clustering"
output_dir.mkdir(parents=True, exist_ok=True)

def recursive_agglomerative_clustering(matrix, sampled_indices, n_neighbors=8):
    """
    Recursively group features by finding n most similar neighbors,
    allowing overlapping clusters, then randomly sample from each group and repeat.

    Args:
        matrix: Full feature matrix (neurons x features)
        sampled_indices: List of feature indices to cluster (128 features)
        n_neighbors: Number of similar features to group together (default 8)

    Returns:
        List of hierarchies, one per level, until we reach a single cluster
    """
    hierarchy = []
    current_indices = sampled_indices.copy()
    level = 0

    while len(current_indices) > 1:
        print(f"Level {level}: {len(current_indices)} features")

        # Extract submatrix for current feature indices
        submatrix = matrix[:, current_indices]

        # Store clusters for this level (overlapping allowed)
        clusters = []

        # For each feature, find its n most similar neighbors
        for i in range(len(current_indices)):
            subset, local_indices, distances = find_most_similar_features(
                matrix=submatrix,
                feature_idx=i,
                n_similar=n_neighbors
            )

            # Skip if all distances are 0 (identical features)
            if distances.mean() == 0:
                continue

            # Map local indices back to global feature indices
            global_indices = [current_indices[idx] for idx in local_indices.tolist()]

            clusters.append({
                'query_feature': current_indices[i],
                'members': global_indices,
                'distances': distances.tolist(),
                'mean_distance': distances.mean().item()
            })

        if len(clusters) == 0:
            print(f"No valid clusters at level {level}, stopping")
            break

        hierarchy.append({
            'level': level,
            'n_features': len(current_indices),
            'n_clusters': len(clusters),
            'clusters': clusters
        })

        # Randomly sample one representative from each cluster for next level
        next_indices = []
        for cluster in clusters:
            representative = random.choice(cluster['members'])
            next_indices.append(representative)

        # Remove duplicates while preserving order
        seen = set()
        current_indices = []
        for idx in next_indices:
            if idx not in seen:
                current_indices.append(idx)
                seen.add(idx)

        level += 1

        # If we end up with the same number of features, we've converged
        if len(current_indices) >= len(hierarchy[-1]['clusters']):
            print(f"Convergence detected at level {level}, stopping")
            break

    print(f"Clustering complete: {level} levels, final cluster size: {len(current_indices)}")
    return hierarchy

def build_linkage_matrix(hierarchy, sampled_indices):
    """
    Build a linkage matrix compatible with scipy.cluster.hierarchy.dendrogram
    from the custom hierarchical clustering.

    Args:
        hierarchy: List of level dictionaries from recursive_agglomerative_clustering
        sampled_indices: Original sampled feature indices

    Returns:
        linkage_matrix: numpy array for dendrogram visualization
    """
    n_samples = len(sampled_indices)

    # Map global feature indices to local indices (0 to n_samples-1)
    global_to_local = {f: i for i, f in enumerate(sampled_indices)}

    # Track which nodes are still available for merging
    # Maps node_id -> current_cluster_id (starts with identity mapping)
    node_mapping = {i: i for i in range(n_samples)}

    linkage_rows = []
    next_cluster_id = n_samples

    for level_idx, level_data in enumerate(hierarchy):
        clusters = level_data['clusters']

        for cluster in clusters:
            members = cluster['members']
            mean_dist = cluster['mean_distance']

            # Convert global indices to local indices, then to current node IDs
            current_nodes = []
            for m in members:
                if m in global_to_local:
                    local_idx = global_to_local[m]
                    if local_idx in node_mapping:
                        current_nodes.append(node_mapping[local_idx])

            # Remove duplicates while preserving order
            seen = set()
            unique_nodes = []
            for n in current_nodes:
                if n not in seen:
                    unique_nodes.append(n)
                    seen.add(n)

            if len(unique_nodes) >= 2:
                # Merge the first two unique nodes
                left = min(unique_nodes[0], unique_nodes[1])
                right = max(unique_nodes[0], unique_nodes[1])

                # Add linkage row: [left_child, right_child, distance, num_leaves]
                # num_leaves is 2 for now (can be computed more accurately if needed)
                linkage_rows.append([left, right, mean_dist, 2])

                # Update node mapping: both merged nodes now point to the new cluster
                for local_idx, current_node in node_mapping.items():
                    if current_node == left or current_node == right:
                        node_mapping[local_idx] = next_cluster_id

                next_cluster_id += 1

    if len(linkage_rows) == 0:
        return None

    return np.array(linkage_rows)

def visualize_dendrogram(hierarchy, sampled_indices, matrix_name, save_path):
    """
    Create a dendrogram visualization of the agglomerative clustering hierarchy.

    Args:
        hierarchy: List of level dictionaries from recursive_agglomerative_clustering
        sampled_indices: Original sampled feature indices
        matrix_name: Name of the matrix being visualized
        save_path: Path to save the visualization
    """
    from scipy.cluster.hierarchy import dendrogram

    if len(hierarchy) == 0:
        print("No hierarchy to visualize")
        return

    fig, ax = plt.subplots(figsize=(20, 10))

    # Build linkage matrix
    linkage_matrix = build_linkage_matrix(hierarchy, sampled_indices)

    if linkage_matrix is None:
        print("Could not build linkage matrix")
        return

    # Create dendrogram
    dendrogram(
        linkage_matrix,
        ax=ax,
        leaf_font_size=6,
        color_threshold=None,
        above_threshold_color='gray'
    )

    ax.set_xlabel('Feature Index (from sampled features)', fontsize=14)
    ax.set_ylabel('Hamming Distance', fontsize=14)
    ax.set_title(f'Agglomerative Clustering Dendrogram - {matrix_name}\n'
                 f'{len(sampled_indices)} sampled features, {len(hierarchy)} levels',
                 fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved dendrogram to {save_path}")

def find_clusters(matrix, n_sample=128, n_neighbors=8):
    """
    Perform recursive agglomerative clustering on randomly sampled features.

    Args:
        matrix: Feature matrix (neurons x features)
        n_sample: Number of features to randomly sample (default 128)
        n_neighbors: Number of neighbors per cluster (default 8)

    Returns:
        tuple: (hierarchy, sampled_indices)
    """
    n_features = matrix.shape[1]

    # Randomly sample n_sample feature indices
    sampled_indices = random.sample(range(n_features), min(n_sample, n_features))
    print(f"Sampled {len(sampled_indices)} features from {n_features} total features")

    # Perform recursive agglomerative clustering
    hierarchy = recursive_agglomerative_clustering(matrix, sampled_indices, n_neighbors)

    # Print summary
    print("\n=== Clustering Hierarchy Summary ===")
    for level_data in hierarchy:
        print(f"Level {level_data['level']}: {level_data['n_features']} features → {level_data['n_clusters']} clusters")
        print(f"  Mean cluster distance: {sum(c['mean_distance'] for c in level_data['clusters']) / len(level_data['clusters']):.4f}")

    return hierarchy, sampled_indices

for matrix_path in wanda_matrices:
    matrix = torch.load(matrix_path)
    matrix_name = Path(matrix_path).stem

    print(f"\n{'='*60}")
    print(f"Processing: {matrix_path}")
    print(f"Matrix shape: {matrix.shape}")
    print(f"{'='*60}")

    hierarchy, sampled_indices = find_clusters(matrix, n_sample=128, n_neighbors=8)

    # Generate dendrogram visualization
    if hierarchy:
        dendrogram_path = output_dir / f"{matrix_name}_dendrogram.png"
        visualize_dendrogram(hierarchy, sampled_indices, matrix_name, dendrogram_path)
        print(f"Dendrogram saved to {dendrogram_path}")
