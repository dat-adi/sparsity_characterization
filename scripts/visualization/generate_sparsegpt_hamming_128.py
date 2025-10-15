import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.cluster import KMeans
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create output directory
output_dir = Path("../../results/visualizations/sparsegpt_hamming_128")
output_dir.mkdir(parents=True, exist_ok=True)

# Load SparseGPT unstructured matrices
matrices = {
    'down_proj': torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-mlp.down_proj.pt"),
    'v_proj': torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-self_attn.v_proj.pt"),
    'o_proj': torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-self_attn.o_proj.pt")
}

def get_most_similar_features(main_feature_idx: int, n_features: int, matrix):
    """Get the n most similar features to the main feature based on Hamming distance"""
    # Convert to binary mask
    binary_matrix = (matrix.abs() > 0).int()
    main_feature = binary_matrix[:, main_feature_idx]

    # Calculate Hamming distance to all other features
    n_total_features = matrix.shape[1]
    hamming_distances = []

    for i in range(n_total_features):
        if i != main_feature_idx:
            hamming_dist = (main_feature != binary_matrix[:, i]).sum().item()
            hamming_distances.append((i, hamming_dist))

    # Sort by Hamming distance (ascending - most similar first)
    hamming_distances.sort(key=lambda x: x[1])

    # Get the n most similar feature indices
    most_similar_indices = [idx for idx, _ in hamming_distances[:n_features]]

    # Return subset with main feature first, then most similar features
    subset = torch.cat([
        binary_matrix[:, main_feature_idx].unsqueeze(1),
        binary_matrix[:, most_similar_indices]
    ], dim=1)

    # Return distances for the selected features
    selected_distances = [dist for _, dist in hamming_distances[:n_features]]

    return subset, [main_feature_idx] + most_similar_indices, selected_distances

def create_clustered_spy_visualization(matrix, matrix_name, k, main_feature_idx=0):
    """Create spy visualization with k-means clustering on most similar features"""

    # Get the 127 most similar features (+ 1 main feature = 128 total)
    feature_subset, original_indices, hamming_distances = get_most_similar_features(
        main_feature_idx, 127, matrix
    )

    data = feature_subset.cpu().numpy().T  # Shape: (128, n_samples)

    # Run k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(data)

    # Sort features by cluster assignment for visualization
    cluster_sort_idx = np.argsort(labels)
    sorted_labels = labels[cluster_sort_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Reorder data by cluster
    reordered_data = data[cluster_sort_idx]

    # Create spy plot (binary visualization)
    ax.imshow(reordered_data, cmap='binary', aspect='auto', interpolation='nearest')

    # Add cluster boundaries
    cluster_boundaries = np.where(np.diff(sorted_labels))[0] + 1
    for boundary in cluster_boundaries:
        ax.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)

    # Calculate sparsity
    total_elements = reordered_data.size
    nonzero_elements = np.count_nonzero(reordered_data)
    sparsity = 1 - (nonzero_elements / total_elements)

    # Hamming distance statistics
    min_hamming = min(hamming_distances)
    max_hamming = max(hamming_distances)
    mean_hamming = np.mean(hamming_distances)

    ax.set_xlabel('Output Features (Samples)', fontsize=12)
    ax.set_ylabel('Input Features (sorted by cluster)', fontsize=12)
    ax.set_title(
        f'SparseGPT {matrix_name.upper()} - K-Means Clustering (k={k}, Layer 1)\n'
        f'128 Most Similar Features | Sparsity: {sparsity:.2%} | '
        f'Hamming: [{int(min_hamming)}, {int(max_hamming)}], mean={int(mean_hamming)}',
        fontsize=13, fontweight='bold'
    )

    ax.grid(False)
    plt.tight_layout()

    # Save the figure
    output_path = output_dir / f"sparsegpt_{matrix_name}_k{k}_spy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Calculate cluster statistics
    unique, counts = np.unique(labels, return_counts=True)

    print(f"Saved: {output_path}")
    print(f"  Matrix: {matrix_name}, k={k}")
    print(f"  Shape: {reordered_data.shape}")
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Hamming distance: min={int(min_hamming)}, max={int(max_hamming)}, mean={int(mean_hamming):.0f}")
    print(f"  K-means inertia: {kmeans.inertia_:.2f}")
    print(f"  K-means iterations: {kmeans.n_iter_}")
    print(f"  Cluster sizes: {dict(zip(unique, counts))}")
    print(f"  Mean cluster size: {np.mean(counts):.1f} ± {np.std(counts):.1f}")
    print()

# Generate visualizations for each matrix and k value
print("Generating SparseGPT K-Means Clustering Spy Visualizations")
print("(128 Most Similar Features)")
print("="*70)

k_values = [4, 8, 16]

for matrix_name, matrix in matrices.items():
    print(f"\n{matrix_name.upper()} (shape: {matrix.shape})")
    print("-" * 70)
    for k in k_values:
        create_clustered_spy_visualization(matrix, matrix_name, k)

print("="*70)
print(f"All visualizations saved to: {output_dir}")
print(f"\nFiles created:")
for matrix_name in matrices.keys():
    for k in k_values:
        print(f"  - sparsegpt_{matrix_name}_k{k}_spy.png")
