import torch
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
import numpy as np

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Load SparseGPT down projection matrix
sparsegpt_down_proj = torch.load("../../data/clustering/sparsegpt/layer1-mlp.down_proj.pt")

def create_feature_subset(main_feature_idx: int, n_comparitive_features: int, matrix):
    """Create a subset from a weights matrix with randomly selected features"""
    n_features = matrix.shape[1]
    feature_range = list(range(0, n_features))
    random_feature_range = feature_range[:main_feature_idx] + feature_range[main_feature_idx+1:]
    random_feature_indices = random.sample(random_feature_range, min(n_comparitive_features, len(random_feature_range)))

    # In our setup, we always have the feature that we're comparing every other feature with
    # at the start of the subset.
    return torch.cat([
        matrix[:, main_feature_idx].unsqueeze(1),
        matrix[:, random_feature_indices]
    ], dim=1)

def get_coactivation_gradient(matrix):
    """Arranges the features for the main feature (located at idx 0) from most relevant to least relevant

    Utilizes the hamming distance to make this measurement.
    """
    hamming_distances = torch.tensor([(matrix[:, 0] != matrix[:, i]).sum() for i in range(1, matrix.shape[1])])
    sorted_indices = torch.argsort(hamming_distances, descending=False)

    # adding in the 0th index to pad the coactivation gradient with the main feature
    sorted_indices = torch.cat([torch.tensor([0]), sorted_indices + 1])
    return matrix[:, sorted_indices], sorted_indices

# Create feature subset (select feature 0 and 127 random comparative features)
feature_subset = create_feature_subset(0, 127, sparsegpt_down_proj)
feature_subset = (feature_subset.abs() > 0).int()

# Get coactivation gradient
coactivation_gradient, sorted_indices = get_coactivation_gradient(feature_subset)
data = coactivation_gradient.cpu()

# Transpose data so features are data points (for clustering)
features_as_points = data.T.numpy()  # Shape: (128, n_samples)

# Run k-means with k=16
kmeans = KMeans(n_clusters=16, random_state=42)
labels = kmeans.fit_predict(features_as_points)

# Sort features by cluster assignment for visualization
cluster_sort_idx = np.argsort(labels)
sorted_labels = labels[cluster_sort_idx]

# Create spy visualization
fig, ax = plt.subplots(figsize=(16, 10))

# Reorder data by cluster
reordered_data = data.numpy()[:, cluster_sort_idx].T

# Create spy plot (binary visualization)
ax.spy(reordered_data, markersize=0.5, aspect='auto')

# Add cluster boundaries
cluster_boundaries = np.where(np.diff(sorted_labels))[0] + 1
for boundary in cluster_boundaries:
    ax.axhline(y=boundary, color='red', linewidth=2, alpha=0.7)

ax.set_xlabel('Samples', fontsize=12)
ax.set_ylabel('Features (sorted by cluster)', fontsize=12)
ax.set_title(f'SparseGPT Down Projection - Spy Visualization (k=16)', fontsize=14, fontweight='bold')

plt.tight_layout()

# Save the figure
output_path = '../../results/visualizations/sparsegpt_k16_down_proj_spy.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Saved spy visualization to: {output_path}")

# Also display cluster statistics
unique, counts = np.unique(labels, return_counts=True)
print(f"\nClustering Statistics (k=16):")
print(f"{'='*50}")
print(f"Total inertia: {kmeans.inertia_:.2f}")
print(f"Iterations: {kmeans.n_iter_}")
print(f"Cluster sizes: {dict(zip(unique, counts))}")
print(f"Mean cluster size: {np.mean(counts):.1f} ± {np.std(counts):.1f}")

plt.show()
