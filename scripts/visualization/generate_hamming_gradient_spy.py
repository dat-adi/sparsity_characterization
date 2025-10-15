import torch
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create output directory
output_dir = Path("../../results/visualizations/sparsegpt_hamming_gradient_spy")
output_dir.mkdir(parents=True, exist_ok=True)

# Load SparseGPT unstructured matrices
matrices = {
    'down_proj': torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-mlp.down_proj.pt"),
    'v_proj': torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-self_attn.v_proj.pt"),
    'o_proj': torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-self_attn.o_proj.pt")
}

def create_feature_subset(main_feature_idx: int, n_comparative_features: int, matrix):
    """Create a subset from a weights matrix with randomly selected features"""
    n_features = matrix.shape[1]
    feature_range = list(range(0, n_features))
    random_feature_range = feature_range[:main_feature_idx] + feature_range[main_feature_idx+1:]
    random_feature_indices = random.sample(random_feature_range, min(n_comparative_features, len(random_feature_range)))

    # In our setup, we always have the feature that we're comparing every other feature with
    # at the start of the subset.
    return torch.cat([
        matrix[:, main_feature_idx].unsqueeze(1),
        matrix[:, random_feature_indices]
    ], dim=1), [main_feature_idx] + random_feature_indices

def get_coactivation_gradient(matrix):
    """Arranges the features for the main feature (located at idx 0) from most relevant to least relevant

    Utilizes the hamming distance to make this measurement.
    Returns the reordered matrix and the sorted indices.
    """
    hamming_distances = torch.tensor([(matrix[:, 0] != matrix[:, i]).sum().item() for i in range(1, matrix.shape[1])])
    sorted_indices = torch.argsort(hamming_distances, descending=False)

    # adding in the 0th index to pad the coactivation gradient with the main feature
    sorted_indices = torch.cat([torch.tensor([0]), sorted_indices + 1])
    return matrix[:, sorted_indices], sorted_indices, hamming_distances

def create_spy_visualization(matrix, matrix_name, main_feature_idx=0, n_features=511):
    """Create a spy visualization with Hamming distance gradient ordering"""

    # Create feature subset
    feature_subset, original_indices = create_feature_subset(main_feature_idx, n_features, matrix)

    # Convert to binary mask
    feature_subset_binary = (feature_subset.abs() > 0).int()

    # Get coactivation gradient (Hamming distance ordering)
    coactivation_gradient, sorted_indices, hamming_distances = get_coactivation_gradient(feature_subset_binary)
    data = coactivation_gradient.cpu().numpy().T

    # Calculate sparsity
    total_elements = data.size
    nonzero_elements = np.count_nonzero(data)
    sparsity = 1 - (nonzero_elements / total_elements)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 10))

    # Create spy plot with proper black/white coloring
    ax.imshow(data, cmap='binary', aspect='auto', interpolation='nearest')

    ax.set_xlabel('Output Features (Samples)', fontsize=12)
    ax.set_ylabel('Input Features (sorted by Hamming distance)', fontsize=12)

    # Add information about the gradient
    hamming_sorted = hamming_distances[sorted_indices[1:] - 1]  # Skip the main feature at idx 0
    min_hamming = hamming_sorted.min().item() if len(hamming_sorted) > 0 else 0
    max_hamming = hamming_sorted.max().item() if len(hamming_sorted) > 0 else 0

    ax.set_title(
        f'SparseGPT {matrix_name.upper()} - Hamming Distance Gradient (Layer 1)\n'
        f'Shape: {data.shape}, Sparsity: {sparsity:.2%}, '
        f'Hamming range: [{int(min_hamming)}, {int(max_hamming)}]',
        fontsize=14, fontweight='bold'
    )

    # Add colorbar to show gradient direction
    # Add text annotation for gradient direction
    ax.text(0.02, 0.98, '← Lowest Hamming Distance',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(0.02, 0.02, '← Highest Hamming Distance',
            transform=ax.transAxes, fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))

    ax.grid(False)

    plt.tight_layout()

    # Save the figure
    output_path = output_dir / f"sparsegpt_{matrix_name}_hamming_gradient_spy.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  Original matrix shape: {matrix.shape}")
    print(f"  Subset shape: {data.shape}")
    print(f"  Main feature index: {main_feature_idx}")
    print(f"  Sparsity: {sparsity:.2%}")
    print(f"  Hamming distance range: [{int(min_hamming)}, {int(max_hamming)}]")
    print()

# Generate visualizations for each matrix
print("Generating SparseGPT Hamming Distance Gradient Spy Visualizations...")
print("="*70)

for matrix_name, matrix in matrices.items():
    create_spy_visualization(matrix, matrix_name)

print("="*70)
print(f"All visualizations saved to: {output_dir}")
print(f"\nFiles created:")
print(f"  1. sparsegpt_down_proj_hamming_gradient_spy.png")
print(f"  2. sparsegpt_v_proj_hamming_gradient_spy.png")
print(f"  3. sparsegpt_o_proj_hamming_gradient_spy.png")
