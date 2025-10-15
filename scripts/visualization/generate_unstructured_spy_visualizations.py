import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Create output directory
output_dir = Path("../../results/visualizations/sparsegpt_unstructured_spy")
output_dir.mkdir(parents=True, exist_ok=True)

# Load SparseGPT unstructured matrices
down_proj = torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-mlp.down_proj.pt")
o_proj = torch.load("../../data/sparsegpt_unstructured/layer-1/layer1-self_attn.o_proj.pt")

# Also load and save the k=16 clustering visualization
import shutil
k16_source = "../../results/visualizations/sparsegpt_k16_down_proj_spy.png"
k16_dest = output_dir / "sparsegpt_k16_clustering_down_proj_spy.png"
if Path(k16_source).exists():
    shutil.copy(k16_source, k16_dest)
    print(f"Copied k=16 clustering visualization to: {k16_dest}")

def create_spy_visualization(matrix, title, output_path):
    """Create a proper spy visualization with binary sparsity pattern"""
    # Convert to binary sparsity mask (1 = non-zero, 0 = zero)
    binary_mask = (matrix.abs() > 0).cpu().numpy()

    # Calculate sparsity
    total_elements = binary_mask.size
    nonzero_elements = np.count_nonzero(binary_mask)
    sparsity = 1 - (nonzero_elements / total_elements)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create spy plot with proper black/white coloring
    # matplotlib spy uses marker='s' for squares, but we want a cleaner look
    ax.imshow(binary_mask, cmap='binary', aspect='auto', interpolation='nearest')

    ax.set_xlabel('Input Features', fontsize=12)
    ax.set_ylabel('Output Features', fontsize=12)
    ax.set_title(f'{title}\nShape: {matrix.shape}, Sparsity: {sparsity:.2%}',
                 fontsize=14, fontweight='bold')

    # Add grid for better visualization
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_path}")
    print(f"  Shape: {matrix.shape}")
    print(f"  Non-zero elements: {nonzero_elements:,} / {total_elements:,}")
    print(f"  Sparsity: {sparsity:.2%}\n")

    return sparsity

# Generate visualizations
print("Generating SparseGPT unstructured spy visualizations...")
print("="*60)

down_proj_sparsity = create_spy_visualization(
    down_proj,
    "SparseGPT Unstructured - Down Projection (Layer 1)",
    output_dir / "sparsegpt_unstructured_down_proj_spy.png"
)

o_proj_sparsity = create_spy_visualization(
    o_proj,
    "SparseGPT Unstructured - O Projection (Layer 1)",
    output_dir / "sparsegpt_unstructured_o_proj_spy.png"
)

print("="*60)
print(f"All visualizations saved to: {output_dir}")
print(f"\nFiles created:")
print(f"  1. sparsegpt_k16_clustering_down_proj_spy.png (k-means clustering)")
print(f"  2. sparsegpt_unstructured_down_proj_spy.png (full matrix)")
print(f"  3. sparsegpt_unstructured_o_proj_spy.png (full matrix)")
