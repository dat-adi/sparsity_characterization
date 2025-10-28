"""
Test the new feature flow visualization on a single matrix.
"""

import sys
import torch
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from the agglomerative_clustering module
from agglomerative.agglomerative_clustering import (
    sample_nonzero_features,
    recursive_agglomerative_clustering,
    visualize_dendrogram,
    visualize_feature_flow
)
from utils.custom import set_seed

# Configuration
N_SEED_FEATURES = 128  # Smaller for faster testing
GROUP_SIZE = 8
RANDOM_SEED = 42

set_seed(RANDOM_SEED)

# Find a matrix to test with
data_dir = Path(__file__).parent.parent.parent / "data"
matrix_path = None

# Try to find a Wanda matrix
wanda_dir = data_dir / "clustering" / "wanda"
if wanda_dir.exists():
    pt_files = list(wanda_dir.glob("*.pt"))
    if pt_files:
        matrix_path = pt_files[0]

if matrix_path is None:
    print("Error: Could not find test matrix")
    sys.exit(1)

print(f"Testing with matrix: {matrix_path}")

# Load matrix
matrix = torch.load(matrix_path)
print(f"Matrix shape: {matrix.shape}")

# Output directory
output_dir = Path(__file__).parent.parent.parent / "results" / "visualizations" / "test_feature_flow"
output_dir.mkdir(parents=True, exist_ok=True)

# Sample features and create initial groups
print(f"\nSampling {N_SEED_FEATURES} seed features with group size {GROUP_SIZE}...")
all_indices, initial_groups = sample_nonzero_features(matrix, N_SEED_FEATURES, GROUP_SIZE)
print(f"Created {len(initial_groups)} groups with {len(all_indices)} total features")

# Perform hierarchical clustering
import math
max_levels = math.ceil(math.log(N_SEED_FEATURES * GROUP_SIZE, GROUP_SIZE))
print(f"\nPerforming hierarchical clustering with max {max_levels} levels...")
hierarchy = recursive_agglomerative_clustering(matrix, initial_groups, GROUP_SIZE, max_levels)

# Create visualizations
matrix_name = matrix_path.stem

print("\nCreating dendrogram visualization...")
dendrogram_path = output_dir / f"{matrix_name}_dendrogram_test.png"
visualize_dendrogram(hierarchy, all_indices, matrix_name, dendrogram_path)

print("\nCreating feature flow visualization...")
flow_path = output_dir / f"{matrix_name}_feature_flow_test.png"
visualize_feature_flow(hierarchy, all_indices, matrix_name, flow_path)

print(f"\n{'='*60}")
print("Test complete!")
print(f"Visualizations saved to: {output_dir}")
print(f"{'='*60}")
