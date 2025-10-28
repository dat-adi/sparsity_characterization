# Agglomerative Clustering with Random Sampling

This directory contains the implementation of hierarchical agglomerative clustering for neural network weight matrices.

## Overview

The script performs hierarchical agglomerative clustering using a custom algorithm that creates non-overlapping groups at each level and visualizes the results.

## Configuration

**Easy to customize!** Simply modify these constants at the top of `agglomerative_clustering.py`:

```python
N_SEED_FEATURES = 512  # Number of seed features to sample
GROUP_SIZE = 8         # Size of groups at each level
RANDOM_SEED = 42       # Random seed for reproducibility
```

The script automatically calculates:
- **Initial features**: `N_SEED_FEATURES × GROUP_SIZE` (e.g., 512 × 8 = 4096)
- **Number of levels**: `ceil(log_GROUP_SIZE(initial_features))` (e.g., log₈(4096) = 4 levels)

### Example Configurations

| N_SEED_FEATURES | GROUP_SIZE | Initial Features | Levels | Progression |
|-----------------|------------|------------------|--------|-------------|
| 128 | 8 | 1,024 | 4 | 1024→128→16→2→1 |
| 512 | 8 | 4,096 | 4 | 4096→512→64→8→1 |
| 256 | 16 | 4,096 | 3 | 4096→256→16→1 |
| 1024 | 8 | 8,192 | 5 | 8192→1024→128→16→2→1 |

## Algorithm

1. **Sample N seed features**: Randomly select N non-zero feature vectors (columns) from the matrix
2. **Create initial groups**: For each seed, find G most similar features (by Hamming distance) → N groups of G features
3. **Random sampling within groups**: From each group of G, randomly sample 1 representative
4. **Recursive clustering**: Repeat the grouping and sampling process until reaching 1 cluster
5. **Visualize**: Create visualization showing the hierarchical clustering structure at each level

## Key Features

- **Fully configurable**: Change N_SEED_FEATURES and GROUP_SIZE to adjust clustering granularity
- **Auto-calculated levels**: Number of hierarchical levels is automatically determined
- **Non-overlapping groups**: At each level, features are partitioned into non-overlapping groups
- **Hamming distance**: Uses Hamming distance for similarity calculations between binary feature vectors
- **Greedy assignment**: After level 0, uses greedy seed-based assignment to form balanced groups
- **Random sampling**: Samples one representative per group to reduce dimensions at each level

## Usage

```bash
# From the project root
uv run python scripts/agglomerative/agglomerative_clustering.py
```

The script will:
1. Process all Wanda matrices from `data/wanda_unstructured/layer-1/`
2. Generate dendrograms for each matrix
3. Save visualizations to `results/visualizations/agglomerative_clustering/`

## Output

For each matrix, the script generates:
- **Visualization PNG**: Custom visualization showing the hierarchical clustering structure
  - Horizontal stacked bars representing groups at each level
  - Color-coded groups with size information
  - Level-by-level statistics (features, groups, mean Hamming distance)
  - Located at: `results/visualizations/agglomerative_clustering/{matrix_name}_dendrogram.png`

## Implementation Details

### Key Functions

- `sample_nonzero_features()`: Samples seed features and creates initial groups using Hamming similarity
- `agglomerative_cluster_step()`: Partitions features into groups and samples representatives
- `recursive_agglomerative_clustering()`: Recursively applies clustering across multiple levels
- `build_linkage_from_hierarchy()`: Converts custom hierarchy to scipy linkage matrix format
- `visualize_dendrogram()`: Creates dendrogram visualization with level annotations

### Dependencies

- `torch`: For loading weight matrices
- `numpy`: For numerical computations
- `matplotlib`: For visualization
- `scipy`: For dendrogram creation
- `utils.hamming_analysis`: For Hamming distance calculations and feature similarity
- `utils.custom`: For loading matrices and setting random seeds

## Example Output

With `N_SEED_FEATURES = 512` and `GROUP_SIZE = 8`:

```
================================================================================
CONFIGURATION
================================================================================
Seed features: 512
Group size: 8
Initial features (Level 0): 4096 (512 × 8)
Maximum levels: 4
Random seed: 42
================================================================================

Level 0: 512 groups of 8 features each
  → 512 representatives sampled
Level 1: 512 features → 64 groups → 64 representatives
Level 2: 64 features → 8 groups → 8 representatives
Level 3: 8 features → 1 groups → 1 representatives
Clustering complete: 4 levels, final size: 1

=== Clustering Hierarchy Summary ===
Level 0: 4096 features → 512 groups → 512 representatives
  Mean intra-group Hamming distance: 0.7422
Level 1: 512 features → 64 groups → 64 representatives
  Mean intra-group Hamming distance: 0.7448
Level 2: 64 features → 8 groups → 8 representatives
  Mean intra-group Hamming distance: 0.7291
Level 3: 8 features → 1 groups → 1 representatives
  Mean intra-group Hamming distance: 0.0000
```

## Notes

- **Number of levels automatically adapts** to the configuration parameters
- Non-overlapping groups are enforced at level 0 by tracking used features
- Subsequent levels use greedy seed-based assignment for balanced partitioning
- Hamming distance is used for all similarity calculations (appropriate for binary/sparse matrices)
- Simply change `N_SEED_FEATURES` and `GROUP_SIZE` at the top of the script to reconfigure
