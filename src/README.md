# Nearest Neighbor Feature Analysis

This package analyzes feature similarity in pruned neural network weight matrices using Hamming distance metrics.

## Structure

```
src/
├── __init__.py           # Package initialization and exports
├── main.py              # Main analysis pipeline script
├── data_loaders.py      # Matrix loading utilities
├── hamming_utils.py     # Hamming distance computations
├── matrix_analysis.py   # Sparsity and pattern metrics
├── visualization.py     # Visualization utilities
├── database.py          # DuckDB storage and queries
└── ingest.py           # Original script (deprecated - use main.py)
```

## Usage

### Basic Usage

Run the analysis on all matrices (both Wanda and SparseGPT):

```bash
cd /home/datadi/burns/aws/workloads
python -m src.main
```

### Command Line Options

```bash
python -m src.main --help

Options:
  --seed SEED             Random seed (default: 42)
  --n-samples N           Number of features to sample per matrix (default: 64)
  --n-neighbors K         Number of nearest neighbors to find (default: 8)
  --db-path PATH          Path to DuckDB database (default: ./sparsity.db)
  --save-viz              Save visualizations of feature subsets
  --viz-dir PATH          Directory for visualizations (default: ./results/visualizations/nearest_neighbor_analysis)
  --method {wanda,sparsegpt,both}  Which method to analyze (default: both)
```

### Examples

**Analyze only Wanda matrices with 128 samples:**
```bash
python -m src.main --method wanda --n-samples 128
```

**Analyze with visualizations saved:**
```bash
python -m src.main --save-viz --viz-dir ./my_visualizations
```

**Custom database path:**
```bash
python -m src.main --db-path ./my_analysis.db
```

## Module Documentation

### data_loaders.py
Functions for loading weight matrices:
- `set_seed(seed)`: Set random seed for reproducibility
- `get_wanda_matrices(files)`: Get paths to Wanda matrices
- `get_sparsegpt_matrices(files)`: Get paths to SparseGPT matrices
- `get_unstructured_matrices_layer_1()`: Get all layer 1 matrices
- `select_feature_columns(matrix, n)`: Randomly sample features

### ingest.py
Standalone script for quick analysis with CDF visualization:
- Computes pairwise Hamming distances for feature neighborhoods
- Generates cumulative distribution functions (CDFs)
- Saves binary matrix spy plots and CDF plots
- Output directory: `./results/hamming_cdf/`

### hamming_utils.py
Hamming distance computations:
- `compute_hamming_distance_batch(vec, matrix)`: Distances from one vector to all columns
- `find_most_similar_features(matrix, feature_idx, n_similar)`: Find k-nearest neighbors
- `compute_pairwise_hamming_distances_efficient(matrix)`: All pairwise distances
- `compute_hamming_distance_cdf(distance_matrix)`: Compute CDF of pairwise distances

### matrix_analysis.py
Matrix statistics and metrics:
- `count_inactive_rows(matrix)`: Count all-zero rows
- `count_fully_active_rows(matrix)`: Count all-one rows
- `count_identical_rows(matrix)`: Count duplicate rows
- `compute_subset_metrics(subset)`: Compute all metrics for a subset

### visualization.py
Visualization utilities:
- `viz_binary_matrix(matrix, title, save_path)`: Create spy plot
- `viz_hamming_distance_cdf(values, cdf, title, save_path)`: Plot CDF of Hamming distances

### database.py
DuckDB storage and queries:
- `get_db_connection(db_path)`: Connect to database
- `create_nearest_neighbor_table(conn)`: Create results table
- `save_batch_results(db_path, results)`: Save analysis results
- `query_results(db_path, method, projection)`: Query stored results

## Database Schema

The analysis results are stored in the `nearest_neighbor_metrics` table:

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| method | VARCHAR | Pruning method (wanda/sparsegpt) |
| matrix_name | VARCHAR | Matrix filename |
| layer | INTEGER | Layer number |
| projection | VARCHAR | Projection type (mlp.down_proj, etc.) |
| feature_idx | INTEGER | Reference feature index |
| n_neighbors | INTEGER | Number of neighbors analyzed |
| mean_pairwise_distance | DOUBLE | Mean Hamming distance |
| zeros | INTEGER | Number of all-zero rows |
| ones | INTEGER | Number of all-one rows |
| duplicates | INTEGER | Number of duplicate row types |
| total_dups | INTEGER | Total duplicate row instances |
| unique_rows | INTEGER | Number of unique rows |
| density | DOUBLE | Proportion of non-zero elements |
| timestamp | TIMESTAMP | Analysis timestamp |

## Programmatic Usage

You can also use the modules directly in your own scripts:

```python
from src import (
    set_seed,
    get_unstructured_matrices_layer_1,
    find_most_similar_features,
    compute_subset_metrics,
    save_batch_results
)

# Set seed
set_seed(42)

# Load matrices
wanda_matrices, sparsegpt_matrices = get_unstructured_matrices_layer_1()

# Analyze a matrix
import torch
matrix = torch.load(wanda_matrices[0], weights_only=True)
matrix = (matrix != 0).int()

# Find nearest neighbors
subset, indices, distances = find_most_similar_features(matrix, feature_idx=0, n_similar=8)

# Compute metrics
metrics = compute_subset_metrics(subset)
print(metrics)

# Save to database
results = [{
    'method': 'wanda',
    'matrix_name': 'layer1-mlp.down_proj.pt',
    'feature_idx': 0,
    'n_neighbors': 8,
    'mean_distance': distances.mean().item(),
    'metrics': metrics
}]
save_batch_results('./sparsity.db', results)
```

## Output

The analysis produces:
1. **Console output**: Real-time progress and metrics for each feature
2. **Database records**: All results stored in DuckDB for querying
3. **Visualizations** (optional): Spy plots of feature subsets

## Notes

- The original `ingest.py` script has been refactored into modular components
- All functions support both CPU and CUDA tensors
- Hamming distances are computed efficiently using matrix operations
- The database allows for easy aggregation and comparison of results across methods and layers
