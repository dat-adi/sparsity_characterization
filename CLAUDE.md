# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a neural network sparsity analysis research codebase that compares different pruning methods (Wanda and SparseGPT) on language models. The project focuses on analyzing activation patterns, feature co-activation, and sparsity structures in transformer model layers.

## Development Setup

### Environment Management
- Python 3.10+ managed via `uv` package manager
- Dependencies are defined in `pyproject.toml`
- Install dependencies: `uv sync`
- Virtual environment is in `.venv/`

### Database
- DuckDB database at `./sparsity.db` stores sparsity analysis results
- Contains tables with naming pattern like `wanda_0_5_down_proj` for different sparsity levels and projection types

## Key Commands

### Running Analysis Scripts
```bash
# Visualize sparsity patterns from .pt files
cd scripts/visualization && python visualize_sparsity.py

# Apply thresholding to create sparsified versions
cd scripts/visualization && python draw_visualization.py

# Run Marimo notebook for interactive sparsity evaluation
marimo edit notebooks/sparsity_evaluation.py

# Run correlation analysis
marimo edit scripts/clustering/correlation_analysis.py

# Run clustering metrics visualization
cd scripts/clustering && python visualize_clustering_metrics.py
```

### Triton Kernel Development
```bash
# Run sparse GEMV kernel implementation
cd scripts/kernels && python kernelize.py
```

## Architecture Overview

### Directory Structure

```
/
├── data/                          # Raw data and model outputs
│   ├── wanda_unstructured/        # Wanda pruning results organized by layer
│   ├── sparsegpt_unstructured/    # SparseGPT pruning results organized by layer
│   ├── sparsegpt-orig-opt-125m/   # Original SparseGPT model outputs
│   ├── ablations/                 # Ablation study data
│   │   ├── sparsity_layer_1_unstructured_{wanda,sparsegpt}/
│   │   └── structure_layer_1_{wanda,sparsegpt}/
│   └── clustering/                # Clustering analysis data
│       ├── wanda/                 # .pt files for Wanda clustering analysis
│       └── sparsegpt/             # .pt files for SparseGPT clustering analysis
│
├── results/                       # Analysis outputs
│   ├── visualizations/            # Generated plots and figures
│   └── metrics/                   # Computed metrics (JSON files)
│
├── scripts/                       # Analysis scripts
│   ├── visualization/             # Sparsity visualization tools
│   │   ├── visualize_sparsity.py
│   │   ├── draw_visualization.py
│   │   ├── generate_unstructured_spy_visualizations.py
│   │   ├── generate_hamming_gradient_spy.py
│   │   ├── generate_sparsegpt_hamming_128.py
│   │   └── generate_sparsegpt_k16_spy.py
│   ├── clustering/                # Feature clustering analysis
│   │   ├── feature_clustering_analysis.py
│   │   ├── correlation_analysis.py
│   │   └── visualize_clustering_metrics.py
│   ├── metrics/                   # Pruning method comparison
│   │   ├── metrics_bw_unstructured_wanda_and_sparsegpt.py
│   │   ├── metrics_bw_structured_wanda_ablations.py
│   │   └── metrics_bw_structured_sparsegpt_ablations.py
│   ├── analysis/                  # Advanced analysis tools
│   │   ├── hamming_cluster_analysis.py
│   │   ├── multi_execution_hamming_analysis.py
│   │   └── generate_comparative_report.py
│   └── kernels/                   # Sparse inference kernels
│       └── kernelize.py
│
├── notebooks/                     # Interactive Marimo notebooks
│   └── sparsity_evaluation.py
│
└── sparsity.db                    # DuckDB database with analysis results
```

### Data Format

**.pt Files**: PyTorch tensors representing weight matrices with sparsity patterns
- Naming convention: `layer{N}-{component}-{indices}-{sparsity}.pt`
- Components: `mlp.down_proj`, `mlp.up_proj`, `mlp.gate_proj`, `self_attn.{q,k,v,o}_proj`
- Example: `layer0-mlp.down_proj-0-0-0.5.pt` (layer 0, down projection, 50% sparsity)

### Key Algorithms

**Sparsity Metrics** (`scripts/metrics/metrics_bw_unstructured_wanda_and_sparsegpt.py`):
- `jaccard_similarity()`: Measures overlap of non-zero positions between two sparse matrices
- `cosine_similarity()`: Measures directional similarity of weight values
- `hamming_distance()`: Measures proportion of differing sparsity patterns
- `compute_metrics_by_feature()`: Computes metrics row-wise (down_proj) or column-wise (other projections)

**Clustering Analysis** (`scripts/clustering/correlation_analysis.py`):
- `create_feature_subset()`: Samples features for correlation analysis
- `get_coactivation_gradient()`: Orders features by Hamming distance to identify co-activation patterns
- `visualize_clusters()`: K-means clustering with t-SNE visualization and block coherence analysis for MMM (Multiply-Mask-Multiply) optimization potential

**Feature Clustering** (`scripts/clustering/feature_clustering_analysis.py`):
- `find_most_similar_features()`: Finds n most similar features by Hamming distance
- `compute_cluster_metrics()`: Performs k-means clustering and computes within/between cluster distances
- `analyze_matrix()`: Full analysis pipeline for a single weight matrix

**Triton Kernel** (`scripts/kernels/kernelize.py`):
- `splitk_sparse_gemv_kernel()`: Autotuned Triton kernel for sparse GEMV with dynamic activation masking
- Uses split-K parallelization across both M and N dimensions
- Threshold-based sparsification at runtime (masks activations below threshold before loading weights)

### Sparsity Thresholds

Hard-coded thresholds for 90% sparsity (from `scripts/visualization/draw_visualization.py`):
```python
{
    'q': 0.0182, 'k': 0.0182, 'v': 0.0182, 'o': 0.0035,
    'up': 0.0601, 'gate': 0.0601, 'down': 0.0032
}
```

### Important Path Notes

- Scripts use relative paths from their location (e.g., `../../data/...` from `scripts/*/`)
- Data files are stored in `data/` subdirectories
- All outputs go to `results/visualizations/` or `results/metrics/`
- Database file `sparsity.db` remains at the project root
- When running scripts, either `cd` to the script's directory first, or use paths relative to the project root

## Common Workflows

### Analyzing a New Pruning Method
1. Generate .pt files for each layer and component at desired sparsity levels
2. Organize into `data/{method}_unstructured/layer-{N}/` directories
3. Run `cd scripts/visualization && python visualize_sparsity.py` to generate sparsity plots
4. Add file paths to scripts in `scripts/metrics/`
5. Run metrics scripts to compare against existing methods

### Interactive Exploration
1. Use Marimo notebooks: `marimo edit <notebook>.py`
2. `notebooks/sparsity_evaluation.py`: Query aggregated results from DuckDB
3. `scripts/clustering/correlation_analysis.py`: Investigate feature co-activation and clustering properties
4. `scripts/clustering/visualize_clustering_metrics.py`: Generate comparative visualizations

### Performance Optimization
1. Modify `scripts/kernels/kernelize.py` Triton configs for new matrix dimensions or sparsity levels
2. Adjust `BLOCK_M`, `BLOCK_N`, `num_warps` in `configs` list
3. Use `SPARSITY_BIN` key to cache compiled kernels per sparsity level

### Adding New Scripts
When adding new analysis scripts:
1. Place them in the appropriate `scripts/` subdirectory
2. Use relative paths (e.g., `../../data/...`, `../../results/...`)
3. Save outputs to `results/visualizations/` or `results/metrics/`
4. Update README.md with usage instructions
5. Consider whether the script should be run from its directory or the project root
