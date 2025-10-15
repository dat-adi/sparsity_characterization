# Neural Network Sparsity Analysis

A research codebase for analyzing and comparing different pruning methods (Wanda and SparseGPT) on language models. This project focuses on analyzing activation patterns, feature co-activation, and sparsity structures in transformer model layers.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
  - [Visualization Scripts](#visualization-scripts)
  - [Clustering Analysis](#clustering-analysis)
  - [Metrics Comparison](#metrics-comparison)
  - [Interactive Notebooks](#interactive-notebooks)
  - [Triton Kernels](#triton-kernels)
- [Data Format](#data-format)
- [Key Algorithms](#key-algorithms)
- [Configuration](#configuration)

## Overview

This project investigates how different neural network pruning techniques affect sparsity patterns and feature interactions in transformer models. The primary methods compared are:

- **Wanda**: Unstructured pruning based on weight magnitude and activation
- **SparseGPT**: Structured pruning with careful layer-wise optimization

### Research Questions

1. How do different pruning methods affect feature co-activation patterns?
2. Can we identify clusters of features that activate together?
3. What is the overlap between sparsity patterns from different methods?
4. How can we optimize sparse inference using these patterns?

## Project Structure

```
/
├── data/                          # Raw data and model outputs
│   ├── wanda_unstructured/        # Wanda pruning results
│   ├── sparsegpt_unstructured/    # SparseGPT pruning results
│   ├── sparsegpt-orig-opt-125m/   # Original SparseGPT model outputs
│   ├── ablations/                 # Ablation study data
│   │   ├── sparsity_layer_1_unstructured_wanda/
│   │   ├── sparsity_layer_1_unstructured_sparsegpt/
│   │   ├── structure_layer_1_wanda/
│   │   └── structure_layer_1_sparsegpt/
│   └── clustering/                # Clustering analysis data
│       ├── wanda/
│       └── sparsegpt/
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
├── sparsity.db                    # DuckDB database with analysis results
├── pyproject.toml                 # Python project configuration
├── uv.lock                        # Dependency lock file
├── CLAUDE.md                      # Instructions for Claude Code
└── REPORT.md                      # Analysis report

```

## Setup

### Requirements

- Python 3.10+
- CUDA-capable GPU (for Triton kernels)

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate
```

### Dependencies

Core libraries:
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- DuckDB
- Marimo (for notebooks)
- Triton (for GPU kernels)

## Usage

### Visualization Scripts

#### Generate Sparsity Plots

```bash
# Visualize sparsity patterns from .pt files
cd scripts/visualization
python visualize_sparsity.py
```

This script:
- Reads `.pt` (PyTorch tensor) files from the current directory
- Generates spy plots showing sparsity patterns
- Saves visualizations as PNG files

#### Create Spy Visualizations with Clustering

```bash
cd scripts/visualization

# Generate k-means clustered visualizations (k=4, 8, 16)
python generate_sparsegpt_hamming_128.py

# Generate Hamming distance gradient visualizations
python generate_hamming_gradient_spy.py

# Generate k=16 clustering visualization
python generate_sparsegpt_k16_spy.py
```

### Clustering Analysis

#### Feature Clustering

```bash
cd scripts/clustering
python feature_clustering_analysis.py
```

This analyzes feature co-activation patterns by:
1. Randomly selecting a feature vector
2. Finding the 128 most similar features by Hamming distance
3. Applying k-means clustering (k=4, 8, 16)
4. Computing within/between cluster distances

#### Interactive Correlation Analysis

```bash
cd scripts/clustering
marimo edit correlation_analysis.py
```

This Marimo notebook provides:
- Interactive clustering visualization
- t-SNE projections of feature embeddings
- Block coherence analysis for MMM (Multiply-Mask-Multiply) optimization

#### Clustering Metrics Visualization

```bash
cd scripts/clustering
python visualize_clustering_metrics.py
```

Generates comparative visualizations:
- Separation ratio plots (between/within cluster distances)
- Cluster size distributions
- Hamming distance distributions
- Heatmaps comparing Wanda vs SparseGPT

### Metrics Comparison

#### Unstructured Pruning Comparison

```bash
cd scripts/metrics
python metrics_bw_unstructured_wanda_and_sparsegpt.py
```

Computes feature-wise metrics:
- **Jaccard similarity**: Overlap of non-zero positions
- **Cosine similarity**: Directional similarity of weight values
- **Hamming distance**: Proportion of differing sparsity patterns

#### Structured Pruning Ablations

```bash
cd scripts/metrics
python metrics_bw_structured_wanda_ablations.py
python metrics_bw_structured_sparsegpt_ablations.py
```

Compares different structured sparsity configurations (2:4, 4:8, etc.)

### Interactive Notebooks

#### Sparsity Evaluation

```bash
marimo edit notebooks/sparsity_evaluation.py
```

Provides SQL-based exploration of the DuckDB database with:
- Interactive queries on sparsity statistics
- Histogram visualizations
- Feature relevance analysis

### Triton Kernels

#### Sparse GEMV Kernel

```bash
cd scripts/kernels
python kernelize.py
```

Runs a sparse matrix-vector multiplication kernel optimized for:
- Dynamic activation masking (threshold-based sparsification)
- Split-K parallelization across M and N dimensions
- Auto-tuning for different matrix sizes and sparsity levels

## Data Format

### PyTorch Tensor Files (.pt)

Weight matrices stored as PyTorch tensors with sparsity patterns:

**Naming Convention:**
```
layer{N}-{component}-{indices}-{sparsity}.pt
```

**Examples:**
- `layer0-mlp.down_proj-0-0-0.5.pt`: Layer 0, down projection, 50% sparsity
- `layer1-self_attn.q_proj-0-0-0.5.pt`: Layer 1, query projection, 50% sparsity

**Components:**
- MLP: `mlp.down_proj`, `mlp.up_proj`, `mlp.gate_proj`
- Attention: `self_attn.q_proj`, `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.o_proj`

### DuckDB Database

The `sparsity.db` file contains aggregated analysis results in tables like:
- `wanda_0_5_down_proj`: Wanda results at 50% sparsity for down projection
- `sparsegpt_0_5_up_proj`: SparseGPT results at 50% sparsity for up projection

## Key Algorithms

### Jaccard Similarity

Measures overlap of non-zero positions:
```python
jaccard(A, B) = |A ∩ B| / |A ∪ B|
```

### Hamming Distance

Proportion of differing sparsity patterns:
```python
hamming(A, B) = (A != B).sum() / len(A)
```

### Feature Co-activation Gradient

Orders features by Hamming distance to identify co-activation patterns:
1. Select a reference feature
2. Compute Hamming distance to all other features
3. Sort by distance (ascending = most similar first)
4. Visualize with spy plots

### k-means Clustering

Groups features by activation patterns:
- **Input**: Binary activation matrices (features × samples)
- **Output**: Cluster assignments
- **Metrics**:
  - Within-cluster distance (cohesion)
  - Between-cluster distance (separation)
  - Separation ratio = between / within

### MMM Block Coherence

Measures potential for Multiply-Mask-Multiply optimization:
```python
coherence = (all_features_zero_in_block).sum() / total_blocks
```

High coherence (>0.3) indicates good MMM applicability.

## Configuration

### Sparsity Thresholds

Hard-coded thresholds for 90% sparsity (from `scripts/visualization/draw_visualization.py`):

```python
thresholds_90pct = {
    'q': 0.0182,
    'k': 0.0182,
    'v': 0.0182,
    'o': 0.0035,
    'up': 0.0601,
    'gate': 0.0601,
    'down': 0.0032
}
```

### Triton Kernel Configuration

Auto-tuned configurations in `scripts/kernels/kernelize.py`:

```python
configs = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 128}, num_warps=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 256}, num_warps=4),
    # ... more configurations
]
```

Tuning parameters:
- `BLOCK_M`, `BLOCK_N`: Block sizes for tiling
- `num_warps`: Number of GPU warps per threadblock
- `SPARSITY_BIN`: Cache key for different sparsity levels

## Contributing

When adding new analysis scripts:
1. Place visualization scripts in `scripts/visualization/`
2. Place clustering analysis in `scripts/clustering/`
3. Place method comparisons in `scripts/metrics/`
4. Update paths to use relative references from script location
5. Save outputs to `results/visualizations/` or `results/metrics/`

## License

See project documentation for licensing information.

## Citation

If you use this codebase in your research, please cite the associated paper (see REPORT.md).
