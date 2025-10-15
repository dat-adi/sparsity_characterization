# Analysis Scripts

This directory contains scripts for analyzing sparsity patterns and feature clustering in pruned neural network weights.

## Directory Structure

```
analysis_scripts/
├── hamming_cluster_analysis.py          # Single-execution analysis
├── multi_execution_hamming_analysis.py  # Multi-execution statistical analysis
├── generate_comparative_report.py       # Comparative report generator
└── README.md                            # This file

../analysis_results/                     # Output directory for JSON results
├── wanda_down_proj.json
├── wanda_up_proj.json
├── sparsegpt_down_proj.json
└── sparsegpt_up_proj.json
```

## Scripts Overview

### 1. `hamming_cluster_analysis.py`
Single-execution analysis that:
- Selects a feature vector from a weight matrix
- Finds top 128 most similar features by Hamming distance
- Performs k-means clustering
- Displays all Hamming distances and absolute cluster metrics

**Usage:**
```bash
python analysis_scripts/hamming_cluster_analysis.py <matrix_path> [options]

# Example
python analysis_scripts/hamming_cluster_analysis.py \
    clustering_characterization/wanda/layer1-mlp.down_proj.pt \
    --feature-idx 42 \
    --n-features 128 \
    --n-clusters 8
```

**Options:**
- `--feature-idx`: Index of main feature to analyze (default: randomly selected)
- `--n-features`: Number of similar features to analyze (default: 128)
- `--n-clusters`: Number of k-means clusters (default: 8)
- `--seed`: Random seed for reproducibility (default: 42)

### 2. `multi_execution_hamming_analysis.py`
Multi-execution statistical analysis that:
- Runs clustering analysis over N random feature selections
- Aggregates statistics (mean ± std) across executions
- Exports results to JSON for further analysis

**Usage:**
```bash
python analysis_scripts/multi_execution_hamming_analysis.py <matrix_path> [options]

# Example
python analysis_scripts/multi_execution_hamming_analysis.py \
    clustering_characterization/wanda/layer1-mlp.down_proj.pt \
    --n-executions 10 \
    --n-clusters 8 \
    --output-json analysis_results/wanda_down_proj.json
```

**Options:**
- `--n-executions`: Number of random feature selections (default: 10)
- `--n-features`: Number of similar features per execution (default: 128)
- `--n-clusters`: Number of k-means clusters (default: 8)
- `--seed`: Random seed for reproducibility (default: 42)
- `--output-json`: Path to save JSON output (optional)

### 3. `generate_comparative_report.py`
Generates comparative analysis across multiple matrices:
- Loads all JSON results from a directory
- Creates comparative table with key metrics
- Provides insights on projection types and pruning methods

**Usage:**
```bash
python analysis_scripts/generate_comparative_report.py [--results-dir <dir>]

# Example
python analysis_scripts/generate_comparative_report.py --results-dir analysis_results
```

## Quick Start: Full Analysis Pipeline

Run all analyses and generate comparative report:

```bash
# 1. Run multi-execution analysis on all matrices
python analysis_scripts/multi_execution_hamming_analysis.py \
    clustering_characterization/wanda/layer1-mlp.down_proj.pt \
    --n-executions 10 --output-json analysis_results/wanda_down_proj.json

python analysis_scripts/multi_execution_hamming_analysis.py \
    clustering_characterization/wanda/layer1-mlp.up_proj.pt \
    --n-executions 10 --output-json analysis_results/wanda_up_proj.json

python analysis_scripts/multi_execution_hamming_analysis.py \
    clustering_characterization/sparsegpt/layer1-mlp.down_proj.pt \
    --n-executions 10 --output-json analysis_results/sparsegpt_down_proj.json

python analysis_scripts/multi_execution_hamming_analysis.py \
    clustering_characterization/sparsegpt/layer1-mlp.up_proj.pt \
    --n-executions 10 --output-json analysis_results/sparsegpt_up_proj.json

# 2. Generate comparative report
python analysis_scripts/generate_comparative_report.py
```

## Metrics Explained

### Hamming Distance Metrics
- **Min/Max Hamming Distance**: Range of differing positions between selected feature and top 128 similar features
- **Hamming Range**: Spread of Hamming distances (max - min)

### Cluster Distance Metrics
- **Mean Within-Cluster Distance**: Average Euclidean distance from each feature to its cluster center (lower = tighter clusters)
- **Mean Between-Cluster Distance**: Average pairwise Euclidean distance between cluster centers (higher = more separated)
- **Separation Ratio**: Between-cluster / Within-cluster distance
  - `> 2.0`: Excellent separation
  - `> 1.0`: Good separation
  - `> 0.5`: Moderate separation
  - `< 0.5`: Poor separation

### Other Metrics
- **Total Inertia**: Sum of squared distances to nearest cluster center (lower = better clustering)
- **Cluster Balance**: Standard deviation of cluster sizes (lower = more balanced)

## Output Files

JSON output files contain:
- Matrix metadata (path, name, shape)
- Aggregated metrics across all executions
- Individual metrics for each execution
- Configuration parameters used

## Dependencies

All scripts require:
- torch
- numpy
- scikit-learn
- rich (for formatted console output)
