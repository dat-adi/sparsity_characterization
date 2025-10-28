# Analysis Scripts

Advanced analysis tools for multi-seed experiments and detailed feature clustering in pruned neural network weights.

## Directory Structure

```
analysis/
├── hamming_cluster_analysis.py             # Single-execution analysis
├── multi_execution_hamming_analysis.py     # Multi-execution statistical analysis
├── multi_seed_clustering_analysis.py       # NEW: Multi-seed clustering analysis
├── generate_comparative_report.py          # Comparative report generator
└── README.md                               # This file
```

## Scripts Overview

### 1. `multi_seed_clustering_analysis.py` ⭐ NEW

**Comprehensive multi-seed clustering analysis with extensive visualizations.**

Performs clustering analysis across multiple random seeds to understand the stability and structure of feature co-activation in sparse weight matrices.

For each random seed:
1. Randomly selects a feature from the weight matrix
2. Finds the top N most similar features (by Hamming distance)
3. Performs k-means clustering with multiple k values
4. Computes cluster metrics (within/between distances, separation ratios)
5. Creates comprehensive visualizations

**Usage:**
```bash
# Basic usage
python scripts/analysis/multi_seed_clustering_analysis.py \
    --method wanda \
    --layer 1 \
    --component mlp.down_proj

# Full configuration
python scripts/analysis/multi_seed_clustering_analysis.py \
    --method wanda \
    --layer 1 \
    --component self_attn.q_proj \
    --seeds 10 \
    --n-similar 128 \
    --k-values 4 8 16
```

**Arguments:**
- `--method`: Pruning method (`wanda` or `sparsegpt`) **[required]**
- `--layer`: Layer number to analyze (default: 1)
- `--component`: Component name (e.g., `mlp.down_proj`, `self_attn.q_proj`) **[required]**
- `--seeds`: Number of random seeds to test (default: 10)
- `--n-similar`: Number of similar features to analyze (default: 128)
- `--k-values`: K values for clustering (default: 4 8 16)
- `--output-dir`: Custom output directory (optional)

**Output Structure:**
```
results/metrics/multi_seed_clustering/{method}/{component}/
├── seed_0/k_4/
│   ├── seed0_k4_feat{N}_spy_plot.png          # Sparsity patterns with cluster boundaries
│   ├── seed0_k4_feat{N}_statistics.png        # Cluster sizes and cohesion
│   ├── seed0_k4_feat{N}_tsne.png              # t-SNE 2D embedding
│   └── seed0_k4_feat{N}_distance_heatmap.png  # Within/between distance matrix
├── seed_0/k_8/...
├── seed_1/...
└── summary/
    ├── summary_separation_ratios.png  # Separation ratios across k values
    ├── summary_distances.png          # Within/between distance trends
    └── metrics_summary.json           # All metrics in JSON format
```

**Visualizations Generated:**
- **Per-seed, per-k**: Spy plot, cluster statistics, t-SNE, distance heatmap
- **Summary**: Separation ratios (mean ± std across seeds), distance trends

**Metrics Computed:**
- Within-cluster distance: Mean Hamming distance within same cluster
- Between-cluster distance: Mean Hamming distance between different clusters
- Separation ratio: Between/within distance ratio (higher = better separation)
- Cluster sizes, inertia, convergence info

**Use Cases:**
- Feature co-activation analysis
- MMM optimization potential assessment
- Method comparison (Wanda vs SparseGPT)
- Robustness testing across random seeds

---

### 2. `hamming_cluster_analysis.py`
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

### 3. `multi_execution_hamming_analysis.py`
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

### 4. `generate_comparative_report.py`
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
