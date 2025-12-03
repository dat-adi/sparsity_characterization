# set seed
# get matrices
# for each matrix:
## select 64 feature vectors randomly
## for each feature vector:
### indices <- find the 8 closest feature vectors by hamming distance
### create a group from these eight feature vectors -> subset
## end
## for each subset:
### pairs <- compute the pairwise hamming distance between all columns.
### cdf <- get cdf from the pairs
### zeros, ones <- compute the number of rows of all 0s and all 1s.
### iden, non_iden <- compute the number of rows which are identical and non-identical.
## end

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import random
import torch

# Import from local utilities
from hamming_utils import (
    find_most_similar_features,
    compute_pairwise_hamming_distances_efficient,
    compute_hamming_distance_cdf
)
from matrix_analysis import (
    count_inactive_rows,
    count_fully_active_rows,
    count_identical_rows
)
from visualization import viz_binary_matrix, viz_hamming_distance_cdf


def set_seed(random_seed: int) -> None:
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

def get_wanda_matrices(files: list[str]) -> list[Path]:
    WANDA_DIR = "/home/datadi/burns/aws/workloads/data/wanda_unstructured/"
    layer_1 = Path(WANDA_DIR) / "layer-1"

    return [layer_1 / f for f in files]

def get_sparsegpt_matrices(files: list[str]) -> list[Path]:
    SPARSEGPT_DIR = "/home/datadi/burns/aws/workloads/data/sparsegpt_unstructured/"
    layer_1 = Path(SPARSEGPT_DIR) / "layer-1"

    return [layer_1 / f for f in files]

def get_unstructured_matrices_layer_1() -> tuple[list[Path], list[Path]]:
    # Matrix files to analyze
    matrix_files = [
        "layer1-mlp.down_proj.pt",
        "layer1-mlp.up_proj.pt",
        "layer1-mlp.gate_proj.pt",
        "layer1-self_attn.q_proj.pt",
        "layer1-self_attn.k_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt",
    ]

    wanda_matrices = get_wanda_matrices(matrix_files)
    sparsegpt_matrices = get_sparsegpt_matrices(matrix_files)

    return wanda_matrices, sparsegpt_matrices

def select_feature_columns(matrix, n):
    # randomly select n columns from the matrix
    return random.sample(range(matrix.shape[1]), min(n, matrix.shape[1]))

def main():
    set_seed(42)
    wanda_m, sparsegpt_m = get_unstructured_matrices_layer_1()

    # Create output directory for visualizations
    output_dir = Path("./results/hamming_cdf")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store all results for later compilation
    all_results = []

    # Process both Wanda and SparseGPT matrices
    for method_name, matrices in [("wanda", wanda_m), ("sparsegpt", sparsegpt_m)]:
        for p in matrices:
            matrix = torch.load(p, weights_only=True)
            matrix = (matrix != 0).int() # convert to binary matrix
            cols = select_feature_columns(matrix=matrix, n=64)

            matrix_name = p.stem  # Get filename without extension
            print(f"\n{'='*80}")
            print(f"Method: {method_name.upper()} | Matrix: {p.name}")
            print(f"{'='*80}")

            for idx, c in enumerate(cols):
                subset, _, _ = find_most_similar_features(matrix, feature_idx=c, n_similar=8)
                dist = compute_pairwise_hamming_distances_efficient(subset)

                # Compute CDF
                cdf_values, cdf_probs = compute_hamming_distance_cdf(dist)

                # Compute row statistics
                zeros = count_inactive_rows(subset)
                ones = count_fully_active_rows(subset)
                duplicates, total_dups = count_identical_rows(subset)
                unique_rows = subset.shape[0] - total_dups

                print(f"\nFeature {idx + 1} (column {c}):")
                print(f"  Pairwise hamming distance mean: {dist.mean():.4f}")
                print(f"  Pairwise hamming distance std: {dist.std():.4f}")
                print(f"  All zero rows: {zeros}")
                print(f"  All ones rows: {ones}")
                print(f"  Rows that appear more than once: {duplicates}")
                print(f"  Total number of redundant/duplicate rows: {total_dups}")
                print(f"  Number of unique rows: {unique_rows}")
                print(f"  CDF points: {len(cdf_values)}")

                # Save visualizations
                binary_matrix_path = output_dir / f"{method_name}_{matrix_name}_col{c}_binary.png"
                cdf_path = output_dir / f"{method_name}_{matrix_name}_col{c}_cdf.png"

                viz_binary_matrix(subset, f"{method_name.upper()} - {matrix_name} - Column {c}", binary_matrix_path)
                viz_hamming_distance_cdf(cdf_values, cdf_probs, f"{method_name.upper()} - {matrix_name} - Column {c}", cdf_path)

                # Store results
                all_results.append({
                    'method': method_name,
                    'matrix': matrix_name,
                    'feature_idx': c,
                    'mean_distance': dist.mean().item(),
                    'std_distance': dist.std().item(),
                    'zeros': zeros,
                    'ones': ones,
                    'duplicates': duplicates,
                    'total_dups': total_dups,
                    'unique_rows': unique_rows,
                    'density': subset.float().mean().item(),
                    'cdf_points': len(cdf_values),
                    'cdf_path': str(cdf_path),
                    'binary_path': str(binary_matrix_path)
                })

                print("---")

    # Save results as JSON for PDF generation
    import json
    results_json_path = output_dir / "all_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis complete! Processed {len(all_results)} feature groups.")
    print(f"Results saved to: {results_json_path}")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    main()

