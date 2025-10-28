#!/usr/bin/env python3
"""
Generate Detailed Per-Seed Tables

Creates comprehensive tables showing clustering metrics for each individual seed
across all matrices for both Wanda and SparseGPT.

Usage:
    python generate_seed_detail_tables.py
"""

import sys
import torch
import numpy as np
import random
from pathlib import Path
from typing import Dict, List
from rich.console import Console
from rich.table import Table
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.hamming_analysis import find_most_similar_features
from utils.clustering import compute_cluster_metrics, ClusterMetrics

console = Console()


def run_single_seed_analysis(
    matrix: torch.Tensor,
    seed: int,
    n_features: int = 128,
    k_values: List[int] = [4, 8, 16]
) -> Dict[int, ClusterMetrics]:
    """
    Run clustering analysis for a single seed.

    Args:
        matrix: Weight matrix [D, N]
        seed: Random seed for reproducibility
        n_features: Number of similar features to find
        k_values: List of k values for clustering

    Returns:
        Dictionary mapping k -> ClusterMetrics
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Randomly select a feature
    n_total_features = matrix.shape[1]
    selected_feature_idx = random.randint(0, n_total_features - 1)

    # Find most similar features
    subset, indices, distances = find_most_similar_features(
        matrix, selected_feature_idx, n_features
    )

    # Perform clustering for each k
    results = {}
    for k in k_values:
        metrics = compute_cluster_metrics(subset, k, random_state=seed)
        results[k] = metrics

    return results, selected_feature_idx


def create_seed_table_for_matrix(
    matrix_name: str,
    wanda_seed_results: List[tuple],
    sparsegpt_seed_results: List[tuple],
    k_values: List[int]
) -> Table:
    """Create a detailed table showing results for each seed."""

    table = Table(
        title=f"Per-Seed Results: {matrix_name}",
        show_header=True,
        header_style="bold cyan"
    )

    # Add columns
    table.add_column("Seed", justify="center", style="yellow")
    table.add_column("Method", justify="center")
    table.add_column("Feature\nIndex", justify="center")
    table.add_column("k", justify="center")
    table.add_column("Within\nDistance", justify="right")
    table.add_column("Between\nDistance", justify="right")
    table.add_column("Separation\nRatio", justify="right")
    table.add_column("Cluster\nSizes", justify="left")

    n_seeds = len(wanda_seed_results)

    for seed_idx in range(n_seeds):
        w_results, w_feature_idx = wanda_seed_results[seed_idx]
        s_results, s_feature_idx = sparsegpt_seed_results[seed_idx]

        # Add Wanda results for this seed
        for k_idx, k in enumerate(k_values):
            w_metrics = w_results[k]
            sep_ratio = (w_metrics.mean_between / w_metrics.mean_within
                        if w_metrics.mean_within > 0 else float('inf'))

            cluster_sizes_str = str(w_metrics.cluster_sizes)

            table.add_row(
                str(seed_idx) if k_idx == 0 else "",
                "Wanda" if k_idx == 0 else "",
                str(w_feature_idx) if k_idx == 0 else "",
                str(k),
                f"{w_metrics.mean_within:.4f}",
                f"{w_metrics.mean_between:.4f}" if not np.isnan(w_metrics.mean_between) else "NaN",
                f"{sep_ratio:.4f}" if not np.isnan(sep_ratio) and not np.isinf(sep_ratio) else "NaN",
                cluster_sizes_str,
                style="green" if k_idx == 0 else ""
            )

        # Add SparseGPT results for this seed
        for k_idx, k in enumerate(k_values):
            s_metrics = s_results[k]
            sep_ratio = (s_metrics.mean_between / s_metrics.mean_within
                        if s_metrics.mean_within > 0 else float('inf'))

            cluster_sizes_str = str(s_metrics.cluster_sizes)

            table.add_row(
                "",
                "SparseGPT" if k_idx == 0 else "",
                str(s_feature_idx) if k_idx == 0 else "",
                str(k),
                f"{s_metrics.mean_within:.4f}",
                f"{s_metrics.mean_between:.4f}" if not np.isnan(s_metrics.mean_between) else "NaN",
                f"{sep_ratio:.4f}" if not np.isnan(sep_ratio) and not np.isinf(sep_ratio) else "NaN",
                cluster_sizes_str,
                style="magenta" if k_idx == 0 else "",
                end_section=(seed_idx == n_seeds - 1 and k_idx == len(k_values) - 1)
            )

    return table


def export_to_csv(
    matrix_name: str,
    wanda_seed_results: List[tuple],
    sparsegpt_seed_results: List[tuple],
    k_values: List[int],
    output_dir: Path
):
    """Export detailed results to CSV."""

    rows = []
    n_seeds = len(wanda_seed_results)

    for seed_idx in range(n_seeds):
        w_results, w_feature_idx = wanda_seed_results[seed_idx]
        s_results, s_feature_idx = sparsegpt_seed_results[seed_idx]

        # Add Wanda results
        for k in k_values:
            w_metrics = w_results[k]
            sep_ratio = (w_metrics.mean_between / w_metrics.mean_within
                        if w_metrics.mean_within > 0 else np.nan)

            rows.append({
                'seed': seed_idx,
                'method': 'Wanda',
                'matrix': matrix_name,
                'feature_idx': w_feature_idx,
                'k': k,
                'within_distance': w_metrics.mean_within,
                'between_distance': w_metrics.mean_between,
                'separation_ratio': sep_ratio,
                'cluster_sizes': str(w_metrics.cluster_sizes),
                'inertia': w_metrics.inertia,
                'n_iter': w_metrics.n_iter
            })

        # Add SparseGPT results
        for k in k_values:
            s_metrics = s_results[k]
            sep_ratio = (s_metrics.mean_between / s_metrics.mean_within
                        if s_metrics.mean_within > 0 else np.nan)

            rows.append({
                'seed': seed_idx,
                'method': 'SparseGPT',
                'matrix': matrix_name,
                'feature_idx': s_feature_idx,
                'k': k,
                'within_distance': s_metrics.mean_within,
                'between_distance': s_metrics.mean_between,
                'separation_ratio': sep_ratio,
                'cluster_sizes': str(s_metrics.cluster_sizes),
                'inertia': s_metrics.inertia,
                'n_iter': s_metrics.n_iter
            })

    df = pd.DataFrame(rows)

    # Save to CSV
    csv_filename = f"seed_details_{matrix_name}.csv"
    df.to_csv(output_dir / csv_filename, index=False)

    return csv_filename


def main():
    """Main execution function."""

    console.print("\n[bold cyan]Generating Per-Seed Detail Tables[/bold cyan]\n")

    # Configuration
    wanda_dir = Path("../../data/wanda_unstructured/layer-1")
    sparsegpt_dir = Path("../../data/sparsegpt_unstructured/layer-1")
    output_dir = Path("../../results/metrics/multi_seed_hamming_comparison")

    matrix_files = [
        "layer1-mlp.down_proj.pt",
        "layer1-mlp.up_proj.pt",
        "layer1-mlp.gate_proj.pt",
        "layer1-self_attn.q_proj.pt",
        "layer1-self_attn.k_proj.pt",
        "layer1-self_attn.v_proj.pt",
        "layer1-self_attn.o_proj.pt",
    ]

    n_seeds = 10
    n_features = 128
    k_values = [4, 8, 16]

    console.print(f"Configuration:")
    console.print(f"  • Seeds: {n_seeds}")
    console.print(f"  • Features per seed: {n_features}")
    console.print(f"  • k values: {k_values}")
    console.print(f"  • Matrices: {len(matrix_files)}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each matrix
    for matrix_file in matrix_files:
        matrix_name = matrix_file.replace('.pt', '')

        console.print(f"\n[bold]Processing: {matrix_name}[/bold]")

        # Load matrices
        wanda_matrix = torch.load(wanda_dir / matrix_file, weights_only=True)
        sparsegpt_matrix = torch.load(sparsegpt_dir / matrix_file, weights_only=True)

        console.print(f"  Wanda matrix shape: {wanda_matrix.shape}, "
                     f"sparsity: {(wanda_matrix == 0).float().mean():.2%}")
        console.print(f"  SparseGPT matrix shape: {sparsegpt_matrix.shape}, "
                     f"sparsity: {(sparsegpt_matrix == 0).float().mean():.2%}")

        # Run analysis for all seeds
        wanda_seed_results = []
        sparsegpt_seed_results = []

        for seed in range(n_seeds):
            w_result, w_idx = run_single_seed_analysis(
                wanda_matrix, seed, n_features, k_values
            )
            wanda_seed_results.append((w_result, w_idx))

            s_result, s_idx = run_single_seed_analysis(
                sparsegpt_matrix, seed, n_features, k_values
            )
            sparsegpt_seed_results.append((s_result, s_idx))

        # Create and display table
        table = create_seed_table_for_matrix(
            matrix_name, wanda_seed_results, sparsegpt_seed_results, k_values
        )
        console.print(table)

        # Export to CSV
        csv_filename = export_to_csv(
            matrix_name, wanda_seed_results, sparsegpt_seed_results,
            k_values, output_dir
        )
        console.print(f"  [green]✓[/green] Saved: {output_dir / csv_filename}")

    # Create a combined CSV with all matrices
    console.print("\n[bold cyan]Creating Combined CSV[/bold cyan]")

    all_dfs = []
    for matrix_file in matrix_files:
        matrix_name = matrix_file.replace('.pt', '')
        csv_path = output_dir / f"seed_details_{matrix_name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    combined_csv_path = output_dir / "seed_details_all_matrices.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    console.print(f"[green]✓[/green] Saved: {combined_csv_path}")

    console.print("\n[bold green]Analysis Complete![/bold green]")
    console.print(f"\nResults saved to: [cyan]{output_dir}[/cyan]")
    console.print(f"\nGenerated {len(matrix_files)} individual CSV files + 1 combined CSV")


if __name__ == "__main__":
    main()
