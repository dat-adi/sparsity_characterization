"""
Generate a comparative report across all analyzed matrices.
"""

import json
from pathlib import Path
from rich import print
from rich.table import Table
from rich.console import Console
import argparse


def load_results(results_dir):
    """Load all JSON results from the results directory."""
    results = {}
    results_path = Path(results_dir)

    for json_file in results_path.glob("*.json"):
        with open(json_file, 'r') as f:
            data = json.load(f)
            key = json_file.stem  # Use filename without extension as key
            results[key] = data

    return results


def generate_comparative_table(results, console):
    """Generate a comparative table across all matrices."""

    # Create main comparison table
    table = Table(
        show_header=True,
        header_style="bold cyan",
        title="[bold]Multi-Execution Comparative Analysis Summary[/bold]",
        title_style="bold magenta"
    )

    table.add_column("Matrix", style="cyan", width=20)
    table.add_column("Shape", justify="center", width=15)
    table.add_column("Hamming\nMin (μ±σ)", justify="right", width=18)
    table.add_column("Hamming\nMax (μ±σ)", justify="right", width=18)
    table.add_column("Within\nCluster (μ±σ)", justify="right", width=18)
    table.add_column("Between\nCluster (μ±σ)", justify="right", width=18)
    table.add_column("Sep.\nRatio", justify="right", width=12)
    table.add_column("Cluster\nBalance", justify="right", width=12)

    # Sort by matrix name for consistent ordering
    sorted_results = sorted(results.items())

    for key, data in sorted_results:
        agg = data['aggregated_metrics']
        shape = f"{data['matrix_shape'][0]}×{data['matrix_shape'][1]}"

        # Format values
        hamming_min = f"{agg['hamming_min_mean']:.0f}±{agg['hamming_min_std']:.0f}"
        hamming_max = f"{agg['hamming_max_mean']:.0f}±{agg['hamming_max_std']:.0f}"
        within = f"{agg['mean_within_cluster_mean']:.2f}±{agg['mean_within_cluster_std']:.2f}"
        between = f"{agg['mean_between_cluster_mean']:.2f}±{agg['mean_between_cluster_std']:.2f}"
        sep_ratio = f"{agg['separation_ratio_mean']:.3f}"
        balance = f"{agg['cluster_balance_mean']:.1f}"

        # Color code separation ratio
        if agg['separation_ratio_mean'] > 1.0:
            sep_style = "green"
        elif agg['separation_ratio_mean'] > 0.5:
            sep_style = "yellow"
        else:
            sep_style = "red"

        # Color code balance
        if agg['cluster_balance_mean'] < 10:
            balance_style = "green"
        elif agg['cluster_balance_mean'] < 20:
            balance_style = "yellow"
        else:
            balance_style = "red"

        table.add_row(
            key.replace("_", " ").title(),
            shape,
            hamming_min,
            hamming_max,
            within,
            between,
            f"[{sep_style}]{sep_ratio}[/{sep_style}]",
            f"[{balance_style}]{balance}[/{balance_style}]"
        )

    console.print("\n")
    console.print(table)
    console.print("\n")


def generate_insights(results, console):
    """Generate key insights from the comparative analysis."""

    console.print("[bold cyan]Key Insights:[/bold cyan]\n")

    # Extract metrics
    metrics_by_type = {
        'down_proj': {},
        'up_proj': {}
    }

    for key, data in results.items():
        proj_type = 'down_proj' if 'down_proj' in key else 'up_proj'
        method = 'wanda' if 'wanda' in key else 'sparsegpt'
        agg = data['aggregated_metrics']

        metrics_by_type[proj_type][method] = {
            'sep_ratio': agg['separation_ratio_mean'],
            'within': agg['mean_within_cluster_mean'],
            'between': agg['mean_between_cluster_mean'],
            'balance': agg['cluster_balance_mean'],
            'hamming_range': agg['hamming_range_mean']
        }

    # Insight 1: Projection type comparison
    console.print("[bold]1. Projection Type Differences:[/bold]")
    console.print("   • [cyan]down_proj matrices[/cyan] show:")
    if metrics_by_type['down_proj']:
        avg_sep_down = sum(m['sep_ratio'] for m in metrics_by_type['down_proj'].values()) / len(metrics_by_type['down_proj'])
        avg_balance_down = sum(m['balance'] for m in metrics_by_type['down_proj'].values()) / len(metrics_by_type['down_proj'])
        console.print(f"     - Average separation ratio: {avg_sep_down:.3f}")
        console.print(f"     - Average cluster balance: {avg_balance_down:.1f} (higher = more imbalanced)")

    console.print("   • [cyan]up_proj matrices[/cyan] show:")
    if metrics_by_type['up_proj']:
        avg_sep_up = sum(m['sep_ratio'] for m in metrics_by_type['up_proj'].values()) / len(metrics_by_type['up_proj'])
        avg_balance_up = sum(m['balance'] for m in metrics_by_type['up_proj'].values()) / len(metrics_by_type['up_proj'])
        console.print(f"     - Average separation ratio: {avg_sep_up:.3f}")
        console.print(f"     - Average cluster balance: {avg_balance_up:.1f} (higher = more imbalanced)")

    # Insight 2: Method comparison
    console.print("\n[bold]2. Pruning Method Comparison:[/bold]")

    for proj_type in ['down_proj', 'up_proj']:
        if 'wanda' in metrics_by_type[proj_type] and 'sparsegpt' in metrics_by_type[proj_type]:
            wanda = metrics_by_type[proj_type]['wanda']
            sparsegpt = metrics_by_type[proj_type]['sparsegpt']

            console.print(f"   • [cyan]{proj_type}[/cyan]:")
            console.print(f"     - Separation ratio: Wanda={wanda['sep_ratio']:.3f}, SparseGPT={sparsegpt['sep_ratio']:.3f}")
            console.print(f"     - Cluster balance: Wanda={wanda['balance']:.1f}, SparseGPT={sparsegpt['balance']:.1f}")

            sep_diff = abs(wanda['sep_ratio'] - sparsegpt['sep_ratio'])
            if sep_diff < 0.1:
                console.print(f"     → [yellow]Similar clustering behavior[/yellow]")
            else:
                better = "Wanda" if wanda['sep_ratio'] > sparsegpt['sep_ratio'] else "SparseGPT"
                console.print(f"     → [green]{better} shows better separation[/green]")

    # Insight 3: Stability analysis
    console.print("\n[bold]3. Stability Across Random Selections:[/bold]")
    for key, data in sorted(results.items()):
        agg = data['aggregated_metrics']
        # Calculate coefficient of variation for separation ratio
        cv_sep = (agg['separation_ratio_std'] / agg['separation_ratio_mean']) * 100 if agg['separation_ratio_mean'] > 0 else 0

        stability = "stable" if cv_sep < 30 else "moderate" if cv_sep < 50 else "unstable"
        style = "green" if cv_sep < 30 else "yellow" if cv_sep < 50 else "red"

        console.print(f"   • [{style}]{key.replace('_', ' ').title()}[/{style}]: {stability} (CV={cv_sep:.1f}%)")

    console.print("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparative report from analysis results"
    )
    parser.add_argument(
        '--results-dir',
        type=str,
        default='analysis_results',
        help='Directory containing JSON result files (default: analysis_results)'
    )

    args = parser.parse_args()

    console = Console()

    # Load results
    results = load_results(args.results_dir)

    if not results:
        console.print(f"[bold red]Error:[/bold red] No results found in {args.results_dir}")
        return

    console.print(f"\n[bold cyan]Loaded {len(results)} result files[/bold cyan]")

    # Generate comparative table
    generate_comparative_table(results, console)

    # Generate insights
    generate_insights(results, console)

    console.print("[dim]Note: μ = mean, σ = standard deviation across 10 executions[/dim]\n")


if __name__ == "__main__":
    main()
