"""
Visualization utilities for binary matrices.

Functions for creating spy plots and other visualizations.
"""

import matplotlib.pyplot as plt
import torch
from pathlib import Path

def viz_group_metrics(results: list[dict], method: str, matrix_name: str, save_path: Path | str, matrix_shape: tuple[int, int] | None = None) -> None:
    """
    Create comprehensive visualization with 5 subplots for group analysis.

    Args:
        results: List of result dictionaries from analyze_matrix containing metrics for each group
        method: Pruning method name ('wanda' or 'sparsegpt')
        matrix_name: Name of the matrix being analyzed
        save_path: Path to save the figure
        matrix_shape: Optional tuple of (num_rows, num_cols) for the original matrix to calculate correct percentages
    """
    # Sort results by mean distance (ascending) for meaningful x-axis ordering
    results = sorted(results, key=lambda x: x['mean_distance'])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f"{method.upper()} - {matrix_name}\nGroup Metrics Analysis (Sorted by Mean Distance)", fontsize=16, fontweight='bold')

    # Extract data from results
    group_numbers = [i + 1 for i in range(len(results))]
    mean_distances = [r['mean_distance'] for r in results]
    all_zero_rows = [r['metrics']['zeros'] for r in results]
    all_one_rows = [r['metrics']['ones'] for r in results]
    duplicate_rows = [r['metrics']['duplicates'] for r in results]
    total_duplicates = [r['metrics']['total_dups'] for r in results]

    # Smart tick configuration based on number of groups
    num_groups = len(results)
    if num_groups <= 20:
        tick_step = 1
        rotation = 0
        fontsize = 9
    elif num_groups <= 50:
        tick_step = 5
        rotation = 45
        fontsize = 8
    elif num_groups <= 100:
        tick_step = 10
        rotation = 90
        fontsize = 7
    else:
        tick_step = 20
        rotation = 90
        fontsize = 6

    tick_positions = group_numbers[::tick_step]
    tick_labels = [str(x) for x in tick_positions]

    # Helper function to configure x-axis consistently
    def configure_xaxis(ax, group_numbers):
        ax.set_xlim(0.5, len(group_numbers) + 0.5)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=rotation, fontsize=fontsize, ha='right' if rotation > 0 else 'center')

    # Subplot 1: Mean Hamming Distance per Group
    ax1 = axes[0, 0]
    bars1 = ax1.bar(group_numbers, mean_distances, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax1.set_ylabel('Mean Hamming Distance', fontsize=11)
    ax1.set_title('Mean Pairwise Hamming Distance by Group', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax1, group_numbers)

    # Subplot 2: All-Zero Rows per Group
    ax2 = axes[0, 1]
    bars2 = ax2.bar(group_numbers, all_zero_rows, color='coral', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('All-Zero Rows by Group', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax2, group_numbers)

    # Subplot 3: All-One Rows per Group
    ax3 = axes[0, 2]
    bars3 = ax3.bar(group_numbers, all_one_rows, color='mediumseagreen', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax3.set_ylabel('Count', fontsize=11)
    ax3.set_title('All-One Rows by Group', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax3, group_numbers)

    # Subplot 4: Duplicate Row Patterns
    ax4 = axes[1, 0]
    bars4 = ax4.bar(group_numbers, duplicate_rows, color='mediumpurple', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax4.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax4.set_ylabel('Count', fontsize=11)
    ax4.set_title('Number of Duplicate Row Patterns by Group', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax4, group_numbers)

    # Subplot 5: Total Duplicate Instances
    ax5 = axes[1, 1]
    bars5 = ax5.bar(group_numbers, total_duplicates, color='gold', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax5.set_xlabel('Group Rank (by Mean Distance)', fontsize=11)
    ax5.set_ylabel('Count', fontsize=11)
    ax5.set_title('Total Duplicate Row Instances by Group', fontsize=12, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3, linestyle='--')
    configure_xaxis(ax5, group_numbers)

    axes[1, 2].axis('off')

    # Add summary statistics in the unused subplot area
    # Calculate total rows: each group has n_neighbors+1 rows (original + neighbors)
    n_neighbors = results[0]['n_neighbors']
    rows_per_group = n_neighbors + 1
    total_groups = len(results)

    # For percentage calculations, use actual matrix rows if provided
    # Otherwise fall back to total group rows (with overlap)
    if matrix_shape is not None:
        actual_matrix_rows = matrix_shape[0]
        total_zeros = sum(all_zero_rows)
        total_ones = sum(all_one_rows)
        total_dups = sum(total_duplicates)

        # Calculate total row instances across all groups (includes overlaps)
        # Each group has n_neighbors+1 rows, and we have total_groups groups
        total_row_instances = 64 * actual_matrix_rows

        summary_text = f"Summary Statistics:\n\n"
        summary_text += f"Total Groups: {total_groups}\n"
        summary_text += f"Rows per Group: {rows_per_group}\n"
        summary_text += f"Total Row Instances: {total_row_instances}\n"
        summary_text += f"Matrix Rows: {actual_matrix_rows}\n\n"
        summary_text += f"Avg Mean Distance: {sum(mean_distances)/len(mean_distances):.4f}\n\n"
        summary_text += f"Total Zero Rows: {total_zeros} ({total_zeros/total_row_instances:.2%})\n"
        summary_text += f"Total One Rows: {total_ones} ({total_ones/total_row_instances:.2%})\n"
        summary_text += f"Total Duplicate Types: {sum(duplicate_rows)}\n"
        summary_text += f"Total Duplicate Instances: {total_dups} ({total_dups/total_row_instances:.2%})"
    else:
        # Fallback: use sum of group rows (may include overlaps)
        total_rows_with_overlap = total_groups * rows_per_group
        total_zeros = sum(all_zero_rows)
        total_ones = sum(all_one_rows)
        total_dups = sum(total_duplicates)

        summary_text = f"Summary Statistics:\n\n"
        summary_text += f"Total Groups: {total_groups}\n"
        summary_text += f"Rows per Group: {rows_per_group}\n"
        summary_text += f"Total Row Instances: {total_rows_with_overlap}\n"
        summary_text += f"(may include overlaps)\n\n"
        summary_text += f"Avg Mean Distance: {sum(mean_distances)/len(mean_distances):.4f}\n\n"
        summary_text += f"Total Zero Rows: {total_zeros}\n"
        summary_text += f"Total One Rows: {total_ones}\n"
        summary_text += f"Total Duplicate Types: {sum(duplicate_rows)}\n"
        summary_text += f"Total Duplicate Instances: {total_dups}"

    axes[1, 2].text(0.1, 0.5, summary_text, fontsize=11,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved group metrics visualization to: {save_path}")

def viz_binary_matrix(matrix: torch.Tensor, title: str, save_path: Path | str) -> None:
    """
    Visualize binary matrix (1=black, 0=white).

    Args:
        matrix: Binary matrix [D, N] to visualize
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Convert to numpy for plotting
    matrix_np = matrix.cpu().numpy() if isinstance(matrix, torch.Tensor) else matrix

    plt.figure(figsize=(10, 8))
    plt.imshow(matrix_np, cmap='binary', aspect='auto', interpolation='nearest')
    plt.title(f"{title}\nShape: {matrix_np.shape}, Density: {matrix_np.mean():.1%}")
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")


def viz_hamming_distance_cdf(
    values: torch.Tensor,
    cdf: torch.Tensor,
    title: str,
    save_path: Path | str
) -> None:
    """
    Visualize cumulative distribution function of Hamming distances.

    Args:
        values: Sorted unique distance values
        cdf: Cumulative probabilities for each value
        title: Title for the plot
        save_path: Path to save the figure
    """
    # Convert to numpy for plotting
    values_np = values.cpu().numpy() if isinstance(values, torch.Tensor) else values
    cdf_np = cdf.cpu().numpy() if isinstance(cdf, torch.Tensor) else cdf

    plt.figure(figsize=(10, 6))
    plt.plot(values_np, cdf_np, linewidth=2, marker='o', markersize=3)
    plt.xlabel('Hamming Distance')
    plt.ylabel('Cumulative Probability')
    plt.title(f"{title}\nCDF of Pairwise Hamming Distances")
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved CDF visualization to: {save_path}")
