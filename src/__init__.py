"""
Nearest Neighbor Feature Analysis Package

This package provides utilities for analyzing feature similarity in
pruned neural network weight matrices using Hamming distance metrics.

Modules:
    data_loaders: Load weight matrices from Wanda and SparseGPT
    hamming_utils: Compute Hamming distances and find similar features
    matrix_analysis: Compute sparsity and pattern metrics
    visualization: Create visualizations of binary matrices
    database: Store and query results in DuckDB
    main: Main analysis pipeline
"""

__version__ = "0.1.0"

from .data_loaders import (
    set_seed,
    get_wanda_matrices,
    get_sparsegpt_matrices,
    get_unstructured_matrices_layer_1,
    select_feature_columns
)

from .hamming_utils import (
    compute_hamming_distance_batch,
    find_most_similar_features,
    compute_pairwise_hamming_distances_efficient
)

from .matrix_analysis import (
    count_inactive_rows,
    count_fully_active_rows,
    count_identical_rows,
    compute_subset_metrics
)

from .visualization import viz_binary_matrix

from .database import (
    get_db_connection,
    create_nearest_neighbor_table,
    insert_analysis_result,
    save_batch_results,
    query_results
)

__all__ = [
    # data_loaders
    'set_seed',
    'get_wanda_matrices',
    'get_sparsegpt_matrices',
    'get_unstructured_matrices_layer_1',
    'select_feature_columns',

    # hamming_utils
    'compute_hamming_distance_batch',
    'find_most_similar_features',
    'compute_pairwise_hamming_distances_efficient',

    # matrix_analysis
    'count_inactive_rows',
    'count_fully_active_rows',
    'count_identical_rows',
    'compute_subset_metrics',

    # visualization
    'viz_binary_matrix',

    # database
    'get_db_connection',
    'create_nearest_neighbor_table',
    'insert_analysis_result',
    'save_batch_results',
    'query_results',
]
