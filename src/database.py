"""
Database utilities for storing analysis metrics.

Functions for creating and populating DuckDB tables with analysis results.
"""

import duckdb
from pathlib import Path
from typing import Any


def get_db_connection(db_path: str = "./sparsity.db") -> duckdb.DuckDBPyConnection:
    """
    Get a connection to the DuckDB database.

    Args:
        db_path: Path to the database file

    Returns:
        DuckDB connection object
    """
    return duckdb.connect(db_path)


def create_nearest_neighbor_table(conn: duckdb.DuckDBPyConnection) -> None:
    """
    Create table for nearest neighbor analysis results.

    Args:
        conn: DuckDB connection
    """
    # Create sequence first
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS seq_nearest_neighbor_metrics_id START 1
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS nearest_neighbor_metrics (
            id INTEGER PRIMARY KEY DEFAULT nextval('seq_nearest_neighbor_metrics_id'),
            method VARCHAR,
            matrix_name VARCHAR,
            layer INTEGER,
            projection VARCHAR,
            feature_idx INTEGER,
            n_neighbors INTEGER,
            mean_pairwise_distance DOUBLE,
            zeros INTEGER,
            ones INTEGER,
            duplicates INTEGER,
            total_dups INTEGER,
            unique_rows INTEGER,
            density DOUBLE,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    print("Created/verified nearest_neighbor_metrics table")


def insert_analysis_result(
    conn: duckdb.DuckDBPyConnection,
    method: str,
    matrix_name: str,
    feature_idx: int,
    n_neighbors: int,
    mean_distance: float,
    metrics: dict[str, Any]
) -> None:
    """
    Insert a single analysis result into the database.

    Args:
        conn: DuckDB connection
        method: Pruning method (e.g., 'wanda', 'sparsegpt')
        matrix_name: Name of the matrix file
        feature_idx: Index of the reference feature
        n_neighbors: Number of neighbors analyzed
        mean_distance: Mean pairwise Hamming distance
        metrics: Dictionary of computed metrics
    """
    # Parse matrix name to extract layer and projection
    # Example: "layer1-mlp.down_proj.pt" -> layer=1, projection="mlp.down_proj"
    parts = matrix_name.replace(".pt", "").split("-")
    layer = int(parts[0].replace("layer", ""))
    projection = "-".join(parts[1:])

    conn.execute("""
        INSERT INTO nearest_neighbor_metrics (
            method, matrix_name, layer, projection, feature_idx, n_neighbors,
            mean_pairwise_distance, zeros, ones, duplicates, total_dups,
            unique_rows, density
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, [
        method,
        matrix_name,
        layer,
        projection,
        feature_idx,
        n_neighbors,
        mean_distance,
        metrics['zeros'],
        metrics['ones'],
        metrics['duplicates'],
        metrics['total_dups'],
        metrics['unique_rows'],
        metrics['density']
    ])


def save_batch_results(
    db_path: str,
    results: list[dict[str, Any]]
) -> None:
    """
    Save a batch of analysis results to the database.

    Args:
        db_path: Path to the database file
        results: List of result dictionaries, each containing:
            - method: Pruning method
            - matrix_name: Matrix filename
            - feature_idx: Feature index
            - n_neighbors: Number of neighbors
            - mean_distance: Mean pairwise distance
            - metrics: Dictionary of computed metrics
    """
    conn = get_db_connection(db_path)
    create_nearest_neighbor_table(conn)

    for result in results:
        insert_analysis_result(
            conn,
            result['method'],
            result['matrix_name'],
            result['feature_idx'],
            result['n_neighbors'],
            result['mean_distance'],
            result['metrics']
        )

    conn.commit()
    conn.close()
    print(f"Saved {len(results)} results to {db_path}")


def query_results(
    db_path: str,
    method: str | None = None,
    projection: str | None = None
) -> list[tuple]:
    """
    Query analysis results from the database.

    Args:
        db_path: Path to the database file
        method: Optional filter by pruning method
        projection: Optional filter by projection type

    Returns:
        List of result tuples
    """
    conn = get_db_connection(db_path)

    query = "SELECT * FROM nearest_neighbor_metrics WHERE 1=1"
    params = []

    if method:
        query += " AND method = ?"
        params.append(method)

    if projection:
        query += " AND projection = ?"
        params.append(projection)

    results = conn.execute(query, params).fetchall()
    conn.close()

    return results
