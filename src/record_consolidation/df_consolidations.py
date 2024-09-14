from typing import Any, Literal

import networkx as nx
import polars as pl

from record_consolidation.graphs import (
    extract_consolidation_mapping_from_graph,
    unconsolidated_df_to_graph,
)


def consolidate_intra_field(df: pl.DataFrame) -> pl.DataFrame:
    """
    Consolidates fields within a DataFrame by mapping each field's values to their canonical values.

    This function converts the DataFrame into a graph where each node represents a unique value and edges represent
    occurences of the values being observed in the same row(s). It then extracts a consolidation mapping from the graph using subgraph analysis,
    and applies this mapping to the DataFrame, replacing each value with its canonical value.

    Args:
        df (pl.DataFrame): The input DataFrame to be consolidated.

    Returns:
        pl.DataFrame: A new DataFrame with values consolidated within each field.
    """
    df_as_graph: nx.Graph = unconsolidated_df_to_graph(df, weight_edges=False)
    consolidation_mapping: dict[str, dict[str, Any]] = (
        extract_consolidation_mapping_from_graph(df_as_graph)
    )
    return df.with_columns(
        pl.col(field).replace_strict(mapping, default=None)
        for field, mapping in consolidation_mapping.items()
    )


def _consolidate_inter_field(
    df: pl.DataFrame,
    confirm_input_was_intra_field_consolidated: bool = False,
) -> pl.DataFrame:
    """
    Consolidates fields within a DataFrame by filling null values based on the field with the least nulls.

    This function identifies the field with the least number of null values and uses it as a reference to fill
    null values in other fields. It ensures that the field with the least nulls has no null values, otherwise,
    it raises a ValueError.

    NOTE: fields must have already been consolidated within themselves ("intra_field"). If not, values will be propagated randomly (and likely incorrectly).

    Args:
        df (pl.DataFrame): The input DataFrame to be consolidated.
        confirm_input_was_intra_field_consolidated: bool = True: If true, will check that the input DataFrame was already consolidated intra-field and will raise an error if not.
    Returns:
        pl.DataFrame: A new DataFrame with null values filled based on the field with the least nulls.

    Raises:
        ValueError: If all fields have some null values, making it unsafe to consolidate with this logic.
    """
    if confirm_input_was_intra_field_consolidated:
        # Create a consolidation mapping from the DataFrame
        df_as_graph: nx.Graph = unconsolidated_df_to_graph(df, weight_edges=False)
        consolidation_mapping: dict[str, dict[str, Any]] = (
            extract_consolidation_mapping_from_graph(df_as_graph)
        )
        # Apply the consolidation mapping to the DataFrame
        consolidated_df: pl.DataFrame = df.with_columns(
            pl.col(field).replace_strict(mapping)
            for field, mapping in consolidation_mapping.items()
        )
        # Check if the DataFrame remains unchanged
        if not df.equals(consolidated_df):
            raise ValueError(
                "The input DataFrame does not appear to be intra-field consolidated."
            )

    null_counts: dict[str, int] = df.select(pl.all().is_null().sum()).to_dicts()[0]
    lf: pl.LazyFrame = df.lazy()
    least_null_field: str = min(null_counts, key=null_counts.get)  # type: ignore
    if null_counts[least_null_field] != 0:
        raise ValueError(
            "All fields have some nulls - not safe to consolidate with this logic."
        )
    for field in df.columns:
        if null_counts[field] > 0:
            lf = lf.with_columns(
                pl.col(field).fill_null(strategy="max").over(pl.col(least_null_field))
            )

    return lf.collect()


def consolidate_normalized_table(
    df: pl.DataFrame,
    depth: Literal["intra_field", "intra_and_inter_field"],
) -> pl.DataFrame:
    """
    Consolidates a normalized table by either intra-field or both intra-field and inter-field consolidation.

    This function first performs intra-field consolidation by mapping each field's values to their canonical values (isolated - via subgraph analysis - within each entity).
    Optionally, it can also perform inter-field consolidation, filling null values based for each entity.

    Args:
        df (pl.DataFrame): The input DataFrame to be consolidated.
        depth (Literal["intra_field", "intra_and_inter_field"]): The depth of consolidation to perform.
            - "intra_field": Only perform intra-field consolidation.
            - "intra_and_inter_field": Perform both intra-field and inter-field consolidation.

    Returns:
        pl.DataFrame: A consolidated DataFrame with unique rows.

    Raises:
        ValueError: If an invalid depth is provided."""
    consolidated_intra_field: pl.DataFrame = df.pipe(consolidate_intra_field)
    match depth:
        case "intra_field":
            return consolidated_intra_field.unique()
        case "intra_and_inter_field":
            return consolidated_intra_field.pipe(_consolidate_inter_field).unique()
        case _:
            raise ValueError(depth)
