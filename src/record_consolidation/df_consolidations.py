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
    already_tried: set[str] = set(),
) -> pl.DataFrame:
    """
    Recursively consolidates fields within a DataFrame by filling null values,
    using the field with the least nulls as a lookup to the canonical.

    This function identifies the field with the least number of null values and uses it as a reference to fill
    null values in other fields. It ensures that no null values remain, otherwise,
    it recursively consolidates the DataFrame until no null values remain or all fields have been tried.

    NOTE: fields must have already been consolidated within themselves ("intra_field").
        If not, values will be propagated randomly (and likely incorrectly).

    Args:
        df (pl.DataFrame): The input DataFrame to be consolidated.
        confirm_input_was_intra_field_consolidated (bool): If true, will check that the input DataFrame was already consolidated intra-field and will raise an error if not. NOTE: this is an expensive operation, so default is False.
        already_tried (set[str]): A set of fields that have already been used as the basis for consolidation in previous recursive calls.

    Returns:
        pl.DataFrame: A new DataFrame with null values filled based on the field with the least nulls.

    Raises:
        ValueError: If the input DataFrame is not intra-field consolidated when required.
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
    # print(f"{null_counts=}")
    print(f"{already_tried=}")
    if all(count == 0 for count in null_counts.values()):
        print("df has been inter-consolidated; returning.")
        return df
    if set(df.columns) == already_tried:
        print(
            f"df has been inter-consolidated to the maximum extent possible; returning.\n{null_counts=}\n{df.shape=}"
        )
        return df

    least_null_field: str = min(null_counts, key=null_counts.get)  # type: ignore
    if least_null_field in already_tried:
        least_null_field = min(
            {k: v for k, v in null_counts.items() if k not in already_tried},
            key=null_counts.get,  # type: ignore
        )  # get next least null field as basis of consolidation

    lf: pl.LazyFrame = df.lazy()
    for field in df.columns:
        if null_counts[field] > 0:
            lf = lf.with_columns(
                pl.col(field).fill_null(strategy="max").over(pl.col(least_null_field))
            )
    output = lf.collect()

    already_tried.add(least_null_field)
    # TODO: qa uniqueness of output vals? But output vals will only be unique if consolidation perfectly takes replaces all nulls, which won't always happen
    return _consolidate_inter_field(
        output, confirm_input_was_intra_field_consolidated, already_tried=already_tried
    )


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
