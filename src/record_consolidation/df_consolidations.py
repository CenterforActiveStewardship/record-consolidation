from typing import Any, Literal
from warnings import warn

import networkx as nx
import polars as pl

from record_consolidation.graphs import (
    _create_field_val_to_canonical_lookup,
    _extract_canonicals_from_subgraph,
    extract_consolidation_mapping_from_graph,
    unconsolidated_df_to_graph,
)
from record_consolidation.utils import (
    assign_columns_if_missing,
    extract_connected_subgraphs,
    extract_null_counts,
)


def _consolidate_intra_field(df: pl.DataFrame) -> pl.DataFrame:
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


def __consolidate_inter_field(
    df: pl.DataFrame,
    confirm_input_was_intra_field_consolidated: bool = False,
    already_tried: set[str] = set(),
    verbose: bool = False,
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
    if verbose:
        print("\n")
        print(f"{already_tried=}")
        print(f"{null_counts=}")
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
    return __consolidate_inter_field(
        output, confirm_input_was_intra_field_consolidated, already_tried=already_tried
    )


def _consolidate_normalized_table_deprecated(
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
    consolidated_intra_field: pl.DataFrame = df.pipe(_consolidate_intra_field)
    match depth:
        case "intra_field":
            return consolidated_intra_field.unique()
        case "intra_and_inter_field":
            return consolidated_intra_field.pipe(__consolidate_inter_field).unique()
        case _:
            raise ValueError(depth)


def extract_normalized_atomic(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts a normalized atomic DataFrame from the input DataFrame.

    This function converts the input DataFrame into a graph, processes its connected subgraphs to extract
    canonical values for each field, and then constructs a new DataFrame from these canonical values.

    Args:
        df (pl.DataFrame): The input DataFrame containing entity attributes.

    Returns:
        pl.DataFrame: A new DataFrame where each row represents a set of canonical values for the fields
                      present in the input DataFrame.
    """
    g: nx.Graph = unconsolidated_df_to_graph(df)
    df_precursor: list[dict[str, Any]] = []
    for subg in extract_connected_subgraphs(g):
        df_precursor.append(_extract_canonicals_from_subgraph(subg, "max_n"))
    return pl.DataFrame(df_precursor)


def _normalize_subset_deprecated(
    df: pl.DataFrame,
    cols_to_normalize: list[str] | Literal["all"] = "all",
    leave_potential_dupes_in_output: bool = True,
) -> pl.DataFrame:
    """
    Replaces a subset of columns in the DataFrame with their normalized (canonical) values.

    Normalization is the process of converting data to a standard or consistent form. This function takes a DataFrame
    and a list of columns to normalize. It creates a lookup table for the normalized values of the specified columns
    and replaces the original values in these columns with their normalized counterparts. If "all" is specified, all columns
    in the DataFrame are normalized.

    The normalization process involves:
    1. Creating a graph where each unique value in the DataFrame is represented as a node, and edges represent co-occurrences
       of these values within the same row.
    2. Extracting connected subgraphs from this graph and determining the normalized/canonical value for each field within these subgraphs.
    3. Constructing a nested dictionary where each field maps to another dictionary that maps each value to its normalized values
       across all fields.
    4. Replacing the original values in the DataFrame with their normalized counterparts based on the lookup table.

    Args:
        df (pl.DataFrame): The input DataFrame containing the data to be normalized.
        cols_to_normalize (list[str] | Literal["all"] = "all"): The columns to be normalized. If "all", all columns in the DataFrame are normalized.
        leave_potential_dupes_in_output (bool): If True, potential duplicate rows are left in the output. If False, duplicate rows are removed.


    Returns:
        pl.DataFrame: A new DataFrame with the specified columns replaced by their normalized values.

    Raises:
        KeyError: If a value in the DataFrame does not have a corresponding normalized value in the lookup table.
    """
    warn(
        "DEPRECATED: use `normalize_subset_via_joins` instead (there's an issue with this function that I can't recall this second :<{ )."
    )
    if cols_to_normalize == "all":
        subset_selector: list[str] = df.columns
    else:
        subset_selector = cols_to_normalize
    subset: pl.DataFrame = df.select(pl.col(subset_selector))

    field_val_to_canonical_lookup: dict[str, dict[Any, dict[str, Any]]] = (
        _create_field_val_to_canonical_lookup(subset)
    )

    normalized_subset_precursor: list[dict[str, Any]] = []
    for row in subset.rows(named=True):
        for field, value in row.items():
            if value is not None:
                normalized_subset_precursor.append(
                    field_val_to_canonical_lookup[field][value]
                )
                break

    normalized_subset: pl.DataFrame = pl.DataFrame(normalized_subset_precursor)

    if cols_to_normalize == "all":
        return normalized_subset

    normalized_subset = assign_columns_if_missing(
        assign_to=normalized_subset, assign_from=df, cols=subset_selector
    )

    reunited_subsets: pl.DataFrame = pl.concat(
        [normalized_subset, df.select(pl.exclude(subset_selector))], how="horizontal"
    ).select(
        pl.col(df.columns)
    )  # reorder cols to original

    if reunited_subsets.shape != df.shape:
        raise ValueError(f"{reunited_subsets.shape=} != {df.shape=}")
    if not reunited_subsets.select(pl.exclude(subset_selector)).equals(
        df.select(pl.exclude(subset_selector))
    ):
        raise ValueError("Columns excluded from normalization should NOT have changed.")

    if not leave_potential_dupes_in_output:
        reunited_subsets = reunited_subsets.unique()

    return reunited_subsets


def normalize_subset(
    df: pl.DataFrame,
    cols_to_normalize: list[str] | Literal["all"] = "all",
    leave_potential_dupes_in_output: bool = True,
) -> pl.DataFrame:
    """
    Normalizes a subset of columns in the DataFrame using join operations to replace values with their canonical counterparts.

    This function takes a DataFrame and a list of columns to normalize. It performs intra-field consolidation on the specified columns,
    extracts canonical values, and uses join operations to replace the original values with their canonical counterparts. If "all" is specified,
    all columns in the DataFrame are normalized.

    The normalization process involves:
    1. Performing intra-field consolidation on the specified columns.
    2. Extracting canonical values for each field within the consolidated columns.
    3. Using join operations to replace the original values in the DataFrame with their canonical counterparts.
    4. Ensuring that the columns excluded from normalization remain unchanged.

    Args:
        df (pl.DataFrame): The input DataFrame containing the data to be normalized.
        cols_to_normalize (list[str] | Literal["all"] = "all"): The columns to be normalized. If "all", all columns in the DataFrame are normalized.
        leave_potential_dupes_in_output (bool): If True, potential duplicate rows are left in the output. If False, duplicate rows are removed.

    Returns:
        pl.DataFrame: A new DataFrame with the specified columns replaced by their normalized values.

    Raises:
        ValueError: If the shape of the reunited subsets does not match the original DataFrame.
        ValueError: If columns excluded from normalization have changed.
    """
    # TODO: add `atomized_subset` as an optional arg

    if cols_to_normalize == "all":
        subset_selector: list[str] = df.columns
    else:
        subset_selector = cols_to_normalize
    subset: pl.DataFrame = (
        df.select(pl.col(subset_selector))
        .pipe(_consolidate_intra_field)
        .with_columns(pl.lit(None).alias("canonical_row"))  # .cast(pl.Struct)
    )

    ##### ADD `canonical_row` STRUCT COL TO `subset`
    subset_null_counts: dict[str, int] = extract_null_counts(subset)

    atomized_subset = extract_normalized_atomic(subset).with_columns(
        pl.struct(pl.all()).alias("canonical_row")
    )
    already_tried: set[str] = set(["canonical_row"])
    while subset.select(
        pl.col("canonical_row").is_null().any()
    ).item() and already_tried != set(subset.columns):
        subset_null_counts = extract_null_counts(subset)
        least_null_col: str = min({k: v for k, v in subset_null_counts.items() if k not in already_tried}, key=subset_null_counts.get)  # type: ignore
        already_tried.update(set([least_null_col]))
        subset = (
            subset.join(
                atomized_subset.select(pl.col([least_null_col, "canonical_row"])),
                how="left",
                on=least_null_col,
                # validate="m:1",
            )
            .with_columns(
                pl.col("canonical_row").fill_null(pl.col("canonical_row_right"))
            )
            .drop("canonical_row_right")
        )

    ##### /ADD `canonical_row` STRUCT COL TO `subset`
    subset = (
        subset.select(pl.col("canonical_row"))
        .unnest("canonical_row")
        .pipe(assign_columns_if_missing, assign_from=df, cols=subset_selector)
    )

    reunited_subsets: pl.DataFrame = pl.concat(
        [subset, df.select(pl.exclude(subset_selector))], how="horizontal"
    ).select(
        pl.col(df.columns)
    )  # reorder cols to original

    if reunited_subsets.shape != df.shape:
        raise ValueError(f"{reunited_subsets.shape=} != {df.shape=}")
    if not reunited_subsets.select(pl.exclude(subset_selector)).equals(
        df.select(pl.exclude(subset_selector))
    ):
        raise ValueError("Columns excluded from normalization should NOT have changed.")

    if not leave_potential_dupes_in_output:
        reunited_subsets = reunited_subsets.unique()

    return reunited_subsets
