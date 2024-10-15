from typing import Any, Literal, cast

import networkx as nx
import polars as pl

from record_consolidation.graphs import (
    _extract_canonicals_from_subgraph,  # TODO: shouldn't be importing private class
)
from record_consolidation.graphs import (
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


def normalize_subset(
    df: pl.DataFrame,
    cols_to_normalize: list[str] | Literal["all"] = "all",
    leave_potential_dupes_in_output: bool = True,
    atomized_subset: pl.DataFrame | None = None,
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
        atomized_subset (Optional[pl.DataFrame]): The atomized subset containing all unique/canonical/normalized relationships which we want to bring
            into `df`. If None, this will be automatically extracted from the `subset` (which should lead to identical `atomized_subset_w_canonical_row` variables,
            and therefore identical output).

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

    # Atomize subset if not passed as arg
    if atomized_subset is None:
        atomized_subset = cast(pl.DataFrame, extract_normalized_atomic(subset))

    atomized_subset_w_canonical_row = atomized_subset.with_columns(
        pl.struct(pl.all()).alias("canonical_row")
    )

    #
    subset_null_counts: dict[str, int] = extract_null_counts(subset)
    already_tried: set[str] = set(["canonical_row"])
    while subset.select(
        pl.col("canonical_row").is_null().any()
    ).item() and already_tried != set(subset.columns):
        subset_null_counts = extract_null_counts(subset)
        least_null_col: str = min({k: v for k, v in subset_null_counts.items() if k not in already_tried}, key=subset_null_counts.get)  # type: ignore
        already_tried.update(set([least_null_col]))
        subset = (
            subset.join(
                atomized_subset_w_canonical_row.select(
                    pl.col([least_null_col, "canonical_row"])
                ),
                how="left",
                on=least_null_col,
                # validate="m:1", # TODO
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
