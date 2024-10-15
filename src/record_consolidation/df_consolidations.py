from typing import Any, Literal, cast

import polars as pl

from record_consolidation._typing import GraphGenerator, SubGraphPostProcessorFnc
from record_consolidation.graphs import (
    extract_consolidation_mapping_from_subgraphs,
    extract_normalized_atomic,
    unconsolidated_df_to_subgraphs,
)
from record_consolidation.utils.polars_df import (
    assign_columns_if_missing,
    extract_null_counts,
)


def _consolidate_intra_field(
    df: pl.DataFrame, connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None
) -> pl.DataFrame:
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
    df_as_subgraphs: GraphGenerator = unconsolidated_df_to_subgraphs(
        df,
        weight_edges=False,
        connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
    )
    consolidation_mapping: dict[str, dict[str, Any]] = (
        extract_consolidation_mapping_from_subgraphs(df_as_subgraphs)
    )
    return df.with_columns(
        pl.col(field).replace_strict(mapping, default=None)
        for field, mapping in consolidation_mapping.items()
    )


def normalize_subset(
    df: pl.DataFrame,
    connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None,
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
        .pipe(
            _consolidate_intra_field,
            connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
        )
        .with_columns(pl.lit(None).alias("canonical_row"))  # .cast(pl.Struct)
    )

    ##### ADD `canonical_row` STRUCT COL TO `subset`

    # Atomize subset if not passed as arg
    if atomized_subset is None:
        atomized_subset = cast(
            pl.DataFrame,
            extract_normalized_atomic(
                subset,
                connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
            ),
        )

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
