from typing import Any, Callable, Literal
from warnings import warn

import polars as pl
import polars.selectors as cs

from record_consolidation._typing import SubGraphPostProcessorFnc
from record_consolidation.graphs import (
    atomize_records,
    extract_consolidation_mapping_from_subgraphs,
    unconsolidated_df_to_subgraphs,
)
from record_consolidation.utils.polars_df import assign_id


def _consolidate_intra_field(
    df: pl.DataFrame,
    connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None,
    pre_processing_fnc: Callable[[pl.DataFrame], pl.DataFrame] | None,
) -> pl.DataFrame:
    """
    Consolidates fields within a DataFrame by mapping each field's values to their canonical values.

    This function converts the DataFrame into a graph where each node represents a unique value and edges represent
    occurences of the values being observed in the same row(s). It then extracts a consolidation mapping from the graph using subgraph analysis,
    and applies this mapping to the DataFrame, replacing each value with its canonical value.

    If a value does not have a corresponding canonical value in the mapping, it remains unchanged.


    Args:
        df (pl.DataFrame): The input DataFrame to be consolidated.

    Returns:
        pl.DataFrame: A new DataFrame with values consolidated within each field.
    """
    consolidation_mapping: dict[str, dict[str, Any]] = (
        extract_consolidation_mapping_from_subgraphs(
            unconsolidated_df_to_subgraphs(
                df,
                connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
                pre_processing_fnc=pre_processing_fnc,
            )
        )
    )
    return df.with_columns(
        pl.col(field).replace_strict(mapping, default=pl.col(field))
        for field, mapping in consolidation_mapping.items()
    )


def check_no_new_nulls(
    old: pl.DataFrame,
    new: pl.DataFrame,
    cols: list[str],
    on_error: Literal["raise", "warn"],
) -> None:
    """Confirm that no new nulls have been added to the data"""
    for col in cols:
        new_null_idxs: pl.Series = new[col].is_null()
        if (
            old.select(pl.col(col))
            .filter(new_null_idxs)
            .to_series()
            .is_not_null()
            .any()
        ):
            warning_str = f"New nulls added -- detected in {col}."
            if on_error == "raise":
                raise ValueError(warning_str)
            else:
                warn(warning_str)


def qa_normalized_subset(
    og: pl.DataFrame,
    normed: pl.DataFrame,
    atomized_subset: pl.DataFrame,
    normed_cols: list[str],
    non_null_fields: list[str],
) -> None:
    ## shape
    if normed.shape != og.shape:
        raise ValueError(f"{normed.shape} != {og.shape}\n{normed}")

    ## no new nulls
    check_no_new_nulls(old=og, new=normed, cols=normed_cols, on_error="warn")

    ## null counts
    new_null_count: int = (
        normed.select(pl.col(normed_cols).is_null().sum()).sum_horizontal().item()
    )
    og_null_count: int = (
        og.select(pl.col(normed_cols).is_null().sum()).sum_horizontal().item()
    )
    if new_null_count > og_null_count:
        if atomized_subset is not None:
            atomized_subset_nulls: int = (
                atomized_subset.select(pl.all().is_null().sum()).sum_horizontal().item()
            )
        raise ValueError(
            f"This function has added nulls:{og_null_count=} --> {new_null_count=}   ({atomized_subset_nulls=})"
        )

    ## non-nullable fields
    for field in non_null_fields:
        null_rows: pl.DataFrame = normed.filter(pl.col(field).is_null())
        if not null_rows.is_empty():
            warning_msg = (
                f"Null values detected in non-nullable column {field}:\n{null_rows}"
            )
            warn(warning_msg)
            # raise ValueError(warning_msg)


def _replace_vals_w_canons_via_atomized(
    df: pl.DataFrame,
    subset: pl.DataFrame,
    atomized_subset: pl.DataFrame,
    subset_selector: list[str],
    non_null_fields: list[str],
) -> pl.DataFrame:
    # create n^2 columns and then coalesce them
    to_return_precursor: pl.DataFrame = subset
    for col in to_return_precursor.columns:
        to_return_precursor = to_return_precursor.join(
            atomized_subset,
            how="left",
            on=col,
            suffix=f"_{col}",
            # validate="m:1",
        )

    # THIS REARRAGENMENT IS IMPORTANT + INTENTIONAL:
    #   if we don't do this, the original (non-canonicalized) values are left-most and thus end up in the normalized outpu
    #   because coalescing happens from left to right.
    to_return_precursor = pl.concat(
        [
            to_return_precursor.select(
                pl.col(
                    col
                    for col in to_return_precursor.columns
                    if col not in subset_selector
                )
            ),
            to_return_precursor.select(pl.col(subset_selector)),
        ],
        how="horizontal",
    )

    # coalesce
    to_return_subset_only: pl.DataFrame = pl.DataFrame(
        {
            col: to_return_precursor.select(pl.coalesce(cs.starts_with(col)))
            for col in subset.columns
        }
    )
    to_return = pl.concat(
        [to_return_subset_only, df.drop(to_return_subset_only.columns)],
        how="horizontal",
    )
    qa_normalized_subset(
        og=df,
        normed=to_return,
        atomized_subset=atomized_subset,
        normed_cols=subset_selector,
        non_null_fields=non_null_fields,
    )

    return to_return


def normalize_subset(
    df: pl.DataFrame,
    cols_to_normalize: list[str] | Literal["all"],
    id_colname: str | None,
    connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None = None,
    pre_processing_fnc_before_clustering: (
        Callable[[pl.DataFrame], pl.DataFrame] | None
    ) = None,
    atomized_subset: pl.DataFrame | None = None,
    non_null_fields: list[str] = [],
    consolidate_twice: bool = False,  # NOTE: this is not performing well as of Nov 1, 2024
) -> pl.DataFrame:
    # TODO: further QA and improvement of the (mapping canonicals into DF) process
    # 1) Nulls remain where they shouldn't (e.g., null CUSIPs next to canonicalized issuer_names)
    # 2) Possible spurious consolidation with `consolidate_twice`
    """
    Normalizes a subset of columns in the DataFrame using atomic joins and post-processing functions.

    This function takes a DataFrame and a list of columns to normalize. It performs pre-processing on the specified subset of columns,
    extracts an atomized subset of canonical values (via `extract_normalized_atomic`), and joins them back into the DataFrame to replace the original values with their normalized counterparts.
    If "all" is specified, all columns in the DataFrame are normalized -- this is *not* recommended.

    The normalization process involves:
    1. Pre-processing the subset of columns (if a function is provided).
    2. Consolidating values within each field to their canonical forms.
    3. Extracting canonical values from a graph-based "atomized_subset" or using a pre-provided "atomized_subset".
    4. Mapping (joining) the "atomized_subset" into the final result by over-joining and then coalescing the non-null values.
    4.5. ~~Repeating (4), but using the output of (4) as input.~~
        - ~~This takes care of straggler nulls, although a better fix should be applied.~~
    5. Ensuring no new null values are introduced and maintaining non-null constraints in specified columns.

    Args:
        df (pl.DataFrame): The input DataFrame containing the data to be normalized.
        cols_to_normalize (list[str] | Literal["all"]): The columns to normalize. If "all", all columns in the DataFrame are normalized.
        id_colname (str | None): Name of the column to assign unique IDs to, if specified.
        connected_subgraphs_postprocessor (SubGraphPostProcessorFnc | None): A function to process subgraphs during the normalization process.
        pre_processing_fnc_before_clustering (Callable[[pl.DataFrame], pl.DataFrame] | None): Function applied to preprocess the subset of columns to normalize, defaulting to `remove_string_nulls_and_uppercase`.
        cols_to_normalize (list[str] | Literal["all"]): The columns to normalize. If "all", all columns in the DataFrame are normalized.
        atomized_subset (pl.LazyFrame | pl.DataFrame | None): Optionally passed precomputed canonical values; if None, the function will compute them.
        non_null_fields (list[str]): Fields that are expected to be non-null. Raises warnings if null values are detected in these fields after processing.

    Returns:
        pl.DataFrame: A new DataFrame with the specified columns replaced by their normalized values.

    Raises:
        ValueError: If the shape of the returned DataFrame does not match the original DataFrame.
        ValueError: If the function introduces new null values into the non-nullable fields.
        Warning: If null values are detected in non-nullable fields without raising an exception.
    Notes:
        - For best performance, limit `cols_to_normalize` to a targeted subset rather than "all."
    """
    # TODO: add `atomized_subset` as an optional arg

    if pre_processing_fnc_before_clustering is None:
        warn("`pre_processing_fnc_before_clustering` is None")
    if connected_subgraphs_postprocessor is None:
        warn("`connected_subgraphs_postprocessor` is None")

    if cols_to_normalize == "all":
        warn(
            "`cols_to_normalize` = 'all' is unusally incorrect usage, and can lead to incorrect output. If this function is taking a long time, you probably meant to only normalize a subset."
        )
        subset_selector: list[str] = df.columns
    else:
        subset_selector = cols_to_normalize

    subset: pl.DataFrame = df.select(pl.col(subset_selector))
    if pre_processing_fnc_before_clustering:
        subset = subset.pipe(pre_processing_fnc_before_clustering)
    subset = _consolidate_intra_field(
        subset,
        connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
        pre_processing_fnc=pre_processing_fnc_before_clustering,
    )

    # TODO: sloppy typing
    if atomized_subset is None:
        atomized_subset = atomize_records(
            subset,
            connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
            pre_processing_fnc=None,
        )

    to_return = _replace_vals_w_canons_via_atomized(
        df=df,
        subset=subset,
        atomized_subset=atomized_subset,
        subset_selector=subset_selector,
        non_null_fields=non_null_fields,
    )

    # Consolidate again to take care of straggler nulls (not sure why they're here...).
    # e.g., "MICROSOFT CORPORATION" w/ null CUSIP
    # TODO: this is hacky, and I'm not certain that running `_replace_vals_w_canons_via_atomized` twice is guaranteed to fix things. plus, this reflects a lack of control over the process
    if consolidate_twice:
        raise NotImplementedError("Not supported due to false-positive linkings.")
        to_return = _replace_vals_w_canons_via_atomized(
            df=to_return,
            subset=subset,
            atomized_subset=atomize_records(
                to_return.select(pl.col(subset_selector)),
                connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
                pre_processing_fnc=pre_processing_fnc_before_clustering,  # probably redundant because it's already been applied, but probably also doesn't hurt
            ),
            subset_selector=subset_selector,
            non_null_fields=non_null_fields,
        )

    # final qa here just to be sure
    qa_normalized_subset(
        og=df,
        normed=to_return,
        atomized_subset=atomized_subset,
        normed_cols=subset_selector,
        non_null_fields=non_null_fields,
    )

    if id_colname is not None:
        to_return = to_return.pipe(
            assign_id, name=id_colname, constituent_cols=subset_selector
        )

    return to_return
