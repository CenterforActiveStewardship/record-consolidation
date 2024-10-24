from typing import Any, Callable, Literal
from warnings import warn

import polars as pl
import polars.selectors as cs

from record_consolidation._typing import GraphGenerator, SubGraphPostProcessorFnc
from record_consolidation.graphs import (
    extract_consolidation_mapping_from_subgraphs,
    extract_normalized_atomic,
    unconsolidated_df_to_subgraphs,
)
from record_consolidation.utils.polars_df import remove_string_nulls_and_uppercase


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
        pl.col(field).replace_strict(
            mapping, default=pl.col(field)
        )  # NOTE: THIS IS A BIG CHANGE!!!!! ( default=None ----> default=pl.col(field) )
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


def normalize_subset(
    df: pl.DataFrame,
    connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None,
    pre_processing_fnc_before_clustering: (
        Callable[[pl.DataFrame], pl.DataFrame] | None
    ) = remove_string_nulls_and_uppercase,
    cols_to_normalize: list[str] | Literal["all"] = "all",
    atomized_subset: pl.DataFrame | None = None,
    non_null_fields: list[str] = [],
) -> pl.DataFrame:
    """
     Normalizes a subset of columns in the DataFrame using atomic joins and post-processing functions.

     This function takes a DataFrame and a list of columns to normalize. It performs pre-processing on the specified subset of columns,
     extracts an atomized subset of canonical values (via `extract_normalized_atomic`), and joins them back into the DataFrame to replace the original values with their normalized counterparts.
     If "all" is specified, all columns in the DataFrame are normalized -- this is *not* recommended.

    The normalization process involves:
     1. Pre-processing the subset of columns (if a function is provided).
     2. Extracting canonical values from a graph-based "atomized_subset" or using a pre-provided "atomized_subset".
     3. Mapping (joining) the "atomized_subset" into the final result by over-joining and then coalescing the non-null values.
     4. Ensuring no new null values are introduced and maintaining non-null constraints in specified columns.

     Args:
         df (pl.DataFrame): The input DataFrame containing the data to be normalized.
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
    """
    # TODO: add `atomized_subset` as an optional arg

    if cols_to_normalize == "all":
        warn(
            "`cols_to_normalize` = 'All' is unusally incorrect usage, and can lead to incorrect output. If this function is taking a long time, you probably meant to only normalize a subset."
        )
        subset_selector: list[str] = df.columns
    else:
        subset_selector = cols_to_normalize

    subset: pl.DataFrame = df.select(pl.col(subset_selector))
    if pre_processing_fnc_before_clustering:
        subset = subset.pipe(pre_processing_fnc_before_clustering)

    # TODO: sloppy typing
    if atomized_subset is None:
        atomized_subset = extract_normalized_atomic(
            subset,
            connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
            pre_processing_fnc=None,
        )

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

    # THIS IS IMPORTANT + INTENTIONAL: if we don't do this, the original (non-canonicalized) values are left-most and thus remain after the coalescing step
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

    # to_return_precursor.with_columns().drop(
    #     subset_selector
    # )

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
    # to_return = to_return_precursor.select(
    #     pl.coalesce(cs.starts_with(col)) for col in subset.columns
    # )

    # qa
    ## shape
    if to_return.shape != df.shape:
        raise ValueError(f"{to_return.shape} != {df.shape}\n{to_return}")

    ## no new nulls
    check_no_new_nulls(old=df, new=to_return, cols=subset_selector, on_error="warn")

    ## null counts
    new_null_count: int = (
        to_return.select(pl.col(subset_selector).is_null().sum())
        .sum_horizontal()
        .item()
    )
    og_null_count: int = (
        df.select(pl.col(subset_selector).is_null().sum()).sum_horizontal().item()
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
        null_rows: pl.DataFrame = to_return.filter(pl.col(field).is_null())
        if not null_rows.is_empty():
            warning_msg = (
                f"Null values detected in non-nullable column {field}:\n{null_rows}"
            )
            warn(warning_msg)
            # raise ValueError(warning_msg)

    return to_return


# def normalize_subset(
#     df: pl.DataFrame,
#     connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None,
#     cols_to_normalize: (
#         list[str] | Literal["all"]
#     ),  # TODO: convert to pl.Expr/selector thing like "pl.col(...)" | "pl.all()" | ...
#     leave_potential_dupes_in_output: bool = True,
#     atomized_subset: pl.DataFrame | None = None,
#     _apply_consolidate_intra_field: bool = True,  # TODO: remove. There's a chance that this is where the Nulls are coming from (Oct 17, 2024)
#     _return_atomized_subset: bool = False,  # TODO: remove
#     _return_joined_subset: bool = False,  # TODO: remove
# ) -> pl.DataFrame:
#     """
#     Normalizes a subset of columns in the DataFrame using join operations to replace values with their canonical counterparts.

#     This function takes a DataFrame and a list of columns to normalize. It performs intra-field consolidation on the specified columns,
#     extracts canonical values, and uses join operations to replace the original values with their canonical counterparts. If "all" is specified,
#     all columns in the DataFrame are normalized.

#     The normalization process involves:
#     1. Performing intra-field consolidation on the specified columns.
#     2. Extracting canonical values for each field within the consolidated columns.
#     3. Using join operations to replace the original values in the DataFrame with their canonical counterparts.
#     4. Ensuring that the columns excluded from normalization remain unchanged.

#     Args:
#         df (pl.DataFrame): The input DataFrame containing the data to be normalized.
#         cols_to_normalize (list[str] | Literal["all"] = "all"): The columns to be normalized. If "all", all columns in the DataFrame are normalized.
#         leave_potential_dupes_in_output (bool): If True, potential duplicate rows are left in the output. If False, duplicate rows are removed.
#         atomized_subset (Optional[pl.DataFrame]): The atomized subset containing all unique/canonical/normalized relationships which we want to bring
#             into `df`. If None, this will be automatically extracted from the `subset` (which should lead to identical `atomized_subset_w_canonical_row` variables,
#             and therefore identical output).

#     Returns:
#         pl.DataFrame: A new DataFrame with the specified columns replaced by their normalized values.

#     Raises:
#         ValueError: If the shape of the reunited subsets does not match the original DataFrame.
#         ValueError: If columns excluded from normalization have changed.
#     """
#     # TODO: add `atomized_subset` as an optional arg

#     if cols_to_normalize == "all":
#         warn(
#             "`cols_to_normalize` = 'All' is unusally incorrect usage, and can lead to incorrect output. If this function is taking a long time, you probably meant to only normalize a subset."
#         )
#         subset_selector: list[str] = df.columns
#     else:
#         subset_selector = cols_to_normalize

#     # SUBSET
#     subset: pl.DataFrame = df.select(pl.col(subset_selector))
#     if _apply_consolidate_intra_field:
#         subset = subset.pipe(  # TODO: FIGURE OUT WHY WE'RE DOING THIS. ~~IT SEEMS TO PROPAGATE AVOIDABLE NULLS!!!!!~~ Ok actually it doesn't seem to propagate nulls; looks like `atomize_subset` is the issue
#             _consolidate_intra_field,
#             connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
#         )
#     subset = subset.select(pl.col(subset_selector)).with_columns(
#         pl.lit(None).alias("canonical_row")
#     )  # .cast(pl.Struct)
#     # / SUBSET

#     ##### ADD `canonical_row` STRUCT COL TO `subset`

#     # Atomize subset if not passed as arg
#     # NOTE: `atomized_subset` IS NOT CORRECTLY CREATED: THERE ARE **NEW** NULL ELEMENTS THAT WEREN'T PRESENT IN THE INPUT
#     if atomized_subset is None:
#         atomized_subset = cast(
#             pl.DataFrame,
#             extract_normalized_atomic(
#                 subset,
#                 connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
#             ),
#         )
#     if _return_atomized_subset:
#         return atomized_subset

#     atomized_subset_w_canonical_row = atomized_subset.with_columns(
#         pl.struct(pl.all()).alias("canonical_row")
#     )

#     #
#     subset_null_counts: dict[str, int] = extract_null_counts(subset)
#     already_tried: set[str] = set(["canonical_row"])
#     while subset.select(
#         pl.col("canonical_row").is_null().any()
#     ).item() and already_tried != set(subset.columns):
#         subset_null_counts = extract_null_counts(subset)
#         least_null_col: str = min({k: v for k, v in subset_null_counts.items() if k not in already_tried}, key=subset_null_counts.get)  # type: ignore
#         already_tried.update(set([least_null_col]))
#         subset = (
#             subset.join(
#                 atomized_subset_w_canonical_row.select(
#                     pl.col([least_null_col, "canonical_row"])
#                 ),
#                 how="left",
#                 on=least_null_col,
#                 # validate="m:1", # TODO
#             )
#             .with_columns(
#                 pl.col("canonical_row").fill_null(pl.col("canonical_row_right"))
#             )
#             .drop("canonical_row_right")
#         )

#     ##### /ADD `canonical_row` STRUCT COL TO `subset`
#     if _return_joined_subset:
#         return subset

#     subset = (
#         subset.select(pl.col("canonical_row"))
#         .unnest("canonical_row")
#         .pipe(assign_columns_if_missing, assign_from=df, cols=subset_selector)
#     )

#     reunited_subsets: pl.DataFrame = pl.concat(
#         [subset, df.select(pl.exclude(subset_selector))], how="horizontal"
#     ).select(
#         pl.col(df.columns)
#     )  # reorder cols to original

#     if reunited_subsets.shape != df.shape:
#         raise ValueError(f"{reunited_subsets.shape=} != {df.shape=}")
#     if not reunited_subsets.select(pl.exclude(subset_selector)).equals(
#         df.select(pl.exclude(subset_selector))
#     ):
#         raise ValueError("Columns excluded from normalization should NOT have changed.")

#     if not leave_potential_dupes_in_output:
#         reunited_subsets = reunited_subsets.unique()

#     return reunited_subsets


# # def normalize_subset2(
# #     df: pl.DataFrame,
# #     connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None,
# #     cols_to_normalize: list[str] | Literal["all"] = "all",
# #     leave_potential_dupes_in_output: bool = True,
# #     atomized_subset: pl.DataFrame | None = None,
# #     _apply_consolidate_intra_field: bool = True,  # TODO: remove. There's a chance that this is where the Nulls are coming from (Oct 17, 2024)
# #     _return_atomized_subset: bool = False,  # TODO: remove
# #     _return_joined_subset: bool = False,  # TODO: remove
# # ) -> pl.DataFrame:
# #     """
# #     Normalizes a subset of columns in the DataFrame using join operations to replace values with their canonical counterparts.

# #     This function takes a DataFrame and a list of columns to normalize. It performs intra-field consolidation on the specified columns,
# #     extracts canonical values, and uses join operations to replace the original values with their canonical counterparts. If "all" is specified,
# #     all columns in the DataFrame are normalized.

# #     The normalization process involves:
# #     1. Performing intra-field consolidation on the specified columns.
# #     2. Extracting canonical values for each field within the consolidated columns.
# #     3. Using join operations to replace the original values in the DataFrame with their canonical counterparts.
# #     4. Ensuring that the columns excluded from normalization remain unchanged.

# #     Args:
# #         df (pl.DataFrame): The input DataFrame containing the data to be normalized.
# #         cols_to_normalize (list[str] | Literal["all"] = "all"): The columns to be normalized. If "all", all columns in the DataFrame are normalized.
# #         leave_potential_dupes_in_output (bool): If True, potential duplicate rows are left in the output. If False, duplicate rows are removed.
# #         atomized_subset (Optional[pl.DataFrame]): The atomized subset containing all unique/canonical/normalized relationships which we want to bring
# #             into `df`. If None, this will be automatically extracted from the `subset` (which should lead to identical `atomized_subset_w_canonical_row` variables,
# #             and therefore identical output).

# #     Returns:
# #         pl.DataFrame: A new DataFrame with the specified columns replaced by their normalized values.

# #     Raises:
# #         ValueError: If the shape of the reunited subsets does not match the original DataFrame.
# #         ValueError: If columns excluded from normalization have changed.
# #     """
# #     # TODO: add `atomized_subset` as an optional arg

# #     if cols_to_normalize == "all":
# #         warn(
# #             "`cols_to_normalize` = 'All' is unusally incorrect usage, and can lead to incorrect output. If this function is taking a long time, you probably meant to only normalize a subset."
# #         )
# #         subset_selector: list[str] = df.columns
# #     else:
# #         subset_selector = cols_to_normalize

# #     # SUBSET
# #     subset: pl.DataFrame = df.select(pl.col(subset_selector))
# #     if _apply_consolidate_intra_field:
# #         subset = subset.pipe(  # TODO: FIGURE OUT WHY WE'RE DOING THIS. IT SEEMS TO PROPAGATE AVOIDABLE NULLS!!!!!
# #             _consolidate_intra_field,
# #             connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
# #         )
# #     subset = subset.select(pl.col(subset_selector)).with_columns(
# #         pl.lit(None).alias("canonical_row")
# #     )  # .cast(pl.Struct)
# #     # / SUBSET

# #     ##### ADD `canonical_row` STRUCT COL TO `subset`

# #     # Atomize subset if not passed as arg
# #     # NOTE: `atomized_subset` IS NOT CORRECTLY CREATED: THERE ARE **NEW** NULL ELEMENTS THAT WEREN'T PRESENT IN THE INPUT
# #     if atomized_subset is None:
# #         atomized_subset = cast(
# #             pl.DataFrame,
# #             extract_normalized_atomic(
# #                 subset,
# #                 connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
# #             ),
# #         )
# #     if _return_atomized_subset:
# #         return atomized_subset

# #     atomized_subset_w_canonical_row = atomized_subset.with_columns(
# #         pl.struct(pl.all()).alias("canonical_row")
# #     )

# #     #
# #     subset_null_counts: dict[str, int] = extract_null_counts(subset)
# #     already_tried: set[str] = set(["canonical_row"])
# #     while subset.select(
# #         pl.col("canonical_row").is_null().any()
# #     ).item() and already_tried != set(subset.columns):
# #         subset_null_counts = extract_null_counts(subset)
# #         least_null_col: str = min({k: v for k, v in subset_null_counts.items() if k not in already_tried}, key=subset_null_counts.get)  # type: ignore
# #         already_tried.update(set([least_null_col]))
# #         subset = (
# #             subset.join(
# #                 atomized_subset_w_canonical_row.select(
# #                     pl.col([least_null_col, "canonical_row"])
# #                 ),
# #                 how="left",
# #                 on=least_null_col,
# #                 # validate="m:1", # TODO
# #             )
# #             .with_columns(
# #                 pl.col("canonical_row").fill_null(pl.col("canonical_row_right"))
# #             )
# #             .drop("canonical_row_right")
# #         )

# #     ##### /ADD `canonical_row` STRUCT COL TO `subset`
# #     if _return_joined_subset:
# #         return subset

# #     subset = (
# #         subset.select(pl.col("canonical_row"))
# #         .unnest("canonical_row")
# #         .pipe(assign_columns_if_missing, assign_from=df, cols=subset_selector)
# #     )

# #     reunited_subsets: pl.DataFrame = pl.concat(
# #         [subset, df.select(pl.exclude(subset_selector))], how="horizontal"
# #     ).select(
# #         pl.col(df.columns)
# #     )  # reorder cols to original

# #     if reunited_subsets.shape != df.shape:
# #         raise ValueError(f"{reunited_subsets.shape=} != {df.shape=}")
# #     if not reunited_subsets.select(pl.exclude(subset_selector)).equals(
# #         df.select(pl.exclude(subset_selector))
# #     ):
# #         raise ValueError("Columns excluded from normalization should NOT have changed.")

# #     if not leave_potential_dupes_in_output:
# #         reunited_subsets = reunited_subsets.unique()

# #     return reunited_subsets


# def check_no_new_nulls(
#     old: pl.DataFrame,
#     new: pl.DataFrame,
#     cols: list[str],
#     on_error: Literal["raise", "warn"],
# ) -> None:
#     for col in cols:
#         new_null_idxs: pl.Series = new[col].is_null()
#         if old.select(pl.col(col)).filter(new_null_idxs).to_series().is_null().any():
#             warning_str = f"New nulls added -- detected in {col}."
#             if on_error == "raise":
#                 raise ValueError(warning_str)
#             else:
#                 warn(warning_str)


# def normalize_subset3(
#     df: pl.DataFrame,
#     connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None,
#     cols_to_normalize: list[str] | Literal["all"] = "all",
#     atomized_subset: pl.DataFrame | None = None,
#     non_null_fields: list[str] = [],
#     _apply_consolidate_intra_field: bool = True,
#     _return_atomized_subset: bool = False,  # TODO: remove
# ) -> pl.DataFrame:
#     """
#     Normalizes a subset of columns in the DataFrame using join operations to replace values with their canonical counterparts.

#     This function takes a DataFrame and a list of columns to normalize. It performs intra-field consolidation on the specified columns,
#     extracts canonical values, and uses join operations to replace the original values with their canonical counterparts. If "all" is specified,
#     all columns in the DataFrame are normalized.

#     The normalization process involves:
#     1. Performing intra-field consolidation on the specified columns.
#     2. Extracting canonical values for each field within the consolidated columns.
#     3. Using join operations to replace the original values in the DataFrame with their canonical counterparts.
#     4. Ensuring that the columns excluded from normalization remain unchanged.

#     Args:
#         df (pl.DataFrame): The input DataFrame containing the data to be normalized.
#         cols_to_normalize (list[str] | Literal["all"] = "all"): The columns to be normalized. If "all", all columns in the DataFrame are normalized.
#         leave_potential_dupes_in_output (bool): If True, potential duplicate rows are left in the output. If False, duplicate rows are removed.
#         atomized_subset (Optional[pl.DataFrame]): The atomized subset containing all unique/canonical/normalized relationships which we want to bring
#             into `df`. If None, this will be automatically extracted from the `subset` (which should lead to identical `atomized_subset_w_canonical_row` variables,
#             and therefore identical output).

#     Returns:
#         pl.DataFrame: A new DataFrame with the specified columns replaced by their normalized values.

#     Raises:
#         ValueError: If the shape of the reunited subsets does not match the original DataFrame.
#         ValueError: If columns excluded from normalization have changed.
#     """
#     # TODO: add `atomized_subset` as an optional arg

#     if cols_to_normalize == "all":
#         warn(
#             "`cols_to_normalize` = 'All' is unusally incorrect usage, and can lead to incorrect output. If this function is taking a long time, you probably meant to only normalize a subset."
#         )
#         subset_selector: list[str] = df.columns
#     else:
#         subset_selector = cols_to_normalize

#     # SUBSET
#     subset: pl.DataFrame = df.select(pl.col(subset_selector))
#     if _apply_consolidate_intra_field:
#         subset = subset.pipe(
#             _consolidate_intra_field,
#             connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
#         )
#     # / SUBSET

#     ##### ADD `canonical_row` STRUCT COL TO `subset`

#     # Atomize subset if not passed as arg
#     if atomized_subset is None:
#         atomized_subset = cast(
#             pl.DataFrame,
#             extract_normalized_atomic(
#                 subset,
#                 connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
#             ),
#         )
#     if _return_atomized_subset:
#         return atomized_subset

#     new_rows: list[dict[str, Any]] = []
#     for row in tqdm(subset.rows(named=True)):
#         for col, val in row.items():
#             corresp_canonical_rows: pl.DataFrame = atomized_subset.filter(
#                 pl.col(col) == pl.lit(val)
#             )
#             match corresp_canonical_rows.shape[0]:
#                 case 0:
#                     continue  # no  continue to next field, val
#                 case 1:
#                     new_row: dict[str, Any] = row.update(
#                         corresp_canonical_rows.rows(named=True)[0]
#                     )
#                     new_rows.append(new_row)
#                     break  # move on to next row
#                 case _:
#                     raise ValueError(
#                         f"Atomized subset is supposed to contain one row per canonical value (regardless of field):\n{corresp_canonical_rows}"
#                     )

#     normalized_subset: pl.DataFrame = pl.DataFrame(
#         new_rows, schema=atomized_subset.schema
#     )

#     to_return: pl.DataFrame = pl.concat(
#         [normalized_subset, df.drop(subset_selector)], how="horizontal"
#     )

#     # qa
#     ## shape
#     if to_return.shape != df.shape:
#         raise ValueError(f"{to_return.shape} != {df.shape}")

#     ## no new nulls
#     check_no_new_nulls(old=df, new=to_return, cols=subset_selector)

#     ## null counts
#     new_null_count: int = (
#         to_return.select(pl.col(subset_selector).is_null().sum())
#         .sum_horizontal()
#         .item()
#     )
#     og_null_count: int = (
#         df.select(pl.col(subset_selector).is_null().sum()).sum_horizontal().item()
#     )
#     print(f"{new_null_count=:_}, {og_null_count=:_}")
#     if new_null_count > og_null_count:
#         atomized_subset_nulls: int = (
#             atomized_subset.select(pl.all().is_null().sum()).sum_horizontal().item()
#         )
#         raise ValueError(
#             f"This function has added nulls:{og_null_count=} --> {new_null_count=}   ({atomized_subset_nulls=})"
#         )

#     ## non-nullable fields
#     for col in non_null_fields:
#         null_rows: pl.DataFrame = to_return.filter(pl.col(col).is_null())
#         if not null_rows.is_empty():
#             raise ValueError(
#                 f"Null values detected in non-nullable column {col}:\n{null_rows}"
#             )

#     return to_return


# def normalize_subset4(
#     df: pl.DataFrame,
#     connected_subgraphs_postprocessor: SubGraphPostProcessorFnc | None,
#     cols_to_normalize: list[str] | Literal["all"] = "all",
#     atomized_subset: pl.DataFrame | None = None,
#     non_null_fields: list[str] = [],
#     _apply_consolidate_intra_field: bool = True,
#     _return_atomized_subset: bool = False,  # TODO: remove
# ) -> pl.DataFrame:
#     """
#     Normalizes a subset of columns in the DataFrame using join operations to replace values with their canonical counterparts.

#     This function takes a DataFrame and a list of columns to normalize. It performs intra-field consolidation on the specified columns,
#     extracts canonical values, and uses join operations to replace the original values with their canonical counterparts. If "all" is specified,
#     all columns in the DataFrame are normalized.

#     The normalization process involves:
#     1. Performing intra-field consolidation on the specified columns.
#     2. Extracting canonical values for each field within the consolidated columns.
#     3. Using join operations to replace the original values in the DataFrame with their canonical counterparts.
#     4. Ensuring that the columns excluded from normalization remain unchanged.

#     Args:
#         df (pl.DataFrame): The input DataFrame containing the data to be normalized.
#         cols_to_normalize (list[str] | Literal["all"] = "all"): The columns to be normalized. If "all", all columns in the DataFrame are normalized.
#         leave_potential_dupes_in_output (bool): If True, potential duplicate rows are left in the output. If False, duplicate rows are removed.
#         atomized_subset (Optional[pl.DataFrame]): The atomized subset containing all unique/canonical/normalized relationships which we want to bring
#             into `df`. If None, this will be automatically extracted from the `subset` (which should lead to identical `atomized_subset_w_canonical_row` variables,
#             and therefore identical output).

#     Returns:
#         pl.DataFrame: A new DataFrame with the specified columns replaced by their normalized values.

#     Raises:
#         ValueError: If the shape of the reunited subsets does not match the original DataFrame.
#         ValueError: If columns excluded from normalization have changed.
#     """
#     # TODO: add `atomized_subset` as an optional arg

#     if cols_to_normalize == "all":
#         warn(
#             "`cols_to_normalize` = 'All' is unusally incorrect usage, and can lead to incorrect output. If this function is taking a long time, you probably meant to only normalize a subset."
#         )
#         subset_selector: list[str] = df.columns
#     else:
#         subset_selector = cols_to_normalize

#     subset: pl.DataFrame = df.select(pl.col(subset_selector))

#     canonical_consolidation_mapping: dict[str, dict[str, Any]] = (
#         extract_consolidation_mapping_from_subgraphs(
#             subgraphs=unconsolidated_df_to_subgraphs(
#                 subset,
#                 connected_subgraphs_postprocessor=connected_subgraphs_postprocessor,
#             )
#         )
#     )

#     def update_nulls(target: dict, overwriter: dict) -> dict:
#         """
#         Update the values in the target dictionary where the current value is None,
#         using values from the overwriter dictionary. If a key in the target has a
#         value of None, it will be updated with the corresponding value from the
#         overwriter. Keys in the overwriter that are not present in the target are ignored.

#         Args:
#             target (dict): The dictionary to update, where values might be None.
#             overwriter (dict): The dictionary containing potential replacement values.

#         Returns:
#             dict: The updated target dictionary.
#         """
#         for k, v in target.items():
#             if v is None:
#                 target[k] = overwriter.get(k, None)
#         return target

#     new_rows: list[dict[str, Any]] = []
#     for row in tqdm(subset.rows(named=True)):
#         for field, val in row.items():
#             canonical_row: dict[str, Any] = canonical_consolidation_mapping[field].get(
#                 val, None
#             )
#             if canonical_row is not None:
#                 canonical_row = update_nulls(
#                     target=canonical_row, overwriter=row
#                 )  # overwrite any nulls in the canonical row with non-nulls in the original row... this shouldn't occur but...
#                 new_rows.append(canonical_row)

#     normalized_subset: pl.DataFrame = pl.DataFrame(new_rows, schema=subset.schema)

#     to_return: pl.DataFrame = pl.concat(
#         [normalized_subset, df.drop(subset_selector)], how="horizontal"
#     )

#     # qa
#     ## shape
#     if to_return.shape != df.shape:
#         raise ValueError(f"{to_return.shape} != {df.shape}")

#     ## no new nulls
#     check_no_new_nulls(old=df, new=to_return, cols=subset_selector)

#     ## null counts
#     new_null_count: int = (
#         to_return.select(pl.col(subset_selector).is_null().sum())
#         .sum_horizontal()
#         .item()
#     )
#     og_null_count: int = (
#         df.select(pl.col(subset_selector).is_null().sum()).sum_horizontal().item()
#     )
#     print(f"{new_null_count=:_}, {og_null_count=:_}")
#     if new_null_count > og_null_count:
#         if atomized_subset is not None:
#             atomized_subset_nulls: int = (
#                 atomized_subset.select(pl.all().is_null().sum()).sum_horizontal().item()
#             )
#         raise ValueError(
#             f"This function has added nulls:{og_null_count=} --> {new_null_count=}   ({atomized_subset_nulls=})"
#         )

#     ## non-nullable fields
#     for field in non_null_fields:
#         null_rows: pl.DataFrame = to_return.filter(pl.col(field).is_null())
#         if not null_rows.is_empty():
#             raise ValueError(
#                 f"Null values detected in non-nullable column {field}:\n{null_rows}"
#             )

#     return to_return
