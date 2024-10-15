from pathlib import Path

import polars as pl
import pytest
from regression_tester import RegressionTestPackage

from record_consolidation.df_consolidations import (
    extract_normalized_atomic,
    normalize_subset,
)


@pytest.fixture()
def MSFTS_AND_AMZNS() -> pl.DataFrame:
    return pl.read_parquet("test_data/intra_field/msfts_and_amzns.parquet")


@pytest.fixture()
def MSFTS() -> pl.DataFrame:
    return pl.read_parquet("test_data/intra_field/msfts_and_amzns.parquet").filter(
        pl.col("issuer_name") == pl.lit("MICROSOFT CORPORATION")
    )


# @pytest.mark.parametrize(
#     "depth",
#     [
#         ("intra_field"),
#         ("intra_and_inter_field"),
#     ],
# )
# def test_normalization(MSFTS, depth) -> None:
#     root_path = Path("test_data") / depth
#     raw_input_path = root_path / "msfts_and_amzns.parquet"
#     reg_tester = RegressionTestPackage(
#         root_path=root_path,
#         extraction_fnc=lambda _: _consolidate_normalized_table_deprecated(
#             MSFTS, depth=depth
#         )
#         .unique()
#         .sort(pl.all()),
#         optional_raw_input_path=raw_input_path,  # have to put an extant path here
#     )
#     reg_tester.execute_regression_test()


def test_normalization_via_normalized_atomizer(MSFTS) -> None:
    """
    Should produce the desired output of `consolidate_normalized_table(df, depth="intra_and_inter_field")`!
    """
    root_path = Path("test_data") / "intra_and_inter_field"
    raw_input_path = root_path / "msfts_and_amzns.parquet"
    reg_tester = RegressionTestPackage(
        root_path=root_path,
        extraction_fnc=lambda x: extract_normalized_atomic(MSFTS)
        .unique()
        .sort(pl.all()),
        optional_raw_input_path=raw_input_path,  # have to put an extant path here
    )
    reg_tester.execute_regression_test()


def test_normalization_via_subset_normalizer(MSFTS) -> None:
    """
    Should produce the desired output of `consolidate_normalized_table(df, depth="intra_and_inter_field")` & extract_normalized_atomic
    """
    root_path = Path("test_data") / "intra_and_inter_field"
    raw_input_path = root_path / "msfts_and_amzns.parquet"
    reg_tester = RegressionTestPackage(
        root_path=root_path,
        extraction_fnc=lambda x: normalize_subset(MSFTS, "all").unique().sort(pl.all()),
        optional_raw_input_path=raw_input_path,  # have to put an extant path here
    )
    reg_tester.execute_regression_test()


# def test_deprecated_subset_normalizer() -> None:
#     root_path = Path("test_data/normalize_subset")
#     input_path = root_path / "input.parquet"
#     snapshot_path = root_path / "processed.parquet"

#     raw_input: pl.DataFrame = pl.read_parquet(input_path)
#     locally_processed: pl.DataFrame = _normalize_subset_deprecated(
#         raw_input, cols_to_normalize=["issuer_name", "cusip", "isin", "figi"]
#     )
#     snapshot: pl.DataFrame = pl.read_parquet(snapshot_path)

#     compare_dataframes(
#         locally_processed,
#         snapshot,
#         "locally_processed",
#         "snapshot",
#         comparison_export_path=root_path,
#         raise_if_schema_difference=False,
#     )


# TODO: get this working again
# def test_subset_normalizer() -> None:
#     root_path = Path("test_data/normalize_subset")
#     input_path = root_path / "input.parquet"
#     snapshot_path = root_path / "processed.parquet"

#     raw_input: pl.DataFrame = pl.read_parquet(input_path).filter(
#         pl.col("issuer_name") == pl.lit("MICROSOFT CORPORATION")
#     )
#     locally_processed: pl.DataFrame = normalize_subset(
#         raw_input, cols_to_normalize=["issuer_name", "cusip", "isin", "figi"]
#     )
#     snapshot: pl.DataFrame = pl.read_parquet(snapshot_path)

#     compare_dataframes(
#         locally_processed,
#         snapshot,
#         "locally_processed",
#         "snapshot",
#         comparison_export_path=root_path / "reg_test_comparison.csv",
#         raise_if_schema_difference=True,
#     )
