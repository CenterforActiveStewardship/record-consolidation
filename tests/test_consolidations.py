import polars as pl

from record_consolidation.df_consolidations import consolidate_normalized_table


def test_intra_consolidation() -> None:
    msfts: pl.DataFrame = pl.read_csv("test_data/msfts_and_amzns.csv").filter(
        pl.col("issuer_name") == pl.lit("MICROSOFT_CORPORATION")
    )
    unique_consolidated: pl.DataFrame = (
        consolidate_normalized_table(msfts, depth="intra_field").unique().sort(pl.all())
    )
    correct_unique_consolidated = pl.DataFrame(
        [
            pl.Series(
                "issuer_name",
                ["MICROSOFT CORPORATION"] * 5,
                dtype=pl.String,
            ),
            pl.Series(
                "cusip", [None, None, None, "594918104", "594918104"], dtype=pl.String
            ),
            pl.Series(
                "isin",
                [None, "US5949181045", "US5949181045", None, "US5949181045"],
                dtype=pl.String,
            ),
            pl.Series("figi", [None, None, "MSFT", None, None], dtype=pl.String),
        ]
    )


def test_intra_and_inter_consolidation():
    pass
