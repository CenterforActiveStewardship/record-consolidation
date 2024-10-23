import polars as pl


def remove_string_nulls(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col(pl.String).str.strip_chars().replace(["", "N/A"], None)
    )


def assign_columns_if_missing(
    assign_to: pl.DataFrame,
    assign_from: pl.DataFrame,
    cols: list[str],
) -> pl.DataFrame:
    # assign columns if they never end up in
    for col in cols:
        if col not in assign_to.columns:
            print(f"{col=} wasn't used.")
            coltype: pl.DataType = assign_from.schema[col]
            assign_to = assign_to.with_columns(pl.lit(None).cast(coltype).alias(col))
    return assign_to


def extract_null_counts(df: pl.DataFrame) -> dict[str, int]:
    return df.select(pl.all().is_null().sum()).to_dicts()[0]
