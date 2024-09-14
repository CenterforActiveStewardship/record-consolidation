from typing import Any, Generator

import networkx as nx
import polars as pl


def extract_connected_subgraphs(G: nx.Graph) -> Generator[nx.Graph, Any, Any]:
    return (G.subgraph(c).copy() for c in nx.connected_components(G))


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
