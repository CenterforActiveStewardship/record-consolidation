import networkx as nx
import polars as pl
import pytest

from record_consolidation.graphs import unconsolidated_df_to_graph


@pytest.fixture()
def MOCK_MSFT_DF() -> pl.DataFrame:
    return pl.DataFrame(
        [
            pl.Series(
                "issuer_name",
                ["MICROSOFT CORPORATION"] * 6,
                dtype=pl.String,
            ),
            pl.Series(
                "cusip",
                ["594918104", "594918105", None, None, None, "594918104"],
                dtype=pl.String,
            ),
            pl.Series(
                "isin",
                ["US5949181045"] * 4 + [None] * 2,
                dtype=pl.String,
            ),
            pl.Series("figi", [None, None, None, "MSFT", None, None], dtype=pl.String),
        ]
    )


def test_unconsolidated_df_to_graph(MOCK_MSFT_DF) -> None:
    g: nx.Graph = unconsolidated_df_to_graph(MOCK_MSFT_DF, weight_edges=False)

    correct_nodes = {
        "MICROSOFT CORPORATION",
        "594918104",
        "594918105",
        "US5949181045",
        "MSFT",
    }
    assert set(g.nodes) == correct_nodes

    correct_edges = {
        ("MICROSOFT CORPORATION", "US5949181045"),
        ("MICROSOFT CORPORATION", "594918104"),
        ("MICROSOFT CORPORATION", "MSFT"),
        ("MICROSOFT CORPORATION", "594918105"),
        ("594918104", "US5949181045"),
        ("MSFT", "US5949181045"),
        ("594918105", "US5949181045"),
    }

    assert set(tuple(sorted(edge)) for edge in g.edges) == set(
        tuple(sorted(edge)) for edge in correct_edges
    )
