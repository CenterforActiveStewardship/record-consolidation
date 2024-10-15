from typing import Any
from record_consolidation._typing import GraphGenerator, SubGraphPostProcessorFnc
import networkx as nx

from record_consolidation.subgraph_post_processing.apply_alg_to_subgraphs import (
    apply_post_processor_to_subgraphs,
)
from record_consolidation.utils.graphs import extract_connected_subgraphs


def _ps(subgraph: nx.Graph) -> None:
    for x in subgraph.nodes.data():
        print(x)


def _psps(l: list[nx.Graph]) -> None:
    for subg in l:
        _ps(subg)


def test_apply_post_processor_to_subgraphs() -> None:
    G = nx.Graph([(0, 1), (1, 2), (2, 0)])
    subgraphs = list(extract_connected_subgraphs(G))
    post_processed = list(
        apply_post_processor_to_subgraphs(subgraphs, graphs_post_processor=lambda g: g)
    )
    _psps(subgraphs)
    _psps(post_processed)
    assert set(subgraphs) == set(post_processed), (
        list(x for x in subg.nodes.data() for subg in subgraphs),
        list(x for x in subg.nodes.data() for subg in post_processed),
    )
