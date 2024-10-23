import networkx as nx

from record_consolidation._typing import GraphGenerator, SubGraphPostProcessorFnc


def extract_connected_subgraphs(
    G: nx.Graph,
    graphs_post_processor: SubGraphPostProcessorFnc | None = None,
) -> GraphGenerator:
    connected_subgraphs = (G.subgraph(c).copy() for c in nx.connected_components(G))
    if graphs_post_processor is None:
        return connected_subgraphs
    return graphs_post_processor(connected_subgraphs)
