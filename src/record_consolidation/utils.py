from typing import Any, Generator

import networkx as nx


def extract_connected_subgraphs(G: nx.Graph) -> Generator[nx.Graph, Any, Any]:
    return (G.subgraph(c).copy() for c in nx.connected_components(G))
