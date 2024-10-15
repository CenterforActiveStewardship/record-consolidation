from typing import Any
from record_consolidation._typing import GraphGenerator, SubGraphPostProcessorFnc
import networkx as nx

from record_consolidation.utils.graphs import extract_connected_subgraphs


def apply_post_processor_to_subgraphs(
    subgraphs: GraphGenerator,
    graphs_post_processor: SubGraphPostProcessorFnc,
) -> GraphGenerator:
    new_subgraphs: list[nx.Graph] = []
    for subg in subgraphs:
        processed_subg: nx.Graph = graphs_post_processor(subg)
        processed_split_subgs: GraphGenerator = extract_connected_subgraphs(
            processed_subg
        )  # split the processed_subg into distinct graph objects if connections have been severed, i.e., if there are now multiple isolated subgraphs
        new_subgraphs += list(processed_split_subgs)
        # for new_subg in processed_split_subgs:
        #     _ps(new_subg)
        # for new_subg in processed_split_subgs:
        #     new_subgraphs.append(new_subg)
    return new_subgraphs
