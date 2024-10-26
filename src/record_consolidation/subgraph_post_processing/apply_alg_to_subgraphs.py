import networkx as nx
from tqdm.auto import tqdm

from record_consolidation._typing import GraphGenerator, SubGraphPostProcessorFnc
from record_consolidation.utils.graphs import extract_connected_subgraphs


def apply_post_processor_to_subgraphs(
    subgraphs: GraphGenerator,
    graphs_post_processor: SubGraphPostProcessorFnc,
) -> GraphGenerator:

    new_subgraphs: list[nx.Graph] = []
    print("Post-processing subgraphs.")
    for subg in tqdm(subgraphs):
        processed_subg: nx.Graph = graphs_post_processor(subg)
        processed_split_subgs: GraphGenerator = extract_connected_subgraphs(
            processed_subg
        )  # split the processed_subg into distinct graph objects if connections have been severed, i.e., if there are now multiple isolated subgraphs
        new_subgraphs += list(processed_split_subgs)
    return (x for x in new_subgraphs)  # TODO: actually use generator or change typehint
