# import networkx as nx

# from record_consolidation.subgraph_post_processing.apply_alg_to_subgraphs import (
#     apply_post_processor_to_subgraphs,
# )
# from record_consolidation.utils.graphs import extract_connected_subgraphs


# def test_apply_post_processor_to_subgraphs() -> None:
#     G = nx.Graph([(0, 1), (1, 2), (2, 0)])
#     subgraphs = list(extract_connected_subgraphs(G))
#     post_processed = list(  # noqa
#         apply_post_processor_to_subgraphs(
#             (x for x in subgraphs), graphs_post_processor=lambda g: g
#         )
#     )  # noqa
#     # TODO: assert isomorphic
