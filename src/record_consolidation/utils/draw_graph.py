# import matplotlib.pyplot as plt
# import networkx as nx
# import numpy as np


# def draw_graph(G: nx.Graph, size: int = 12):  # TODO: remove after development
#     # Use spring_layout with adjusted parameters
#     pos = nx.spring_layout(G, k=0.25, iterations=100, seed=42)
#     edge_weights = [G[u][v]["count"] for u, v in G.edges()]
#     # Draw the graph with the new layout
#     plt.figure(figsize=(size, size))
#     nx.draw_networkx(
#         G,
#         pos=pos,
#         arrows=True,
#         node_size=[np.sqrt(x[1]["count"]) * 50 for x in G.nodes.data()],
#         width=np.sqrt(edge_weights),
#         with_labels=True,
#         font_size=6,
#     )
#     plt.show()


def draw_graph(*args, **kwargs) -> None:
    raise NotImplementedError(
        "Remove this an uncomment the function above (on a plane so can't import matplotlib)."
    )
