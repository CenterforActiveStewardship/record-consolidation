import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def draw_graph(G: nx.Graph, size: int = 12):  # TODO: remove after development
    # Use spring_layout with adjusted parameters
    pos = nx.spring_layout(G, k=0.25, iterations=100, seed=42)
    edge_weights = [G[u][v]["count"] for u, v in G.edges()]
    # Draw the graph with the new layout
    plt.figure(figsize=(size, size))
    nx.draw_networkx(
        G,
        pos=pos,
        arrows=True,
        node_size=[np.sqrt(x[1]["count"]) * 50 for x in G.nodes.data()],
        width=np.sqrt(edge_weights),
        with_labels=True,
        font_size=6,
    )
    plt.show()


def plot_colored_graph(G: nx.Graph, labels: np.ndarray):
    """
    Plot a graph with nodes colored according to their cluster labels.

    Parameters:
    - G: The input graph (networkx.Graph)
    - labels: A list or array of cluster labels corresponding to the nodes in the graph
    """
    # Create a color map based on the cluster labels
    unique_labels = np.unique(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))  # type: ignore
    label_color_map = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Assign colors to nodes based on their labels
    node_colors = [
        label_color_map[labels[list(G.nodes()).index(node)]] for node in G.nodes()
    ]

    # Draw the graph
    pos = nx.spring_layout(G, seed=42)  # Layout for consistent plotting
    plt.figure(figsize=(10, 8))
    nx.draw_networkx(
        G,
        pos,
        node_color=node_colors,
        with_labels=True,
        node_size=500,
        font_size=10,
        font_color="black",
        cmap=plt.cm.rainbow,  # type: ignore
    )

    plt.title("Graph with Colored Clusters")
    plt.show()


def plot_eigenvalues(eigenvalues: np.ndarray, labels=None) -> None:
    plt.plot(range(1, len(eigenvalues) + 1), eigenvalues, "o-", markersize=5)

    if labels is None:
        labels = range(1, len(eigenvalues) + 1)  # Default labels are the indices

    for i, label in enumerate(labels):
        plt.text(
            i + 1, eigenvalues[i], f"{label}", fontsize=7, ha="right", rotation=-45
        )

    plt.title("Eigenvalues of the Laplacian Matrix")
    plt.xlabel("Index")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.show()
