from typing import Literal

import networkx as nx
import numpy as np
import polars as pl
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer

from record_consolidation.subgraph_post_processing.validate_post_processor import (
    repopulate_missing_nodes,
)
from record_consolidation.utils.draw_graph import plot_colored_graph

PartitioningMethod = Literal[
    "kmeans",
    "betweenness_centrality",
    "articulation_points",
    "spectral_clustering",
    "louvain",
    # "girvan_newman",
]
partitioning_methods: list[PartitioningMethod] = [
    "kmeans",
    "betweenness_centrality",
    "articulation_points",
    "spectral_clustering",
    "louvain",
    # "girvan_newman",
]


@repopulate_missing_nodes("G")
def partition_via_louvain(G: nx.Graph, verbose: bool, seed: int = 42) -> nx.Graph:
    communities: list[set] = nx.community.louvain_communities(
        G,
        seed=seed,
        weight="count",
    )
    node_to_community = {}
    for community_idx, community in enumerate(communities):
        for node in community:
            node_to_community[node] = community_idx

    # Step 2: Sever edges between different communities
    edges_to_remove = [
        (u, v)
        for u, v in G.edges()
        if node_to_community[u]
        != node_to_community[v]  # Remove edge if nodes are in different communities
    ]

    if verbose:
        num_communities = len(communities)
        print(f"Louvain detected {num_communities} communities")
        print(f"Severing edges between different communities:\n{edges_to_remove} ")

    G.remove_edges_from(edges_to_remove)

    return G


@repopulate_missing_nodes("G")
def partition_via_spectral_clustering(G: nx.Graph, k: int, verbose: bool) -> nx.Graph:
    """
    Partition the graph G into k clusters using spectral clustering.

    Parameters:
    - G: The input graph (networkx Graph).
    - k: The number of clusters.
    - verbose: Whether to print intermediate steps.

    Returns:
    - G: The graph with nodes labeled by their cluster membership.
    """

    # Step 1: Compute the Laplacian matrix
    laplacian_matrix = nx.normalized_laplacian_matrix(G).astype(float)

    # Step 2: Compute the top k eigenvectors of the Laplacian matrix
    eigenvalues, eigenvectors = eigsh(laplacian_matrix, k=k, which="SM")

    if verbose:
        print(f"{eigenvalues=}")

    # Step 3: Apply k-means clustering to the top k eigenvectors
    kmeans = KMeans(n_clusters=k, random_state=42).fit(eigenvectors)
    labels = kmeans.labels_

    # Step 4: Assign the cluster labels to the nodes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["cluster"] = labels[i]

    return G


# @repopulate_missing_nodes("G") # don't need - this doesn't remove nodes
def partition_via_kmeans(G: nx.Graph, k: int, verbose: bool) -> nx.Graph:
    # warn("`kmeans` does not produce good results -- it mixes labels within clusters.")

    # # Use the adjacency matrix of the graph as features
    # adjacency_matrix = nx.to_numpy_array(G, weight="count")

    # lower_nodes: list[str] = [n.lower for n in G.nodes]

    vectorizer = CountVectorizer(
        analyzer="char_wb",
        strip_accents="ascii",
        ngram_range=(1, 1),
    )
    char_count_embeddings: np.ndarray = vectorizer.fit_transform(G.nodes).toarray()
    # node_embeddings: dict[str, np.ndarray] =

    # Apply k-means clustering to node features
    # kmeans = KMeans(n_clusters=k, random_state=0).fit(adjacency_matrix)
    kmeans = KMeans(n_clusters=k, random_state=0).fit(char_count_embeddings)
    labels = kmeans.labels_

    # Create a dictionary of clusters
    clusters: dict[int, list[str]] = {i: [] for i in range(k)}
    for node, label in zip(G.nodes(), labels):
        clusters[label].append(node)

    if verbose:
        print("Performing k-means clustering")
        plot_colored_graph(G, labels)

    # remove edges between different clusters
    for u, v in list(G.edges()):
        if labels[list(G.nodes()).index(u)] != labels[list(G.nodes()).index(v)]:
            G.remove_edge(u, v)
            if verbose:
                print(f"Removed edge between {u} and {v} (different clusters)")
    return G


@repopulate_missing_nodes("G")
def partition_via_betweenness_centrality(
    G: nx.Graph, k: int, verbose: bool
) -> nx.Graph:
    """Basically node-modified Girvan-Newman"""
    removed_nodes = []
    while nx.number_connected_components(G) < k:
        # Compute betweenness centrality for all nodes
        betweennesses = dict(nx.betweenness_centrality(G, weight="count"))
        if verbose:
            betweennesses = {
                k: v for k, v in sorted(betweennesses.items(), key=lambda item: item[1])
            }

        # Find the node with the highest betweenness centrality
        max_node: str = max(betweennesses, key=betweennesses.get)  # type: ignore

        # Remove the node with the highest betweenness centrality
        G.remove_node(max_node)
        removed_nodes.append(max_node)

        if verbose:
            print(
                f"Removed node {max_node} with betweenness centrality {betweennesses[max_node]:.4f}"
            )
    return G


@repopulate_missing_nodes("G")
def partition_via_articulation_points(
    G: nx.Graph,
    k: int,
    verbose: bool,
    sort_articulation_points_by: Literal["degree", "betweenness_centrality"],
) -> nx.Graph:
    """
    Basically, node-based Girvan-Newman.
    This will likely perform better than Girvan-Newman, because the graphs we're working on
        are basically split multi-graphs
    """
    i = 0
    while nx.number_connected_components(G) < k:
        i += 1
        cut_points = list(nx.articulation_points(G))
        if verbose:
            # draw_graph(G, 5)
            print(f"{i=}")
            print(f"{cut_points=}")
        cut_points = [
            node
            for node in cut_points
            if G.degree(node)
            > 1  # more than one connection because it has to bridge multiple clusters - this also avoids creating "archipelagos"
        ]

        match sort_articulation_points_by:
            # TODO: weighted meaasure of both?
            case "betweenness_centrality":
                betweenness_centrality = nx.betweenness_centrality(G, weight="count")
                cut_points = sorted(
                    cut_points,
                    key=lambda node: betweenness_centrality[node],
                    reverse=True,
                )
            case "degree":
                cut_points = sorted(
                    cut_points, key=lambda node: G.degree(node, weight="count")
                )

        if verbose:
            betweenness_centralities = nx.betweenness_centrality(G, weight="count")
            degrees = [G.degree(node, weight="count") for node in cut_points]
            asdasdasd = pl.DataFrame(
                {
                    "point": cut_points,
                    "betweenness_centrality": [
                        betweenness_centralities[x] for x in cut_points
                    ],
                    "degree": degrees,
                }
            )
            print(asdasdasd)
        if len(cut_points) == 0:
            break
        cut_point: str = cut_points[0]
        if verbose:
            print(f"{cut_point=}")

        # get cut point
        data = G.nodes.data()
        cut_point_counts: dict[str, int] = {
            name: data[name]["count"] for name in cut_points
        }
        to_cut: str = min(cut_point_counts, key=cut_point_counts.get)  # type: ignore

        # remove
        G.remove_node(to_cut)

    return G
