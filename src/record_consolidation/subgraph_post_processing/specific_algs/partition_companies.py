from typing import Iterable, Literal
from warnings import warn

import networkx as nx
import numpy as np
import polars as pl
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

from record_consolidation.utils.draw_graph import (
    draw_graph,
    plot_colored_graph,
    plot_eigenvalues,
)


def find_k_clusters(
    G: nx.Graph,
    finding_method: Literal["n_0val_eigeinvalues", "eigen_gaps"],
    verbose: bool,
    eigenval_threshold: float | None = 0.05,
) -> int:
    """
    Finds the number of *nearly* disconnected components in a graph
      by finding the number of near-zero eigenvalues in its Laplacian.
    """
    A = nx.adjacency_matrix(G, weight="count")

    # Step 1: Compute the Laplacian and its eigenvalues/eigenvectors
    L = csgraph.laplacian(A, normed=True)
    eigenvalues, _ = np.linalg.eigh(L.toarray())

    # Step 1.5: Plot eigenvalues
    if verbose:
        labels = [
            x[0] if x[1]["field"] == "issuer_name" else "" for x in G.nodes.data()
        ]
        plot_eigenvalues(eigenvalues, labels=labels)

    # Step 2: Find "k"
    k: int
    match finding_method:
        case "eigen_gaps":
            warning_str = (
                f"{finding_method=} is deprecated; use `n_0val_eigeinvalues` instead."
            )
            warn(warning_str)
            eigen_gaps = np.diff(eigenvalues)
            k = int(
                np.argmax(eigen_gaps) + 1
            )  # The index of the largest gap indicates number of clusters
        case "n_0val_eigeinvalues":
            k = (
                eigenvalues < eigenval_threshold
            ).sum()  # they won't be exactly 0 - just close TODO: normalize this & make more robust & check edge cases
        case _:
            raise ValueError(finding_method)
    if verbose:
        print(f"{sorted(eigenvalues)[:5]=}")
        print(f"{k=}")
        if k > 5:
            print(f"{L=}")
    return k


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


def _partition_via_louvain(G: nx.Graph, verbose: bool, seed: int) -> nx.Graph:
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


def _partition_via_spectral_clustering(G: nx.Graph, k: int, verbose: bool) -> nx.Graph:
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
    kmeans = KMeans(n_clusters=k, random_state=0).fit(eigenvectors)
    labels = kmeans.labels_

    # Step 4: Assign the cluster labels to the nodes
    for i, node in enumerate(G.nodes()):
        G.nodes[node]["cluster"] = labels[i]

    return G


def _partition_via_kmeans(G: nx.Graph, k: int, verbose: bool) -> nx.Graph:
    warn("`kmeans` does not produce good results -- it mixes labels within clusters.")

    # Use the adjacency matrix of the graph as features
    adjacency_matrix = nx.to_numpy_array(G, weight="count")

    # Apply k-means clustering to node features
    kmeans = KMeans(n_clusters=k, random_state=0).fit(adjacency_matrix)
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


def _partition_via_betweenness_centrality(
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


def _partition_via_articulation_points(
    G: nx.Graph,
    k: int,
    verbose: bool,
    sort_articulation_points_by: Literal["degree", "betweenness_centrality"],
) -> nx.Graph:
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
        if verbose:
            draw_graph(G, 5)

    return G


def _partition_companies_graph(
    G: nx.Graph,
    k: int,
    method: PartitioningMethod,
    sort_articulation_points_by: Literal["degree", "betweenness_centrality"],
    seed: int,
    verbose: bool = True,
) -> nx.Graph:
    match method:
        case "kmeans":
            G = _partition_via_kmeans(G, k=k, verbose=verbose)
        case "betweenness_centrality":
            G = _partition_via_betweenness_centrality(G, k=k, verbose=verbose)
        case "articulation_points":
            G = _partition_via_articulation_points(
                G,
                k=k,
                verbose=verbose,
                sort_articulation_points_by=sort_articulation_points_by,  # TODO: inject this at the correct level
            )
        case "spectral_clustering":
            G = _partition_via_spectral_clustering(G, k=k, verbose=verbose)
        case "louvain":
            G = _partition_via_louvain(G, verbose=verbose, seed=seed)
        # case "girvan_newman":
        #     G = _partition_via_girvan_newman(G, verbose=verbose, seed=seed)
        case _:
            raise NotImplementedError(method)
    return G


def partition_companies_graph_where_necessary(
    G: nx.Graph,
    k_finding_method: Literal["n_0val_eigeinvalues", "eigen_gaps"],
    sort_articulation_points_by: Literal["degree", "betweenness_centrality"],
    partitioning_methods: Iterable[PartitioningMethod] = partitioning_methods,
    verbose: bool = False,
    seed: int = 42,
) -> nx.Graph:
    draw_graph(G, 10)
    k: int = find_k_clusters(G.copy(), finding_method=k_finding_method, verbose=verbose)
    if k == 1:
        if verbose:
            print("Not partitioning because k=1.")
        return G

    partition_attempts: dict[PartitioningMethod, nx.Graph] = {}
    method_scores: dict[PartitioningMethod, float] = {}
    for method in partitioning_methods:
        print(
            "-" * 100,
            method,
            "-" * 100,
        )

        partition_attempts[method] = _partition_companies_graph(
            G.copy(),
            k=k,
            method=method,
            verbose=verbose,
            sort_articulation_points_by=sort_articulation_points_by,
            seed=seed,
        )
        method_scores[method] = nx.community.modularity(
            partition_attempts[method],
            communities=nx.connected_components(partition_attempts[method]),
            weight="count",
        )
        # ensure that the result produced the correct number of partitions
        if nx.number_connected_components(partition_attempts[method]) != k:
            method_scores[method] = 0
        # if verbose:
        if True:
            print(
                "-" * 100, method, "score =", round(method_scores[method], 3), "-" * 100
            )
            draw_graph(partition_attempts[method], 4)

    best_score: float = max(method_scores.values())
    best_partition: nx.Graph
    if best_score == 0:
        warning_str = f"Returning unmodified graph, because `best_score` == 0\n{k=}\n{best_score=}\n{method_scores=}"
        warn(warning_str)
        best_partition = G
    else:
        best_method: PartitioningMethod = max(method_scores, key=method_scores.get)  # type: ignore
        best_partition = partition_attempts[best_method]
        print(f"{method_scores=}")
        print(f"{best_method=}")
    draw_graph(best_partition, 10)
    return best_partition
