from typing import Iterable, Literal
from warnings import warn

import networkx as nx
import numpy as np
from scipy.sparse import csgraph

from record_consolidation.utils.draw_graph import draw_graph, plot_eigenvalues

from .specific_partitioning_algs import (
    PartitioningMethod,
    partition_via_articulation_points,
    partition_via_betweenness_centrality,
    partition_via_kmeans,
    partition_via_louvain,
    partition_via_spectral_clustering,
    partitioning_methods,
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
            raise ValueError(
                f"{finding_method=} is deprecated; use `n_0val_eigeinvalues` instead."
            )
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


def _partition_ensemble_router(
    G: nx.Graph,
    k: int,
    method: PartitioningMethod,
    sort_articulation_points_by: Literal["degree", "betweenness_centrality"],
    seed: int,
    verbose: bool = True,
) -> nx.Graph:
    match method:
        case "kmeans":
            G = partition_via_kmeans(G, k=k, verbose=verbose)
        case "betweenness_centrality":
            G = partition_via_betweenness_centrality(G, k=k, verbose=verbose)
        case "articulation_points":
            G = partition_via_articulation_points(
                G,
                k=k,
                verbose=verbose,
                sort_articulation_points_by=sort_articulation_points_by,  # TODO: inject this at the correct level
            )
        case "spectral_clustering":
            G = partition_via_spectral_clustering(G, k=k, verbose=verbose)
        case "louvain":
            G = partition_via_louvain(G, verbose=verbose, seed=seed)
        case _:
            raise NotImplementedError(method)
    return G


def partition_subgraphs(
    G: nx.Graph,
    k_finding_method: Literal[
        "n_0val_eigeinvalues", "eigen_gaps"
    ] = "n_0val_eigeinvalues",
    eigenval_threshold: float | None = 0.05,
    sort_articulation_points_by: Literal[
        "degree", "betweenness_centrality"
    ] = "betweenness_centrality",  # TODO: weighted consideration of both?
    partitioning_methods: Iterable[PartitioningMethod] = partitioning_methods,
    verbose: bool = False,
    verbose_within_partitioning_algs: bool = False,
    seed: int = 42,
) -> nx.Graph:
    if verbose:
        draw_graph(G, 10)
    k: int = find_k_clusters(
        G.copy(),
        finding_method=k_finding_method,
        eigenval_threshold=eigenval_threshold,
        verbose=verbose_within_partitioning_algs,
    )
    if k == 1:
        if verbose:
            print("Not partitioning because k=1.")
        return G

    partition_attempts: dict[PartitioningMethod, nx.Graph] = {}
    method_scores: dict[PartitioningMethod, float] = {}
    for method in partitioning_methods:
        if verbose:
            print(
                "-" * 100,
                method,
                "-" * 100,
            )

        partition_attempts[method] = _partition_ensemble_router(
            G.copy(),
            k=k,
            method=method,
            verbose=verbose_within_partitioning_algs,
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
        if verbose:
            print(
                "-" * 100, method, "score =", round(method_scores[method], 3), "-" * 100
            )
            draw_graph(partition_attempts[method], 4)

    best_score: float = max(method_scores.values())
    best_partition: nx.Graph
    if best_score == 0:
        if verbose:
            warning_str = f"Returning unmodified graph, because `best_score` == 0\n{k=}\n{best_score=}\n{method_scores=}"
            warn(warning_str)
        best_partition = G
    else:
        best_method: PartitioningMethod = max(method_scores, key=method_scores.get)  # type: ignore
        best_partition = partition_attempts[best_method]
        if verbose:
            print(f"{method_scores=}")
            print(f"{best_method=}")
    if verbose:
        draw_graph(best_partition, 10)
    return best_partition
