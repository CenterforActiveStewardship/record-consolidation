from functools import partial
from typing import Callable

import networkx as nx

from record_consolidation.graphs import GraphGenerator, extract_connected_subgraphs
from record_consolidation.utils.draw_graph import draw_graph


def _has_high_max_betweenness(G: nx.Graph, threshold: float) -> bool:
    # TODO: this decision rule could be improved for the case of JPMorgan
    # Surely there's something that can measure this perfectly or very well
    betweennesses: dict[tuple, float] = nx.edge_betweenness_centrality(
        G, weight="count", normalized=True
    )
    max_betweenness: float = max(betweennesses.values())
    print(f"{max_betweenness=}\n{threshold=}")
    return max_betweenness >= threshold


def _total_count(G: nx.Graph) -> int:
    data = G.nodes.data()
    return sum(data[key[0]]["count"] for key in data)


def _total_count_ge(G: nx.Graph, than: int) -> bool:
    return _total_count(G) >= than


def _extract_issuer_names(G: nx.Graph, just_words: bool = True) -> set[str]:
    issuer_names: set[str] = set()
    for node in G.nodes.data():
        if node[1]["field"] == "issuer_name":
            issuer_names.add(node[0])
    if not just_words:
        return issuer_names

    issuer_name_words: set[str] = set()
    for name in issuer_names:
        for word in name.split():
            issuer_name_words.add(word)
    return issuer_name_words


def split_subgraph_where_necessary(
    G: nx.Graph,
    should_consider_cutting_decider: Callable[[nx.Graph], bool] = partial(
        _total_count_ge, than=5000
    ),
    should_cut: Callable[[nx.Graph], bool] = partial(
        _has_high_max_betweenness,
        threshold=0.06,
    ),
    max_cuts: int = 3,
    verbose: bool = True,
    extra_verbose: bool = False,
) -> GraphGenerator:
    # TODO: save cut_points when possible (rather than removing them from the graph). Low priority though, because they account for such a small portion of the total.
    if verbose:
        print("-" * 120)
        issuer_names: set[str] = _extract_issuer_names(G)
        print(f"{issuer_names=}")
        print(f"{_total_count(G)=}")

    if not should_consider_cutting_decider(G):
        if extra_verbose:
            print("Not splitting graph because `consider_cutting_heuristic==False`")
        return G

    i = 0
    if extra_verbose:
        draw_graph(G, size=10)

    while should_cut(G):
        if extra_verbose:
            print(f"Iteration: {i}")

        cut_points = tuple(nx.articulation_points(G))
        if len(cut_points) == 0 or i > max_cuts:
            break

        # TODO(?): assess betweenness_centrality of cut_points; only cut if total counts > some X

        data = G.nodes.data()
        cut_point_counts: dict[str, int] = {
            name: data[name]["count"] for name in cut_points
        }
        to_cut: str = min(cut_point_counts, key=cut_point_counts.get)  # type: ignore
        G.remove_node(to_cut)

        cut_points = tuple(nx.articulation_points(G))
        if extra_verbose:
            draw_graph(G, size=10)
        i += 1
    return G
