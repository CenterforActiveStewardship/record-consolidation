import inspect
from functools import wraps
from typing import Callable, Iterable, Literal

import networkx as nx
from pydantic import BaseModel
from rapidfuzz.distance.JaroWinkler import normalized_similarity as jw_similarity


def _get_graph_arg(func, graph_arg_name, *args, **kwargs):
    """
    Extracts the nx.Graph argument from the provided args and kwargs based on its name.

    Args:
        graph_arg_name (str): The name of the argument that contains the nx.Graph.
        *args: Positional arguments passed to the function.
        **kwargs: Keyword arguments passed to the function.

    Returns:
        nx.Graph: The graph argument if found.

    Raises:
        ValueError: If the specified argument name is not found.
        TypeError: If the found argument is not a networkx.Graph.
    """
    # Bind the arguments to the function's signature
    bound_args = inspect.signature(func).bind(*args, **kwargs)
    bound_args.apply_defaults()

    # Extract and validate the graph argument
    graph = bound_args.arguments.get(graph_arg_name)
    if graph is None:
        raise ValueError(
            f"Argument '{graph_arg_name}' not found in function signature."
        )
    if not isinstance(graph, nx.Graph):
        raise TypeError(f"Argument '{graph_arg_name}' must be of type networkx.Graph.")

    return graph


class OutputHasMoreNodes(Exception):
    pass


class OurCustomNode(BaseModel):
    name: str
    field: str
    count: int

    @classmethod
    def from_node_tuple(
        cls, node_tuple: tuple[str, dict[Literal["field", "count"], str | int]]
    ) -> "OurCustomNode":
        return cls(  # type: ignore
            name=node_tuple[0],
            **node_tuple[1],  # type: ignore
        )


class OurCustomNodes(BaseModel):
    nodes: Iterable[OurCustomNode]

    @classmethod
    def from_graph(cls, G: nx.graph) -> "OurCustomNodes":
        return cls(
            nodes=[
                OurCustomNode(
                    name=node_data.get("name", f"Node_{node}"),
                    field=node_data.get("field", "default_field"),
                    count=node_data.get("count", 0),
                )
                for node, node_data in G.nodes(data=True)
            ]
        )


def assign_missing_node(
    node_to_add: OurCustomNode,  # TODO
    G: nx.Graph,
    method: Literal["text_similarities"],
    find_closest_via: Literal["single_node", "cluster_avg"] = "single_node",
) -> nx.Graph:
    """Find and create the best edge."""
    print(f"{node_to_add=}")
    FIELD: str = node_to_add.field
    match method:
        case "text_similarities":
            nodes_of_same_field = (n for n in G.nodes.data() if n["field"] == FIELD)
            similarities: dict[str, float] = {
                n: jw_similarity(node_to_add.name, n.name) for n in nodes_of_same_field
            }
            print(f"{similarities=}")
            match find_closest_via:
                case "single_node":
                    closest_match = max(similarities, key=similarities.get)  # type: ignore
                    G.add_node(node_to_add, count=0, field=FIELD)
                    G.add_edge(node_to_add, closest_match, count=0)
                case _:
                    raise NotImplementedError()
        case _:
            raise NotImplementedError(method)


def _repopulate_missing_nodes(
    og: nx.Graph, out: nx.Graph, verbose: bool = False
) -> nx.Graph:

    og_nodes: list[tuple[str, dict[Literal["field", "count"], str | int]]] = list(
        og.nodes.data()
    )
    out_nodes: list[tuple[str, dict[Literal["field", "count"], str | int]]] = list(
        out.nodes.data()
    )

    if extra_node_names := (
        set((x[0] for x in out_nodes)) - set((name[0] for name in og_nodes))
    ):
        raise OutputHasMoreNodes(extra_node_names)

    if missing_node_names := (
        set((name[0] for name in og_nodes)) - set((name[0] for name in out_nodes))
    ):
        print(f"{missing_node_names=}")
        for node_name in missing_node_names:
            if verbose:
                print(f"Reassigning removed node: {node_name}")
            print(f"{og[node_name]=}")
            node_to_add: OurCustomNode = OurCustomNode.from_node_tuple(
                node_tuple=og[node_name]
            )
            print(f"{node_to_add=}")
            out = assign_missing_node(
                node_to_add=node_to_add,
                G=out,
                method="text_similarities",
            )
    else:
        if verbose:
            print("NO MISSING NODES.")

    return out


def repopulate_missing_nodes(graph_arg_name="G") -> Callable:
    """
    Args:
        graph_arg_name (str): The name of the argument that contains the nx.Graph (default: 'G').
    """

    def decorator(func) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> nx.Graph:
            original_graph = _get_graph_arg(func, graph_arg_name, *args, **kwargs)

            output_graph = func(*args, **kwargs)

            repopulated: nx.Graph = _repopulate_missing_nodes(
                og=original_graph, out=output_graph
            )

            return repopulated

        return wrapper

    return decorator
