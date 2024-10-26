from typing import Any, Callable, Generator

import networkx as nx

GraphGenerator = Generator[nx.Graph, Any, Any]
SubGraphPostProcessorFnc = Callable[[nx.Graph], nx.Graph]
