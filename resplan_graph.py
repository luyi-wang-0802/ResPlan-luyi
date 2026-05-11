"""Room graph helpers for ResPlan-style floorplan datasets."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from shapely.geometry import Polygon

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import networkx as nx
except ImportError:
    nx = None

from resplan_geometry import _require, get_geometries, normalize_keys
from resplan_plot import plot_plan


def plan_to_graph(plan: Dict[str, Any], buffer_factor: float = 0.75) -> Any:
    """Create a simple room graph with room/front-door nodes and connection edges."""
    _require(nx, "networkx", "plan_to_graph")
    plan = normalize_keys(plan)
    graph = nx.Graph()

    wall_width = float(plan.get("wall_width", plan.get("wall_depth", 0.1)) or 0.1)
    buf = max(wall_width * buffer_factor, 0.01)

    nodes_by_type: Dict[str, List[str]] = {
        key: [] for key in ["living", "kitchen", "bedroom", "bathroom", "balcony", "front_door"]
    }

    for room_type in ["living", "kitchen", "bedroom", "bathroom", "balcony"]:
        for i, geom in enumerate(get_geometries(plan.get(room_type))):
            if isinstance(geom, Polygon) and not geom.is_empty:
                node_id = f"{room_type}_{i}"
                graph.add_node(node_id, geometry=geom, type=room_type, area=geom.area)
                nodes_by_type[room_type].append(node_id)

    for i, geom in enumerate(get_geometries(plan.get("front_door"))):
        node_id = f"front_door_{i}"
        graph.add_node(node_id, geometry=geom, type="front_door", area=getattr(geom, "area", 0.0))
        nodes_by_type["front_door"].append(node_id)

    doors = get_geometries(plan.get("door"))
    windows = get_geometries(plan.get("window"))
    connectors = [(geom, "via_door") for geom in doors] + [(geom, "via_window") for geom in windows]

    for front_door in nodes_by_type["front_door"]:
        fd_geom = graph.nodes[front_door]["geometry"]
        for living in nodes_by_type["living"]:
            living_geom = graph.nodes[living]["geometry"]
            if fd_geom.intersects(living_geom.buffer(buf)):
                graph.add_edge(front_door, living, type="direct")

    for room_type in ["kitchen", "bedroom"]:
        for room in nodes_by_type[room_type]:
            room_geom = graph.nodes[room]["geometry"].buffer(buf)
            for living in nodes_by_type["living"]:
                living_geom = graph.nodes[living]["geometry"]
                if room_geom.buffer(buf).intersects(living_geom.buffer(buf)):
                    graph.add_edge(room, living, type="adjacency")

    for room_type in ["bathroom", "balcony"]:
        for room in nodes_by_type[room_type]:
            room_geom = graph.nodes[room]["geometry"].buffer(buf)
            for conn_geom, conn_type in connectors:
                if not conn_geom.intersects(room_geom):
                    continue
                for target_type in ["living", "bedroom"]:
                    for target in nodes_by_type[target_type]:
                        target_geom = graph.nodes[target]["geometry"].buffer(buf)
                        if conn_geom.intersects(target_geom) and not graph.has_edge(room, target):
                            graph.add_edge(room, target, type=conn_type)

    return graph


def plot_plan_and_graph(
    plan: Dict[str, Any],
    ax: Optional[Any] = None,
    node_scale: Tuple[float, float] = (150, 1000),
    title: Optional[str] = None,
) -> Any:
    """Plot a plan and overlay the room graph."""
    _require(nx, "networkx", "plot_plan_and_graph")
    _require(plt, "matplotlib", "plot_plan_and_graph")

    graph = plan["graph"] if "graph" in plan else plan_to_graph(plan)
    ax = plot_plan(plan, legend=True, ax=ax, title=title)

    pos = {}
    for node, data in graph.nodes(data=True):
        geom = data.get("geometry")
        if geom is None or geom.is_empty:
            continue
        c = geom.centroid
        pos[node] = (c.x, c.y)

    node_style = {
        "living": dict(color="white", shape="o", size=400, edgecolor="black"),
        "bedroom": dict(color="cyan", shape="s", size=300, edgecolor="black"),
        "bathroom": dict(color="magenta", shape="D", size=260, edgecolor="black"),
        "kitchen": dict(color="yellow", shape="^", size=300, edgecolor="black"),
        "balcony": dict(color="lightgray", shape="X", size=260, edgecolor="black"),
        "front_door": dict(color="red", shape="*", size=420, edgecolor="black"),
    }

    areas = [graph.nodes[node].get("area", 0.0) for node in graph.nodes]
    area_min = min(areas) if areas else 0.0
    area_max = max(areas) if areas else 1.0
    size_min, size_max = node_scale

    def scale_size(area: float) -> float:
        if area_max <= area_min:
            return (size_min + size_max) / 2
        return size_min + ((area - area_min) / (area_max - area_min)) * (size_max - size_min)

    for node_type, style in node_style.items():
        nodes = [node for node, data in graph.nodes(data=True) if data.get("type") == node_type and node in pos]
        if not nodes:
            continue
        nx.draw_networkx_nodes(
            graph,
            pos,
            nodelist=nodes,
            node_size=[scale_size(graph.nodes[node].get("area", 0.0)) for node in nodes],
            node_shape=style["shape"],
            node_color=style["color"],
            edgecolors=style["edgecolor"],
            linewidths=1.0,
            ax=ax,
            alpha=0.9,
        )

    edge_style = {
        "direct": dict(color="darkred", width=2.0, style="-"),
        "adjacency": dict(color="darkgreen", width=1.5, style="--"),
        "via_door": dict(color="darkblue", width=1.2, style="-"),
        "via_window": dict(color="orange", width=1.0, style=":"),
    }
    for edge_type, style in edge_style.items():
        edges = [
            (u, v)
            for u, v, data in graph.edges(data=True)
            if data.get("type") == edge_type and u in pos and v in pos
        ]
        if edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=edges,
                width=style["width"],
                edge_color=style["color"],
                style=style["style"],
                ax=ax,
                alpha=0.8,
            )

    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax
