"""Plot helpers for ResPlan plans and wall segmentation debugging."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, Point, Polygon

try:
    import geopandas as gpd
except ImportError:
    gpd = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

from resplan_constants import CATEGORY_COLORS
from resplan_geometry import _dedupe_points, _require, get_geometries, normalize_keys
from resplan_wall import (
    _line_from_segment,
    _plan_wall_depth,
    split_wall_faces,
    split_wall_segments,
    validate_wall_segmentation,
)


def _segment_endpoint_points(seg: Dict[str, Any]) -> List[Point]:
    line = _line_from_segment(seg)
    if line is None:
        return []
    return [Point(line.coords[0]), Point(line.coords[-1])]


def _opening_geometries(plan: Dict[str, Any]) -> List[Any]:
    geoms: List[Any] = []
    for key in ("door", "window", "front_door"):
        geoms.extend(get_geometries(plan.get(key)))
    return [geom for geom in geoms if geom is not None and not geom.is_empty]


def _point_on_opening(pt: Point, opening_geoms: List[Any], tolerance: float) -> bool:
    return any(pt.distance(geom) <= tolerance for geom in opening_geoms)


def plot_plan(
    plan: Dict[str, Any],
    categories: Optional[List[str]] = None,
    colors: Dict[str, str] = CATEGORY_COLORS,
    ax: Optional[Any] = None,
    legend: bool = True,
    title: Optional[str] = None,
    tight: bool = True,
) -> Any:
    """Plot a single plan with colored layers."""
    _require(plt, "matplotlib", "plot_plan")
    plan = normalize_keys(plan)

    if categories is None:
        categories = [
            "living",
            "bedroom",
            "bathroom",
            "kitchen",
            "door",
            "window",
            "wall",
            "front_door",
            "balcony",
            "storage",
        ]

    geoms, color_list, present = [], [], []

    for key in categories:
        parts = get_geometries(plan.get(key))
        if not parts:
            continue

        geoms.extend(parts)
        color_list.extend([colors.get(key, "#000000")] * len(parts))
        present.append(key)

    if not geoms:
        raise ValueError("No geometries to plot.")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if gpd is not None:
        gpd.GeoSeries(geoms).plot(ax=ax, color=color_list, edgecolor="black", linewidth=0.5)
    else:
        from matplotlib.patches import Polygon as MplPolygon

        for geom, color in zip(geoms, color_list):
            if isinstance(geom, Polygon):
                patch = MplPolygon(
                    np.asarray(geom.exterior.coords),
                    closed=True,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(patch)

                for ring in geom.interiors:
                    hole = MplPolygon(
                        np.asarray(ring.coords),
                        closed=True,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.5,
                    )
                    ax.add_patch(hole)

            elif isinstance(geom, LineString):
                xs, ys = geom.xy
                ax.plot(xs, ys, color=color, linewidth=1.0)

            elif isinstance(geom, Point):
                ax.scatter([geom.x], [geom.y], color=color, s=15)

        xs0, ys0, xs1, ys1 = zip(*(g.bounds for g in geoms))
        pad = max(max(xs1) - min(xs0), max(ys1) - min(ys0)) * 0.03
        ax.set_xlim(min(xs0) - pad, max(xs1) + pad)
        ax.set_ylim(min(ys0) - pad, max(ys1) + pad)

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    if legend:
        from matplotlib.patches import Patch

        uniq_present = list(dict.fromkeys(present))
        handles = [
            Patch(
                facecolor=colors.get(k, "#000000"),
                edgecolor="black",
                label=k.replace("_", " "),
            )
            for k in uniq_present
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    if tight:
        plt.tight_layout()

    return ax


def plot_wall_segments(
    plan: Dict[str, Any],
    segments: Optional[List[Dict[str, Any]]] = None,
    mode: str = "segments",
    ax: Optional[Any] = None,
    label: bool = True,
    title: Optional[str] = None,
) -> Any:
    """Plot the floorplan and overlay wall faces or wall segments."""
    _require(plt, "matplotlib", "plot_wall_segments")

    mode = mode.lower()
    if segments is None:
        segments = split_wall_faces(plan) if mode in ("face", "faces") else split_wall_segments(plan)

    ax = plot_plan(plan, ax=ax, legend=True, title=title or "Wall segments")

    for seg in segments:
        line = seg.get("face_line") if mode in ("face", "faces") else _line_from_segment(seg)
        if not isinstance(line, LineString):
            continue

        xs, ys = line.xy
        is_split = bool(seg.get("split_at_wall_intersection"))
        is_paired = bool(seg.get("paired"))

        linewidth = 2.8 if is_split else 1.8
        linestyle = "-" if is_paired else "--"
        color = "red" if is_split else "black"

        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, zorder=15)

        if label:
            mid = line.interpolate(line.length / 2)
            ax.text(mid.x, mid.y, str(seg.get("id", "")), fontsize=7, color="black", zorder=20)

    if title:
        ax.set_title(title)

    plt.tight_layout()
    return ax


def plot_wall_segmentation_debug(
    plan: Dict[str, Any],
    segments: Optional[List[Dict[str, Any]]] = None,
    ax: Optional[Any] = None,
    label: bool = True,
    show_endpoints: bool = True,
    show_split_points: bool = False,
    show_issue_points: bool = True,
    dedupe_endpoint_markers: bool = True,
    endpoint_dedupe_tolerance: Optional[float] = None,
    hide_opening_endpoints: bool = True,
    opening_endpoint_tolerance: Optional[float] = None,
    title: Optional[str] = None,
) -> Any:
    """Debug overlay with wall endpoints, split markers, and issue points."""
    _require(plt, "matplotlib", "plot_wall_segmentation_debug")

    if segments is None:
        segments = split_wall_segments(plan)

    ax = plot_plan(plan, ax=ax, legend=True, title=title or "Wall segmentation debug")

    wall_depth = _plan_wall_depth(plan)
    endpoint_tol = (
        endpoint_dedupe_tolerance
        if endpoint_dedupe_tolerance is not None
        else max(wall_depth * 1.10, 2.0)
    )
    opening_tol = (
        opening_endpoint_tolerance
        if opening_endpoint_tolerance is not None
        else max(wall_depth * 0.35, 0.75)
    )

    if show_endpoints:
        points = []
        for seg in segments:
            points.extend(_segment_endpoint_points(seg))
        if hide_opening_endpoints:
            openings = _opening_geometries(plan)
            points = [
                pt for pt in points
                if not _point_on_opening(pt, openings, opening_tol)
            ]
        if dedupe_endpoint_markers:
            points = _dedupe_points(points, endpoint_tol)

        if points:
            ax.scatter(
                [pt.x for pt in points],
                [pt.y for pt in points],
                s=22,
                color="black",
                marker="o",
                zorder=20,
                label="wall endpoints",
            )

    if show_split_points:
        points = []
        for seg in segments:
            if seg.get("split_at_wall_intersection"):
                points.extend(_segment_endpoint_points(seg))
        if points:
            ax.scatter(
                [pt.x for pt in points],
                [pt.y for pt in points],
                s=34,
                color="#4daf4a",
                marker="x",
                zorder=21,
                label="intersection splits",
            )

    if show_issue_points:
        report = validate_wall_segmentation(plan, segments)
        issue_points = [
            issue.get("point")
            for issue in report.get("unsplit_intersections", [])
            if isinstance(issue.get("point"), Point)
        ]

        if issue_points:
            ax.scatter(
                [pt.x for pt in issue_points],
                [pt.y for pt in issue_points],
                s=70,
                color="red",
                marker="X",
                zorder=25,
                label="unsplit intersections",
            )

    return ax


def demo_wall_segments(
    pkl_path: str = "ResPlan.pkl",
    sample_index: int = 0,
    out_path: str = "assets/wall_segments_sample.png",
) -> List[Dict[str, Any]]:
    """Load a ResPlan sample, split walls, and save a debug overlay."""
    import pickle
    import zipfile

    _require(plt, "matplotlib", "demo_wall_segments")

    if pkl_path.lower().endswith(".zip"):
        with zipfile.ZipFile(pkl_path) as zf:
            with zf.open("ResPlan.pkl") as f:
                data = pickle.load(f)
    else:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

    plan = data[sample_index]
    segments = split_wall_segments(plan)

    fig, ax = plt.subplots(figsize=(8, 8))
    plot_wall_segmentation_debug(
        plan,
        segments,
        ax=ax,
        title=f"Wall segmentation sample {sample_index}",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return segments
