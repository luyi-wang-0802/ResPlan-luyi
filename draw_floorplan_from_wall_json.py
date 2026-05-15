"""Draw a 2D floor plan from a ResPlan wall-segment JSON file.

Example:
    python draw_floorplan_from_wall_json.py ^
        assets/wall_segments_json/resplan_to_JSON_001.json ^
        --output assets/resplan_to_JSON_001_floorplan.png
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import zipfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


WALL_COLORS = {
    "exterior": "#202020",
    "interior": "#595959",
    "unknown": "#404040",
}

OPENING_COLORS = {
    "door": "#d95f02",
    "front_door": "#2ca25f",
    "window": "#1f78b4",
    "opening": "#6a3d9a",
}

VISUAL_WALL_THICKNESS_MM = {
    "exterior": 300,
    "interior": 150,
    "unknown": 150,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw walls, doors, and windows from a wall-segment JSON file."
    )
    parser.add_argument(
        "json_path",
        nargs="?",
        default="assets/wall_segments_json/resplan_to_JSON_001.json",
        help="Input wall-segment JSON path.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="assets/floorplan_from_wall_json.png",
        help="Output image path. Use .png, .jpg, .pdf, or .svg.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for raster formats.",
    )
    parser.add_argument(
        "--show-labels",
        action="store_true",
        help="Show wall IDs and opening IDs.",
    )
    parser.add_argument(
        "--no-openings",
        action="store_true",
        help="Draw only walls.",
    )
    parser.add_argument(
        "--line-scale",
        type=float,
        default=1.0,
        help="Scale factor for all drawn line widths.",
    )
    parser.add_argument(
        "--compare-segmentation",
        action="store_true",
        help="Recompute the source plan wall segmentation and compare it with JSON walls before drawing.",
    )
    parser.add_argument(
        "--data",
        default="ResPlan.zip",
        help="Source ResPlan.zip or ResPlan.pkl used by --compare-segmentation.",
    )
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail instead of warning when JSON quality or segmentation comparison fails.",
    )
    parser.add_argument(
        "--validation-overlay",
        action="store_true",
        help="Draw source plan polygons with JSON wall segments overlaid for conversion checking.",
    )
    parser.add_argument(
        "--validation-clean",
        action="store_true",
        help="Draw JSON walls, openings, and endpoints only, without source plan polygons.",
    )
    return parser.parse_args()


def wall_start_end(wall: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    """Read wall endpoints from either converted_schema or compact JSON format."""
    if "geometry" in wall:
        start = wall["geometry"]["start"]
        end = wall["geometry"]["end"]
    else:
        start = wall["start_point"]
        end = wall["end_point"]
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def wall_location(wall: dict[str, Any]) -> str:
    if "physical" in wall:
        return str(wall["physical"].get("wall_location", "unknown"))
    return str(wall.get("wall_location", "unknown"))


def wall_thickness_mm(wall: dict[str, Any], defaults: dict[str, Any]) -> float:
    if "physical" in wall and wall["physical"].get("thickness_mm") is not None:
        return float(wall["physical"]["thickness_mm"])

    thickness_defaults = defaults.get("wall_thickness_mm", VISUAL_WALL_THICKNESS_MM)
    return float(
        thickness_defaults.get(
            wall_location(wall),
            thickness_defaults.get("unknown", VISUAL_WALL_THICKNESS_MM["unknown"]),
        )
    )


def opening_ratios(opening: dict[str, Any], host_len: float) -> tuple[float, float]:
    """Return opening start/end ratios along its host wall."""
    if "position_on_wall" in opening:
        pos = opening["position_on_wall"]
        return float(pos.get("start_ratio", 0.0)), float(pos.get("end_ratio", 0.0))

    # Older compact JSON stores distances along wall.
    if "start_position" in opening and "end_position" in opening and host_len:
        return float(opening["start_position"]) / host_len, float(opening["end_position"]) / host_len

    if "position_ratio" in opening:
        ratio = float(opening["position_ratio"])
        return ratio, ratio

    return float(opening.get("start_ratio", 0.0)), float(opening.get("end_ratio", 0.0))


def opening_host_policy(opening: dict[str, Any]) -> str:
    if "position_on_wall" not in opening:
        return "cut_wall"
    return str(opening["position_on_wall"].get("host_span_policy", "cut_wall"))


def opening_geometry_start_end(
    opening: dict[str, Any],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    geom = opening.get("opening_geometry")
    if not geom:
        return None
    start = geom.get("start")
    end = geom.get("end")
    if not start or not end:
        return None
    return (float(start[0]), float(start[1])), (float(end[0]), float(end[1]))


def opening_type(opening: dict[str, Any]) -> str:
    return str(opening.get("opening_type", "opening"))


def interpolate(
    start: tuple[float, float], end: tuple[float, float], ratio: float
) -> tuple[float, float]:
    return (
        start[0] + (end[0] - start[0]) * ratio,
        start[1] + (end[1] - start[1]) * ratio,
    )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def opening_width_data_units(opening: dict[str, Any], data: dict[str, Any], span: float) -> float:
    pos = opening.get("position_on_wall", {})
    width_mm = float(pos.get("width_mm", 0.0) or 0.0)
    scale_to_mm = float(data.get("coordinate_system", {}).get("scale_to_mm", 0.0) or 0.0)
    if width_mm <= 0 or scale_to_mm <= 0:
        return span * 0.035
    return width_mm / scale_to_mm


def opening_visual_start_end(
    wall_start: tuple[float, float],
    wall_end: tuple[float, float],
    opening: dict[str, Any],
    data: dict[str, Any],
    span: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    host_len = distance(wall_start, wall_end)
    if host_len == 0:
        return wall_start, wall_start

    start_ratio, end_ratio = opening_ratios(opening, host_len)
    center_ratio = float(opening.get("position_on_wall", {}).get("center_ratio", (start_ratio + end_ratio) / 2.0))
    width_ratio = abs(end_ratio - start_ratio)

    minimum_width = opening_width_data_units(opening, data, span)
    minimum_ratio = minimum_width / host_len if host_len > 0 else 0.0
    if opening_type(opening) in ("door", "front_door") and minimum_ratio > width_ratio:
        half = minimum_ratio / 2.0
        start_ratio = center_ratio - half
        end_ratio = center_ratio + half

        if start_ratio < 0.0:
            end_ratio -= start_ratio
            start_ratio = 0.0
        if end_ratio > 1.0:
            start_ratio -= end_ratio - 1.0
            end_ratio = 1.0
        start_ratio = max(0.0, start_ratio)
        end_ratio = min(1.0, end_ratio)

    if end_ratio < start_ratio:
        start_ratio, end_ratio = end_ratio, start_ratio

    return interpolate(wall_start, wall_end, start_ratio), interpolate(wall_start, wall_end, end_ratio)


def collect_bounds(walls: list[dict[str, Any]]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for wall in walls:
        start, end = wall_start_end(wall)
        xs.extend([start[0], end[0]])
        ys.extend([start[1], end[1]])
    return min(xs), min(ys), max(xs), max(ys)


def load_source_plans(path: str) -> list[dict[str, Any]]:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            with zf.open("ResPlan.pkl") as f:
                return pickle.load(f)

    with open(path, "rb") as f:
        return pickle.load(f)


def wall_key(wall: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    start, end = wall_start_end(wall)
    a = (round(start[0], 6), round(start[1], 6))
    b = (round(end[0], 6), round(end[1], 6))
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def is_generated_closure_wall(wall: dict[str, Any]) -> bool:
    return wall.get("generated", {}).get("type") == "exterior_closure_wall"


def compare_json_to_source_segmentation(
    data: dict[str, Any],
    source_data_path: str,
) -> dict[str, Any]:
    import export_wall_segments_json as exporter
    import resplan_utils as ru

    plan_index = int(data.get("metadata", {}).get("plan_index"))
    plans = load_source_plans(source_data_path)
    plan = plans[plan_index]
    segments = ru.split_wall_segments(
        plan,
        split_openings=False,
        filter_short_isolated_artifacts=True,
    )
    bounds = exporter.normalization_bounds(segments)

    expected = []
    for seg in segments:
        line = exporter.line_from_segment(seg)
        if line is None:
            continue
        start = exporter.normalize_point(line.coords[0], bounds)
        end = exporter.normalize_point(line.coords[-1], bounds)
        expected.append(tuple(sorted((tuple(start), tuple(end)))))

    generated_closures = [
        wall for wall in data.get("walls", [])
        if is_generated_closure_wall(wall)
    ]
    actual = [
        wall_key(wall)
        for wall in data.get("walls", [])
        if not is_generated_closure_wall(wall)
    ]
    expected_set = set(expected)
    actual_set = set(actual)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)

    return {
        "ok": not missing and not extra,
        "expected_segment_count": len(expected),
        "json_wall_count": len(actual),
        "generated_closure_count": len(generated_closures),
        "missing_segments": missing,
        "extra_segments": extra,
    }


def source_plan_for_json(data: dict[str, Any], source_data_path: str) -> dict[str, Any]:
    plan_index = int(data.get("metadata", {}).get("plan_index"))
    return load_source_plans(source_data_path)[plan_index]


def validate_before_draw(
    data: dict[str, Any],
    compare_segmentation: bool,
    source_data_path: str,
    strict_validation: bool,
) -> None:
    quality = data.get("quality_check")
    if quality is not None and not quality.get("ok", False):
        message = "JSON quality_check is not ok."
        if strict_validation:
            raise ValueError(message)
        print(f"Warning: {message}")

    if not compare_segmentation:
        return

    report = compare_json_to_source_segmentation(data, source_data_path)
    message = (
        "Segmentation comparison: "
        f"expected={report['expected_segment_count']} "
        f"json={report['json_wall_count']} "
        f"generated_closures={report['generated_closure_count']} "
        f"missing={len(report['missing_segments'])} "
        f"extra={len(report['extra_segments'])}"
    )
    if report["ok"]:
        print(f"{message} [ok]")
        return

    if strict_validation:
        raise ValueError(f"{message} [failed]")
    print(f"Warning: {message} [failed]")


def wall_linewidth(thickness_mm: float, line_scale: float) -> float:
    # Matplotlib linewidth uses points, not data units. Keep the visual readable
    # while still making exterior walls thicker than interior walls.
    return max(1.5, thickness_mm / 55.0) * line_scale


def draw_floorplan(
    data: dict[str, Any],
    output_path: Path,
    dpi: int,
    show_labels: bool,
    draw_openings: bool,
    line_scale: float,
) -> None:
    walls = data.get("walls", [])
    if not walls:
        raise ValueError("No walls found in JSON.")

    defaults = data.get("defaults", {})
    min_x, min_y, max_x, max_y = collect_bounds(walls)
    span = max(max_x - min_x, max_y - min_y)
    pad = span * 0.06 if span else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#fbfaf7")

    # Draw walls first.
    for wall in walls:
        start, end = wall_start_end(wall)
        location = wall_location(wall)
        linewidth = wall_linewidth(wall_thickness_mm(wall, defaults), line_scale)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=WALL_COLORS.get(location, WALL_COLORS["unknown"]),
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=2,
        )

        if show_labels:
            mid = interpolate(start, end, 0.5)
            ax.text(mid[0], mid[1], wall.get("wall_id", ""), fontsize=5, color="#333333", zorder=8)

    # Draw openings as small colored segments on top of the wall centerlines.
    if draw_openings:
        for wall in walls:
            start, end = wall_start_end(wall)
            host_len = distance(start, end)
            if host_len == 0:
                continue

            base_width = wall_linewidth(wall_thickness_mm(wall, defaults), line_scale)
            for opening in wall.get("openings", []):
                start_ratio, end_ratio = opening_ratios(opening, host_len)
                start_ratio = max(0.0, min(1.0, start_ratio))
                end_ratio = max(0.0, min(1.0, end_ratio))

                if end_ratio < start_ratio:
                    start_ratio, end_ratio = end_ratio, start_ratio

                o_start = interpolate(start, end, start_ratio)
                o_end = interpolate(start, end, end_ratio)
                kind = opening_type(opening)
                policy = opening_host_policy(opening)

                explicit_opening = opening_geometry_start_end(opening)
                if explicit_opening is not None:
                    o_start, o_end = explicit_opening
                    ax.plot(
                        [o_start[0], o_end[0]],
                        [o_start[1], o_end[1]],
                        color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                        linewidth=max(1.5, base_width * 0.45),
                        solid_capstyle="butt",
                        zorder=5,
                    )

                    if show_labels:
                        mid = interpolate(o_start, o_end, 0.5)
                        ax.text(
                            mid[0],
                            mid[1],
                            opening.get("opening_id", ""),
                            fontsize=5,
                            color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                            zorder=9,
                        )
                    continue

                o_start, o_end = opening_visual_start_end(start, end, opening, data, span)

                if math.isclose(start_ratio, end_ratio) or policy == "endpoint_insert":
                    center_ratio = float(
                        opening.get("position_on_wall", {}).get("center_ratio", start_ratio)
                    )
                    anchor = interpolate(start, end, max(0.0, min(1.0, center_ratio)))
                    ux = (end[0] - start[0]) / host_len
                    uy = (end[1] - start[1]) / host_len
                    if center_ratio <= 0.0:
                        ux, uy = -ux, -uy

                    mark_len = opening_width_data_units(opening, data, span)
                    mark_end = (anchor[0] + ux * mark_len, anchor[1] + uy * mark_len)
                    ax.plot(
                        [anchor[0], mark_end[0]],
                        [anchor[1], mark_end[1]],
                        color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                        linewidth=max(1.5, base_width * 0.45),
                        solid_capstyle="butt",
                        zorder=5,
                    )

                    if show_labels:
                        ax.text(
                            anchor[0],
                            anchor[1],
                            opening.get("opening_id", ""),
                            fontsize=5,
                            color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                            zorder=9,
                        )
                    continue

                # First erase the wall line at the opening, then draw a thinner
                # colored mark so doors/windows are visible.
                ax.plot(
                    [o_start[0], o_end[0]],
                    [o_start[1], o_end[1]],
                    color=ax.get_facecolor(),
                    linewidth=base_width + 1.4,
                    solid_capstyle="butt",
                    zorder=4,
                )
                ax.plot(
                    [o_start[0], o_end[0]],
                    [o_start[1], o_end[1]],
                    color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                    linewidth=max(1.2, base_width * 0.45),
                    solid_capstyle="butt",
                    zorder=5,
                )

                if show_labels:
                    mid = interpolate(o_start, o_end, 0.5)
                    ax.text(
                        mid[0],
                        mid[1],
                        opening.get("opening_id", ""),
                        fontsize=5,
                        color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                        zorder=9,
                    )

    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)

    coord = data.get("coordinate_system", {})
    if coord.get("origin") == "top_left":
        ax.invert_yaxis()

    ax.axis("off")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=WALL_COLORS["exterior"], lw=4, label="exterior wall"),
        Line2D([0], [0], color=WALL_COLORS["interior"], lw=3, label="interior wall"),
    ]
    if draw_openings:
        handles.extend(
            [
                Line2D([0], [0], color=OPENING_COLORS["door"], lw=3, label="door"),
                Line2D([0], [0], color=OPENING_COLORS["front_door"], lw=3, label="front door"),
                Line2D([0], [0], color=OPENING_COLORS["window"], lw=3, label="window"),
            ]
        )
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_clean_validation(
    data: dict[str, Any],
    output_path: Path,
    dpi: int,
    show_labels: bool,
    draw_openings: bool,
    line_scale: float,
) -> None:
    walls = data.get("walls", [])
    if not walls:
        raise ValueError("No walls found in JSON.")

    defaults = data.get("defaults", {})
    min_x, min_y, max_x, max_y = collect_bounds(walls)
    span = max(max_x - min_x, max_y - min_y)
    pad = span * 0.06 if span else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("white")

    endpoint_points: list[tuple[float, float]] = []
    for wall in walls:
        start, end = wall_start_end(wall)
        endpoint_points.extend([start, end])
        location = wall_location(wall)
        linewidth = wall_linewidth(wall_thickness_mm(wall, defaults), line_scale)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=WALL_COLORS.get(location, WALL_COLORS["unknown"]),
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=2,
        )

        if show_labels:
            mid = interpolate(start, end, 0.5)
            ax.text(mid[0], mid[1], wall.get("wall_id", ""), fontsize=5, color="#111111", zorder=8)

    if draw_openings:
        for wall in walls:
            start, end = wall_start_end(wall)
            host_len = distance(start, end)
            if host_len == 0:
                continue

            base_width = wall_linewidth(wall_thickness_mm(wall, defaults), line_scale)
            for opening in wall.get("openings", []):
                kind = opening_type(opening)
                explicit = opening_geometry_start_end(opening)
                if explicit is not None:
                    o_start, o_end = explicit
                else:
                    o_start, o_end = opening_visual_start_end(start, end, opening, data, span)

                ax.plot(
                    [o_start[0], o_end[0]],
                    [o_start[1], o_end[1]],
                    color="white",
                    linewidth=base_width + 1.8,
                    solid_capstyle="butt",
                    zorder=4,
                )
                ax.plot(
                    [o_start[0], o_end[0]],
                    [o_start[1], o_end[1]],
                    color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                    linewidth=max(2.0, base_width * 0.55),
                    solid_capstyle="butt",
                    zorder=5,
                )

                if show_labels:
                    mid = interpolate(o_start, o_end, 0.5)
                    ax.text(
                        mid[0],
                        mid[1],
                        opening.get("opening_id", ""),
                        fontsize=5,
                        color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                        zorder=9,
                    )

    if endpoint_points:
        unique_points = sorted(
            {
                (round(pt[0], 6), round(pt[1], 6))
                for pt in endpoint_points
            }
        )
        ax.scatter(
            [pt[0] for pt in unique_points],
            [pt[1] for pt in unique_points],
            s=18,
            color="black",
            marker="o",
            zorder=10,
        )

    ax.set_xlim(min_x - pad, max_x + pad)
    ax.set_ylim(min_y - pad, max_y + pad)
    coord = data.get("coordinate_system", {})
    if coord.get("origin") == "top_left":
        ax.invert_yaxis()
    ax.axis("off")

    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=WALL_COLORS["exterior"], lw=4, label="exterior wall"),
        Line2D([0], [0], color=WALL_COLORS["interior"], lw=3, label="interior wall"),
    ]
    if draw_openings:
        handles.extend(
            [
                Line2D([0], [0], color=OPENING_COLORS["door"], lw=3, label="door"),
                Line2D([0], [0], color=OPENING_COLORS["front_door"], lw=3, label="front door"),
                Line2D([0], [0], color=OPENING_COLORS["window"], lw=3, label="window"),
            ]
        )
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def draw_validation_overlay(
    data: dict[str, Any],
    source_data_path: str,
    output_path: Path,
    dpi: int,
    show_labels: bool,
    draw_openings: bool,
    line_scale: float,
) -> None:
    import export_wall_segments_json as exporter
    import resplan_plot

    plan = source_plan_for_json(data, source_data_path)
    walls = data.get("walls", [])
    if not walls:
        raise ValueError("No walls found in JSON.")

    segments = ru_segments_for_overlay(data, source_data_path)
    bounds = exporter.normalization_bounds(segments)
    min_x, min_y, max_x, max_y, span = bounds
    coord = data.get("coordinate_system", {})
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0

    def denormalize(xy: tuple[float, float]) -> tuple[float, float]:
        if coord.get("origin") == "bbox_center":
            return center_x + xy[0] * span, center_y + xy[1] * span
        return min_x + xy[0] * span, min_y + xy[1] * span

    fig, ax = plt.subplots(figsize=(8, 8))
    resplan_plot.plot_plan(
        plan,
        ax=ax,
        legend=True,
        title=f"JSON conversion validation {data.get('metadata', {}).get('plan_index', '')}",
    )

    defaults = data.get("defaults", {})
    json_points: list[tuple[float, float]] = []
    for wall in walls:
        start_norm, end_norm = wall_start_end(wall)
        start = denormalize(start_norm)
        end = denormalize(end_norm)
        json_points.extend([start, end])
        linewidth = max(1.4, wall_linewidth(wall_thickness_mm(wall, defaults), line_scale) * 0.55)
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="#111111",
            linewidth=linewidth,
            solid_capstyle="butt",
            zorder=30,
        )

        if show_labels:
            mid = interpolate(start, end, 0.5)
            ax.text(mid[0], mid[1], wall.get("wall_id", ""), fontsize=5, color="#111111", zorder=35)

        if not draw_openings:
            continue

        for opening in wall.get("openings", []):
            explicit = opening_geometry_start_end(opening)
            if explicit is not None:
                o_start = denormalize(explicit[0])
                o_end = denormalize(explicit[1])
            else:
                host_len = distance(start_norm, end_norm)
                if host_len == 0:
                    continue
                start_ratio, end_ratio = opening_ratios(opening, host_len)
                o_start = denormalize(interpolate(start_norm, end_norm, start_ratio))
                o_end = denormalize(interpolate(start_norm, end_norm, end_ratio))

            kind = opening_type(opening)
            ax.plot(
                [o_start[0], o_end[0]],
                [o_start[1], o_end[1]],
                color=OPENING_COLORS.get(kind, OPENING_COLORS["opening"]),
                linewidth=3.0,
                solid_capstyle="butt",
                zorder=40,
            )

    if json_points:
        ax.scatter(
            [pt[0] for pt in json_points],
            [pt[1] for pt in json_points],
            s=18,
            color="black",
            marker="o",
            zorder=45,
            label="JSON wall endpoints",
        )
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def ru_segments_for_overlay(data: dict[str, Any], source_data_path: str) -> list[dict[str, Any]]:
    import resplan_utils as ru

    plan = source_plan_for_json(data, source_data_path)
    return ru.split_wall_segments(
        plan,
        split_openings=False,
        filter_short_isolated_artifacts=True,
    )


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    output_path = Path(args.output)

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    validate_before_draw(
        data=data,
        compare_segmentation=args.compare_segmentation,
        source_data_path=args.data,
        strict_validation=args.strict_validation,
    )

    if args.validation_clean:
        draw_clean_validation(
            data=data,
            output_path=output_path,
            dpi=args.dpi,
            show_labels=args.show_labels,
            draw_openings=not args.no_openings,
            line_scale=args.line_scale,
        )
    elif args.validation_overlay:
        draw_validation_overlay(
            data=data,
            source_data_path=args.data,
            output_path=output_path,
            dpi=args.dpi,
            show_labels=args.show_labels,
            draw_openings=not args.no_openings,
            line_scale=args.line_scale,
        )
    else:
        draw_floorplan(
            data=data,
            output_path=output_path,
            dpi=args.dpi,
            show_labels=args.show_labels,
            draw_openings=not args.no_openings,
            line_scale=args.line_scale,
        )
    print(f"Saved floor plan to {output_path}")


if __name__ == "__main__":
    main()
