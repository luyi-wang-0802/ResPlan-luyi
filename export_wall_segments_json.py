"""Export ResPlan wall segmentation results as Vectorworks-friendly JSON.

Examples:
    python export_wall_segments_json.py --indices 1
    python export_wall_segments_json.py --indices 0 1 2 --out-dir assets/wall_segments_json
    python export_wall_segments_json.py --random 10 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import zipfile
import math
from statistics import median
from typing import Any, Dict, List, Tuple

from shapely.geometry import LineString, Point

import resplan_utils as ru


SCALE_TO_MM = 10000
WALL_HEIGHT_MM = 3000
WALL_THICKNESS_MM = {
    "exterior": 300,
    "interior": 150,
    "unknown": 150,
}
OPENING_DEFAULTS = {
    "door": {
        "width_mm": 900,
        "height_mm": 2100,
        "sill_height_mm": 0,
        "thickness_mode": "use_vectorworks_default",
    },
    "front_door": {
        "width_mm": 1000,
        "height_mm": 2100,
        "sill_height_mm": 0,
        "thickness_mode": "use_vectorworks_default",
    },
    "window": {
        "height_mm": 1200,
        "sill_height_mm": 900,
        "thickness_mode": "use_vectorworks_default",
    },
    "opening": {
        "height_mm": 2100,
        "sill_height_mm": 0,
        "thickness_mode": "use_vectorworks_default",
    },
}


def load_plans(path: str) -> List[Dict[str, Any]]:
    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            with zf.open("ResPlan.pkl") as f:
                return pickle.load(f)

    with open(path, "rb") as f:
        return pickle.load(f)


def choose_indices(total: int, args: argparse.Namespace) -> List[int]:
    if args.indices:
        return args.indices

    count = min(args.random, total)
    rng = random.Random(args.seed)
    return sorted(rng.sample(range(total), count))


def line_to_coords(line: Any) -> Any:
    if isinstance(line, LineString):
        return [[float(x), float(y)] for x, y in line.coords]
    return None


def point_to_xy(point: Any) -> Any:
    return [float(point[0]), float(point[1])]


def room_name(room_type: str, index: int) -> str:
    return f"{room_type.replace('_', ' ').title()} {index + 1}"


def line_from_segment(seg: Dict[str, Any]) -> Any:
    line = seg.get("baseline") or seg.get("face_line") or seg.get("line")
    if isinstance(line, LineString) and not line.is_empty and line.length > 0:
        return line
    return None


def source_opening_ids(plan: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for opening_type in ("door", "window", "front_door"):
        for i, geom in enumerate(ru.get_geometries(plan.get(opening_type))):
            if geom is None or geom.is_empty:
                continue
            ids.append(f"{opening_type}_{i}")
    return ids


def segment_node_key(xy: Any, precision: int = 6) -> Tuple[float, float]:
    return round(float(xy[0]), precision), round(float(xy[1]), precision)


def normalized_segment_node_keys(
    segments: List[Dict[str, Any]],
    bounds: Tuple[float, float, float, float, float],
) -> List[Tuple[float, float]]:
    nodes: List[Tuple[float, float]] = []
    seen = set()
    for seg in segments:
        line = line_from_segment(seg)
        if line is None:
            continue
        for xy in (line.coords[0], line.coords[-1]):
            key = segment_node_key(normalize_point(xy, bounds))
            if key in seen:
                continue
            seen.add(key)
            nodes.append(key)
    return nodes


def line_angle(line: LineString) -> float:
    p0, p1 = list(line.coords)[0], list(line.coords)[-1]
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0


def collect_rooms(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rooms: List[Dict[str, Any]] = []
    for room_type in ru.DEFAULT_ROOM_KEYS:
        for i, geom in enumerate(ru.get_geometries(plan.get(room_type))):
            if geom is None or geom.is_empty:
                continue
            name = room_name(room_type, i)
            rooms.append(
                {
                    "room_id": f"{room_type}_{i}",
                    "room_type": room_type,
                    "room_name": name,
                    "ifc_space_name": name,
                    "ifc_space_generation": "by_vectorworks_space_boundary",
                }
            )
    return rooms


def room_geometries(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    rooms: List[Dict[str, Any]] = []
    for room_type in ru.DEFAULT_ROOM_KEYS:
        for i, geom in enumerate(ru.get_geometries(plan.get(room_type))):
            if geom is None or geom.is_empty:
                continue
            rooms.append(
                {
                    "room_id": f"{room_type}_{i}",
                    "room_type": room_type,
                    "geometry": geom,
                }
            )
    return rooms


def adjacent_rooms(plan: Dict[str, Any], line: LineString, wall_depth: float) -> List[Dict[str, Any]]:
    rooms = room_geometries(plan)
    if not rooms:
        return []

    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    dx = x1 - x0
    dy = y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1e-9)
    nx = -dy / length
    ny = dx / length
    offset = max(wall_depth * 0.70, 1.0)
    sample_count = max(3, min(24, int(line.length / max(wall_depth * 2.0, 4.0)) + 1))
    hits: Dict[str, Dict[str, Any]] = {}

    for i in range(sample_count):
        d = line.length * (i + 0.5) / sample_count
        p = line.interpolate(d)
        side_points = {
            "left": (p.x + nx * offset, p.y + ny * offset),
            "right": (p.x - nx * offset, p.y - ny * offset),
        }

        for side, xy in side_points.items():
            for room in rooms:
                geom = room["geometry"]
                if not geom.covers(Point(xy)):
                    continue
                rid = room["room_id"]
                item = hits.setdefault(
                    rid,
                    {
                        "room_id": rid,
                        "room_type": room["room_type"],
                        "sides": set(),
                        "sample_hits": 0,
                    },
                )
                item["sides"].add(side)
                item["sample_hits"] += 1

    out = []
    for item in hits.values():
        out.append(
            {
                "room_id": item["room_id"],
                "room_type": item["room_type"],
                "sides": sorted(item["sides"]),
                "sample_hits": item["sample_hits"],
            }
        )

    out.sort(key=lambda item: (-item["sample_hits"], item["room_id"]))
    return out


def wall_location(plan: Dict[str, Any], line: LineString, adjacent: List[Dict[str, Any]], wall_depth: float) -> str:
    inner = plan.get("inner")
    mid = line.interpolate(line.length / 2.0)
    if inner is not None and not inner.is_empty and not inner.buffer(wall_depth * 0.25).covers(mid):
        return "exterior"

    sides = {side for room in adjacent for side in room.get("sides", [])}
    if "left" in sides and "right" in sides:
        return "interior"
    if adjacent:
        return "exterior"
    return "unknown"


def normalization_bounds(segments: List[Dict[str, Any]]) -> Tuple[float, float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for seg in segments:
        line = line_from_segment(seg)
        if line is None:
            continue
        for x, y in line.coords:
            xs.append(float(x))
            ys.append(float(y))

    if not xs or not ys:
        raise ValueError("Cannot normalize coordinates because no wall segment endpoints were found.")

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span = max(max_x - min_x, max_y - min_y)
    if span <= 0:
        raise ValueError("Cannot normalize coordinates because the wall bounding box is empty.")

    return min_x, min_y, max_x, max_y, span


def normalize_point(xy: Any, bounds: Tuple[float, float, float, float, float]) -> List[float]:
    min_x, min_y, _, _, span = bounds
    return [
        round((float(xy[0]) - min_x) / span, 6),
        round((float(xy[1]) - min_y) / span, 6),
    ]


def normalized_line(
    line: LineString,
    bounds: Tuple[float, float, float, float, float],
) -> Tuple[List[float], List[float], float]:
    start = normalize_point(line.coords[0], bounds)
    end = normalize_point(line.coords[-1], bounds)
    length_ratio = math.hypot(end[0] - start[0], end[1] - start[1])
    return start, end, round(length_ratio, 6)


def geometry_major_axis_width(geom: Any) -> float:
    if geom is None or geom.is_empty:
        return 0.0
    x0, y0, x1, y1 = geom.bounds
    return max(float(x1 - x0), float(y1 - y0))


def width_to_mm(width: float, bounds: Tuple[float, float, float, float, float]) -> int:
    return int(round((width / bounds[-1]) * SCALE_TO_MM))


def plan_default_opening_widths(
    plan: Dict[str, Any],
    bounds: Tuple[float, float, float, float, float],
) -> Dict[str, int]:
    defaults = {
        "door": int(OPENING_DEFAULTS["door"]["width_mm"]),
        "front_door": int(OPENING_DEFAULTS["front_door"]["width_mm"]),
    }

    for opening_type in ("door", "front_door"):
        widths = [
            width_to_mm(geometry_major_axis_width(geom), bounds)
            for geom in ru.get_geometries(plan.get(opening_type))
            if geom is not None and not geom.is_empty
        ]
        widths = [width for width in widths if width > 0]
        if widths:
            defaults[opening_type] = int(round(median(widths)))

    return defaults


def opening_axis_width(opening: Dict[str, Any], line: LineString) -> float:
    interval = opening_axis_interval(opening, line)
    if interval is None:
        return 0.0
    return interval[1] - interval[0]


def opening_axis_interval(opening: Dict[str, Any], line: LineString) -> Tuple[float, float] | None:
    geom = opening.get("geometry")
    if geom is None or geom.is_empty or line.length <= 0:
        return None

    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    ux = (x1 - x0) / line.length
    uy = (y1 - y0) / line.length

    def axis_projection(xy: Any) -> float:
        return (float(xy[0]) - x0) * ux + (float(xy[1]) - y0) * uy

    projections: List[float] = []
    if isinstance(geom, LineString):
        projections.extend(axis_projection(xy) for xy in geom.coords)
    elif hasattr(geom, "exterior"):
        projections.extend(axis_projection(xy) for xy in geom.exterior.coords)
    else:
        try:
            bx0, by0, bx1, by1 = geom.bounds
            projections.extend(
                [
                    axis_projection((bx0, by0)),
                    axis_projection((bx0, by1)),
                    axis_projection((bx1, by0)),
                    axis_projection((bx1, by1)),
                ]
            )
        except Exception:
            return None

    if not projections:
        return None

    return min(projections), max(projections)


def point_at_axis_position(line: LineString, position: float) -> Tuple[float, float]:
    (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
    ux = (x1 - x0) / line.length
    uy = (y1 - y0) / line.length
    return x0 + ux * position, y0 + uy * position


def opening_axis_geometry(
    opening: Dict[str, Any],
    line: LineString,
    bounds: Tuple[float, float, float, float, float],
) -> Dict[str, Any] | None:
    interval = opening_axis_interval(opening, line)
    if interval is None:
        return None

    start_pos, end_pos = interval
    start_xy = point_at_axis_position(line, start_pos)
    end_xy = point_at_axis_position(line, end_pos)
    start = normalize_point(start_xy, bounds)
    end = normalize_point(end_xy, bounds)
    length_ratio = math.hypot(end[0] - start[0], end[1] - start[1])

    return {
        "start": start,
        "end": end,
        "length_ratio": round(length_ratio, 6),
    }


def opening_raw_axis_geometry(
    opening: Dict[str, Any],
    bounds: Tuple[float, float, float, float, float],
) -> Dict[str, Any] | None:
    geom = opening.get("geometry")
    if geom is None or geom.is_empty:
        return None

    minx, miny, maxx, maxy = geom.bounds
    width = float(maxx - minx)
    height = float(maxy - miny)
    if width <= 0 and height <= 0:
        return None

    if width >= height:
        y = (float(miny) + float(maxy)) / 2.0
        start_xy = (float(minx), y)
        end_xy = (float(maxx), y)
    else:
        x = (float(minx) + float(maxx)) / 2.0
        start_xy = (x, float(miny))
        end_xy = (x, float(maxy))

    start = normalize_point(start_xy, bounds)
    end = normalize_point(end_xy, bounds)
    length_ratio = math.hypot(end[0] - start[0], end[1] - start[1])

    return {
        "start": start,
        "end": end,
        "length_ratio": round(length_ratio, 6),
    }


def opening_host_interval(opening: Dict[str, Any], line: LineString) -> Tuple[float, float, str]:
    line_len = float(line.length)
    start = opening.get("start_position")
    end = opening.get("end_position")

    if start is None or end is None:
        center = opening.get("position")
        if center is None:
            geom = opening.get("geometry")
            center = float(line.project(geom.representative_point())) if geom is not None else 0.0
        start = end = float(center)

    start = max(0.0, min(line_len, float(start)))
    end = max(0.0, min(line_len, float(end)))
    if end < start:
        start, end = end, start

    if end > start:
        return start, end, "projection_overlap"

    width = opening_axis_width(opening, line)
    if width <= 0:
        return start, end, "zero_width_unresolved"

    return start, end, "endpoint_gap"


def expand_interval_to_width(
    start: float,
    end: float,
    line_length: float,
    width: float,
) -> Tuple[float, float]:
    if line_length <= 0 or width <= 0:
        return start, end

    width = min(width, line_length)
    center = (start + end) / 2.0
    start = center - width / 2.0
    end = center + width / 2.0

    if start < 0.0:
        end -= start
        start = 0.0
    if end > line_length:
        start -= end - line_length
        end = line_length

    return max(0.0, start), min(line_length, end)


def mm_to_data_units(width_mm: float, bounds: Tuple[float, float, float, float, float]) -> float:
    return (float(width_mm) / SCALE_TO_MM) * bounds[-1]


def normalized_opening_type(opening: Dict[str, Any]) -> str:
    return str(opening.get("type", "opening"))


def serialize_opening(
    opening: Dict[str, Any],
    line: LineString,
    line_length_ratio: float,
    bounds: Tuple[float, float, float, float, float],
    room_membership: List[Dict[str, Any]],
    default_widths_mm: Dict[str, int],
) -> Dict[str, Any]:
    start, end, _ = opening_host_interval(opening, line)
    opening_type = normalized_opening_type(opening)

    measured_width = opening_axis_width(opening, line)
    width_for_mm = measured_width if measured_width > 0 else opening_axis_width(opening, line)
    if width_for_mm <= 0:
        raw_geom = opening_raw_axis_geometry(opening, bounds)
        width_mm = int(default_widths_mm.get(opening_type, OPENING_DEFAULTS.get(opening_type, {}).get("width_mm", 0)))
        if raw_geom is not None:
            width_mm = int(round(raw_geom["length_ratio"] * SCALE_TO_MM))
    else:
        width_mm = width_to_mm(width_for_mm, bounds)

    if opening_type in ("door", "front_door") and width_mm <= 0:
        width_mm = int(default_widths_mm[opening_type])
    if opening_type == "window" and width_mm <= 0:
        width_mm = 1200

    center = (start + end) / 2.0
    host_width = end - start

    start_ratio = start / line.length if line.length > 0 else 0.0
    end_ratio = end / line.length if line.length > 0 else 0.0
    center_ratio = center / line.length if line.length > 0 else 0.0
    width_ratio = host_width / line.length if line.length > 0 else 0.0
    opening_geometry = opening_raw_axis_geometry(opening, bounds)

    belongs_to_rooms = [room["room_id"] for room in room_membership]
    connects_rooms = belongs_to_rooms if opening_type in ("door", "front_door") else []

    return {
        "opening_id": opening.get("id"),
        "opening_type": opening_type,
        "host_wall_id": opening.get("attached_wall_id"),
        "opening_geometry": opening_geometry,
        "position_on_wall": {
            "start_ratio": round(start_ratio, 6),
            "end_ratio": round(end_ratio, 6),
            "center_ratio": round(center_ratio, 6),
            "width_ratio": round(width_ratio, 6),
            "width_mm": width_mm,
        },
        "semantic": {
            "connects_rooms": connects_rooms,
            "belongs_to_rooms": belongs_to_rooms,
        },
    }


def serialize_wall(
    plan: Dict[str, Any],
    seg: Dict[str, Any],
    wall_depth: float,
    bounds: Tuple[float, float, float, float, float],
    default_widths_mm: Dict[str, int],
) -> Dict[str, Any]:
    line = line_from_segment(seg)
    if line is None:
        return {}

    rooms = adjacent_rooms(plan, line, wall_depth)
    location = wall_location(plan, line, rooms, wall_depth)
    start, end, length_ratio = normalized_line(line, bounds)
    openings = [
        serialize_opening(o, line, length_ratio, bounds, rooms, default_widths_mm)
        for o in seg.get("openings", [])
    ]

    return {
        "wall_id": seg.get("id"),
        "geometry": {
            "start": start,
            "end": end,
            "length_ratio": length_ratio,
            "angle_deg": round(line_angle(line), 6),
        },
        "physical": {
            "wall_location": location,
            "thickness_mm": WALL_THICKNESS_MM.get(location, WALL_THICKNESS_MM["unknown"]),
            "height_mm": WALL_HEIGHT_MM,
        },
        "room_membership": [
            {
                "room_id": room["room_id"],
                "room_type": room["room_type"],
                "side": room["sides"][0] if room.get("sides") else "unknown",
                "relation": "bounds_room",
            }
            for room in rooms
        ],
        "openings": openings,
        "quality_check": {
            "shared_by_multiple_rooms": len(rooms) > 1,
            "room_count": len(rooms),
            "has_openings": bool(openings),
        },
    }


def is_generated_closure_wall(wall: Dict[str, Any]) -> bool:
    return wall.get("generated", {}).get("type") == "exterior_closure_wall"


def point_distance(a: List[float], b: List[float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def endpoint_cluster_key(point: List[float], tolerance: float) -> Tuple[int, int]:
    return (
        int(round(float(point[0]) / tolerance)),
        int(round(float(point[1]) / tolerance)),
    )


def exterior_endpoint_closure_candidates(
    dangling: List[Dict[str, Any]],
    axis_tolerance: float,
    max_gap: float,
) -> List[Tuple[float, int, int, str]]:
    candidates: List[Tuple[float, int, int, str]] = []

    for i, a in enumerate(dangling):
        for j in range(i + 1, len(dangling)):
            b = dangling[j]
            if a["wall_id"] == b["wall_id"]:
                continue

            dx = abs(float(a["point"][0]) - float(b["point"][0]))
            dy = abs(float(a["point"][1]) - float(b["point"][1]))
            distance = math.hypot(dx, dy)
            if distance <= 1e-9 or distance > max_gap:
                continue

            if dx <= axis_tolerance:
                axis = "vertical"
            elif dy <= axis_tolerance:
                axis = "horizontal"
            else:
                continue

            candidates.append((distance, i, j, axis))

    return sorted(candidates, key=lambda item: item[0])


def add_exterior_closure_walls(
    walls: List[Dict[str, Any]],
    wall_depth: float,
    bounds: Tuple[float, float, float, float, float],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    span = bounds[-1]
    if span <= 0:
        return walls, {"added_count": 0, "reason": "empty_bounds"}

    wall_depth_ratio = wall_depth / span
    node_tolerance = max(wall_depth_ratio * 0.05, 1e-5)
    axis_tolerance = max(wall_depth_ratio * 0.50, 1e-4)
    max_gap = min(max(wall_depth_ratio * 2.50, 0.015), 0.06)

    endpoints: List[Dict[str, Any]] = []
    for wall in walls:
        if wall.get("physical", {}).get("wall_location") != "exterior":
            continue
        geometry = wall.get("geometry", {})
        for endpoint_name in ("start", "end"):
            point = geometry.get(endpoint_name)
            if not point:
                continue
            endpoints.append(
                {
                    "wall_id": wall.get("wall_id"),
                    "endpoint": endpoint_name,
                    "point": [float(point[0]), float(point[1])],
                }
            )

    clusters: Dict[Tuple[int, int], List[int]] = {}
    for i, endpoint in enumerate(endpoints):
        clusters.setdefault(endpoint_cluster_key(endpoint["point"], node_tolerance), []).append(i)

    dangling = [
        endpoints[indexes[0]]
        for indexes in clusters.values()
        if len(indexes) == 1
    ]

    candidates = exterior_endpoint_closure_candidates(
        dangling,
        axis_tolerance=axis_tolerance,
        max_gap=max_gap,
    )

    used: set[int] = set()
    closure_walls: List[Dict[str, Any]] = []

    def add_closure_wall(
        p0: List[float],
        p1: List[float],
        a: Dict[str, Any],
        b: Dict[str, Any],
        axis: str,
        segment_index: int | None = None,
    ) -> None:
        if axis == "vertical":
            x = round((p0[0] + p1[0]) / 2.0, 6)
            p0 = [x, round(p0[1], 6)]
            p1 = [x, round(p1[1], 6)]
        elif axis == "horizontal":
            y = round((p0[1] + p1[1]) / 2.0, 6)
            p0 = [round(p0[0], 6), y]
            p1 = [round(p1[0], 6), y]
        else:
            p0 = [round(p0[0], 6), round(p0[1], 6)]
            p1 = [round(p1[0], 6), round(p1[1], 6)]

        length_ratio = point_distance(p0, p1)
        if length_ratio <= 1e-9:
            return

        angle_deg = 90.0 if abs(p0[0] - p1[0]) <= abs(p0[1] - p1[1]) else 0.0
        closure_id = f"exterior_closure_{len(closure_walls):04d}"
        closure_walls.append(
            {
                "wall_id": closure_id,
                "geometry": {
                    "start": p0,
                    "end": p1,
                    "length_ratio": round(length_ratio, 6),
                    "angle_deg": angle_deg,
                },
                "physical": {
                    "wall_location": "exterior",
                    "thickness_mm": WALL_THICKNESS_MM["exterior"],
                    "height_mm": WALL_HEIGHT_MM,
                },
                "room_membership": [],
                "openings": [],
                "quality_check": {
                    "shared_by_multiple_rooms": False,
                    "room_count": 0,
                    "has_openings": False,
                },
                "generated": {
                    "type": "exterior_closure_wall",
                    "purpose": "slab_boundary_closure",
                    "from": {
                        "wall_id": a["wall_id"],
                        "endpoint": a["endpoint"],
                    },
                    "to": {
                        "wall_id": b["wall_id"],
                        "endpoint": b["endpoint"],
                    },
                    "axis": axis,
                    "max_gap_ratio": round(max_gap, 6),
                    "axis_tolerance_ratio": round(axis_tolerance, 6),
                    "segment_index": segment_index,
                },
            }
        )

    for _, i, j, axis in candidates:
        if i in used or j in used:
            continue

        a = dangling[i]
        b = dangling[j]
        add_closure_wall(list(a["point"]), list(b["point"]), a, b, axis)
        used.update([i, j])

    remaining = [i for i in range(len(dangling)) if i not in used]
    l_candidates: List[Tuple[float, int, int]] = []
    for pos, i in enumerate(remaining):
        a = dangling[i]
        for j in remaining[pos + 1:]:
            b = dangling[j]
            if a["wall_id"] == b["wall_id"]:
                continue
            dx = abs(float(a["point"][0]) - float(b["point"][0]))
            dy = abs(float(a["point"][1]) - float(b["point"][1]))
            if dx <= max_gap and dy <= max_gap and dx + dy <= max_gap * 2.0:
                l_candidates.append((math.hypot(dx, dy), i, j))

    for _, i, j in sorted(l_candidates, key=lambda item: item[0]):
        if i in used or j in used:
            continue
        a = dangling[i]
        b = dangling[j]
        p0 = list(a["point"])
        p1 = list(b["point"])
        corner = [round(p0[0], 6), round(p1[1], 6)]
        add_closure_wall(p0, corner, a, b, "orthogonal_l", segment_index=0)
        add_closure_wall(corner, p1, a, b, "orthogonal_l", segment_index=1)
        used.update([i, j])

    report = {
        "enabled": True,
        "source": "exterior_dangling_endpoint_pairing",
        "exterior_endpoint_count": len(endpoints),
        "dangling_endpoint_count_before": len(dangling),
        "added_count": len(closure_walls),
        "unpaired_dangling_endpoint_count": len(dangling) - len(used),
        "max_gap_ratio": round(max_gap, 6),
        "axis_tolerance_ratio": round(axis_tolerance, 6),
    }

    return walls + closure_walls, report


def build_export_quality_check(
    plan: Dict[str, Any],
    segments: List[Dict[str, Any]],
    walls: List[Dict[str, Any]],
    bounds: Tuple[float, float, float, float, float],
    exterior_closure_report: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    segmentation_report = ru.validate_wall_segmentation(
        plan,
        segments=segments,
        check_openings=True,
    )
    expected_opening_ids = set(source_opening_ids(plan))
    attached_segment_opening_ids = {
        oid
        for seg in segments
        for oid in seg.get("opening_ids", [])
    }
    exported_opening_ids = {
        opening.get("opening_id")
        for wall in walls
        for opening in wall.get("openings", [])
        if opening.get("opening_id") is not None
    }
    host_mismatches = [
        {
            "opening_id": opening.get("opening_id"),
            "opening_host_wall_id": opening.get("host_wall_id"),
            "containing_wall_id": wall.get("wall_id"),
        }
        for wall in walls
        for opening in wall.get("openings", [])
        if opening.get("host_wall_id") != wall.get("wall_id")
    ]

    segment_nodes = normalized_segment_node_keys(segments, bounds)
    json_nodes = []
    seen_json_nodes = set()
    for wall in walls:
        if is_generated_closure_wall(wall):
            continue
        geometry = wall.get("geometry", {})
        for key in ("start", "end"):
            if key not in geometry:
                continue
            node = segment_node_key(geometry[key])
            if node in seen_json_nodes:
                continue
            seen_json_nodes.add(node)
            json_nodes.append(node)

    missing_json_nodes = [
        list(node)
        for node in segment_nodes
        if node not in seen_json_nodes
    ]
    extra_json_nodes = [
        list(node)
        for node in json_nodes
        if node not in set(segment_nodes)
    ]

    unattached_opening_ids = sorted(expected_opening_ids - attached_segment_opening_ids)
    unexported_attached_opening_ids = sorted(attached_segment_opening_ids - exported_opening_ids)
    unexported_source_opening_ids = sorted(expected_opening_ids - exported_opening_ids)

    ok = (
        bool(segmentation_report.get("ok"))
        and not unattached_opening_ids
        and not unexported_attached_opening_ids
        and not host_mismatches
        and not missing_json_nodes
        and not extra_json_nodes
    )

    return {
        "ok": ok,
        "segmentation_report": {
            "segment_count": segmentation_report.get("segment_count"),
            "split_by_opening_count": segmentation_report.get("split_by_opening_count"),
            "split_intersection_part_count": segmentation_report.get("split_intersection_part_count"),
            "endpoint_repaired_count": segmentation_report.get("endpoint_repaired_count"),
            "dangling_wall_count": segmentation_report.get("dangling_wall_count"),
            "opening_count": segmentation_report.get("opening_count"),
            "unattached_openings": segmentation_report.get("unattached_openings", []),
            "too_short_segments": segmentation_report.get("too_short_segments", []),
            "unsplit_intersections": [
                {
                    "type": issue.get("type"),
                    "segment_ids": issue.get("segment_ids"),
                    "point": normalize_point(issue["point"].coords[0], bounds)
                    if isinstance(issue.get("point"), Point)
                    else None,
                }
                for issue in segmentation_report.get("unsplit_intersections", [])
            ],
            "ok": segmentation_report.get("ok"),
        },
        "opening_attachment": {
            "source_opening_count": len(expected_opening_ids),
            "attached_opening_count": len(attached_segment_opening_ids),
            "exported_opening_count": len(exported_opening_ids),
            "unattached_opening_ids": unattached_opening_ids,
            "unexported_attached_opening_ids": unexported_attached_opening_ids,
            "unexported_source_opening_ids": unexported_source_opening_ids,
            "host_wall_mismatches": host_mismatches,
        },
        "node_consistency": {
            "source": "same split_wall_segments output used by debug plot and JSON export",
            "segment_node_count": len(segment_nodes),
            "json_node_count": len(json_nodes),
            "missing_json_nodes": missing_json_nodes,
            "extra_json_nodes": extra_json_nodes,
            "ok": not missing_json_nodes and not extra_json_nodes,
        },
        "exterior_closure": exterior_closure_report or {"enabled": False},
    }


def export_one(plan: Dict[str, Any], index: int, out_dir: str, strict_validation: bool = False) -> str:
    segments = ru.split_wall_segments(
        plan,
        split_openings=False,
        filter_short_isolated_artifacts=True,
    )
    wall_depth = float(plan.get("wall_depth") or plan.get("wall_width") or 6.0)
    bounds = normalization_bounds(segments)
    default_widths_mm = plan_default_opening_widths(plan, bounds)
    walls = [
        serialize_wall(plan, seg, wall_depth, bounds, default_widths_mm)
        for seg in segments
    ]
    walls = [wall for wall in walls if wall]
    walls, exterior_closure_report = add_exterior_closure_walls(
        walls,
        wall_depth=wall_depth,
        bounds=bounds,
    )
    openings_index = [
        {
            "opening_id": opening["opening_id"],
            "opening_type": opening["opening_type"],
            "host_wall_id": wall["wall_id"],
        }
        for wall in walls
        for opening in wall.get("openings", [])
    ]
    quality_check = build_export_quality_check(
        plan,
        segments,
        walls,
        bounds,
        exterior_closure_report=exterior_closure_report,
    )
    if strict_validation and not quality_check["ok"]:
        raise ValueError(
            f"Export validation failed for plan {index}: "
            f"{json.dumps(quality_check, ensure_ascii=False)}"
        )

    payload = {
        "schema_version": "1.2",
        "metadata": {
            "plan_id": f"plan_{index:04d}",
            "plan_index": index,
            "source_dataset": "ResPlan",
            "unit": "mm",
        },
        "coordinate_system": {
            "input_coordinates": "normalized",
            "origin": "bottom_left",
            "x_axis": "right",
            "y_axis": "up",
            "normalization_method": "bbox_uniform_scale",
            "scale_to_mm": SCALE_TO_MM,
            "target_coordinate_system": {
                "origin": "model_origin",
                "x_axis": "right",
                "y_axis": "up",
                "z_axis": "up",
            },
        },
        "defaults": {
            "wall_height_mm": WALL_HEIGHT_MM,
            "wall_thickness_mm": WALL_THICKNESS_MM,
            "opening_defaults": {
                **OPENING_DEFAULTS,
                "door": {
                    **OPENING_DEFAULTS["door"],
                    "width_mm": default_widths_mm["door"],
                },
                "front_door": {
                    **OPENING_DEFAULTS["front_door"],
                    "width_mm": default_widths_mm["front_door"],
                },
            },
        },
        "rooms": collect_rooms(plan),
        "walls": walls,
        "openings_index": openings_index,
        "quality_check": quality_check,
    }

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"resplan_to_JSON_{index:03d}.json")
    with open(out_path, "w", encoding="utf-8") as f:
      json.dump(payload, f, indent=2)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="ResPlan.zip", help="Path to ResPlan.zip or ResPlan.pkl.")
    parser.add_argument("--indices", type=int, nargs="*", help="Explicit plan indices to export.")
    parser.add_argument("--random", type=int, default=1, help="Random plan count when --indices is not set.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--out-dir", default="assets/wall_segments_json", help="Output JSON directory.")
    parser.add_argument(
        "--strict-validation",
        action="store_true",
        help="Fail export when openings are unattached or JSON nodes diverge from split wall nodes.",
    )
    args = parser.parse_args()

    plans = load_plans(args.data)
    indices = choose_indices(len(plans), args)

    print(f"Loaded {len(plans)} plans from {args.data}")
    for index in indices:
        out_path = export_one(plans[index], index, args.out_dir, args.strict_validation)
        with open(out_path, "r", encoding="utf-8") as f:
            quality_ok = json.load(f).get("quality_check", {}).get("ok")
        status = "ok" if quality_ok else "check_failed"
        print(f"{index:>5} -> {out_path} [{status}]")


if __name__ == "__main__":
    main()
