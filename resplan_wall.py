"""Wall splitting utilities for ResPlan-style floorplan datasets.

Primary geometry source:
    plan["wall"]

Output:
    - wall centerline segments
    - openings attached as metadata:
        seg["openings"]
        seg["opening_ids"]

Design:
    - wall centerlines are derived from paired opposite wall polygon edges
    - openings are used as topology splitters
    - dangling endpoints are repaired by snapping/extending to nearby
      perpendicular wall centerline intersections
    - repaired endpoints must lie in wall center area, not on wall polygon boundary
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import unary_union

from resplan_constants import DEFAULT_ROOM_KEYS
from resplan_geometry import (
    _dedupe_points,
    _is_near_perpendicular,
    _line_angle,
    _line_unit_and_normal,
    _normalized_angle_delta,
    _projection_interval,
    get_geometries,
    normalize_keys,
)


# ---------------------------------------------------------------------
# Parameters / basic geometry
# ---------------------------------------------------------------------

def _plan_wall_depth(plan: Dict[str, Any], fallback: float = 6.0) -> float:
    for key in ("wall_depth", "wall_width"):
        value = plan.get(key)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return fallback


def _wall_params(plan: Dict[str, Any]) -> Dict[str, float]:
    d = _plan_wall_depth(plan)
    return {
        "depth": d,
        "min_length": max(d * 0.30, 0.5),
        "max_gap": max(d * 4.0, 8.0),
        "max_opening_gap": max(d * 8.0, 20.0),
        "split_tolerance": max(d * 0.10, 0.15),
        "endpoint_snap": max(d * 1.25, 2.0),
        "offset_tolerance": max(d * 0.40, 1.0),
        "opening_gap_tolerance": max(d * 0.75, 1.0),
        "opening_attach_distance": max(d * 2.50, 2.0),
        "opening_attach_buffer": max(d * 0.90, 1.0),
        "connectivity_tolerance": max(d * 0.35, 1.0),
        "wall_support_tolerance": max(d * 0.35, 0.75),
        "dangling_repair_distance": max(d * 3.5, 8.0),
        "wall_boundary_clearance": max(d * 0.22, 0.75),
        "axis_endpoint_alignment_tolerance": max(d * 0.50, 1.5),
    }


def _wall_geometry(plan: Dict[str, Any]) -> Optional[Any]:
    parts = [
        geom
        for geom in get_geometries(plan.get("wall"))
        if geom is not None and not geom.is_empty
    ]
    return unary_union(parts) if parts else None


def _opening_geometries(plan: Dict[str, Any]) -> List[Any]:
    geoms: List[Any] = []
    for key in ("door", "window", "front_door"):
        geoms.extend(get_geometries(plan.get(key)))
    return [g for g in geoms if g is not None and not g.is_empty]


def _wall_centerline_geometry(plan: Dict[str, Any]) -> Optional[Any]:
    parts = [
        geom
        for geom in get_geometries(plan.get("wall"))
        if geom is not None and not geom.is_empty
    ]
    parts.extend(_opening_geometries(plan))
    return unary_union(parts) if parts else None


def _iter_openings(
    plan: Dict[str, Any],
    opening_keys: Iterable[str] = ("door", "window", "front_door"),
) -> Iterable[Dict[str, Any]]:
    for opening_type in opening_keys:
        for i, geom in enumerate(get_geometries(plan.get(opening_type))):
            if geom is None or geom.is_empty:
                continue
            yield {
                "id": f"{opening_type}_{i}",
                "type": opening_type,
                "geometry": geom,
                "length": float(getattr(geom, "length", 0.0) or 0.0),
                "area": float(getattr(geom, "area", 0.0) or 0.0),
            }


def _line_from_segment(seg: Dict[str, Any]) -> Optional[LineString]:
    line = seg.get("baseline") or seg.get("face_line") or seg.get("line")
    if isinstance(line, LineString) and not line.is_empty and line.length > 0:
        return line
    return None


def _make_segment(
    sid: str,
    line: LineString,
    wall_depth: Optional[float] = None,
    **extra: Any,
) -> Dict[str, Any]:
    seg = {
        "id": sid,
        "type": "wall_segment",
        "baseline": line,
        "line": line,
        "length": float(line.length),
        "angle": float(_line_angle(line)),
        "paired": True,
        "face_ids": [],
        "source_rooms": [],
        "source_room_types": ["wall"],
        "thickness": wall_depth,
        "overlap_length": None,
        "openings": [],
        "opening_ids": [],
        "source": "wall_geometry",
    }
    seg.update(extra)
    return seg


def _geometry_to_points(geom: Any) -> List[Point]:
    if geom is None or geom.is_empty:
        return []

    if isinstance(geom, Point):
        return [geom]

    if isinstance(geom, MultiPoint):
        return [pt for pt in geom.geoms if not pt.is_empty]

    if isinstance(geom, LineString):
        return [Point(geom.coords[0]), Point(geom.coords[-1])]

    points: List[Point] = []
    for part in get_geometries(geom):
        if isinstance(part, Point):
            points.append(part)
        elif isinstance(part, LineString) and not part.is_empty:
            points.extend([Point(part.coords[0]), Point(part.coords[-1])])

    return points


def _axis_aligned(line: LineString, angle_tolerance: float = 8.0) -> bool:
    angle = _line_angle(line)
    return (
        min(abs(angle), abs(angle - 180.0)) <= angle_tolerance
        or abs(angle - 90.0) <= angle_tolerance
    )


def _dedupe_lines(lines: List[LineString], precision: int = 4) -> List[LineString]:
    seen = set()
    out: List[LineString] = []

    for line in lines:
        a = tuple(round(v, precision) for v in line.coords[0])
        b = tuple(round(v, precision) for v in line.coords[-1])
        key = tuple(sorted([a, b]))

        if key in seen:
            continue

        seen.add(key)
        out.append(line)

    return out


def _point_supported_by_wall(pt: Point, wall_geom: Any, tolerance: float) -> bool:
    if wall_geom is None or pt is None or pt.is_empty:
        return False
    return wall_geom.buffer(tolerance, join_style=2, cap_style=2).covers(pt)


def _point_too_close_to_wall_boundary(
    pt: Point,
    wall_geom: Any,
    tolerance: float,
) -> bool:
    if wall_geom is None or pt is None or pt.is_empty:
        return True

    boundaries = []
    for geom in get_geometries(wall_geom):
        if isinstance(geom, Polygon) and not geom.is_empty:
            boundaries.append(geom.boundary)

    if not boundaries:
        return False

    boundary = unary_union(boundaries)
    return pt.distance(boundary) <= tolerance


def _point_supported_by_wall_center_area(
    pt: Point,
    wall_geom: Any,
    wall_depth: float,
    boundary_clearance_ratio: float = 0.22,
) -> bool:
    if wall_geom is None or pt is None or pt.is_empty:
        return False

    support_tol = max(wall_depth * 0.35, 0.75)
    boundary_tol = max(wall_depth * boundary_clearance_ratio, 0.75)

    if not wall_geom.buffer(support_tol, join_style=2, cap_style=2).covers(pt):
        return False

    if _point_too_close_to_wall_boundary(pt, wall_geom, boundary_tol):
        return False

    return True


def _line_supported_by_wall(
    line: Optional[LineString],
    wall_geom: Any,
    wall_depth: float,
    min_overlap_ratio: float = 0.50,
) -> bool:
    if wall_geom is None or line is None or line.is_empty:
        return False

    buf = line.buffer(max(wall_depth * 0.35, 0.75), cap_style=2, join_style=2)
    if buf.area <= 0:
        return False

    return (buf.intersection(wall_geom).area / buf.area) >= min_overlap_ratio


def _line_axis_intersection(l1: LineString, l2: LineString) -> Optional[Point]:
    p1, u1, _ = _line_unit_and_normal(l1)
    p2, u2, _ = _line_unit_and_normal(l2)
    mat = np.column_stack((u1, -u2))

    try:
        t, _ = np.linalg.solve(mat, p2 - p1)
    except np.linalg.LinAlgError:
        return None

    p = p1 + t * u1
    return Point(float(p[0]), float(p[1]))


def _replace_line_endpoint(
    line: LineString,
    endpoint_index: int,
    new_pt: Point,
    min_length: float,
) -> Optional[LineString]:
    coords = [
        np.asarray(line.coords[0], dtype=float),
        np.asarray(line.coords[-1], dtype=float),
    ]

    coords[endpoint_index] = np.asarray(new_pt.coords[0], dtype=float)

    if np.linalg.norm(coords[1] - coords[0]) < min_length:
        return None

    return LineString([tuple(coords[0]), tuple(coords[1])])


# ---------------------------------------------------------------------
# Wall polygon -> centerlines
# ---------------------------------------------------------------------

def _wall_boundary_edges(wall_geom: Any, min_length: float) -> List[LineString]:
    edges: List[LineString] = []

    for geom in get_geometries(wall_geom):
        if not isinstance(geom, Polygon) or geom.is_empty:
            continue

        rings = [geom.exterior] + list(geom.interiors)
        for ring in rings:
            coords = list(ring.coords)
            for p0, p1 in zip(coords[:-1], coords[1:]):
                edge = LineString([p0, p1])
                if edge.length >= min_length:
                    edges.append(edge)

    return edges


def _centerlines_from_opposite_wall_edges(
    wall_geom: Any,
    wall_depth: float,
    min_length: float,
    angle_tolerance: float = 8.0,
    thickness_tolerance: Optional[float] = None,
) -> List[LineString]:
    edges = [
        edge
        for edge in _wall_boundary_edges(wall_geom, min_length=min_length)
        if _axis_aligned(edge, angle_tolerance)
    ]

    if not edges:
        return []

    thick_tol = thickness_tolerance if thickness_tolerance is not None else max(wall_depth * 0.65, 1.5)
    min_sep = max(wall_depth - thick_tol, wall_depth * 0.20)
    max_sep = wall_depth + thick_tol
    centerlines: List[LineString] = []

    for i, e1 in enumerate(edges):
        p0, unit, normal = _line_unit_and_normal(e1)
        a0, a1 = _projection_interval(e1, p0, unit)

        for e2 in edges[i + 1:]:
            if _normalized_angle_delta(_line_angle(e1), _line_angle(e2)) > angle_tolerance:
                continue

            sep_signed = float(np.dot(np.asarray(e2.coords[0], dtype=float) - p0, normal))
            sep = abs(sep_signed)

            if sep < min_sep or sep > max_sep:
                continue

            b0, b1 = _projection_interval(e2, p0, unit)
            o0, o1 = max(a0, b0), min(a1, b1)

            if o1 - o0 < min_length:
                continue

            q0 = p0 + unit * o0 + normal * (sep_signed / 2.0)
            q1 = p0 + unit * o1 + normal * (sep_signed / 2.0)
            center = LineString([tuple(q0), tuple(q1)])

            if _line_supported_by_wall(center, wall_geom, wall_depth, min_overlap_ratio=0.35):
                centerlines.append(center)

    return _dedupe_lines(centerlines)


def split_wall_faces(
    plan: Dict[str, Any],
    room_keys: Iterable[str] = DEFAULT_ROOM_KEYS,
    min_length: Optional[float] = None,
    split_tolerance: Optional[float] = None,
    wall_overlap_ratio: float = 0.20,
    use_wall_filter: bool = True,
    use_openings_for_centerlines: bool = True,
) -> List[Dict[str, Any]]:
    plan = normalize_keys(plan)
    params = _wall_params(plan)
    wall_geom = (
        _wall_centerline_geometry(plan)
        if use_openings_for_centerlines
        else _wall_geometry(plan)
    )

    if wall_geom is None:
        return []

    lines = _centerlines_from_opposite_wall_edges(
        wall_geom=wall_geom,
        wall_depth=params["depth"],
        min_length=min_length or params["min_length"],
    )

    faces: List[Dict[str, Any]] = []
    for line in lines:
        fid = f"wf_{len(faces):04d}"
        faces.append(
            {
                "id": fid,
                "type": "wall_face",
                "face_line": line,
                "line": line,
                "length": float(line.length),
                "angle": float(_line_angle(line)),
                "source_room": None,
                "source_room_type": "wall",
                "source_ring": None,
                "raw_edge_index": None,
                "wall_overlap_ratio": 1.0,
                "paired": True,
                "pair_segment_id": None,
                "source": "wall_geometry",
            }
        )

    return faces


def _faces_to_segments(
    faces: List[Dict[str, Any]],
    wall_depth: float,
) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    for face in faces:
        line = face.get("face_line") or face.get("line")
        if not isinstance(line, LineString) or line.is_empty:
            continue

        segments.append(
            _make_segment(
                sid=f"ws_{len(segments):04d}",
                line=line,
                wall_depth=wall_depth,
                face_ids=[face.get("id")],
            )
        )

    return segments


# ---------------------------------------------------------------------
# Collinear merge
# ---------------------------------------------------------------------

def _gap_line(
    origin: np.ndarray,
    unit: np.ndarray,
    normal: np.ndarray,
    offset: float,
    start: float,
    end: float,
) -> LineString:
    p0 = origin + unit * start + normal * offset
    p1 = origin + unit * end + normal * offset
    return LineString([tuple(p0), tuple(p1)])


def _gap_supported_by_opening(
    gap_line: LineString,
    opening_geoms: Optional[List[Any]],
    tolerance: float,
) -> bool:
    if not opening_geoms:
        return False

    buf = gap_line.buffer(tolerance, cap_style=2)
    return any(
        geom is not None and not geom.is_empty and geom.intersects(buf)
        for geom in opening_geoms
    )


def _gap_blocked_by_perpendicular_wall(
    gap_line: LineString,
    base_angle: float,
    all_segments: List[Dict[str, Any]],
    perpendicular_tolerance: float,
    block_tolerance: float,
    min_block_length: float,
    opening_geoms: Optional[List[Any]],
    opening_tolerance: float,
) -> bool:
    if gap_line.is_empty:
        return False

    for seg in all_segments:
        line = _line_from_segment(seg)
        if line is None:
            continue

        if (
            line.length < min_block_length
            and _gap_supported_by_opening(line, opening_geoms, opening_tolerance)
        ):
            continue

        if not _is_near_perpendicular(base_angle, _line_angle(line), perpendicular_tolerance):
            continue

        if gap_line.distance(line) <= block_tolerance:
            return True

    return False


def _build_merged_segment(
    group: List[Tuple[int, Dict[str, Any], float, float]],
    origin: np.ndarray,
    unit: np.ndarray,
    normal: np.ndarray,
    offset: float,
    index: int,
) -> Dict[str, Any]:
    start = min(item[2] for item in group)
    end = max(item[3] for item in group)
    line = _gap_line(origin, unit, normal, offset, start, end)

    face_ids: List[str] = []
    parent_ids: List[str] = []
    thicknesses: List[float] = []

    for _, seg, _, _ in group:
        parent_ids.append(seg.get("id"))
        face_ids.extend(seg.get("face_ids", []))
        if seg.get("thickness") is not None:
            thicknesses.append(float(seg["thickness"]))

    return _make_segment(
        sid=f"ws_{index:04d}",
        line=line,
        wall_depth=float(np.mean(thicknesses)) if thicknesses else None,
        face_ids=list(dict.fromkeys(face_ids)),
        parent_ids=parent_ids,
        merged_collinear=len(group) > 1,
    )


def merge_collinear_segments(
    segments: List[Dict[str, Any]],
    angle_tolerance: float,
    offset_tolerance: float,
    max_gap: float,
    max_opening_gap: float,
    min_length: float,
    opening_geoms: Optional[List[Any]] = None,
    opening_gap_tolerance: float = 2.0,
    respect_junctions: bool = False,
    preserve_opening_segments: bool = False,
    perpendicular_tolerance: float = 12.0,
    junction_tolerance: float = 1.0,
    min_junction_wall_length: float = 0.0,
) -> List[Dict[str, Any]]:
    lines = [_line_from_segment(seg) for seg in segments]
    remaining = [
        (i, seg, line)
        for i, (seg, line) in enumerate(zip(segments, lines))
        if line is not None
    ]

    used = set()
    merged: List[Dict[str, Any]] = []

    for i, _, seed_line in remaining:
        if i in used:
            continue

        origin, unit, normal = _line_unit_and_normal(seed_line)
        seed_angle = _line_angle(seed_line)
        seed_offset = float(np.dot(np.asarray(seed_line.coords[0], dtype=float) - origin, normal))

        group: List[Tuple[int, Dict[str, Any], float, float]] = []

        for j, seg, line in remaining:
            if j in used:
                continue

            if _normalized_angle_delta(seed_angle, _line_angle(line)) > angle_tolerance:
                continue

            coords = np.asarray(line.coords, dtype=float)
            offsets = [
                float(np.dot(p - origin, normal))
                for p in (coords[0], coords[-1])
            ]

            if max(abs(v - seed_offset) for v in offsets) > offset_tolerance:
                continue

            a, b = _projection_interval(line, origin, unit)
            group.append((j, seg, a, b))

        group.sort(key=lambda item: item[2])

        current: List[Tuple[int, Dict[str, Any], float, float]] = []
        cur_start = cur_end = None

        for item in group:
            _, item_seg, a, b = item

            if not current:
                current = [item]
                cur_start, cur_end = a, b
                continue

            gap = max(0.0, a - cur_end)
            gap_line = _gap_line(origin, unit, normal, seed_offset, cur_end, a)

            opening_gap = (
                gap <= max_opening_gap
                and _gap_supported_by_opening(gap_line, opening_geoms, opening_gap_tolerance)
            )

            mergeable = gap <= max_gap or opening_gap

            if preserve_opening_segments:
                prev_seg = current[-1][1]
                if prev_seg.get("opening_ids") or item_seg.get("opening_ids"):
                    mergeable = False

            if respect_junctions and mergeable:
                blocked = _gap_blocked_by_perpendicular_wall(
                    gap_line=gap_line,
                    base_angle=seed_angle,
                    all_segments=segments,
                    perpendicular_tolerance=perpendicular_tolerance,
                    block_tolerance=junction_tolerance,
                    min_block_length=min_junction_wall_length,
                    opening_geoms=opening_geoms,
                    opening_tolerance=opening_gap_tolerance,
                )
                mergeable = not blocked

            if mergeable:
                current.append(item)
                cur_end = max(cur_end, b)
                continue

            used.update(x[0] for x in current)
            if cur_end - cur_start >= min_length:
                merged.append(
                    _build_merged_segment(
                        current,
                        origin,
                        unit,
                        normal,
                        seed_offset,
                        len(merged),
                    )
                )

            current = [item]
            cur_start, cur_end = a, b

        if current:
            used.update(x[0] for x in current)
            if cur_end - cur_start >= min_length:
                merged.append(
                    _build_merged_segment(
                        current,
                        origin,
                        unit,
                        normal,
                        seed_offset,
                        len(merged),
                    )
                )

    return merged


# ---------------------------------------------------------------------
# Opening split / attach
# ---------------------------------------------------------------------

def _opening_sample_points(geom: Any) -> List[Point]:
    points: List[Point] = []

    if isinstance(geom, LineString):
        points.extend(Point(xy) for xy in geom.coords)

    elif hasattr(geom, "exterior"):
        points.extend(Point(xy) for xy in geom.exterior.coords)

    else:
        try:
            x0, y0, x1, y1 = geom.bounds
            points.extend(
                [
                    Point(x0, y0),
                    Point(x0, y1),
                    Point(x1, y0),
                    Point(x1, y1),
                ]
            )
        except Exception:
            pass

    try:
        points.append(geom.representative_point())
    except Exception:
        pass

    return points


def _opening_interval_on_wall_centerline(
    opening_geom: Any,
    wall_line: LineString,
    tolerance: float,
) -> Optional[Tuple[float, float]]:
    if wall_line.length <= 0:
        return None

    projections: List[float] = []

    for pt in _opening_sample_points(opening_geom):
        p = float(wall_line.project(pt))
        q = wall_line.interpolate(p)

        if pt.distance(q) <= tolerance:
            projections.append(p)

    if len(projections) < 2:
        try:
            center = opening_geom.representative_point()
        except Exception:
            return None

        if center.distance(wall_line) <= tolerance:
            p = float(wall_line.project(center))
            return p, p

        return None

    a = max(0.0, min(projections))
    b = min(float(wall_line.length), max(projections))

    if b < a:
        a, b = b, a

    return a, b


def split_wall_segments_at_openings(
    plan: Dict[str, Any],
    segments: List[Dict[str, Any]],
    min_length: float,
    projection_tolerance: float,
    min_opening_projection: float,
) -> List[Dict[str, Any]]:
    openings = list(_iter_openings(plan))
    out: List[Dict[str, Any]] = []

    for seg in segments:
        line = _line_from_segment(seg)
        if line is None or line.is_empty:
            continue

        split_distances = [0.0, float(line.length)]
        opening_refs: List[Tuple[Dict[str, Any], float, float]] = []

        for opening in openings:
            geom = opening["geometry"]

            if line.distance(geom) > projection_tolerance:
                continue

            interval = _opening_interval_on_wall_centerline(
                geom,
                line,
                tolerance=projection_tolerance,
            )

            if interval is None:
                continue

            a, b = interval
            projection = b - a

            if projection < min_opening_projection:
                continue

            split_distances.extend([a, b])
            opening_refs.append((opening, a, b))

        split_distances = sorted(split_distances)

        deduped: List[float] = []
        dedupe_tol = max(min_length * 0.25, 1e-6)

        for d in split_distances:
            if not deduped or abs(d - deduped[-1]) > dedupe_tol:
                deduped.append(d)

        for a, b in zip(deduped[:-1], deduped[1:]):
            if b - a < min_length:
                continue

            p0 = line.interpolate(a)
            p1 = line.interpolate(b)
            part = LineString([(p0.x, p0.y), (p1.x, p1.y)])

            child = dict(seg)
            child.update(
                {
                    "parent_id": seg.get("id"),
                    "id": f"ws_{len(out):04d}",
                    "baseline": part,
                    "line": part,
                    "length": float(part.length),
                    "angle": float(_line_angle(part)),
                    "openings": [],
                    "opening_ids": [],
                    "split_by_opening": bool(opening_refs),
                    "wall_centerline_start": tuple(part.coords[0]),
                    "wall_centerline_end": tuple(part.coords[-1]),
                }
            )

            for opening, oa, ob in opening_refs:
                overlap = max(0.0, min(b, ob) - max(a, oa))

                if overlap <= 0:
                    continue

                local_start = max(a, oa) - a
                local_end = min(b, ob) - a
                local_center = (local_start + local_end) / 2.0

                attached = dict(opening)
                attached.update(
                    {
                        "attached_wall_id": child["id"],
                        "attachment_method": "opening_centerline_split",
                        "distance": float(part.distance(opening["geometry"])),
                        "position": float(local_center),
                        "position_ratio": float(local_center / part.length) if part.length > 0 else None,
                        "start_position": float(local_start),
                        "end_position": float(local_end),
                        "start_ratio": float(local_start / part.length) if part.length > 0 else None,
                        "end_ratio": float(local_end / part.length) if part.length > 0 else None,
                        "projection_overlap": float(overlap),
                    }
                )

                child["openings"].append(attached)
                child["opening_ids"].append(opening["id"])

            out.append(child)

    return out


def _opening_already_attached(segments: List[Dict[str, Any]], opening_id: str) -> bool:
    return any(opening_id in seg.get("opening_ids", []) for seg in segments)


def attach_openings_to_wall_segments(
    plan: Dict[str, Any],
    segments: List[Dict[str, Any]],
    opening_keys: Iterable[str] = ("door", "window", "front_door"),
    max_distance: Optional[float] = None,
    buffer_width: Optional[float] = None,
) -> List[Dict[str, Any]]:
    wall_depth = _plan_wall_depth(plan)
    max_dist = max_distance if max_distance is not None else max(wall_depth * 1.5, 2.0)
    buf_width = buffer_width if buffer_width is not None else max(wall_depth * 0.9, 1.0)
    projection_tol = max(buf_width, wall_depth * 0.9, 1.0)
    min_projection = max(wall_depth * 0.15, 0.35)

    for seg in segments:
        seg["openings"] = list(seg.get("openings", []))
        seg["opening_ids"] = list(seg.get("opening_ids", []))

    for opening in _iter_openings(plan, opening_keys):
        oid = opening["id"]

        if _opening_already_attached(segments, oid):
            continue

        geom = opening["geometry"]
        best_seg = None
        best_score = None
        nearest_seg = None
        nearest_dist = None

        for seg in segments:
            line = _line_from_segment(seg)
            if line is None:
                continue

            dist = float(line.distance(geom))

            if nearest_dist is None or dist < nearest_dist:
                nearest_dist = dist
                nearest_seg = seg

            if dist > max_dist:
                continue

            interval = _opening_interval_on_wall_centerline(geom, line, projection_tol)
            if interval is None:
                continue

            start, end = interval
            projection = max(0.0, end - start)

            if projection < min_projection and dist > max(wall_depth * 0.25, 0.5):
                continue

            overlap_area = float(geom.intersection(line.buffer(buf_width, cap_style=2)).area)
            center_pos = (start + end) / 2.0
            projected = line.interpolate(center_pos)

            endpoint_penalty = (
                min(
                    projected.distance(Point(line.coords[0])),
                    projected.distance(Point(line.coords[-1])),
                )
                < max(wall_depth * 0.25, 0.75)
            )

            score = (
                -projection,
                -overlap_area,
                dist,
                1.0 if endpoint_penalty else 0.0,
                -line.length,
            )

            if best_score is None or score < best_score:
                best_score = score
                best_seg = seg

        if best_seg is None:
            best_seg = nearest_seg
            if best_seg is None:
                continue
            method = "nearest_fallback"
        else:
            method = "projection_fallback"

        line = _line_from_segment(best_seg)
        if line is None:
            continue

        interval = _opening_interval_on_wall_centerline(geom, line, projection_tol)

        if interval is None:
            p = float(line.project(geom.representative_point()))
            interval = (p, p)

        start, end = interval
        center = (start + end) / 2.0

        attached = dict(opening)
        attached.update(
            {
                "attached_wall_id": best_seg["id"],
                "attachment_method": method,
                "distance": float(line.distance(geom)),
                "position": float(center),
                "position_ratio": float(center / line.length) if line.length > 0 else None,
                "start_position": float(start),
                "end_position": float(end),
                "start_ratio": float(start / line.length) if line.length > 0 else None,
                "end_ratio": float(end / line.length) if line.length > 0 else None,
                "projection_overlap": float(max(0.0, end - start)),
            }
        )

        best_seg["openings"].append(attached)
        best_seg["opening_ids"].append(oid)

    return segments


def filter_opening_cap_segments(
    segments: List[Dict[str, Any]],
    opening_geoms: Optional[List[Any]],
    wall_depth: float,
    tolerance: float,
) -> List[Dict[str, Any]]:
    if not opening_geoms:
        return segments

    max_cap_length = max(wall_depth * 1.20, wall_depth + tolerance)
    kept: List[Dict[str, Any]] = []

    for seg in segments:
        line = _line_from_segment(seg)
        if line is None:
            continue

        is_short_cap = line.length <= max_cap_length
        touches_opening = _gap_supported_by_opening(line, opening_geoms, tolerance)

        if is_short_cap and touches_opening:
            continue

        kept.append(seg)

    return kept


def infer_unlabeled_opening_gaps(
    segments: List[Dict[str, Any]],
    wall_geom: Any,
    opening_geoms: List[Any],
    wall_depth: float,
    angle_tolerance: float,
    offset_tolerance: float,
    min_gap: float,
    max_gap: float,
    min_length: float,
    opening_gap_tolerance: float,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    remaining = [
        (i, seg, _line_from_segment(seg))
        for i, seg in enumerate(segments)
    ]
    remaining = [(i, seg, line) for i, seg, line in remaining if line is not None]
    used = set()
    out: List[Dict[str, Any]] = []
    inferred_index = 0

    for i, seed_seg, seed_line in remaining:
        if i in used:
            continue

        origin, unit, normal = _line_unit_and_normal(seed_line)
        seed_angle = _line_angle(seed_line)
        seed_offset = float(np.dot(np.asarray(seed_line.coords[0], dtype=float) - origin, normal))

        group: List[Tuple[int, Dict[str, Any], float, float]] = []
        for j, seg, line in remaining:
            if j in used:
                continue
            if _normalized_angle_delta(seed_angle, _line_angle(line)) > angle_tolerance:
                continue

            coords = np.asarray(line.coords, dtype=float)
            offsets = [
                float(np.dot(p - origin, normal))
                for p in (coords[0], coords[-1])
            ]
            if max(abs(v - seed_offset) for v in offsets) > offset_tolerance:
                continue

            a, b = _projection_interval(line, origin, unit)
            group.append((j, seg, a, b))

        group.sort(key=lambda item: item[2])
        if len(group) <= 1:
            used.add(i)
            out.append(seed_seg)
            continue

        current: List[Tuple[int, Dict[str, Any], float, float]] = []
        inferred_gaps: List[Tuple[float, float]] = []
        cur_start = cur_end = None

        def flush_current() -> None:
            nonlocal inferred_index
            if not current:
                return

            used.update(item[0] for item in current)
            start = min(item[2] for item in current)
            end = max(item[3] for item in current)
            if end - start < min_length:
                return

            if len(current) == 1 and not inferred_gaps:
                out.append(current[0][1])
                return

            line = _gap_line(origin, unit, normal, seed_offset, start, end)
            face_ids: List[str] = []
            parent_ids: List[str] = []
            thicknesses: List[float] = []
            for _, seg, _, _ in current:
                parent_ids.append(seg.get("id"))
                face_ids.extend(seg.get("face_ids", []))
                if seg.get("thickness") is not None:
                    thicknesses.append(float(seg["thickness"]))

            merged = _make_segment(
                sid=f"ws_{len(out):04d}",
                line=line,
                wall_depth=float(np.mean(thicknesses)) if thicknesses else wall_depth,
                face_ids=list(dict.fromkeys(face_ids)),
                parent_ids=parent_ids,
                merged_collinear=len(current) > 1,
            )

            inferred_openings = []
            for gap_start, gap_end in inferred_gaps:
                local_start = max(0.0, gap_start - start)
                local_end = min(line.length, gap_end - start)
                if local_end <= local_start:
                    continue
                center = (local_start + local_end) / 2.0
                inferred_openings.append(
                    {
                        "id": f"opening_inferred_{inferred_index:04d}",
                        "type": "opening",
                        "geometry": _gap_line(origin, unit, normal, seed_offset, gap_start, gap_end),
                        "length": float(gap_end - gap_start),
                        "area": 0.0,
                        "attached_wall_id": merged["id"],
                        "attachment_method": "inferred_collinear_gap",
                        "distance": 0.0,
                        "position": float(center),
                        "position_ratio": float(center / line.length) if line.length > 0 else None,
                        "start_position": float(local_start),
                        "end_position": float(local_end),
                        "start_ratio": float(local_start / line.length) if line.length > 0 else None,
                        "end_ratio": float(local_end / line.length) if line.length > 0 else None,
                        "projection_overlap": float(local_end - local_start),
                        "inferred": True,
                    }
                )
                inferred_index += 1

            if inferred_openings:
                merged["inferred_openings"] = inferred_openings

            out.append(merged)

        for item in group:
            _, _, a, b = item
            if not current:
                current = [item]
                inferred_gaps = []
                cur_start, cur_end = a, b
                continue

            gap = max(0.0, a - cur_end)
            gap_line = _gap_line(origin, unit, normal, seed_offset, cur_end, a)
            known_opening_gap = _gap_supported_by_opening(
                gap_line,
                opening_geoms,
                opening_gap_tolerance,
            )
            wall_supported_gap = _line_supported_by_wall(
                gap_line,
                wall_geom=wall_geom,
                wall_depth=wall_depth,
                min_overlap_ratio=0.20,
            )
            blocked = _gap_blocked_by_perpendicular_wall(
                gap_line=gap_line,
                base_angle=seed_angle,
                all_segments=segments,
                perpendicular_tolerance=15.0,
                block_tolerance=max(wall_depth * 0.35, 1.0),
                min_block_length=max(wall_depth * 1.5, 4.0),
                opening_geoms=opening_geoms,
                opening_tolerance=opening_gap_tolerance,
            )
            inferred_opening_gap = (
                min_gap <= gap <= max_gap
                and not known_opening_gap
                and not wall_supported_gap
                and not blocked
            )

            if inferred_opening_gap:
                current.append(item)
                inferred_gaps.append((cur_end, a))
                cur_end = max(cur_end, b)
                continue

            flush_current()
            current = [item]
            inferred_gaps = []
            cur_start, cur_end = a, b

        flush_current()

    for i, seg, _ in remaining:
        if i not in used:
            out.append(seg)

    return out


# ---------------------------------------------------------------------
# Wall intersection splitting
# ---------------------------------------------------------------------

def _split_line_at_points(
    line: LineString,
    points: Iterable[Point],
    tolerance: float,
    min_length: float,
) -> List[LineString]:
    if line.length <= min_length:
        return []

    distances = [0.0, line.length]

    for pt in _dedupe_points(points, tolerance):
        if line.distance(pt) <= tolerance:
            d = line.project(pt)
            if tolerance < d < line.length - tolerance:
                distances.append(d)

    distances = sorted(distances)
    deduped: List[float] = []

    for d in distances:
        if not deduped or abs(d - deduped[-1]) > tolerance:
            deduped.append(d)

    parts: List[LineString] = []

    for a, b in zip(deduped[:-1], deduped[1:]):
        if b - a < min_length:
            continue

        p0 = line.interpolate(a)
        p1 = line.interpolate(b)
        part = LineString([(p0.x, p0.y), (p1.x, p1.y)])

        if part.length >= min_length:
            parts.append(part)

    return parts


def _snap_lines_to_supported_junctions(
    lines: List[Optional[LineString]],
    wall_geom: Any,
    endpoint_snap_distance: float,
    perpendicular_tolerance: float,
    wall_support_tolerance: float,
) -> List[Optional[LineString]]:
    endpoints: List[Optional[List[np.ndarray]]] = []
    best: List[List[float]] = []

    for line in lines:
        if line is None:
            endpoints.append(None)
            best.append([float("inf"), float("inf")])
            continue

        coords = np.asarray(line.coords, dtype=float)
        endpoints.append([coords[0].copy(), coords[-1].copy()])
        best.append([float("inf"), float("inf")])

    for i, l1 in enumerate(lines):
        if l1 is None or endpoints[i] is None:
            continue

        for j, l2 in enumerate(lines):
            if i == j or l2 is None:
                continue

            if not _is_near_perpendicular(
                _line_angle(l1),
                _line_angle(l2),
                perpendicular_tolerance,
            ):
                continue

            q = _line_axis_intersection(l1, l2)

            if q is None or q.distance(l2) > endpoint_snap_distance:
                continue

            if not _point_supported_by_wall(q, wall_geom, wall_support_tolerance):
                continue

            for k, xy in enumerate(endpoints[i]):
                pt = Point(float(xy[0]), float(xy[1]))
                dist = pt.distance(q)

                if dist <= endpoint_snap_distance and dist < best[i][k]:
                    endpoints[i][k] = np.asarray(q.coords[0], dtype=float)
                    best[i][k] = dist

    snapped: List[Optional[LineString]] = []

    for line, pts in zip(lines, endpoints):
        if line is None or pts is None or np.linalg.norm(pts[1] - pts[0]) <= 1e-9:
            snapped.append(line)
        else:
            snapped.append(LineString([tuple(pts[0]), tuple(pts[1])]))

    return snapped


def _near_perpendicular_split_points(
    l1: LineString,
    l2: LineString,
    wall_geom: Any,
    wall_support_tolerance: float,
    tolerance: float,
    perpendicular_tolerance: float,
    endpoint_snap_distance: float,
) -> Tuple[List[Point], List[Point]]:
    if not _is_near_perpendicular(_line_angle(l1), _line_angle(l2), perpendicular_tolerance):
        return [], []

    points1: List[Point] = []
    points2: List[Point] = []

    for pt in _geometry_to_points(l1.intersection(l2)):
        if _point_supported_by_wall(pt, wall_geom, wall_support_tolerance):
            points1.append(pt)
            points2.append(pt)

    for source, target, out_source, out_target in (
        (l1, l2, points1, points2),
        (l2, l1, points2, points1),
    ):
        for xy in (source.coords[0], source.coords[-1]):
            pt = Point(xy)

            if pt.distance(target) > endpoint_snap_distance:
                continue

            q = _line_axis_intersection(source, target) or target.interpolate(target.project(pt))

            if not _point_supported_by_wall(q, wall_geom, wall_support_tolerance):
                continue

            if q.distance(source) <= endpoint_snap_distance:
                out_source.append(source.interpolate(source.project(q)))

            if q.distance(target) <= endpoint_snap_distance:
                out_target.append(target.interpolate(target.project(q)))

    return points1, points2


def split_wall_segments_at_intersections(
    segments: List[Dict[str, Any]],
    wall_geom: Any = None,
    wall_depth: float = 6.0,
    min_length: float = 0.5,
    tolerance: float = 0.25,
    perpendicular_tolerance: float = 12.0,
    endpoint_snap_distance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    snap = endpoint_snap_distance if endpoint_snap_distance is not None else tolerance
    support_tol = max(wall_depth * 0.35, 0.75)

    original_lines = [_line_from_segment(seg) for seg in segments]
    lines = _snap_lines_to_supported_junctions(
        original_lines,
        wall_geom=wall_geom,
        endpoint_snap_distance=snap,
        perpendicular_tolerance=perpendicular_tolerance,
        wall_support_tolerance=support_tol,
    )

    split_points: Dict[int, List[Point]] = {i: [] for i in range(len(segments))}

    for i, l1 in enumerate(lines):
        if l1 is None:
            continue

        for j in range(i + 1, len(lines)):
            l2 = lines[j]

            if l2 is None or l1.distance(l2) > snap:
                continue

            pts1, pts2 = _near_perpendicular_split_points(
                l1,
                l2,
                wall_geom=wall_geom,
                wall_support_tolerance=support_tol,
                tolerance=tolerance,
                perpendicular_tolerance=perpendicular_tolerance,
                endpoint_snap_distance=snap,
            )

            split_points[i].extend(pts1)
            split_points[j].extend(pts2)

    out: List[Dict[str, Any]] = []

    for i, seg in enumerate(segments):
        line = lines[i]
        if line is None:
            continue

        parts = _split_line_at_points(
            line,
            _dedupe_points(split_points[i], tolerance),
            tolerance,
            min_length,
        ) or [line]

        for part_index, part in enumerate(parts):
            child = dict(seg)
            child.update(
                {
                    "parent_id": seg.get("id"),
                    "id": f"ws_{len(out):04d}",
                    "baseline": part,
                    "line": part,
                    "length": float(part.length),
                    "angle": float(_line_angle(part)),
                    "split_part_index": part_index,
                    "split_part_count": len(parts),
                    "split_at_wall_intersection": len(parts) > 1,
                }
            )
            out.append(child)

    return out


# ---------------------------------------------------------------------
# Dangling endpoint repair
# ---------------------------------------------------------------------

def repair_dangling_wall_endpoints_to_perpendicular_centers(
    segments: List[Dict[str, Any]],
    wall_geom: Any,
    wall_depth: float,
    endpoint_tolerance: float,
    search_distance: float,
    angle_tolerance: float = 15.0,
    min_length: float = 0.5,
    boundary_clearance_ratio: float = 0.22,
    max_axis_extension: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not segments:
        return segments

    max_ext = max_axis_extension if max_axis_extension is not None else search_distance
    repaired = [dict(seg) for seg in segments]
    lines = [_line_from_segment(seg) for seg in repaired]
    target_split_points: Dict[int, List[Point]] = {i: [] for i in range(len(repaired))}

    def endpoint_connected_to_endpoint(seg_index: int, endpoint: Point) -> bool:
        for other_index, other in enumerate(lines):
            if other_index == seg_index or other is None:
                continue
            if any(
                endpoint.distance(Point(xy)) <= endpoint_tolerance
                for xy in (other.coords[0], other.coords[-1])
            ):
                return True
        return False

    def set_segment_line(seg_index: int, line: LineString, **extra: Any) -> None:
        repaired[seg_index].update(
            {
                "baseline": line,
                "line": line,
                "length": float(line.length),
                "angle": float(_line_angle(line)),
                **extra,
            }
        )
        lines[seg_index] = line

    for i in range(len(repaired)):
        line = lines[i]

        if line is None:
            continue

        base_angle = _line_angle(line)
        new_line = line
        changed = False
        repair_records = []

        for endpoint_index in (0, 1):
            endpoint = Point(new_line.coords[endpoint_index])
            if endpoint_connected_to_endpoint(i, endpoint):
                continue

            origin, unit, _ = _line_unit_and_normal(new_line)
            line_start, line_end = _projection_interval(new_line, origin, unit)
            is_start = endpoint_index == 0

            best = None
            best_score = None

            for j, other_seg in enumerate(segments):
                if i == j:
                    continue

                other = _line_from_segment(other_seg)
                if other is None:
                    continue

                if not _is_near_perpendicular(
                    base_angle,
                    _line_angle(other),
                    angle_tolerance,
                ):
                    continue

                q = _line_axis_intersection(new_line, other)
                if q is None:
                    continue

                dist = float(endpoint.distance(q))
                if dist > search_distance:
                    continue

                other_origin, other_unit, _ = _line_unit_and_normal(other)
                oq = float(np.dot(np.asarray(q.coords[0], dtype=float) - other_origin, other_unit))
                o0, o1 = _projection_interval(other, other_origin, other_unit)

                outside_other = max(o0 - oq, oq - o1, 0.0)
                if outside_other > max_ext:
                    continue

                q_proj = float(np.dot(np.asarray(q.coords[0], dtype=float) - origin, unit))

                if is_start:
                    extension = line_start - q_proj
                else:
                    extension = q_proj - line_end

                if extension < -endpoint_tolerance:
                    continue

                if extension > max_ext:
                    continue

                if not _point_supported_by_wall_center_area(
                    q,
                    wall_geom=wall_geom,
                    wall_depth=wall_depth,
                    boundary_clearance_ratio=boundary_clearance_ratio,
                ):
                    continue

                candidate_line = _replace_line_endpoint(
                    new_line,
                    endpoint_index,
                    q,
                    min_length=min_length,
                )
                if candidate_line is None:
                    continue

                if not _line_supported_by_wall(
                    candidate_line,
                    wall_geom=wall_geom,
                    wall_depth=wall_depth,
                    min_overlap_ratio=0.42,
                ):
                    continue

                other_endpoint_points = [Point(other.coords[0]), Point(other.coords[-1])]
                nearest_other_endpoint_index = int(
                    np.argmin([q.distance(pt) for pt in other_endpoint_points])
                )
                other_endpoint_distance = float(q.distance(other_endpoint_points[nearest_other_endpoint_index]))
                target_action = "existing_endpoint"

                if other_endpoint_distance > endpoint_tolerance:
                    if outside_other <= endpoint_tolerance:
                        target_action = "split_target_centerline"
                    else:
                        target_candidate = _replace_line_endpoint(
                            other,
                            nearest_other_endpoint_index,
                            q,
                            min_length=min_length,
                        )
                        if target_candidate is None:
                            continue
                        if not _line_supported_by_wall(
                            target_candidate,
                            wall_geom=wall_geom,
                            wall_depth=wall_depth,
                            min_overlap_ratio=0.42,
                        ):
                            continue
                        target_action = "extend_target_endpoint"

                score = (
                    dist,
                    abs(extension),
                    min(outside_other, other_endpoint_distance),
                    other_endpoint_distance,
                )

                if best_score is None or score < best_score:
                    best_score = score
                    best = {
                        "point": q,
                        "target_segment_index": j,
                        "target_segment_id": other_seg.get("id"),
                        "distance": dist,
                        "extension": extension,
                        "outside_other": outside_other,
                        "target_action": target_action,
                        "target_endpoint_index": nearest_other_endpoint_index,
                        "target_endpoint_distance": other_endpoint_distance,
                    }

            if best is None:
                continue

            candidate_line = _replace_line_endpoint(
                new_line,
                endpoint_index,
                best["point"],
                min_length=min_length,
            )

            if candidate_line is None:
                continue

            new_line = candidate_line
            changed = True

            target_index = int(best["target_segment_index"])
            target_line = lines[target_index]
            if target_line is not None:
                if best["target_action"] == "extend_target_endpoint":
                    target_candidate = _replace_line_endpoint(
                        target_line,
                        int(best["target_endpoint_index"]),
                        best["point"],
                        min_length=min_length,
                    )
                    if target_candidate is not None:
                        set_segment_line(
                            target_index,
                            target_candidate,
                            target_endpoint_aligned_to_repaired_endpoint=True,
                        )
                elif best["target_action"] == "split_target_centerline":
                    target_split_points[target_index].append(best["point"])

            repair_records.append(
                {
                    "endpoint_index": endpoint_index,
                    "target_segment_id": best["target_segment_id"],
                    "distance": float(best["distance"]),
                    "extension": float(best["extension"]),
                    "outside_other": float(best["outside_other"]),
                    "target_endpoint_distance": float(best["target_endpoint_distance"]),
                    "target_action": best["target_action"],
                    "point": tuple(best["point"].coords[0]),
                    "method": "align_dangling_endpoint_to_perpendicular_centerline_junction",
                }
            )

        if changed:
            set_segment_line(
                i,
                new_line,
                endpoint_repaired=True,
                endpoint_repair_records=list(repaired[i].get("endpoint_repair_records", [])) + repair_records,
            )

    out: List[Dict[str, Any]] = []
    for i, seg in enumerate(repaired):
        line = lines[i]
        if line is None:
            continue

        parts = _split_line_at_points(
            line,
            _dedupe_points(target_split_points.get(i, []), endpoint_tolerance),
            endpoint_tolerance,
            min_length,
        )

        if not parts or len(parts) == 1:
            out.append(seg)
            continue

        for part_index, part in enumerate(parts):
            child = dict(seg)
            child.update(
                {
                    "parent_id": seg.get("id"),
                    "id": f"{seg.get('id', f'ws_{i:04d}')}_repair_{part_index}",
                    "baseline": part,
                    "line": part,
                    "length": float(part.length),
                    "angle": float(_line_angle(part)),
                    "split_for_endpoint_alignment": True,
                    "split_part_index": part_index,
                    "split_part_count": len(parts),
                }
            )
            out.append(child)

    return out


def repair_dangling_endpoints_to_perpendicular_centerlines(
    segments: List[Dict[str, Any]],
    wall_geom: Any,
    wall_depth: float = 6.0,
    endpoint_tolerance: float = 2.0,
    search_distance: Optional[float] = None,
    perpendicular_tolerance: float = 15.0,
    wall_support_tolerance: Optional[float] = None,
    min_length: float = 0.5,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    """Backward-compatible public name for dangling endpoint repair."""
    return repair_dangling_wall_endpoints_to_perpendicular_centers(
        segments=segments,
        wall_geom=wall_geom,
        wall_depth=wall_depth,
        endpoint_tolerance=endpoint_tolerance,
        search_distance=search_distance if search_distance is not None else max(wall_depth * 2.5, endpoint_tolerance),
        angle_tolerance=perpendicular_tolerance,
        min_length=min_length,
        max_axis_extension=kwargs.get("max_axis_extension"),
    )


def _cluster_scalar_values(values: List[float], tolerance: float) -> List[Tuple[float, float]]:
    if not values:
        return []

    clusters: List[List[float]] = []
    for value in sorted(values):
        if not clusters or abs(value - float(np.median(clusters[-1]))) > tolerance:
            clusters.append([value])
        else:
            clusters[-1].append(value)

    return [
        (min(cluster), float(np.median(cluster)))
        for cluster in clusters
    ]


def _nearest_cluster_center(value: float, clusters: List[Tuple[float, float]], tolerance: float) -> float:
    best_center = value
    best_distance = float("inf")

    for _, center in clusters:
        distance = abs(value - center)
        if distance <= tolerance and distance < best_distance:
            best_center = center
            best_distance = distance

    return best_center


def align_axis_aligned_wall_endpoint_coordinates(
    segments: List[Dict[str, Any]],
    wall_geom: Any,
    wall_depth: float,
    tolerance: float,
    angle_tolerance: float = 8.0,
    min_length: float = 0.5,
) -> List[Dict[str, Any]]:
    """Put nearly collinear horizontal/vertical wall endpoints on shared rows/columns."""
    if not segments:
        return segments

    horizontal_values: List[float] = []
    vertical_values: List[float] = []
    lines = [_line_from_segment(seg) for seg in segments]

    for line in lines:
        if line is None:
            continue

        angle = _line_angle(line)
        if min(abs(angle), abs(angle - 180.0)) <= angle_tolerance:
            horizontal_values.extend([line.coords[0][1], line.coords[-1][1]])
        elif abs(angle - 90.0) <= angle_tolerance:
            vertical_values.extend([line.coords[0][0], line.coords[-1][0]])

    horizontal_clusters = _cluster_scalar_values(horizontal_values, tolerance)
    vertical_clusters = _cluster_scalar_values(vertical_values, tolerance)

    aligned: List[Dict[str, Any]] = []
    for seg, line in zip(segments, lines):
        if line is None:
            aligned.append(seg)
            continue

        angle = _line_angle(line)
        coords = [
            np.asarray(line.coords[0], dtype=float),
            np.asarray(line.coords[-1], dtype=float),
        ]
        changed = False

        if min(abs(angle), abs(angle - 180.0)) <= angle_tolerance:
            y_values = [coords[0][1], coords[1][1]]
            target_y = _nearest_cluster_center(float(np.median(y_values)), horizontal_clusters, tolerance)
            for coord in coords:
                if abs(coord[1] - target_y) <= tolerance and abs(coord[1] - target_y) > 1e-9:
                    coord[1] = target_y
                    changed = True
                target_x = _nearest_cluster_center(float(coord[0]), vertical_clusters, tolerance)
                if abs(coord[0] - target_x) <= tolerance and abs(coord[0] - target_x) > 1e-9:
                    coord[0] = target_x
                    changed = True

        elif abs(angle - 90.0) <= angle_tolerance:
            x_values = [coords[0][0], coords[1][0]]
            target_x = _nearest_cluster_center(float(np.median(x_values)), vertical_clusters, tolerance)
            for coord in coords:
                if abs(coord[0] - target_x) <= tolerance and abs(coord[0] - target_x) > 1e-9:
                    coord[0] = target_x
                    changed = True
                target_y = _nearest_cluster_center(float(coord[1]), horizontal_clusters, tolerance)
                if abs(coord[1] - target_y) <= tolerance and abs(coord[1] - target_y) > 1e-9:
                    coord[1] = target_y
                    changed = True

        if not changed:
            aligned.append(seg)
            continue

        candidate = LineString([tuple(coords[0]), tuple(coords[1])])
        if candidate.length < min_length:
            aligned.append(seg)
            continue

        if not _line_supported_by_wall(
            candidate,
            wall_geom=wall_geom,
            wall_depth=wall_depth,
            min_overlap_ratio=0.42,
        ):
            aligned.append(seg)
            continue

        child = dict(seg)
        child.update(
            {
                "baseline": candidate,
                "line": candidate,
                "length": float(candidate.length),
                "angle": float(_line_angle(candidate)),
                "axis_endpoint_coordinates_aligned": True,
            }
        )
        aligned.append(child)

    return aligned


# ---------------------------------------------------------------------
# Connectivity / validation
# ---------------------------------------------------------------------

def annotate_wall_connectivity(
    segments: List[Dict[str, Any]],
    endpoint_tolerance: float = 2.0,
    line_tolerance: float = 2.0,
) -> List[Dict[str, Any]]:
    lines = [_line_from_segment(seg) for seg in segments]

    for i, seg in enumerate(segments):
        line = lines[i]

        if line is None:
            seg.update(
                {
                    "endpoint_connected": [False, False],
                    "endpoint_degree": [0, 0],
                    "dangling_endpoint_count": 2,
                    "is_dangling_wall": True,
                }
            )
            continue

        connected = [False, False]
        degree = [0, 0]

        for k, xy in enumerate((line.coords[0], line.coords[-1])):
            pt = Point(xy)

            for j, other in enumerate(lines):
                if i == j or other is None:
                    continue

                other_endpoints = [Point(other.coords[0]), Point(other.coords[-1])]
                endpoint_hit = any(pt.distance(op) <= endpoint_tolerance for op in other_endpoints)
                body_hit = other.distance(pt) <= line_tolerance

                if endpoint_hit or body_hit:
                    connected[k] = True
                    degree[k] += 1

        seg["endpoint_connected"] = connected
        seg["endpoint_degree"] = degree
        seg["dangling_endpoint_count"] = connected.count(False)
        seg["is_dangling_wall"] = seg["dangling_endpoint_count"] > 0

    return segments


def summarize_wall_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for seg in segments:
        line = _line_from_segment(seg)
        if line is None:
            continue

        rows.append(
            {
                "id": seg.get("id"),
                "parent_id": seg.get("parent_id"),
                "length": round(float(seg.get("length", line.length)), 3),
                "angle": round(float(seg.get("angle", _line_angle(line))), 2),
                "split_by_opening": bool(seg.get("split_by_opening")),
                "split_at_wall_intersection": bool(seg.get("split_at_wall_intersection")),
                "endpoint_repaired": bool(seg.get("endpoint_repaired")),
                "endpoint_repair_records": seg.get("endpoint_repair_records"),
                "source": seg.get("source"),
                "thickness": None if seg.get("thickness") is None else round(float(seg["thickness"]), 3),
                "opening_ids": ",".join(seg.get("opening_ids", [])),
                "opening_count": len(seg.get("opening_ids", [])),
                "endpoint_connected": seg.get("endpoint_connected"),
                "endpoint_degree": seg.get("endpoint_degree"),
                "dangling_endpoint_count": seg.get("dangling_endpoint_count"),
                "is_dangling_wall": bool(seg.get("is_dangling_wall")),
                "start": tuple(round(v, 3) for v in line.coords[0]),
                "end": tuple(round(v, 3) for v in line.coords[-1]),
                "wall_centerline_start": seg.get("wall_centerline_start"),
                "wall_centerline_end": seg.get("wall_centerline_end"),
            }
        )

    return rows


def find_unsplit_wall_intersections(
    segments: List[Dict[str, Any]],
    tolerance: float = 0.25,
    perpendicular_tolerance: float = 12.0,
    endpoint_tolerance: float = 0.75,
) -> List[Dict[str, Any]]:
    issues = []
    lines = [_line_from_segment(seg) for seg in segments]

    for i, l1 in enumerate(lines):
        if l1 is None:
            continue

        for j in range(i + 1, len(lines)):
            l2 = lines[j]

            if l2 is None:
                continue

            if not _is_near_perpendicular(_line_angle(l1), _line_angle(l2), perpendicular_tolerance):
                continue

            if l1.distance(l2) > tolerance:
                continue

            for pt in _geometry_to_points(l1.intersection(l2)):
                d1 = min(pt.distance(Point(l1.coords[0])), pt.distance(Point(l1.coords[-1])))
                d2 = min(pt.distance(Point(l2.coords[0])), pt.distance(Point(l2.coords[-1])))

                if d1 > endpoint_tolerance and d2 > endpoint_tolerance:
                    issues.append(
                        {
                            "type": "unsplit_perpendicular_intersection",
                            "segment_ids": [segments[i]["id"], segments[j]["id"]],
                            "point": pt,
                            "dist_to_segment_1_endpoint": float(d1),
                            "dist_to_segment_2_endpoint": float(d2),
                        }
                    )

    return issues


def validate_wall_segmentation(
    plan: Dict[str, Any],
    segments: Optional[List[Dict[str, Any]]] = None,
    tolerance: Optional[float] = None,
    check_openings: bool = False,
) -> Dict[str, Any]:
    if segments is None:
        segments = split_wall_segments(plan)

    wall_depth = _plan_wall_depth(plan)
    tol = tolerance if tolerance is not None else max(wall_depth * 0.10, 0.15)

    unsplit = find_unsplit_wall_intersections(
        segments,
        tolerance=tol,
        endpoint_tolerance=max(wall_depth * 0.35, 0.75),
    )

    too_short = [
        {"id": seg.get("id"), "length": float(seg.get("length", 0.0))}
        for seg in segments
        if float(seg.get("length", 0.0)) < max(wall_depth * 0.30, 0.5)
    ]

    unattached_openings = []
    if check_openings:
        attached = {
            oid
            for seg in segments
            for oid in seg.get("opening_ids", [])
        }
        for opening in _iter_openings(normalize_keys(plan)):
            if opening["id"] not in attached:
                unattached_openings.append(opening["id"])

    return {
        "segment_count": len(segments),
        "split_by_opening_count": sum(1 for seg in segments if seg.get("split_by_opening")),
        "split_intersection_part_count": sum(1 for seg in segments if seg.get("split_at_wall_intersection")),
        "endpoint_repaired_count": sum(1 for seg in segments if seg.get("endpoint_repaired")),
        "dangling_wall_count": sum(1 for seg in segments if seg.get("is_dangling_wall")),
        "opening_count": sum(len(seg.get("opening_ids", [])) for seg in segments),
        "unattached_openings": unattached_openings,
        "too_short_segments": too_short,
        "unsplit_intersections": unsplit,
        "ok": len(unsplit) == 0 and len(too_short) == 0 and len(unattached_openings) == 0,
    }


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def split_wall_segments(
    plan: Dict[str, Any],
    faces: Optional[List[Dict[str, Any]]] = None,
    room_keys: Iterable[str] = DEFAULT_ROOM_KEYS,
    min_overlap: Optional[float] = None,
    angle_tolerance: float = 8.0,
    thickness_tolerance: Optional[float] = None,
    keep_unpaired: bool = True,
    merge_collinear: bool = True,
    split_intersections: bool = True,
    split_openings: bool = False,
    attach_openings: bool = True,
    annotate_connectivity: bool = True,
    repair_dangling_endpoints: bool = True,
    filter_short_isolated_artifacts: bool = False,
) -> List[Dict[str, Any]]:
    plan = normalize_keys(plan)
    params = _wall_params(plan)
    wall_depth = params["depth"]
    wall_geom = _wall_geometry(plan)

    if wall_geom is None:
        return []

    opening_geoms = _opening_geometries(plan)
    centerline_support_geom = _wall_centerline_geometry(plan) or wall_geom

    if faces is None:
        faces = split_wall_faces(
            plan,
            min_length=params["min_length"],
        )

    segments = _faces_to_segments(
        faces,
        wall_depth=wall_depth,
    )

    if merge_collinear:
        segments = merge_collinear_segments(
            segments,
            angle_tolerance=angle_tolerance,
            offset_tolerance=params["offset_tolerance"],
            max_gap=params["max_gap"],
            max_opening_gap=params["max_opening_gap"],
            min_length=params["min_length"],
            opening_geoms=opening_geoms,
            opening_gap_tolerance=params["opening_gap_tolerance"],
            respect_junctions=False,
            preserve_opening_segments=False,
        )

    if split_openings:
        segments = split_wall_segments_at_openings(
            plan=plan,
            segments=segments,
            min_length=params["min_length"],
            projection_tolerance=params["opening_gap_tolerance"],
            min_opening_projection=max(wall_depth * 0.25, 0.75),
        )

    if split_intersections:
        segments = split_wall_segments_at_intersections(
            segments,
            wall_geom=centerline_support_geom,
            wall_depth=wall_depth,
            min_length=params["min_length"],
            tolerance=params["split_tolerance"],
            perpendicular_tolerance=15.0,
            endpoint_snap_distance=params["endpoint_snap"],
        )

        segments = merge_collinear_segments(
            segments,
            angle_tolerance=angle_tolerance,
            offset_tolerance=params["offset_tolerance"],
            max_gap=params["max_gap"],
            max_opening_gap=params["max_opening_gap"],
            min_length=params["min_length"],
            opening_geoms=opening_geoms,
            opening_gap_tolerance=params["opening_gap_tolerance"],
            respect_junctions=True,
            preserve_opening_segments=True,
            perpendicular_tolerance=15.0,
            junction_tolerance=params["connectivity_tolerance"],
            min_junction_wall_length=max(wall_depth * 1.5, 4.0),
        )

    if repair_dangling_endpoints:
        segments = repair_dangling_wall_endpoints_to_perpendicular_centers(
            segments=segments,
            wall_geom=centerline_support_geom,
            wall_depth=wall_depth,
            endpoint_tolerance=params["connectivity_tolerance"],
            search_distance=params["dangling_repair_distance"],
            angle_tolerance=15.0,
            min_length=params["min_length"],
            boundary_clearance_ratio=0.22,
        )

    segments = align_axis_aligned_wall_endpoint_coordinates(
        segments=segments,
        wall_geom=centerline_support_geom,
        wall_depth=wall_depth,
        tolerance=params["axis_endpoint_alignment_tolerance"],
        angle_tolerance=angle_tolerance,
        min_length=params["min_length"],
    )

    segments = [
        seg
        for seg in segments
        if _line_supported_by_wall(
            _line_from_segment(seg),
            wall_geom=centerline_support_geom,
            wall_depth=wall_depth,
            min_overlap_ratio=0.42,
        )
    ]

    segments = filter_opening_cap_segments(
        segments,
        opening_geoms=opening_geoms,
        wall_depth=wall_depth,
        tolerance=params["opening_gap_tolerance"],
    )

    segments = infer_unlabeled_opening_gaps(
        segments=segments,
        wall_geom=wall_geom,
        opening_geoms=opening_geoms,
        wall_depth=wall_depth,
        angle_tolerance=angle_tolerance,
        offset_tolerance=params["offset_tolerance"],
        min_gap=max(wall_depth * 1.20, params["max_gap"]),
        max_gap=params["max_opening_gap"],
        min_length=params["min_length"],
        opening_gap_tolerance=params["opening_gap_tolerance"],
    )

    for i, seg in enumerate(segments):
        seg["id"] = f"ws_{i:04d}"
        for opening in seg.get("inferred_openings", []):
            opening["attached_wall_id"] = seg["id"]

    if attach_openings:
        for seg in segments:
            seg["openings"] = []
            seg["opening_ids"] = []

        segments = attach_openings_to_wall_segments(
            plan,
            segments,
            max_distance=params["opening_attach_distance"],
            buffer_width=params["opening_attach_buffer"],
        )

        for seg in segments:
            for opening in seg.get("inferred_openings", []):
                if opening["id"] in seg.get("opening_ids", []):
                    continue
                seg["openings"].append(opening)
                seg["opening_ids"].append(opening["id"])

    if annotate_connectivity:
        segments = annotate_wall_connectivity(
            segments,
            endpoint_tolerance=params["connectivity_tolerance"],
            line_tolerance=params["connectivity_tolerance"],
        )

    return segments
