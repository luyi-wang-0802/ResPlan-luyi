"""Geometry helpers for ResPlan-style floorplan datasets.

This module contains only reusable geometry, rasterization, and low-level line
helpers. Wall-specific algorithms are in ``resplan_wall.py``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from shapely import affinity
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
    base,
)

from resplan_constants import DEFAULT_CANVAS_SIZE

SINGLE_GEOMS = (Polygon, LineString, Point)
MULTI_GEOMS = (MultiPolygon, MultiLineString, MultiPoint, GeometryCollection)


def _require(module: Any, package_name: str, feature: str) -> None:
    """Raise a clear import error for optional dependencies."""
    if module is None:
        raise ImportError(f"{feature} requires {package_name}.")


def normalize_keys(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize common key typos / variations in-place."""
    if "balacony" in plan and "balcony" not in plan:
        plan["balcony"] = plan.pop("balacony")
    return plan


def get_plan_width(plan: Dict[str, Any]) -> float:
    """Return max(width, height) of the inner polygon bounds."""
    inner = plan.get("inner")
    if inner is None or inner.is_empty:
        return 0.0
    x1, y1, x2, y2 = inner.bounds
    return max(x2 - x1, y2 - y1)


def get_geometries(geom_data: Any) -> List[base.BaseGeometry]:
    """Safely extract individual geometries from single/multi/collections."""
    if geom_data is None or getattr(geom_data, "is_empty", False):
        return []
    if isinstance(geom_data, SINGLE_GEOMS):
        return [geom_data]
    if isinstance(geom_data, MULTI_GEOMS):
        return [g for g in geom_data.geoms if g is not None and not g.is_empty]
    return []


def centroid(poly: Union[Polygon, MultiPolygon]) -> Point:
    """Return centroid for Polygon/MultiPolygon, using the largest part for MultiPolygon."""
    if isinstance(poly, Polygon):
        return poly.centroid
    if isinstance(poly, MultiPolygon) and len(poly.geoms) > 0:
        return max(poly.geoms, key=lambda p: p.area).centroid
    return Point(-1e6, -1e6)


def perturb_polygon(
    polygon: Polygon,
    x_range: Tuple[float, float] = (-2, 2),
    y_range: Tuple[float, float] = (-2, 2),
) -> Polygon:
    """Apply random per-vertex perturbation to a polygon."""
    coords = np.asarray(polygon.exterior.coords, dtype=float)
    dx = np.random.uniform(x_range[0], x_range[1], size=len(coords))
    dy = np.random.uniform(y_range[0], y_range[1], size=len(coords))
    return Polygon(np.column_stack([coords[:, 0] + dx, coords[:, 1] + dy]))


def noise(point: Point, noise_scale: float = 10.0) -> Point:
    """Jitter a point by uniform noise within ±noise_scale."""
    return Point(
        point.x + np.random.uniform(-noise_scale, noise_scale),
        point.y + np.random.uniform(-noise_scale, noise_scale),
    )


def augment_geom(
    geom: base.BaseGeometry,
    degree: float = 0.0,
    flip_vertical: bool = False,
    scale: float = 1.0,
    size: int = 256,
) -> base.BaseGeometry:
    """Rotate around image center, optional vertical flip, then scale."""
    if geom is None:
        return Point(-1e6, -1e6)
    g = affinity.rotate(geom, degree, origin=(size / 2, size / 2))
    y_scale = -scale if flip_vertical else scale
    return affinity.scale(g, xfact=scale, yfact=y_scale, origin=(size / 2, size / 2))


def buffer_shrink_expand(
    geom: base.BaseGeometry,
    w: float,
    join_style: int = 2,
    cap_style: int = 2,
) -> base.BaseGeometry:
    """Shrink then expand by w."""
    return geom.buffer(-w, join_style=join_style, cap_style=cap_style).buffer(
        +w,
        join_style=join_style,
        cap_style=cap_style,
    )


def buffer_expand_shrink(
    geom: base.BaseGeometry,
    w: float,
    join_style: int = 2,
    cap_style: int = 2,
) -> base.BaseGeometry:
    """Expand then shrink by w."""
    return geom.buffer(+w, join_style=join_style, cap_style=cap_style).buffer(
        -w,
        join_style=join_style,
        cap_style=cap_style,
    )


def _poly_to_mask(poly: Polygon, shape: Tuple[int, int], line_thickness: int = 0) -> np.ndarray:
    _require(cv2, "opencv-python (cv2)", "geometry_to_mask")

    h, w = shape
    img = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(poly.exterior.coords, dtype=np.int32)

    if line_thickness > 0:
        cv2.polylines(img, [pts], isClosed=True, color=255, thickness=line_thickness)
    else:
        cv2.fillPoly(img, [pts], color=255)

    for interior in poly.interiors:
        pts_in = np.array(interior.coords, dtype=np.int32)
        if line_thickness > 0:
            cv2.polylines(img, [pts_in], isClosed=True, color=0, thickness=line_thickness)
        else:
            cv2.fillPoly(img, [pts_in], color=0)

    return img


def _line_to_mask(line: LineString, out: np.ndarray, thickness: int) -> None:
    _require(cv2, "opencv-python (cv2)", "geometry_to_mask")
    pts = np.array(line.coords, dtype=np.int32)
    cv2.polylines(out, [pts], isClosed=False, color=255, thickness=max(1, thickness or 1))


def geometry_to_mask(
    geom: Any,
    shape: Tuple[int, int] = DEFAULT_CANVAS_SIZE,
    point_radius: int = 5,
    line_thickness: int = 0,
) -> np.ndarray:
    """Rasterize geometry or iterable of geometries to a binary mask [0, 255]."""
    _require(cv2, "opencv-python (cv2)", "geometry_to_mask")

    h, w = shape
    out = np.zeros((h, w), dtype=np.uint8)

    if isinstance(geom, Polygon):
        return _poly_to_mask(geom, shape, line_thickness)

    if isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            out = np.maximum(out, _poly_to_mask(p, shape, line_thickness))
        return out

    if isinstance(geom, LineString):
        _line_to_mask(geom, out, line_thickness)
        return out

    if isinstance(geom, MultiLineString):
        for line in geom.geoms:
            _line_to_mask(line, out, line_thickness)
        return out

    if isinstance(geom, Point):
        cx, cy = int(round(geom.x)), int(round(geom.y))
        cv2.circle(out, (cx, cy), point_radius, 255, -1)
        return out

    if isinstance(geom, MultiPoint):
        for pt in geom.geoms:
            out = np.maximum(out, geometry_to_mask(pt, shape, point_radius, line_thickness))
        return out

    if isinstance(geom, Iterable):
        for g in geom:
            out = np.maximum(out, geometry_to_mask(g, shape, point_radius, line_thickness))
        return out

    return out


def _ring_edges(ring: Any) -> Iterable[LineString]:
    """Yield non-degenerate LineString edges from a polygon ring."""
    coords = list(ring.coords)
    for p0, p1 in zip(coords[:-1], coords[1:]):
        if Point(p0).distance(Point(p1)) > 1e-9:
            yield LineString([p0, p1])


def _line_angle(line: LineString) -> float:
    """Return line angle in degrees, normalized to [0, 180)."""
    p0, p1 = list(line.coords)[0], list(line.coords)[-1]
    return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0])) % 180.0


def _normalized_angle_delta(a: float, b: float) -> float:
    """Return smallest angular difference modulo 180 degrees."""
    d = abs((a - b) % 180.0)
    return min(d, 180.0 - d)


def _point_key(pt: Point, precision: int = 6) -> Tuple[float, float]:
    return (round(pt.x, precision), round(pt.y, precision))


def _line_unit_and_normal(line: LineString) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return origin, unit direction, and left-hand normal for a line."""
    coords = np.asarray(line.coords, dtype=float)
    p0, p1 = coords[0], coords[-1]
    v = p1 - p0
    norm = np.linalg.norm(v)

    if norm <= 1e-12:
        return p0, np.array([1.0, 0.0]), np.array([0.0, 1.0])

    u = v / norm
    n = np.array([-u[1], u[0]])
    return p0, u, n


def _projection_interval(line: LineString, origin: np.ndarray, unit: np.ndarray) -> Tuple[float, float]:
    """Project both line endpoints onto an axis and return the interval."""
    coords = np.asarray(line.coords, dtype=float)
    vals = [float(np.dot(p - origin, unit)) for p in (coords[0], coords[-1])]
    return min(vals), max(vals)


def _is_near_perpendicular(a: float, b: float, tolerance: float) -> bool:
    """Return whether two normalized line angles are near perpendicular."""
    return abs(_normalized_angle_delta(a, b) - 90.0) <= tolerance


def _dedupe_points(points: Iterable[Point], tolerance: float) -> List[Point]:
    """Cluster nearby points and return averaged representative points."""
    pts = [pt for pt in points if pt is not None and not pt.is_empty]
    if not pts:
        return []

    parent = list(range(len(pts)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if pts[i].distance(pts[j]) <= tolerance:
                union(i, j)

    clusters: Dict[int, List[Point]] = {}
    for i, pt in enumerate(pts):
        clusters.setdefault(find(i), []).append(pt)

    return [
        Point(
            sum(p.x for p in cluster) / len(cluster),
            sum(p.y for p in cluster) / len(cluster),
        )
        for cluster in clusters.values()
    ]
