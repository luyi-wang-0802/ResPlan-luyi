"""Compatibility facade for ResPlan helper modules.

The implementation has been split into focused modules:
    - ``resplan_constants.py``: colors and default constants
    - ``resplan_geometry.py``: generic geometry, augmentation, and mask helpers
    - ``resplan_plot.py``: plan and wall QA plotting helpers
    - ``resplan_graph.py``: room graph helpers
    - ``resplan_wall.py``: wall segmentation and wall QA logic

This file intentionally re-exports the public API used by the original demo
notebook and older scripts so ``import resplan_utils`` keeps working.
"""

from __future__ import annotations

from resplan_constants import CATEGORY_COLORS, DEFAULT_CANVAS_SIZE, DEFAULT_ROOM_KEYS
from resplan_geometry import (
    augment_geom,
    buffer_expand_shrink,
    buffer_shrink_expand,
    centroid,
    cv2,
    geometry_to_mask,
    get_geometries,
    get_plan_width,
    normalize_keys,
    noise,
    perturb_polygon,
)
from resplan_graph import plan_to_graph, plot_plan_and_graph
from resplan_plot import (
    demo_wall_segments,
    gpd,
    plot_plan,
    plot_wall_segments,
    plot_wall_segmentation_debug,
    plt,
)
from resplan_wall import (
    attach_openings_to_wall_segments,
    find_unsplit_wall_intersections,
    merge_collinear_segments,
    repair_dangling_endpoints_to_perpendicular_centerlines,
    split_wall_faces,
    split_wall_segments,
    split_wall_segments_at_openings,
    split_wall_segments_at_intersections,
    summarize_wall_segments,
    validate_wall_segmentation,
)

merge_collinear_wall_segments = merge_collinear_segments

__all__ = [
    "CATEGORY_COLORS",
    "DEFAULT_CANVAS_SIZE",
    "DEFAULT_ROOM_KEYS",
    "attach_openings_to_wall_segments",
    "augment_geom",
    "buffer_expand_shrink",
    "buffer_shrink_expand",
    "centroid",
    "cv2",
    "demo_wall_segments",
    "find_unsplit_wall_intersections",
    "geometry_to_mask",
    "get_geometries",
    "get_plan_width",
    "gpd",
    "merge_collinear_segments",
    "merge_collinear_wall_segments",
    "normalize_keys",
    "noise",
    "perturb_polygon",
    "plan_to_graph",
    "plot_plan",
    "plot_plan_and_graph",
    "repair_dangling_endpoints_to_perpendicular_centerlines",
    "plot_wall_segments",
    "plot_wall_segmentation_debug",
    "plt",
    "split_wall_faces",
    "split_wall_segments",
    "split_wall_segments_at_openings",
    "split_wall_segments_at_intersections",
    "summarize_wall_segments",
    "validate_wall_segmentation",
]
