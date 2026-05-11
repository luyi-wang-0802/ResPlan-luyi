"""Constants for ResPlan-style geometry and wall processing."""

from __future__ import annotations

from typing import Dict, List, Tuple

CATEGORY_COLORS: Dict[str, str] = {
    "living": "#d9d9d9",
    "bedroom": "#66c2a5",
    "bathroom": "#fc8d62",
    "kitchen": "#8da0cb",
    "door": "#e78ac3",
    "window": "#a6d854",
    "wall": "#ffd92f",
    "front_door": "#a63603",
    "balcony": "#b3b3b3",
    "storage": "#cccccc",
}

DEFAULT_CANVAS_SIZE: Tuple[int, int] = (256, 256)  # (H, W)

DEFAULT_ROOM_KEYS: List[str] = [
    "living",
    "bedroom",
    "bathroom",
    "kitchen",
    "balcony",
    "storage",
]
