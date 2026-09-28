from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Structure:
    label: int
    name: str
    color: tuple[float, float, float]
    visible: bool = True

def structures_from_config(config: dict[str, Any]) -> dict[int, Structure]:
    raw = config.get('labels', {})
    structures: dict[int, Structure] = {}
    for (key, meta) in raw.items():
        label = int(key)
        color = tuple((float(c) for c in meta.get('color', [1.0, 1.0, 1.0])))
        structures[label] = Structure(label=label, name=str(meta.get('name', f'Class {label}')), color=(color[0], color[1], color[2]), visible=bool(meta.get('visible', label != 0)))
    return structures