from __future__ import annotations
from typing import Any, Mapping
import pyvista as pv
from ..model.labels import structures_from_config

def apply_scene_style(plotter: pv.Plotter, config: dict[str, Any]) -> None:
    vis = config.get('visualization', {})
    background = vis.get('background', [0.08, 0.09, 0.11])
    plotter.set_background(tuple((float(c) for c in background)))
    plotter.enable_trackball_style()
    plotter.add_axes()
    
    try:
        plotter.enable_depth_peeling()
        plotter.enable_anti_aliasing('msaa')
    except Exception:
        pass

def add_heart_meshes(plotter: pv.Plotter, meshes: Mapping[int, pv.PolyData], config: dict[str, Any], visibility: Mapping[int, bool] | None=None) -> dict[int, str]:
    plotter.clear()
    apply_scene_style(plotter, config)
    structures = structures_from_config(config)
    opacity = float(config.get('visualization', {}).get('opacity', 0.85))
    actor_names: dict[int, str] = {}
    for (label, mesh) in meshes.items():
        structure = structures.get(label)
        if structure is None:
            continue
        visible = True if visibility is None else bool(visibility.get(label, structure.visible))
        name = f'structure-{label}'
        ascii_label = {1: 'LV cavity', 2: 'LV myocardium', 3: 'RV cavity'}.get(label, f'class-{label}')
        
        plotter.add_mesh(
            mesh, 
            color=structure.color, 
            opacity=opacity, 
            name=name, 
            label=ascii_label, 
            smooth_shading=True,
            specular=0.5,
            specular_power=15.0,
            ambient=0.15,
            diffuse=0.8
        )
        actor = plotter.actors.get(name)
        if actor is not None:
            actor.visibility = visible
        actor_names[label] = name
    if actor_names:
        plotter.add_legend(bcolor=None, border=False, size=(0.15, 0.15))
    plotter.reset_camera()
    return actor_names