# -*- coding: utf-8 -*-
"""
Two functions for visualization of the mesh, either before partitioning without curvature data or after with curvature data

@author: joepugar
"""

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.colors
import os
from anato_utils import *
from anato_mesh import *

def mesh_plot(mesh, mesh_color='lightblue', mesh_opacity=0.7, grid_color='black', grid_size=1):
    """Interactive 3D mesh plot of the anatomy."""
    v = mesh.vertices
    f = mesh.faces
    #- Create mesh plot
    mesh = go.Mesh3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        i=f[:, 0],
        j=f[:, 1], 
        k=f[:, 2],
        color=mesh_color, 
        opacity=mesh_opacity, 
        flatshading=True)
    
    scatter = go.Scatter3d(
        x=v[:, 0],
        y=v[:, 1],
        z=v[:, 2],
        mode='markers',
        marker=dict(size=grid_size, color=grid_color))
    edge_x = []
    edge_y = []
    edge_z = []
    for t in f:
        edge_x += [v[t[0]][0], v[t[1]][0], None,
                   v[t[0]][0], v[t[2]][0], None,
                   v[t[1]][0], v[t[2]][0], None]
        edge_y += [v[t[0]][1], v[t[1]][1], None,
                   v[t[0]][1], v[t[2]][1], None,
                   v[t[1]][1], v[t[2]][1], None]
        edge_z += [v[t[0]][2], v[t[1]][2], None,
                   v[t[0]][2], v[t[2]][2], None,
                   v[t[1]][2], v[t[2]][2], None]
    
    edge = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color=grid_color, width=grid_size))
    fig = go.Figure(data=[mesh, scatter, edge])
    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=10, backgroundcolor='white', gridcolor='white'),
            yaxis=dict(nticks=10, backgroundcolor='white', gridcolor='white'),
            zaxis=dict(nticks=10, backgroundcolor='white', gridcolor='white'),
            camera=dict(
                eye=dict(x=2, y=-1.5, z=1)
            )
        ),
        height=1000,
        width=1000,
        paper_bgcolor='white'
    )
    #fig.write_image('mesh_fig.png', format='png', scale=2, engine='kaleido')
    fig.show()
    
def patch_plot(mesh, manifold_df, cluster_ids, var, scan_name, color_scale='Plasma', mesh_opacity=0.7, edge_thickness=3):
    """Interactive 3D mesh plot of the anatomy with colored partitions for surface curvature values and a color bar."""
    v = mesh.vertices
    f = mesh.faces
    min_value = manifold_df[var].min()
    max_value = manifold_df[var].max()
    
    color_map = {cluster_id: plotly.colors.sample_colorscale(color_scale, (value - min_value) / (max_value - min_value))[0]
                 for cluster_id, value in zip(manifold_df.index, manifold_df[var])}
    face_colors = np.array([color_map[cluster_id] for cluster_id in cluster_ids])
    
    fig = go.Figure()
    mesh_trace = go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2],
                           i=f[:, 0], j=f[:, 1], k=f[:, 2],
                           facecolor=face_colors, opacity=mesh_opacity, flatshading=True,
    )
    fig.add_trace(mesh_trace)

    # Draw black edges between clusters
    edge_x, edge_y, edge_z, edge_set = [], [], [], {}

    for tri, cluster_id in zip(f, cluster_ids):
        for j in range(3):
            edge = tuple(sorted((tri[j], tri[(j+1)%3])))
            if edge in edge_set:
                if edge_set[edge] != cluster_id:
                    edge_x += [v[edge[0]][0], v[edge[1]][0], None]
                    edge_y += [v[edge[0]][1], v[edge[1]][1], None]
                    edge_z += [v[edge[0]][2], v[edge[1]][2], None]
            else:
                edge_set[edge] = cluster_id

    edge_trace = go.Scatter3d(x=edge_x, y=edge_y, z=edge_z,
        mode='lines', line=dict(color='black', width=edge_thickness), hoverinfo='none'
    )
    fig.add_trace(edge_trace)

    colorbar_trace = go.Scatter3d(
        x=[None], y=[None], z=[None], 
        mode='markers',
        marker=dict(
            colorscale=color_scale,
            cmin=min_value,   
            cmax=max_value,    
            colorbar=dict(
                title=var,
                titleside='right',
                ticks='outside',
                len=0.75
            ),
            size=0.0001
        ),
        hoverinfo='none'
    )
    fig.add_trace(colorbar_trace)

    # Update layout
    fig.update_layout(
        title=f"3D Mesh Plot: {scan_name}",
        scene=dict(
            xaxis=dict(nticks=10, backgroundcolor='white', gridcolor='white'),
            yaxis=dict(nticks=10, backgroundcolor='white', gridcolor='white'),
            zaxis=dict(nticks=10, backgroundcolor='white', gridcolor='white'),
            camera=dict(
                eye=dict(x=2, y=-1.5, z=1)
            )
        ),
        height=1000,
        width=1000,
        paper_bgcolor='white'
    )
    #fig.write_image('patch_fig.png', format='png', scale=2, engine='kaleido')
    fig.show()
    return manifold_df, fig
