# -*- coding: utf-8 -*-
"""
Two functions for visualization of the mesh, either before partitioning without curvature data or after with curvature data

@author: joepugar
"""

import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.colors
from aorta_utils import *
from aorta_mesh import *

def mesh_plot(svs, sts, mesh_color='lightblue', mesh_opacity=0.7, grid_color='black', grid_size=1):
    """Interactive 3D mesh plot of the anatomy."""
    #- Create mesh plot
    mesh = go.Mesh3d(
        x=svs[:, 0],
        y=svs[:, 1],
        z=svs[:, 2],
        i=sts[:, 0],
        j=sts[:, 1], 
        k=sts[:, 2],
        color=mesh_color, 
        opacity=mesh_opacity, 
        flatshading=True)
    
    #- Create scatter plot of vertices
    scatter = go.Scatter3d(
        x=svs[:, 0],
        y=svs[:, 1],
        z=svs[:, 2],
        mode='markers',
        marker=dict(size=grid_size, color=grid_color))
    
    #- Create edges
    edge_x = []
    edge_y = []
    edge_z = []
    for t in sts:
        edge_x += [svs[t[0]][0], svs[t[1]][0], None,
                   svs[t[0]][0], svs[t[2]][0], None,
                   svs[t[1]][0], svs[t[2]][0], None]
        edge_y += [svs[t[0]][1], svs[t[1]][1], None,
                   svs[t[0]][1], svs[t[2]][1], None,
                   svs[t[1]][1], svs[t[2]][1], None]
        edge_z += [svs[t[0]][2], svs[t[1]][2], None,
                   svs[t[0]][2], svs[t[2]][2], None,
                   svs[t[1]][2], svs[t[2]][2], None]
    
    #- Edges
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
    
def patch_plot(svs, sts, pcs, m, var, scan_name, color_scale='Plasma', mesh_opacity=0.7, edge_thickness=3):
    """Interactive 3D mesh plot of the anatomy with colored partitions for surface curvature values."""
    svsc, stsc, pcsc, mask = edge_cleanup(svs, sts, pcs, tol1=0.4, tol2=tol(pcs))
    k = calc_num_patches(svsc, stsc, pcsc, m=m)
    triangle_COMs = calcCOMs(svsc, stsc)
    km = MiniBatchKMeans(n_clusters=k, max_iter=100, batch_size=1536).fit(triangle_COMs)
    cluster_centers, cluster_ids = km.cluster_centers_, km.labels_
    mesh_quants, patch_areas, num_elem_patch, avg_elem_area = calculate_quantities(svsc, stsc, pcsc, [var], k, cluster_ids)
    manifold_df = organize_data(scan_name, k, cluster_centers, mesh_quants, num_elem_patch, avg_elem_area, patch_areas, [var])
    
    #- Normalize the cluster curvature values for color mapping
    min_value = manifold_df[var].min()
    max_value = manifold_df[var].max()
    manifold_df['normalized_value'] = (manifold_df[var] - min_value) / (max_value - min_value)
    
    #- Create a color map for each cluster based on the normalized curvature values
    color_map = {cluster_id: plotly.colors.sample_colorscale(color_scale, value)[0]
                 for cluster_id, value in zip(manifold_df.index, manifold_df['normalized_value'])}

    fig = go.Figure()

    #- Create the mesh with face colors based on the cluster curvature values
    face_colors = np.array([color_map[cluster_id] for cluster_id in cluster_ids])
    
    mesh = go.Mesh3d(
        x=svsc[:, 0],
        y=svsc[:, 1],
        z=svsc[:, 2],
        i=stsc[:, 0],
        j=stsc[:, 1], 
        k=stsc[:, 2],
        facecolor=face_colors,
        opacity=mesh_opacity, 
        flatshading=True,
    )
    fig.add_trace(mesh)

    #- Identify and add thick black edges only between different clusters
    edge_x = []
    edge_y = []
    edge_z = []
    edge_set = {}

    for tri, cluster_id in zip(stsc, cluster_ids):
        for j in range(3):
            edge = tuple(sorted((tri[j], tri[(j+1)%3])))
            if edge in edge_set:
                if edge_set[edge] != cluster_id:
                    edge_x += [svsc[edge[0]][0], svsc[edge[1]][0], None]
                    edge_y += [svsc[edge[0]][1], svsc[edge[1]][1], None]
                    edge_z += [svsc[edge[0]][2], svsc[edge[1]][2], None]
            else:
                edge_set[edge] = cluster_id

    #- Add dark edges only at cluster boundaries
    edge_trace = go.Scatter3d(
        x=edge_x,
        y=edge_y,
        z=edge_z,
        mode='lines',
        line=dict(color='black', width=edge_thickness),
        hoverinfo='none'
    )
    fig.add_trace(edge_trace)

    #- Update layout for better visualization
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

    #fig.write_image('patch_fig.png', format='png', scale=2, engine='kaleido')
    fig.show()
