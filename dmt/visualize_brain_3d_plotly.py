#!/usr/bin/env python3
"""
Visualización 3D interactiva con Plotly (alternativa web)
Para exploración interactiva en navegador
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

from plot import (
    subject_syncro,
    subject_phases,
    stc_coords_3d,
    node_colors,
    label_names,
    rej,
    num_epochs,
    band,
    cond,
    subject,
    FRAMES_DIR
)


def create_interactive_brain_3d(epoch=0, show_edges=True, edge_threshold=85):
    """
    Crea una visualización 3D interactiva con Plotly.
    
    Parameters
    ----------
    epoch : int
        Época a visualizar
    show_edges : bool
        Mostrar conexiones
    edge_threshold : float
        Percentil para filtrar conexiones
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figura interactiva
    """
    print(f"[Plotly] Creando visualización 3D para época {epoch}...")
    
    # Obtener datos
    stc_syncro_mat = subject_syncro["syncros_stc"][band][0][epoch]
    stc_kuramoto_epoch = subject_syncro["kuramoto_stc"][band][0][epoch]
    
    # Fixed: Asegurar que matriz y coordenadas tengan el mismo tamaño
    n_coords = len(stc_coords_3d)
    n_matrix = stc_syncro_mat.shape[0]
    n_parcels = min(n_coords, n_matrix)
    
    stc_syncro_mat = stc_syncro_mat[:n_parcels, :n_parcels]
    stc_coords_limited = stc_coords_3d[:n_parcels]
    label_names_limited = label_names[:n_parcels]
    
    # Calcular sincronización por parcela
    parcel_sync = stc_syncro_mat.mean(axis=1)
    parcel_sync_norm = (parcel_sync - parcel_sync.min()) / \
                      (parcel_sync.max() - parcel_sync.min() + 1e-8)
    
    # Tamaños proporcionales a sincronización
    sizes = 5 + 15 * parcel_sync_norm
    
    # Crear figura
    fig = go.Figure()
    
    # Agregar edges si está habilitado
    if show_edges:
        threshold = np.percentile(stc_syncro_mat, edge_threshold)
        strong_edges = np.where(stc_syncro_mat > threshold)
        
        edge_x = []
        edge_y = []
        edge_z = []
        
        for i, j in zip(strong_edges[0], strong_edges[1]):
            if i < j and i < n_parcels and j < n_parcels:  # Evitar duplicados y verificar límites
                # Agregar línea
                edge_x.extend([stc_coords_limited[i, 0], stc_coords_limited[j, 0], None])
                edge_y.extend([stc_coords_limited[i, 1], stc_coords_limited[j, 1], None])
                edge_z.extend([stc_coords_limited[i, 2], stc_coords_limited[j, 2], None])
        
        # Plotear edges
        fig.add_trace(go.Scatter3d(
            x=edge_x,
            y=edge_y,
            z=edge_z,
            mode='lines',
            line=dict(color='lightblue', width=1),
            opacity=0.3,
            hoverinfo='none',
            name='Conexiones'
        ))
    
    # Agregar parcelas como scatter3d
    fig.add_trace(go.Scatter3d(
        x=stc_coords_limited[:, 0],
        y=stc_coords_limited[:, 1],
        z=stc_coords_limited[:, 2],
        mode='markers',
        marker=dict(
            size=sizes,
            color=parcel_sync_norm,
            colorscale='Plasma',
            colorbar=dict(title="Sincronización", x=1.02),
            line=dict(color='black', width=0.5),
            opacity=0.9
        ),
        text=[f"{name}<br>Sync: {sync:.3f}" 
              for name, sync in zip(label_names_limited, parcel_sync)],
        hovertemplate='<b>%{text}</b><extra></extra>',
        name='Parcelas'
    ))
    
    # Configuración del layout
    r_mean = np.mean(stc_kuramoto_epoch)
    
    fig.update_layout(
        title=dict(
            text=f"{subject} - {cond} - {band} - Epoch {epoch}<br>"
                 f"Kuramoto Order Parameter: r = {r_mean:.3f}",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor='white',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='data'
        ),
        width=1400,
        height=900,
        showlegend=True,
        hovermode='closest',
        paper_bgcolor='white',
        plot_bgcolor='white'
    )
    
    return fig


def create_animated_brain_3d(start_epoch=0, end_epoch=10, show_edges=False):
    """
    Crea una animación 3D interactiva con slider temporal.
    
    Parameters
    ----------
    start_epoch : int
        Época inicial
    end_epoch : int
        Época final
    show_edges : bool
        Mostrar conexiones (puede ser lento con animación)
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figura con animación
    """
    print(f"[Plotly] Creando animación 3D para épocas {start_epoch} a {end_epoch-1}...")
    
    # Filtrar épocas válidas
    valid_epochs = [e for e in range(start_epoch, end_epoch) if e not in rej]
    
    # Crear frames para cada época
    frames = []
    
    for epoch in valid_epochs:
        stc_syncro_mat = subject_syncro["syncros_stc"][band][0][epoch]
        stc_kuramoto_epoch = subject_syncro["kuramoto_stc"][band][0][epoch]
        
        # Fixed: Asegurar tamaño consistente
        n_coords = len(stc_coords_3d)
        n_matrix = stc_syncro_mat.shape[0]
        n_parcels = min(n_coords, n_matrix)
        
        stc_syncro_mat = stc_syncro_mat[:n_parcels, :n_parcels]
        stc_coords_limited = stc_coords_3d[:n_parcels]
        label_names_limited = label_names[:n_parcels]
        
        parcel_sync = stc_syncro_mat.mean(axis=1)
        parcel_sync_norm = (parcel_sync - parcel_sync.min()) / \
                          (parcel_sync.max() - parcel_sync.min() + 1e-8)
        
        sizes = 5 + 15 * parcel_sync_norm
        r_mean = np.mean(stc_kuramoto_epoch)
        
        # Crear frame
        frame_data = [go.Scatter3d(
            x=stc_coords_limited[:, 0],
            y=stc_coords_limited[:, 1],
            z=stc_coords_limited[:, 2],
            mode='markers',
            marker=dict(
                size=sizes,
                color=parcel_sync_norm,
                colorscale='Plasma',
                colorbar=dict(title="Sincronización"),
                line=dict(color='black', width=0.5),
                opacity=0.9,
                cmin=0,
                cmax=1
            ),
            text=[f"{name}<br>Sync: {sync:.3f}" 
                  for name, sync in zip(label_names_limited, parcel_sync)],
            hovertemplate='<b>%{text}</b><extra></extra>',
            name='Parcelas'
        )]
        
        frames.append(go.Frame(
            data=frame_data,
            name=str(epoch),
            layout=go.Layout(
                title=f"{subject} - {cond} - {band} - Epoch {epoch}<br>"
                      f"Kuramoto r = {r_mean:.3f}"
            )
        ))
    
    # Crear figura inicial con primer frame
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=frames[0].layout.title,
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='white',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='data'
            ),
            width=1400,
            height=900,
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(
                            label='Play',
                            method='animate',
                            args=[None, dict(
                                frame=dict(duration=500, redraw=True),
                                fromcurrent=True,
                                mode='immediate'
                            )]
                        ),
                        dict(
                            label='Pause',
                            method='animate',
                            args=[[None], dict(
                                frame=dict(duration=0, redraw=False),
                                mode='immediate'
                            )]
                        )
                    ],
                    x=0.1,
                    y=0.05,
                    xanchor='left',
                    yanchor='bottom'
                )
            ],
            sliders=[dict(
                active=0,
                steps=[dict(
                    args=[[f.name], dict(
                        frame=dict(duration=0, redraw=True),
                        mode='immediate'
                    )],
                    label=f"Epoch {f.name}",
                    method='animate'
                ) for f in frames],
                x=0.1,
                y=0,
                len=0.85,
                xanchor='left',
                yanchor='bottom'
            )]
        ),
        frames=frames
    )
    
    return fig


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualización 3D interactiva con Plotly")
    parser.add_argument("--mode", choices=["static", "animated"], default="static",
                       help="Modo: 'static' (una época) o 'animated' (slider temporal)")
    parser.add_argument("--epoch", type=int, default=0, help="Época para modo static")
    parser.add_argument("--start", type=int, default=0, help="Época inicial para animated")
    parser.add_argument("--end", type=int, default=10, help="Época final para animated")
    parser.add_argument("--edges", action="store_true", help="Mostrar conexiones")
    parser.add_argument("--output", type=str, default="brain_3d_interactive.html",
                       help="Archivo HTML de salida")
    
    args = parser.parse_args()
    
    if args.mode == "static":
        fig = create_interactive_brain_3d(
            epoch=args.epoch,
            show_edges=args.edges,
            edge_threshold=85
        )
    else:
        fig = create_animated_brain_3d(
            start_epoch=args.start,
            end_epoch=args.end,
            show_edges=args.edges
        )
    
    # Guardar como HTML interactivo
    output_path = FRAMES_DIR / args.output
    fig.write_html(str(output_path))
    
    print(f"\n[Plotly] ✓ Visualización guardada: {output_path}")
    print(f"[Plotly] Abrir en navegador para interactuar (rotar, zoom, hover)")

