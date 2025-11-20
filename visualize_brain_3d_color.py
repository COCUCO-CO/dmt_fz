#!/usr/bin/env python3
"""
Visualización 3D con COLORES DE SINCRONIZACIÓN MAPEADOS sobre la superficie cerebral
Los colores se interpolan suavemente desde las parcelas a la corteza
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
import os
from scipy.spatial import distance_matrix

# Forzar uso de GPU
os.environ['VTK_USE_GPU'] = '1'
os.environ['MESA_GL_VERSION_OVERRIDE'] = '4.5'

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
    RESULTS_DIR,
    FRAMES_DIR
)


def map_parcel_values_to_surface(surface_vertices, parcel_coords, parcel_values, max_distance=0.03):
    """
    Mapea valores de parcelas a vértices de la superficie usando interpolación inversa por distancia.
    
    Parameters:
    -----------
    surface_vertices : array (n_vertices, 3)
        Coordenadas de vértices de la superficie
    parcel_coords : array (n_parcels, 3)
        Coordenadas de las parcelas
    parcel_values : array (n_parcels,)
        Valores a mapear (sincronización)
    max_distance : float
        Distancia máxima para interpolar (en metros)
    
    Returns:
    --------
    vertex_values : array (n_vertices,)
        Valores interpolados en cada vértice
    """
    n_vertices = len(surface_vertices)
    vertex_values = np.zeros(n_vertices)
    
    print(f"  Mapeando {len(parcel_coords)} parcelas a {n_vertices} vértices...")
    
    # Procesar en chunks para no consumir mucha RAM
    chunk_size = 5000
    
    for start_idx in range(0, n_vertices, chunk_size):
        end_idx = min(start_idx + chunk_size, n_vertices)
        chunk_vertices = surface_vertices[start_idx:end_idx]
        
        # Calcular distancias entre vértices del chunk y parcelas
        distances = distance_matrix(chunk_vertices, parcel_coords)
        
        # Para cada vértice, usar interpolación inversa por distancia (IDW)
        for i, dist_row in enumerate(distances):
            # Encontrar parcelas cercanas
            close_parcels = dist_row < max_distance
            
            if not np.any(close_parcels):
                # Si no hay parcelas cercanas, usar la más cercana
                closest = np.argmin(dist_row)
                vertex_values[start_idx + i] = parcel_values[closest]
            else:
                # Interpolación inversa por distancia (IDW)
                close_distances = dist_row[close_parcels]
                close_values = parcel_values[close_parcels]
                
                # Evitar división por cero
                close_distances = np.maximum(close_distances, 1e-6)
                
                # Pesos inversamente proporcionales a la distancia al cuadrado
                weights = 1.0 / (close_distances ** 2)
                weights /= weights.sum()
                
                vertex_values[start_idx + i] = np.sum(weights * close_values)
        
        if (end_idx // chunk_size) % 10 == 0:
            progress = (end_idx / n_vertices) * 100
            print(f"    Progreso: {progress:.1f}%", end='\r')
    
    print(f"    Progreso: 100.0% - ✓")
    
    return vertex_values


def generate_brain_color_video(
    subject_id="S01",
    condition="DMT",
    band_name="Alpha",
    output_file="brain_3d_color.mp4",
    start_epoch=0,
    end_epoch=20,
    fps=2,
    resolution=(1920, 1080)
):
    """
    Genera video con colores de sincronización mapeados sobre la superficie cerebral.
    Las transiciones son suaves y continuas.
    """
    print(f"\n{'='*70}")
    print(f"GENERADOR DE VIDEO 3D - COLORES SOBRE SUPERFICIE CEREBRAL")
    print(f"{'='*70}\n")
    print(f"[GPU] Sujeto: {subject_id}, Condición: {condition}, Banda: {band_name}")
    print(f"[GPU] Épocas: {start_epoch} a {end_epoch-1}")
    
    # Definir vistas (azimuth, elevation, nombre)
    views = [
        (0, 0, "Frontal"),
        (90, 0, "Lateral Izq"),
        (0, 90, "Superior"),
        (180, 0, "Posterior"),
        (-90, 0, "Lateral Der"),
        (0, -90, "Inferior")
    ]
    
    # Cargar superficie cerebral
    print(f"[GPU] Cargando superficie cortical...")
    try:
        import mne
        from mne.datasets import fetch_fsaverage
        
        fs_dir = fetch_fsaverage(verbose=False)
        subjects_dir = Path(fs_dir).parent
        
        surf_lh = mne.read_surface(subjects_dir / 'fsaverage' / 'surf' / 'lh.pial')
        surf_rh = mne.read_surface(subjects_dir / 'fsaverage' / 'surf' / 'rh.pial')
        
        vertices_lh, faces_lh = surf_lh
        vertices_rh, faces_rh = surf_rh
        
        n_verts_lh = len(vertices_lh)
        vertices = np.vstack([vertices_lh, vertices_rh])
        faces = np.vstack([faces_lh, faces_rh + n_verts_lh])
        
        # Convertir a metros (MNE usa mm)
        vertices = vertices / 1000.0
        
        print(f"[GPU] ✓ Superficie cargada: {len(vertices)} vértices, {len(faces)} caras")
    except Exception as e:
        print(f"[GPU] ✗ Error cargando superficie: {e}")
        return None
    
    # Preparar para guardar frames
    temp_frames_dir = FRAMES_DIR / "temp_color_frames"
    temp_frames_dir.mkdir(exist_ok=True)
    
    frame_files = []
    frame_count = 0
    
    # Filtrar épocas rechazadas
    valid_epochs = [e for e in range(start_epoch, end_epoch) if e not in rej]
    
    print(f"[GPU] Generando {len(valid_epochs)} épocas con 6 vistas c/u...\n")
    
    import time
    total_start = time.time()
    
    # Procesar cada época
    for epoch_idx, epoch in enumerate(valid_epochs):
        epoch_start = time.time()
        
        print(f"[GPU] Época {epoch}/{end_epoch-1} ({epoch_idx+1}/{len(valid_epochs)})")
        
        # Obtener datos de esta época
        stc_syncro_mat = subject_syncro["syncros_stc"][band_name][0][epoch]
        stc_kuramoto_epoch = subject_syncro["kuramoto_stc"][band_name][0][epoch]
        
        # Limitar tamaño
        n_coords = len(stc_coords_3d)
        n_matrix = stc_syncro_mat.shape[0]
        n_parcels = min(n_coords, n_matrix)
        
        stc_syncro_mat_limited = stc_syncro_mat[:n_parcels, :n_parcels]
        stc_coords_limited = stc_coords_3d[:n_parcels]
        
        # Calcular sincronización promedio para cada parcela
        parcel_sync = stc_syncro_mat_limited.mean(axis=1)
        parcel_sync_norm = (parcel_sync - parcel_sync.min()) / \
                          (parcel_sync.max() - parcel_sync.min() + 1e-8)
        
        # Kuramoto promedio
        r_mean = np.mean(stc_kuramoto_epoch)
        
        # Mapear valores de sincronización a la superficie
        print(f"[GPU]   Interpolando valores a superficie cerebral...")
        vertex_sync = map_parcel_values_to_surface(
            vertices,
            stc_coords_limited,
            parcel_sync_norm,
            max_distance=0.025  # 2.5 cm
        )
        
        # Crear mesh de superficie con valores
        faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
        brain_mesh = pv.PolyData(vertices, faces_pv)
        brain_mesh['sync'] = vertex_sync
        
        # Crear figura con 6 vistas + espacio para colorbar
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        # Dejar espacio a la derecha para la barra de colores
        gs = GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.05, 
                     right=0.92)  # Dejar 8% a la derecha para colorbar
        
        # Renderizar cada vista
        for view_idx, (azimuth, elevation, view_name) in enumerate(views):
            row = view_idx // 3
            col = view_idx % 3
            
            # Crear plotter para esta vista
            plotter = pv.Plotter(
                window_size=(640, 360),
                off_screen=True,
                lighting='three lights'
            )
            plotter.enable_anti_aliasing('ssaa')
            plotter.set_background('white')
            
            # Agregar cerebro con colores de sincronización (SIN barra individual)
            plotter.add_mesh(
                brain_mesh,
                scalars='sync',
                cmap='plasma',
                smooth_shading=True,
                specular=0.5,
                specular_power=15,
                show_scalar_bar=False,  # NO mostrar barra en cada vista
                clim=[0, 1]
            )
            
            # Configurar cámara
            plotter.camera_position = 'iso'
            plotter.camera.azimuth = azimuth
            plotter.camera.elevation = elevation
            plotter.camera.zoom(1.5)
            
            # Agregar título de vista
            plotter.add_text(
                f"{view_name}",
                position='upper_left',
                font_size=10,
                color='black'
            )
            
            # Capturar imagen
            img = plotter.screenshot(return_img=True)
            plotter.close()
            
            # Agregar al subplot
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.axis('off')
        
        # Agregar título general
        fig.suptitle(
            f"{subject_id} | {condition} | {band_name} | Época {epoch+1}/{num_epochs} | Kuramoto r={r_mean:.3f}",
            fontsize=16,
            fontweight='bold'
        )
        
        # Agregar UNA SOLA barra de colores a la derecha
        from matplotlib import cm
        from matplotlib.colors import Normalize
        
        # Crear un eje para la barra de colores
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        
        # Crear normalización y colormap
        norm = Normalize(vmin=0, vmax=1)
        cmap = cm.get_cmap('plasma')
        
        # Crear colorbar
        cb = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cbar_ax,
            orientation='vertical'
        )
        cb.set_label('Sincronización', fontsize=14, fontweight='bold')
        cb.ax.tick_params(labelsize=11)
        
        # Guardar frame
        frame_file = temp_frames_dir / f"frame_{frame_count:06d}.png"
        fig.savefig(frame_file, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        frame_files.append(str(frame_file))
        frame_count += 1
        
        epoch_time = time.time() - epoch_start
        print(f"[GPU]   ✓ Completada en {epoch_time:.1f}s\n")
    
    total_render_time = time.time() - total_start
    
    # Combinar frames en video
    print(f"[GPU] Combinando {frame_count} frames en video...")
    import cv2
    
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # Probar diferentes codecs
    codecs = [
        ('mp4v', '.mp4'),
        ('XVID', '.avi'),
        ('MJPG', '.avi'),
    ]
    
    video_created = False
    output_path = None
    
    for codec_name, ext in codecs:
        try:
            if output_file.endswith('.mp4') and ext != '.mp4':
                output_name = output_file.replace('.mp4', ext)
            else:
                output_name = output_file
            
            output_path = FRAMES_DIR / output_name
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            
            video_writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            if not video_writer.isOpened():
                continue
            
            print(f"[GPU]   Usando codec: {codec_name}")
            
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                video_writer.write(frame)
            
            video_writer.release()
            
            if output_path.exists() and output_path.stat().st_size > 0:
                video_created = True
                break
        except Exception as e:
            continue
    
    if not video_created:
        print(f"\n[GPU] ⚠ No se pudo crear video, guardando frames...")
        final_frames_dir = FRAMES_DIR / f"color_frames_{subject}_{cond}_{band}"
        if final_frames_dir.exists():
            import shutil
            shutil.rmtree(final_frames_dir)
        temp_frames_dir.rename(final_frames_dir)
        output_path = final_frames_dir
        print(f"[GPU] ✓ Frames en: {final_frames_dir}")
    else:
        print(f"[GPU] Limpiando archivos temporales...")
        import shutil
        shutil.rmtree(temp_frames_dir)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"[GPU] ✓ Generación completada!")
    print(f"[GPU] Ubicación: {output_path}")
    print(f"[GPU] Frames: {frame_count}")
    print(f"[GPU] Tiempo total: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Video 3D con sincronización coloreada sobre superficie cerebral"
    )
    parser.add_argument("--start", type=int, default=0, help="Época inicial")
    parser.add_argument("--end", type=int, default=20, help="Época final")
    parser.add_argument("--fps", type=int, default=2, help="FPS del video")
    parser.add_argument("--output", type=str, default="brain_3d_color.mp4",
                       help="Archivo de salida")
    
    args = parser.parse_args()
    
    print(f"\n🎨 Colores de sincronización mapeados sobre cerebro 3D")
    print(f"🚀 GPU RTX 4090 - Interpolación suave y continua\n")
    
    generate_brain_color_video(
        subject_id=subject,
        condition=cond,
        band_name=band,
        output_file=args.output,
        start_epoch=args.start,
        end_epoch=args.end,
        fps=args.fps
    )

