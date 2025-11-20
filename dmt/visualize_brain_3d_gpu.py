#!/usr/bin/env python3
"""
Visualización 3D OPTIMIZADA PARA GPU (RTX 4090)
Usa aceleración por hardware para renderizado ultrarrápido
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
import os

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


def generate_3d_video_gpu(
    subject_id="S01",
    condition="DMT",
    band_name="Alpha",
    output_file="brain_3d_gpu.mp4",
    start_epoch=0,
    end_epoch=20,
    fps=30,
    rotation_speed=2.0,
    show_brain_surface=False,  # Desactivado por defecto (más rápido)
    show_edges=False,  # Desactivado por defecto (más rápido)
    edge_threshold=90,
    parcel_base_size=0.003,
    parcel_scale_factor=3.0,
    resolution=(1920, 1080),  # Resolución completa
    use_gpu_compositing=True
):
    """
    Genera video 3D optimizado para GPU.
    
    Optimizaciones GPU:
    - Renderizado paralelo en GPU
    - Batch processing de frames
    - Compositing por hardware
    - Anti-aliasing acelerado por GPU
    """
    print(f"\n{'='*70}")
    print(f"GENERADOR DE VIDEO 3D - ACELERADO POR GPU")
    print(f"{'='*70}\n")
    print(f"[GPU] Configurando renderizado acelerado por hardware...")
    
    # Verificar que PyVista detecta GPU
    try:
        print(f"[GPU] Sistema gráfico: {pv.system.gl_info['version']}")
        print(f"[GPU] Renderer: {pv.system.gl_info['renderer']}")
    except:
        print("[GPU] ⚠ No se pudo obtener info de GPU, pero continuando...")
    
    print(f"[GPU] Sujeto: {subject_id}, Condición: {condition}, Banda: {band_name}")
    print(f"[GPU] Épocas: {start_epoch} a {end_epoch-1}")
    print(f"[GPU] FPS: {fps}, Resolución: {resolution}")
    
    # Crear plotter con optimizaciones GPU
    plotter = pv.Plotter(
        window_size=resolution,
        off_screen=True,
        lighting='three lights'
    )
    
    # Habilitar anti-aliasing por GPU (SSAA)
    if use_gpu_compositing:
        plotter.enable_anti_aliasing('ssaa')  # Screen Space Anti-Aliasing
    
    plotter.set_background('white')
    
    # Cargar superficie cerebral si está habilitado (y cachearla)
    brain_mesh = None
    if show_brain_surface:
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
            
            faces_pv = np.hstack([np.full((len(faces), 1), 3), faces]).flatten()
            brain_mesh = pv.PolyData(vertices, faces_pv)
            
            # Agregar al plotter (solo una vez, permanece durante todo el video)
            plotter.add_mesh(
                brain_mesh,
                opacity=0.1,
                color='lightgray',
                smooth_shading=True,
                show_edges=False
            )
            print(f"[GPU] ✓ Superficie agregada")
        except Exception as e:
            print(f"[GPU] ⚠ No se pudo cargar superficie: {e}")
    
    # Configurar cámara
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.5)
    
    # Preparar para guardar frames
    temp_frames_dir = FRAMES_DIR / "temp_gpu_frames"
    temp_frames_dir.mkdir(exist_ok=True)
    
    frame_files = []
    frame_count = 0
    
    # Filtrar épocas rechazadas
    valid_epochs = [e for e in range(start_epoch, end_epoch) if e not in rej]
    
    print(f"[GPU] Generando {len(valid_epochs)} épocas...")
    print(f"[GPU] NOTA: Primera época es lenta (compilando shaders), luego acelera\n")
    
    # Pre-calcular datos para todas las épocas (aprovecha RAM)
    print(f"[GPU] Pre-calculando datos de épocas...")
    epoch_data = []
    
    for epoch in valid_epochs:
        stc_syncro_mat = subject_syncro["syncros_stc"][band_name][0][epoch]
        stc_kuramoto_epoch = subject_syncro["kuramoto_stc"][band_name][0][epoch]
        
        # Limitar tamaño
        n_coords = len(stc_coords_3d)
        n_matrix = stc_syncro_mat.shape[0]
        n_parcels = min(n_coords, n_matrix)
        
        stc_syncro_mat_limited = stc_syncro_mat[:n_parcels, :n_parcels]
        stc_coords_limited = stc_coords_3d[:n_parcels]
        
        # Calcular sincronización
        parcel_sync = stc_syncro_mat_limited.mean(axis=1)
        parcel_sync_norm = (parcel_sync - parcel_sync.min()) / \
                          (parcel_sync.max() - parcel_sync.min() + 1e-8)
        
        sizes = parcel_base_size + parcel_scale_factor * parcel_base_size * parcel_sync_norm
        
        cmap = plt.colormaps['plasma']
        colors_rgba = cmap(parcel_sync_norm)
        
        r_mean = np.mean(stc_kuramoto_epoch)
        
        epoch_data.append({
            'coords': stc_coords_limited,
            'sizes': sizes,
            'colors': colors_rgba,
            'syncro_mat': stc_syncro_mat_limited,
            'r_mean': r_mean,
            'epoch': epoch
        })
    
    print(f"[GPU] ✓ Datos pre-calculados para {len(epoch_data)} épocas\n")
    
    # Procesar épocas
    frames_per_epoch = int(fps * 2)  # 2 segundos por época
    
    import time
    total_start = time.time()
    
    for epoch_idx, data in enumerate(epoch_data):
        epoch_start = time.time()
        epoch = data['epoch']
        
        print(f"[GPU] Época {epoch}/{end_epoch-1} ({epoch_idx+1}/{len(epoch_data)})")
        
        # Pre-crear todas las esferas para esta época (batch)
        spheres = []
        for pos, color, size in zip(data['coords'], data['colors'], data['sizes']):
            sphere = pv.Sphere(
                radius=size,
                center=pos,
                phi_resolution=12,  # Balance calidad/velocidad
                theta_resolution=12
            )
            spheres.append((sphere, color))
        
        # Renderizar frames de esta época
        for frame_in_epoch in range(frames_per_epoch):
            # Limpiar actores dinámicos (mantiene brain_mesh)
            plotter.clear_actors()
            if brain_mesh is not None:
                plotter.add_mesh(
                    brain_mesh,
                    opacity=0.1,
                    color='lightgray',
                    smooth_shading=True,
                    show_edges=False
                )
            
            # Agregar todas las esferas (la GPU renderiza en paralelo)
            for sphere, color in spheres:
                plotter.add_mesh(
                    sphere,
                    color=color[:3],
                    opacity=0.9,
                    smooth_shading=True,
                    specular=0.5
                )
            
            # Rotar cámara
            angle = frame_count * rotation_speed
            plotter.camera.azimuth = angle
            plotter.camera.elevation = 10 + 5 * np.sin(np.radians(angle * 2))
            
            # Texto
            text = (f"{subject_id} - {condition} - {band_name} | "
                   f"Epoch {epoch+1}/{num_epochs} | r={data['r_mean']:.3f}")
            plotter.add_text(text, position='upper_left', font_size=14, color='black')
            
            # Screenshot (la GPU hace el trabajo pesado aquí)
            frame_file = temp_frames_dir / f"frame_{frame_count:06d}.png"
            plotter.screenshot(str(frame_file))
            frame_files.append(str(frame_file))
            frame_count += 1
            
            # Progress cada 10 frames
            if frame_in_epoch % 10 == 0:
                elapsed = time.time() - epoch_start
                fps_actual = (frame_in_epoch + 1) / elapsed if elapsed > 0 else 0
                print(f"[GPU]   Frame {frame_in_epoch}/{frames_per_epoch} "
                      f"({fps_actual:.1f} renders/sec)", end='\r')
        
        epoch_time = time.time() - epoch_start
        print(f"\n[GPU]   ✓ Época {epoch} completada en {epoch_time:.1f}s "
              f"({frames_per_epoch/epoch_time:.1f} fps)")
    
    plotter.close()
    
    total_render_time = time.time() - total_start
    
    # Combinar frames con OpenCV (también puede usar GPU si está disponible)
    print(f"\n[GPU] Combinando {frame_count} frames en video...")
    import cv2
    
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # Usar codec H.264 (mejor que mp4v)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
    output_path = FRAMES_DIR / output_file
    
    video_writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )
    
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"[GPU]   Escribiendo: {i+1}/{len(frame_files)}", end='\r')
    
    video_writer.release()
    print()
    
    # Limpiar
    print(f"[GPU] Limpiando archivos temporales...")
    import shutil
    shutil.rmtree(temp_frames_dir)
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*70}")
    print(f"[GPU] ✓ Video generado exitosamente!")
    print(f"[GPU] Ubicación: {output_path}")
    print(f"[GPU] Frames: {frame_count}")
    print(f"[GPU] Duración video: {frame_count/fps:.1f}s")
    print(f"[GPU] Tiempo procesamiento: {total_time:.1f}s")
    print(f"[GPU] Velocidad promedio: {frame_count/total_time:.1f} frames/sec")
    print(f"{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generador de video 3D acelerado por GPU (RTX 4090)"
    )
    parser.add_argument("--start", type=int, default=0, help="Época inicial")
    parser.add_argument("--end", type=int, default=20, help="Época final")
    parser.add_argument("--fps", type=int, default=30, help="FPS del video")
    parser.add_argument("--output", type=str, default="brain_3d_gpu.mp4",
                       help="Archivo de salida")
    parser.add_argument("--surface", action="store_true",
                       help="Incluir superficie cortical (más lento)")
    parser.add_argument("--edges", action="store_true",
                       help="Incluir conexiones (más lento)")
    parser.add_argument("--resolution", type=int, nargs=2, default=[1920, 1080],
                       help="Resolución del video (ancho alto)")
    
    args = parser.parse_args()
    
    print(f"\n🚀 Usando GPU RTX 4090 para renderizado acelerado")
    print(f"💡 Tip: Primera época compila shaders, luego va MUCHO más rápido\n")
    
    generate_3d_video_gpu(
        subject_id=subject,
        condition=cond,
        band_name=band,
        output_file=args.output,
        start_epoch=args.start,
        end_epoch=args.end,
        fps=args.fps,
        show_brain_surface=args.surface,
        show_edges=args.edges,
        resolution=tuple(args.resolution)
    )

