#!/usr/bin/env python3
"""
Versión RÁPIDA de visualización 3D - para testing
Sin superficie cerebral, menos frames, renderizado optimizado
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt

from plot import (
    subject_syncro,
    stc_coords_3d,
    node_colors,
    rej,
    band,
    cond,
    subject,
    FRAMES_DIR
)

def generate_quick_3d_video(
    epoch_start=0,
    epoch_end=5,
    fps=10,  # Reducido para ser más rápido
    output_file="brain_3d_quick.mp4"
):
    """
    Genera video 3D rápido para testing.
    
    - Sin superficie cortical
    - Menos FPS (10 en lugar de 30)
    - Menos frames por época (0.5 seg en lugar de 2)
    - Sin edges (solo parcelas)
    """
    print(f"\n{'='*70}")
    print(f"GENERADOR RÁPIDO DE VIDEO 3D")
    print(f"{'='*70}\n")
    print(f"[QUICK] Modo rápido: Sin superficie, 10 FPS, sin edges")
    print(f"[QUICK] Épocas: {epoch_start} a {epoch_end-1}")
    
    # Crear plotter offscreen
    plotter = pv.Plotter(window_size=(1280, 720), off_screen=True)
    plotter.set_background('white')
    
    # Configurar cámara
    plotter.camera_position = 'iso'
    plotter.camera.zoom(1.5)
    
    # Preparar directorio temporal
    temp_dir = FRAMES_DIR / "temp_quick"
    temp_dir.mkdir(exist_ok=True)
    
    frame_files = []
    frame_count = 0
    
    # Filtrar épocas rechazadas
    valid_epochs = [e for e in range(epoch_start, epoch_end) if e not in rej]
    
    print(f"[QUICK] Procesando {len(valid_epochs)} épocas...")
    
    for epoch_idx, epoch in enumerate(valid_epochs):
        print(f"\n[QUICK] Época {epoch}/{epoch_end-1} ({epoch_idx+1}/{len(valid_epochs)})")
        
        # Obtener datos
        stc_syncro_mat = subject_syncro["syncros_stc"][band][0][epoch]
        stc_kuramoto_epoch = subject_syncro["kuramoto_stc"][band][0][epoch]
        
        # Limitar tamaño
        n_parcels = min(len(stc_coords_3d), stc_syncro_mat.shape[0])
        stc_syncro_mat = stc_syncro_mat[:n_parcels, :n_parcels]
        coords = stc_coords_3d[:n_parcels]
        
        # Calcular sincronización
        parcel_sync = stc_syncro_mat.mean(axis=1)
        parcel_sync_norm = (parcel_sync - parcel_sync.min()) / \
                          (parcel_sync.max() - parcel_sync.min() + 1e-8)
        
        # Tamaños y colores
        sizes = 0.002 + 2 * 0.002 * parcel_sync_norm
        cmap = plt.colormaps['plasma']
        colors = cmap(parcel_sync_norm)
        
        r_mean = np.mean(stc_kuramoto_epoch)
        
        # Solo 5 frames por época (0.5 segundos a 10 FPS)
        frames_per_epoch = 5
        
        for frame_in_epoch in range(frames_per_epoch):
            print(f"[QUICK]   Frame {frame_in_epoch+1}/{frames_per_epoch}...", end='\r')
            
            # Limpiar todo
            plotter.clear()
            
            # Agregar parcelas como esferas simples
            for i, (pos, color, size) in enumerate(zip(coords, colors, sizes)):
                sphere = pv.Sphere(radius=size, center=pos, phi_resolution=8, theta_resolution=8)
                plotter.add_mesh(
                    sphere,
                    color=color[:3],
                    opacity=0.8,
                    smooth_shading=False  # Más rápido
                )
            
            # Rotar cámara
            angle = frame_count * 5  # Más rápido
            plotter.camera.azimuth = angle
            
            # Texto simple
            plotter.add_text(
                f"{subject} - {cond} - {band} | Epoch {epoch} | r={r_mean:.3f}",
                position='upper_left',
                font_size=12,
                color='black'
            )
            
            # Guardar frame
            frame_file = temp_dir / f"frame_{frame_count:04d}.png"
            plotter.screenshot(str(frame_file))
            frame_files.append(str(frame_file))
            frame_count += 1
        
        print(f"\n[QUICK]   ✓ Época {epoch} completada")
    
    plotter.close()
    
    # Combinar con OpenCV
    print(f"\n[QUICK] Combinando {frame_count} frames en video...")
    import cv2
    
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = FRAMES_DIR / output_file
    video = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_file in frame_files:
        video.write(cv2.imread(frame_file))
    
    video.release()
    
    # Limpiar
    print(f"[QUICK] Limpiando archivos temporales...")
    import shutil
    shutil.rmtree(temp_dir)
    
    print(f"\n[QUICK] ✓ Video generado!")
    print(f"[QUICK] Ubicación: {output_path}")
    print(f"[QUICK] Duración: {frame_count/fps:.1f} segundos")
    print(f"\n{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generador RÁPIDO de video 3D")
    parser.add_argument("--start", type=int, default=0, help="Época inicial")
    parser.add_argument("--end", type=int, default=5, help="Época final")
    parser.add_argument("--output", type=str, default="brain_3d_quick.mp4", help="Archivo de salida")
    
    args = parser.parse_args()
    
    generate_quick_3d_video(
        epoch_start=args.start,
        epoch_end=args.end,
        output_file=args.output
    )


