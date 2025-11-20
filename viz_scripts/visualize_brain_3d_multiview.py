#!/usr/bin/env python3

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


def create_parcels_mesh(coords, sync_values, sizes):
    """
    Crea un mesh con todas las parcelas como esferas con colores de sincronización.
    """
    # Crear mesh combinado
    combined_mesh = pv.PolyData()
    
    for i, (pos, sync, size) in enumerate(zip(coords, sync_values, sizes)):
        sphere = pv.Sphere(
            radius=size,
            center=pos,
            phi_resolution=16,
            theta_resolution=16
        )
        # Agregar el valor de sincronización como scalar
        sphere['sync'] = np.full(sphere.n_points, sync)
        
        if i == 0:
            combined_mesh = sphere
        else:
            combined_mesh = combined_mesh.merge(sphere)
    
    return combined_mesh


def generate_multiview_video(
    subject_id="S01",
    condition="DMT",
    band_name="Alpha",
    output_file="brain_3d_multiview.mp4",
    start_epoch=0,
    end_epoch=20,
    fps=2,  # Más lento para ver mejor los cambios
    show_brain_surface=True,  # ACTIVADO por defecto para ver el cerebro
    parcel_base_size=0.004,  # Un poco más grandes
    parcel_scale_factor=2.5,
    resolution=(1920, 1080)
):
    """
    Genera video con 6 vistas fijas del cerebro mostrando sincronización en colores.
    
    Layout:
    [Frontal]  [Lateral Izq]  [Superior]
    [Posterior][Lateral Der]  [Inferior]
    """
    print(f"\n{'='*70}")
    print(f"GENERADOR DE VIDEO 3D - VISTAS MÚLTIPLES")
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
    
    # Preparar para guardar frames
    temp_frames_dir = FRAMES_DIR / "temp_multiview_frames"
    temp_frames_dir.mkdir(exist_ok=True)
    
    frame_files = []
    frame_count = 0
    
    # Filtrar épocas rechazadas
    valid_epochs = [e for e in range(start_epoch, end_epoch) if e not in rej]
    
    print(f"[GPU] Generando {len(valid_epochs)} épocas con 6 vistas c/u...\n")
    
    # Cargar superficie cerebral si está habilitado
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
            
            print(f"[GPU] ✓ Superficie cargada")
        except Exception as e:
            print(f"[GPU] ⚠ No se pudo cargar superficie: {e}")
    
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
        
        # Tamaños de esferas basados en sincronización
        sizes = parcel_base_size + parcel_scale_factor * parcel_base_size * parcel_sync_norm
        
        # Kuramoto promedio
        r_mean = np.mean(stc_kuramoto_epoch)
        
        # Crear mesh con parcelas coloreadas
        parcels_mesh = create_parcels_mesh(stc_coords_limited, parcel_sync_norm, sizes)
        
        # Crear un plotter para cada vista
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
        gs = GridSpec(2, 3, figure=fig, hspace=0.05, wspace=0.05,
                     right=0.92)  # Dejar espacio a la derecha para colorbar
        
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
            
            # Agregar superficie cerebral si está habilitada
            if brain_mesh is not None:
                plotter.add_mesh(
                    brain_mesh,
                    opacity=0.3,  # Más opaco para ver mejor el cerebro
                    color='lightgray',
                    smooth_shading=True,
                    specular=0.3,
                    diffuse=0.8
                )
            
            # Agregar parcelas con colormap (SIN barra individual)
            plotter.add_mesh(
                parcels_mesh,
                scalars='sync',
                cmap='plasma',
                opacity=0.85,  # Un poco transparentes para ver el cerebro debajo
                show_scalar_bar=False,  # NO mostrar barra en cada vista
                smooth_shading=True,
                specular=0.8,
                specular_power=20,
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
        
        cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
        norm = Normalize(vmin=0, vmax=1)
        cmap_mpl = cm.get_cmap('plasma')
        cb = plt.colorbar(
            cm.ScalarMappable(norm=norm, cmap=cmap_mpl),
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
        print(f"[GPU]   ✓ Completada en {epoch_time:.1f}s")
    
    total_render_time = time.time() - total_start
    
    # Combinar frames en video
    print(f"\n[GPU] Combinando {frame_count} frames en video...")
    import cv2
    
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # Probar diferentes codecs en orden de preferencia
    codecs = [
        ('mp4v', '.mp4'),  # MPEG-4, más compatible
        ('XVID', '.avi'),  # Xvid
        ('MJPG', '.avi'),  # Motion JPEG
    ]
    
    video_created = False
    output_path = None
    
    for codec_name, ext in codecs:
        try:
            # Cambiar extensión del output si es necesario
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
            
            # Verificar que se creó correctamente
            if not video_writer.isOpened():
                print(f"[GPU]   ⚠ Codec {codec_name} no disponible")
                continue
            
            print(f"[GPU]   Usando codec: {codec_name}")
            
            for i, frame_file in enumerate(frame_files):
                frame = cv2.imread(frame_file)
                video_writer.write(frame)
                if (i + 1) % 10 == 0:
                    print(f"[GPU]   Escribiendo: {i+1}/{len(frame_files)}", end='\r')
            
            video_writer.release()
            print()
            
            # Verificar que el archivo se creó
            if output_path.exists() and output_path.stat().st_size > 0:
                video_created = True
                break
            else:
                print(f"[GPU]   ⚠ Video vacío con {codec_name}")
        except Exception as e:
            print(f"[GPU]   ⚠ Error con codec {codec_name}: {e}")
            continue
    
    if not video_created:
        print(f"\n[GPU] ⚠ No se pudo crear video con ningún codec")
        print(f"[GPU] Guardando frames individuales en: {temp_frames_dir}")
        
        # Mover frames a directorio permanente
        final_frames_dir = FRAMES_DIR / f"multiview_frames_{subject}_{cond}_{band}"
        if final_frames_dir.exists():
            import shutil
            shutil.rmtree(final_frames_dir)
        temp_frames_dir.rename(final_frames_dir)
        output_path = final_frames_dir
        
        print(f"[GPU] ✓ Frames guardados exitosamente")
        print(f"[GPU] Para crear video manualmente:")
        print(f"[GPU]   ffmpeg -framerate {fps} -i {final_frames_dir}/frame_%06d.png -c:v libx264 -pix_fmt yuv420p output.mp4")
    else:
        # Limpiar frames temporales solo si el video se creó
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
    print(f"{'='*70}\n")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generador de video 3D multi-vista con colores de sincronización"
    )
    parser.add_argument("--start", type=int, default=0, help="Época inicial")
    parser.add_argument("--end", type=int, default=20, help="Época final")
    parser.add_argument("--fps", type=int, default=2, help="FPS del video (default: 2)")
    parser.add_argument("--output", type=str, default="brain_3d_multiview.mp4",
                       help="Archivo de salida")
    parser.add_argument("--no-surface", action="store_true",
                       help="NO incluir superficie cortical (por defecto SÍ se incluye)")
    
    args = parser.parse_args()
    
    print(f"\n🎨 Generando video con 6 vistas fijas y colores de sincronización")
    print(f"🚀 Usando GPU RTX 4090\n")
    
    generate_multiview_video(
        subject_id=subject,
        condition=cond,
        band_name=band,
        output_file=args.output,
        start_epoch=args.start,
        end_epoch=args.end,
        fps=args.fps,
        show_brain_surface=not args.no_surface  # Invertido: por defecto True
    )

