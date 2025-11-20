#!/usr/bin/env python3
"""
Visualización 3D avanzada del cerebro usando PyVista
Estado del arte para visualización neurocientífica 3D
"""

import numpy as np
import pyvista as pv
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm

# MNE para cargar la superficie cortical
import mne
from mne.datasets import fetch_fsaverage

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


class Brain3DVisualizer:
    """Visualizador 3D avanzado de actividad cerebral."""
    
    def __init__(self, use_fsaverage=True):
        """
        Inicializa el visualizador.
        
        Parameters
        ----------
        use_fsaverage : bool
            Si True, carga el cerebro template fsaverage de MNE
        """
        print("[3D] Inicializando visualizador 3D...")
        
        self.plotter = None
        self.brain_mesh = None
        self.parcel_actors = []
        self.edge_actors = []
        
        if use_fsaverage:
            self._load_fsaverage_brain()
        
    def _load_fsaverage_brain(self):
        """Carga la superficie cortical de fsaverage."""
        print("[3D] Cargando superficie cortical fsaverage...")
        
        try:
            # Descargar fsaverage si no existe
            fs_dir = fetch_fsaverage(verbose=True)
            subjects_dir = Path(fs_dir).parent
            
            # Cargar superficie pial (más externa)
            # Opciones: 'pial', 'white', 'inflated', 'sphere'
            surf_lh = mne.read_surface(
                subjects_dir / 'fsaverage' / 'surf' / 'lh.pial'
            )
            surf_rh = mne.read_surface(
                subjects_dir / 'fsaverage' / 'surf' / 'rh.pial'
            )
            
            # Convertir a PyVista mesh
            vertices_lh, faces_lh = surf_lh
            vertices_rh, faces_rh = surf_rh
            
            # Combinar hemisferios
            n_verts_lh = len(vertices_lh)
            vertices = np.vstack([vertices_lh, vertices_rh])
            faces = np.vstack([faces_lh, faces_rh + n_verts_lh])
            
            # Crear mesh PyVista
            # PyVista requiere faces en formato [n_points, i1, i2, i3, ...]
            faces_pv = np.hstack([
                np.full((len(faces), 1), 3),  # Cada cara tiene 3 vértices
                faces
            ]).flatten()
            
            self.brain_mesh = pv.PolyData(vertices, faces_pv)
            print(f"[3D] ✓ Superficie cargada: {len(vertices)} vértices, {len(faces)} caras")
            
        except Exception as e:
            print(f"[3D] ⚠ No se pudo cargar fsaverage: {e}")
            print("[3D] Usando visualización simplificada...")
            self.brain_mesh = None
    
    def create_plotter(self, window_size=(1920, 1080), off_screen=False):
        """Crea el plotter PyVista."""
        self.plotter = pv.Plotter(
            window_size=window_size,
            off_screen=off_screen,
            lighting='three lights'
        )
        
        # Configurar iluminación
        self.plotter.set_background('white')
        
        return self.plotter
    
    def add_brain_surface(self, opacity=0.15, color='lightgray'):
        """Agrega la superficie cortical al plotter."""
        if self.brain_mesh is None or self.plotter is None:
            return
        
        self.plotter.add_mesh(
            self.brain_mesh,
            opacity=opacity,
            color=color,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
            show_edges=False
        )
    
    def add_parcels(self, positions, colors, sizes, opacity=0.9):
        """
        Agrega las parcelas cerebrales como esferas.
        
        Parameters
        ----------
        positions : array (n_parcels, 3)
            Posiciones XYZ de las parcelas
        colors : array (n_parcels, 3) o (n_parcels, 4)
            Colores RGB o RGBA
        sizes : array (n_parcels,)
            Tamaños de las esferas
        opacity : float
            Opacidad de las parcelas
        """
        if self.plotter is None:
            return
        
        # Limpiar parcelas anteriores
        for actor in self.parcel_actors:
            self.plotter.remove_actor(actor)
        self.parcel_actors = []
        
        # Agregar cada parcela como esfera
        for i, (pos, color, size) in enumerate(zip(positions, colors, sizes)):
            sphere = pv.Sphere(radius=size, center=pos)
            
            # Normalizar color si es necesario
            if isinstance(color, (list, tuple, np.ndarray)):
                if len(color) == 3:
                    color = list(color) + [opacity]
                elif len(color) == 4:
                    color = list(color[:3]) + [opacity]
            
            actor = self.plotter.add_mesh(
                sphere,
                color=color[:3] if len(color) > 3 else color,
                opacity=color[3] if len(color) == 4 else opacity,
                smooth_shading=True,
                specular=0.8,
                specular_power=20
            )
            self.parcel_actors.append(actor)
    
    def add_edges(self, positions, connectivity_matrix, threshold_percentile=90, 
                  color='steelblue', opacity=0.3, line_width=2):
        """
        Agrega conexiones entre parcelas.
        
        Parameters
        ----------
        positions : array (n_parcels, 3)
            Posiciones de las parcelas
        connectivity_matrix : array (n_parcels, n_parcels)
            Matriz de conectividad
        threshold_percentile : float
            Percentil para filtrar conexiones débiles
        color : str or array
            Color de las líneas
        opacity : float
            Opacidad de las líneas
        line_width : float
            Grosor de las líneas
        """
        if self.plotter is None:
            return
        
        # Limpiar edges anteriores
        for actor in self.edge_actors:
            self.plotter.remove_actor(actor)
        self.edge_actors = []
        
        # Fixed: Asegurar que matriz y posiciones tengan el mismo tamaño
        n_positions = len(positions)
        n_matrix = connectivity_matrix.shape[0]
        n_parcels = min(n_positions, n_matrix)
        
        # Limitar matriz a las posiciones disponibles
        connectivity_matrix = connectivity_matrix[:n_parcels, :n_parcels]
        
        # Umbral de conectividad
        threshold = np.percentile(connectivity_matrix, threshold_percentile)
        
        # Encontrar conexiones fuertes
        strong_edges = np.where(connectivity_matrix > threshold)
        
        # Crear líneas
        for i, j in zip(strong_edges[0], strong_edges[1]):
            if i < j and i < n_parcels and j < n_parcels:  # Evitar duplicados y verificar límites
                line = pv.Line(positions[i], positions[j])
                
                # Intensidad basada en conectividad
                strength = (connectivity_matrix[i, j] - threshold) / \
                          (connectivity_matrix.max() - threshold + 1e-8)
                
                actor = self.plotter.add_mesh(
                    line,
                    color=color,
                    opacity=opacity * strength,
                    line_width=line_width
                )
                self.edge_actors.append(actor)
    
    def set_camera_position(self, position='isometric', zoom=1.3):
        """Configura la posición de la cámara."""
        if self.plotter is None:
            return
        
        if position == 'isometric':
            self.plotter.camera_position = 'iso'
        elif position == 'front':
            self.plotter.view_xy()
        elif position == 'side':
            self.plotter.view_xz()
        elif position == 'top':
            self.plotter.view_yz()
        else:
            self.plotter.camera_position = position
        
        self.plotter.camera.zoom(zoom)
    
    def add_text(self, text, position='upper_left', font_size=14, color='black'):
        """Agrega texto a la visualización."""
        if self.plotter is None:
            return
        
        self.plotter.add_text(
            text,
            position=position,
            font_size=font_size,
            color=color,
            font='arial'
        )
    
    def screenshot(self, filename, transparent_background=False):
        """Guarda una captura de pantalla."""
        if self.plotter is None:
            return
        
        self.plotter.screenshot(
            filename,
            transparent_background=transparent_background,
            return_img=False
        )
        print(f"[3D] ✓ Screenshot guardado: {filename}")
    
    def show(self):
        """Muestra la visualización interactiva."""
        if self.plotter is None:
            return
        
        self.plotter.show()
    
    def close(self):
        """Cierra el plotter."""
        if self.plotter is not None:
            self.plotter.close()


def generate_3d_brain_video(
    subject_id="S01",
    condition="DMT",
    band_name="Alpha",
    output_file="brain_3d_animation.mp4",
    start_epoch=0,
    end_epoch=10,
    fps=30,
    rotation_speed=2.0,  # grados por frame
    show_brain_surface=True,
    show_edges=True,
    edge_threshold=85,
    parcel_base_size=0.003,
    parcel_scale_factor=3.0,
    use_fsaverage=True
):
    """
    Genera video 3D del cerebro con sincronización dinámica.
    
    Parameters
    ----------
    subject_id : str
        ID del sujeto
    condition : str
        Condición experimental
    band_name : str
        Banda de frecuencia
    output_file : str
        Nombre del archivo de salida
    start_epoch : int
        Época inicial
    end_epoch : int
        Época final
    fps : int
        Frames por segundo
    rotation_speed : float
        Velocidad de rotación en grados/frame
    show_brain_surface : bool
        Mostrar superficie cortical
    show_edges : bool
        Mostrar conexiones entre parcelas
    edge_threshold : float
        Percentil para filtrar conexiones
    parcel_base_size : float
        Tamaño base de las parcelas
    parcel_scale_factor : float
        Factor de escala según sincronización
    use_fsaverage : bool
        Usar superficie fsaverage
    """
    print(f"\n{'='*70}")
    print(f"GENERADOR DE VIDEO 3D - CEREBRO")
    print(f"{'='*70}\n")
    print(f"[3D] Sujeto: {subject_id}, Condición: {condition}, Banda: {band_name}")
    print(f"[3D] Épocas: {start_epoch} a {end_epoch-1}")
    print(f"[3D] FPS: {fps}, Rotación: {rotation_speed}°/frame")
    
    # Crear visualizador
    viz = Brain3DVisualizer(use_fsaverage=use_fsaverage)
    plotter = viz.create_plotter(window_size=(1920, 1080), off_screen=True)
    
    # Agregar superficie cerebral
    if show_brain_surface and viz.brain_mesh is not None:
        viz.add_brain_surface(opacity=0.1, color='lightgray')
    
    # Configurar cámara inicial
    viz.set_camera_position('isometric', zoom=1.5)
    
    # Fixed: Usar método alternativo para guardar frames (más robusto)
    # En lugar de open_movie (que tiene problemas de compatibilidad con imageio),
    # guardamos frames individuales y luego los combinamos con opencv
    output_path = FRAMES_DIR / output_file
    temp_frames_dir = FRAMES_DIR / "temp_3d_frames"
    temp_frames_dir.mkdir(exist_ok=True)
    
    print(f"[3D] Generando video...")
    
    # Filtrar épocas rechazadas
    valid_epochs = [e for e in range(start_epoch, end_epoch) if e not in rej]
    
    frame_count = 0
    frame_files = []
    
    for epoch_idx, epoch in enumerate(valid_epochs):
        print(f"[3D] Procesando época {epoch}/{end_epoch-1} ({epoch_idx+1}/{len(valid_epochs)})")
        
        # Obtener datos de sincronización
        stc_syncro_mat = subject_syncro["syncros_stc"][band_name][0][epoch]
        stc_kuramoto_epoch = subject_syncro["kuramoto_stc"][band_name][0][epoch]
        
        # Fixed: Asegurar que matriz y coordenadas tengan el mismo tamaño
        n_coords = len(stc_coords_3d)
        n_matrix = stc_syncro_mat.shape[0]
        n_parcels = min(n_coords, n_matrix)
        
        # Limitar a las parcelas disponibles
        stc_syncro_mat_limited = stc_syncro_mat[:n_parcels, :n_parcels]
        stc_coords_limited = stc_coords_3d[:n_parcels]
        
        # Calcular sincronización por parcela (promedio de conexiones)
        parcel_sync = stc_syncro_mat_limited.mean(axis=1)
        parcel_sync_norm = (parcel_sync - parcel_sync.min()) / \
                          (parcel_sync.max() - parcel_sync.min() + 1e-8)
        
        # Tamaños de parcelas según sincronización
        sizes = parcel_base_size + parcel_scale_factor * parcel_base_size * parcel_sync_norm
        
        # Colores según sincronización (usar colormap plasma)
        # Fixed: usar matplotlib.colormaps en lugar de cm.get_cmap (deprecado)
        cmap = plt.colormaps['plasma']
        colors_rgba = cmap(parcel_sync_norm)
        
        # Calcular order parameter promedio de la época (una vez por época)
        r_mean = np.mean(stc_kuramoto_epoch)
        
        # Generar múltiples frames por época con rotación
        frames_per_epoch = int(fps * 2)  # 2 segundos por época
        
        print(f"[3D]   Generando {frames_per_epoch} frames para época {epoch}...")
        
        for frame_in_epoch in range(frames_per_epoch):
            if frame_in_epoch % 10 == 0:
                print(f"[3D]     Frame {frame_in_epoch}/{frames_per_epoch} de época {epoch}...", end='\r')
            
            # Limpiar y redibujar (usar coordenadas y matriz limitadas)
            viz.add_parcels(stc_coords_limited, colors_rgba, sizes, opacity=0.9)
            
            # Agregar edges si está habilitado
            if show_edges:
                viz.add_edges(
                    stc_coords_limited,
                    stc_syncro_mat_limited,
                    threshold_percentile=edge_threshold,
                    color='steelblue',
                    opacity=0.2,
                    line_width=1
                )
            
            # Rotar cámara
            angle = frame_count * rotation_speed
            plotter.camera.azimuth = angle
            plotter.camera.elevation = 10 + 5 * np.sin(np.radians(angle * 2))
            
            # Agregar texto informativo
            text = (f"Subject: {subject_id} | Condition: {condition} | Band: {band_name}\n"
                   f"Epoch: {epoch+1}/{num_epochs} | Kuramoto r: {r_mean:.3f}")
            viz.add_text(text, position='upper_left', font_size=16)
            
            # Fixed: Guardar frame como imagen individual
            frame_file = temp_frames_dir / f"frame_{frame_count:06d}.png"
            plotter.screenshot(str(frame_file))
            frame_files.append(str(frame_file))
            frame_count += 1
        
        print(f"\n[3D]   ✓ Época {epoch} completada ({frames_per_epoch} frames)")
    
    # Cerrar plotter
    plotter.close()
    
    # Combinar frames en video usando opencv
    print(f"[3D] Combinando {frame_count} frames en video...")
    import cv2
    
    # Leer primer frame para obtener dimensiones
    first_frame = cv2.imread(frame_files[0])
    height, width, _ = first_frame.shape
    
    # Crear video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Escribir todos los frames
    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(frame_file)
        video_writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"[3D]   Escribiendo frames al video: {i+1}/{len(frame_files)}")
    
    video_writer.release()
    
    # Limpiar frames temporales
    print(f"[3D] Limpiando frames temporales...")
    import shutil
    shutil.rmtree(temp_frames_dir)
    
    print(f"\n[3D] ✓ Video generado exitosamente!")
    print(f"[3D] Ubicación: {output_path}")
    print(f"[3D] Frames totales: {frame_count}")
    print(f"[3D] Duración: {frame_count/fps:.1f} segundos")
    print(f"\n{'='*70}\n")
    
    return output_path


def generate_3d_snapshot(epoch=0, views=['iso', 'front', 'side', 'top']):
    """
    Genera capturas estáticas desde múltiples ángulos.
    
    Parameters
    ----------
    epoch : int
        Época a visualizar
    views : list
        Lista de vistas a generar
    """
    print(f"[3D] Generando snapshots para época {epoch}...")
    
    viz = Brain3DVisualizer(use_fsaverage=True)
    
    # Obtener datos
    stc_syncro_mat = subject_syncro["syncros_stc"][band][0][epoch]
    
    # Fixed: Asegurar que matriz y coordenadas tengan el mismo tamaño
    n_coords = len(stc_coords_3d)
    n_matrix = stc_syncro_mat.shape[0]
    n_parcels = min(n_coords, n_matrix)
    
    stc_syncro_mat_limited = stc_syncro_mat[:n_parcels, :n_parcels]
    stc_coords_limited = stc_coords_3d[:n_parcels]
    
    parcel_sync = stc_syncro_mat_limited.mean(axis=1)
    parcel_sync_norm = (parcel_sync - parcel_sync.min()) / \
                      (parcel_sync.max() - parcel_sync.min() + 1e-8)
    
    sizes = 0.003 + 3 * 0.003 * parcel_sync_norm
    cmap = plt.colormaps['plasma']
    colors = cmap(parcel_sync_norm)
    
    for view in views:
        plotter = viz.create_plotter(window_size=(1920, 1080), off_screen=True)
        
        if viz.brain_mesh is not None:
            viz.add_brain_surface(opacity=0.1)
        
        viz.add_parcels(stc_coords_limited, colors, sizes)
        viz.add_edges(stc_coords_limited, stc_syncro_mat_limited, threshold_percentile=85)
        
        viz.set_camera_position(view, zoom=1.5)
        viz.add_text(f"{subject} - {cond} - {band} - Epoch {epoch} - View: {view}")
        
        filename = FRAMES_DIR / f"3d_snapshot_{subject}_{cond}_{band}_epoch{epoch}_{view}.png"
        viz.screenshot(filename)
        
        viz.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualización 3D avanzada del cerebro")
    parser.add_argument("--mode", choices=["video", "snapshot"], default="snapshot",
                       help="Modo: 'video' (animación) o 'snapshot' (capturas estáticas)")
    parser.add_argument("--start", type=int, default=0, help="Época inicial")
    parser.add_argument("--end", type=int, default=10, help="Época final")
    parser.add_argument("--epoch", type=int, default=0, help="Época para snapshot")
    parser.add_argument("--fps", type=int, default=30, help="FPS del video")
    parser.add_argument("--output", type=str, default="brain_3d.mp4", help="Archivo de salida")
    parser.add_argument("--no-surface", action="store_true", help="No mostrar superficie cortical")
    parser.add_argument("--no-edges", action="store_true", help="No mostrar conexiones")
    
    args = parser.parse_args()
    
    if args.mode == "video":
        generate_3d_brain_video(
            subject_id=subject,
            condition=cond,
            band_name=band,
            output_file=args.output,
            start_epoch=args.start,
            end_epoch=args.end,
            fps=args.fps,
            show_brain_surface=not args.no_surface,
            show_edges=not args.no_edges
        )
    else:
        generate_3d_snapshot(epoch=args.epoch)

