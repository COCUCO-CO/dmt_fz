#!/usr/bin/env python3
"""
Genera VIDEO de la evolución temporal de la transformada de Hilbert en 3D.
Muestra cómo cambia la trayectoria en el espacio complejo época a época.

Interpretación científica:
- Cambios en amplitud → activación/arousal
- Cambios en fase → coherencia/sincronización
- Trayectoria caótica → irregular, fuera de banda
- Trayectoria suave → oscilación estable en banda

Aplicaciones:
- Ver efectos temporales de DMT (onset, peak, offset)
- Detectar transiciones de estado cerebral
- Identificar patrones recurrentes
- Comparar DMT vs EC/EO dinámicamente
"""

import pickle
from pathlib import Path
import sys
import subprocess
import argparse
from tqdm import tqdm

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import RESULTS_DIR, VISUALIZATIONS_DIR, ensure_dir


def load_file(file):
    """Load pickle file."""
    with open(file, "rb") as handle:
        return pickle.load(handle)


def get_epoch_data(subject_data, band, epoch_idx, roi_idx, trim=0.15):
    """
    Extrae datos de Hilbert para una época específica.
    
    Parameters:
    -----------
    subject_data : dict
        Datos del sujeto con phases_stc y amplitudes_stc
    band : str
        Banda de frecuencia
    epoch_idx : int
        Índice de época
    roi_idx : int
        Índice de ROI
    trim : float
        Fracción a recortar en cada extremo (evitar edge artifacts)
    
    Returns:
    --------
    time, real, imag, amplitude, phase : arrays
        Datos procesados de la época
    """
    phases = subject_data["phases_stc"][band][epoch_idx][roi_idx, :]
    amplitudes = subject_data["amplitudes_stc"][band][epoch_idx][roi_idx, :]
    
    fs = 500.0
    time = np.arange(len(phases)) / fs
    
    # Trim edges to avoid filter artifacts
    start = int(len(time) * trim)
    end = int(len(time) * (1 - trim))
    
    time_seg = time[start:end]
    phase_seg = phases[start:end]
    amp_seg = amplitudes[start:end]
    
    # Analytical signal
    analytic_seg = amp_seg * np.exp(1j * phase_seg)
    real_seg = analytic_seg.real
    imag_seg = analytic_seg.imag
    
    return time_seg, real_seg, imag_seg, amp_seg, phase_seg


def create_hilbert_frame(time, real, imag, amplitude, epoch_num, total_epochs,
                         roi_name, subject, condition, band, 
                         show_projections=True, show_markers=True):
    """
    Crea un frame de Plotly con la trayectoria de Hilbert.
    
    Parameters:
    -----------
    time, real, imag, amplitude : arrays
        Datos de la señal analítica
    epoch_num : int
        Número de época (para el título)
    total_epochs : int
        Total de épocas (para el título)
    roi_name : str
        Nombre de la ROI
    subject, condition, band : str
        Metadatos
    show_projections : bool
        Mostrar proyecciones ortogonales
    show_markers : bool
        Mostrar marcadores inicio/fin
    
    Returns:
    --------
    fig : plotly Figure
        Figura lista para renderizar
    """
    # Trayectoria principal (coloreada por tiempo)
    trajectory = go.Scatter3d(
        x=time,
        y=real,
        z=imag,
        mode="lines",
        line=dict(
            color=time,
            colorscale="Viridis",
            width=5,
            showscale=False  # No mostrar colorbar (ocupa espacio)
        ),
        name="Trayectoria",
        hovertemplate="t=%{x:.3f}s<br>Real=%{y:.4f}<br>Imag=%{z:.4f}<extra></extra>"
    )
    
    data = [trajectory]
    
    # Proyecciones (opcionales)
    if show_projections:
        # Piso (tiempo-real)
        z_floor = np.full_like(time, imag.min() - 0.02)
        proj_time_real = go.Scatter3d(
            x=time, y=real, z=z_floor,
            mode="lines",
            line=dict(color="rgba(31, 119, 180, 0.5)", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        )
        
        # Pared lateral (tiempo-imag)
        y_wall = np.full_like(time, real.min() - 0.02)
        proj_time_imag = go.Scatter3d(
            x=time, y=y_wall, z=imag,
            mode="lines",
            line=dict(color="rgba(214, 39, 40, 0.5)", width=2, dash="dot"),
            showlegend=False,
            hoverinfo="skip"
        )
        
        # Pared trasera (plano complejo)
        x_wall = np.full_like(real, time.max() + 0.02)
        proj_real_imag = go.Scatter3d(
            x=x_wall, y=real, z=imag,
            mode="lines",
            line=dict(color="rgba(148, 103, 189, 0.5)", width=2, dash="longdash"),
            showlegend=False,
            hoverinfo="skip"
        )
        
        data.extend([proj_time_real, proj_time_imag, proj_real_imag])
    
    # Marcadores inicio/fin (opcionales)
    if show_markers:
        markers = go.Scatter3d(
            x=[time[0], time[-1]],
            y=[real[0], real[-1]],
            z=[imag[0], imag[-1]],
            mode="markers",
            marker=dict(
                size=8,
                color=["#2ca02c", "#ff7f0e"],
                symbol=["circle", "square"],
                line=dict(width=2, color="black")
            ),
            name="Inicio/Final",
            showlegend=False,
            hoverinfo="skip"
        )
        data.append(markers)
    
    # Crear figura
    fig = go.Figure(data=data)
    
    # Layout
    fig.update_layout(
        width=1920,  # Full HD
        height=1080,
        template="plotly_white",
        scene=dict(
            xaxis=dict(
                title=dict(text="Tiempo (s)", font=dict(size=20)),
                tickfont=dict(size=14),
                backgroundcolor="rgba(250,250,250,0.8)",
                gridcolor="rgba(0,0,0,0.1)"
            ),
            yaxis=dict(
                title=dict(text="Re(z) = A·cos(φ)", font=dict(size=20)),
                tickfont=dict(size=14),
                backgroundcolor="rgba(250,250,250,0.8)",
                gridcolor="rgba(0,0,0,0.1)"
            ),
            zaxis=dict(
                title=dict(text="Im(z) = A·sin(φ)", font=dict(size=20)),
                tickfont=dict(size=14),
                backgroundcolor="rgba(250,250,250,0.8)",
                gridcolor="rgba(0,0,0,0.1)"
            ),
            camera=dict(eye=dict(x=1.6, y=1.4, z=0.9)),
            aspectmode="manual",
            aspectratio=dict(x=1.5, y=1, z=1)
        ),
        margin=dict(l=0, r=0, b=0, t=120),
        title=dict(
            text=(
                f"<b>Transformada de Hilbert - Evolución Temporal</b>"
                f"<br><span style='font-size:18px;'>"
                f"Sujeto: {subject} | Condición: {condition} | Banda: {band} | "
                f"Época {epoch_num}/{total_epochs} | ROI: {roi_name}"
                f"</span>"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=26)
        ),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    
    return fig


def generate_frames(subject, condition, band, roi_idx, roi_name,
                    start_epoch=0, end_epoch=50, output_dir=None,
                    show_projections=True, show_markers=True):
    """
    Genera frames PNG para un rango de épocas.
    
    Parameters:
    -----------
    subject : str
        ID del sujeto (ej: "S01-DMT")
    condition : str
        Condición (DMT, EC, EO)
    band : str
        Banda de frecuencia
    roi_idx : int
        Índice de ROI
    roi_name : str
        Nombre de ROI
    start_epoch, end_epoch : int
        Rango de épocas
    output_dir : Path
        Directorio de salida
    show_projections : bool
        Mostrar proyecciones ortogonales
    show_markers : bool
        Mostrar marcadores inicio/fin
    
    Returns:
    --------
    frames_dir : Path
        Directorio con los frames generados
    """
    # Load data
    print(f"\n[1/3] Cargando datos...")
    path = RESULTS_DIR
    subject_file = path / condition / f"phases-{subject}.pkl"
    subject_data = load_file(subject_file)
    print(f"      ✓ {subject_file.name}")
    
    # Crear directorio de frames
    if output_dir is None:
        output_dir = ensure_dir(VISUALIZATIONS_DIR / "visualize_hilbert_video")
    frames_dir = ensure_dir(output_dir / "temp_frames")
    
    # Info
    epochs = subject_data["phases_stc"][band]
    total_epochs = len(epochs)
    end_epoch = min(end_epoch, total_epochs)
    num_frames = end_epoch - start_epoch
    
    print(f"\n[2/3] Generando {num_frames} frames...")
    print(f"      Épocas: {start_epoch} → {end_epoch-1}")
    print(f"      ROI: {roi_name} (idx={roi_idx})")
    print(f"      Output: {frames_dir}")
    
    # Generate frames con progress bar
    for epoch_idx in tqdm(range(start_epoch, end_epoch), desc="      Frames"):
        # Get epoch data
        time, real, imag, amp, phase = get_epoch_data(
            subject_data, band, epoch_idx, roi_idx
        )
        
        # Create figure
        fig = create_hilbert_frame(
            time, real, imag, amp,
            epoch_num=epoch_idx + 1,
            total_epochs=total_epochs,
            roi_name=roi_name,
            subject=subject,
            condition=condition,
            band=band,
            show_projections=show_projections,
            show_markers=show_markers
        )
        
        # Save frame
        frame_file = frames_dir / f"hilbert_{epoch_idx:04d}.png"
        pio.write_image(fig, str(frame_file), width=1920, height=1080, scale=1)
    
    print(f"      ✓ {num_frames} frames generados")
    
    return frames_dir


def create_video_ffmpeg(frames_dir, output_file, fps=2, crf=18, gpu=False):
    """
    Une frames en video MP4 usando ffmpeg.
    
    Parameters:
    -----------
    frames_dir : Path
        Directorio con frames (hilbert_0000.png, hilbert_0001.png, ...)
    output_file : Path
        Archivo de salida (.mp4)
    fps : int
        Frames por segundo (default: 2, permite apreciar cada época)
    crf : int
        Calidad (18 = visualmente perfecto)
    gpu : bool
        Usar encoding GPU (Nvidia NVENC)
    
    Returns:
    --------
    output_file : Path
        Ruta al video generado
    """
    print(f"\n[3/3] Creando video con ffmpeg...")
    print(f"      FPS: {fps}")
    print(f"      Encoder: {'GPU (NVENC)' if gpu else 'CPU (libx264)'}")
    
    # Pattern de input
    input_pattern = str(frames_dir / "hilbert_%04d.png")
    
    # Build command
    if gpu:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'h264_nvenc',
            '-preset', 'fast',
            '-rc', 'vbr',
            '-cq', str(crf),
            '-b:v', '0',
            '-pix_fmt', 'yuv420p',
            str(output_file)
        ]
    else:
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-i', input_pattern,
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            str(output_file)
        ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"      ✓ Video creado: {output_file.name} ({size_mb:.1f} MB)")
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"      ✗ Error en ffmpeg:")
        print(e.stderr)
        return None
    except FileNotFoundError:
        print(f"      ✗ ffmpeg no instalado (sudo apt install ffmpeg)")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Genera video de evolución de transformada de Hilbert en 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Video corto de prueba (10 épocas, 2 FPS)
  python visualize_hilbert_video.py --start 0 --end 10 --fps 2
  
  # Video completo (todas las épocas disponibles)
  python visualize_hilbert_video.py --end 200 --fps 3
  
  # Con GPU para encoding más rápido
  python visualize_hilbert_video.py --end 100 --fps 5 --gpu
  
  # ROI específica (ver nombres en visualize_hilbert_improved.py)
  python visualize_hilbert_video.py --roi 10 --end 50
  
  # Sin proyecciones (más limpio, más rápido)
  python visualize_hilbert_video.py --end 50 --no-projections

Interpretación científica:
  - Amplitud variable → cambios en arousal/activación
  - Fase estable → oscilación coherente en banda
  - Trayectoria caótica → actividad irregular
  - Transiciones bruscas → cambios de estado cerebral
  
Velocidades recomendadas:
  - 1-2 FPS: apreciar cada época individualmente
  - 3-5 FPS: ver evolución temporal fluida
  - 10+ FPS: overview rápido de todo el experimento
        """
    )
    
    parser.add_argument("--subject", type=str, default="S01-DMT",
                       help="ID del sujeto (default: S01-DMT)")
    parser.add_argument("--condition", type=str, default="DMT",
                       choices=["DMT", "EC", "EO"],
                       help="Condición experimental (default: DMT)")
    parser.add_argument("--band", type=str, default="Alpha",
                       choices=["Delta", "Theta", "Alpha", "Beta", "Gamma"],
                       help="Banda de frecuencia (default: Alpha)")
    parser.add_argument("--roi", type=int, default=None,
                       help="Índice de ROI (default: auto-select Default/Control network)")
    parser.add_argument("--start", type=int, default=0,
                       help="Época inicial (default: 0)")
    parser.add_argument("--end", type=int, default=50,
                       help="Época final (default: 50)")
    parser.add_argument("--fps", type=int, default=2,
                       help="Frames por segundo (default: 2)")
    parser.add_argument("--crf", type=int, default=18,
                       help="Calidad del video, menor=mejor (default: 18)")
    parser.add_argument("--gpu", action="store_true",
                       help="Usar encoding GPU (Nvidia NVENC)")
    parser.add_argument("--no-projections", action="store_true",
                       help="No mostrar proyecciones ortogonales (más rápido)")
    parser.add_argument("--no-markers", action="store_true",
                       help="No mostrar marcadores inicio/fin")
    parser.add_argument("--keep-frames", action="store_true",
                       help="No eliminar frames temporales después")
    
    args = parser.parse_args()
    
    # Load data to get ROI info
    print("=" * 80)
    print("  GENERADOR DE VIDEO - HILBERT 3D TEMPORAL")
    print("=" * 80)
    
    path = RESULTS_DIR
    [node_colors, label_names, label_names_short,
     stc_coords_3d, ch_names, mapping, eeg_coords_2d] = load_file(path / "extra.pkl")
    
    # Auto-select ROI if not specified
    if args.roi is None:
        roi_idx = 0
        for idx, name in enumerate(label_names[:120]):
            if "Default" in name or "Cont" in name:
                roi_idx = idx
                break
    else:
        roi_idx = args.roi
    
    roi_name = label_names[roi_idx]
    
    print(f"\n📊 Configuración:")
    print(f"   Sujeto:    {args.subject}")
    print(f"   Condición: {args.condition}")
    print(f"   Banda:     {args.band}")
    print(f"   ROI:       {roi_name} (idx={roi_idx})")
    print(f"   Épocas:    {args.start} → {args.end-1} ({args.end - args.start} frames)")
    print(f"   FPS:       {args.fps}")
    print(f"   GPU:       {'✓ NVENC' if args.gpu else '✗ CPU'}")
    
    # Estimate video duration
    duration = (args.end - args.start) / args.fps
    print(f"\n⏱️  Duración estimada del video: {duration:.1f}s ({duration/60:.1f} min)")
    
    # Output
    output_dir = ensure_dir(VISUALIZATIONS_DIR / "visualize_hilbert_video")
    output_file = output_dir / f"hilbert_{args.subject}_{args.band}_{args.condition}_epochs{args.start}-{args.end}.mp4"
    
    print(f"\n💾 Output: {output_file}")
    print("=" * 80)
    
    # Generate frames
    import time
    start_time = time.time()
    
    frames_dir = generate_frames(
        subject=args.subject,
        condition=args.condition,
        band=args.band,
        roi_idx=roi_idx,
        roi_name=roi_name,
        start_epoch=args.start,
        end_epoch=args.end,
        output_dir=output_dir,
        show_projections=not args.no_projections,
        show_markers=not args.no_markers
    )
    
    # Create video
    video_path = create_video_ffmpeg(
        frames_dir=frames_dir,
        output_file=output_file,
        fps=args.fps,
        crf=args.crf,
        gpu=args.gpu
    )
    
    # Cleanup
    if video_path and not args.keep_frames:
        import shutil
        print(f"\n🗑️  Limpiando frames temporales...")
        shutil.rmtree(frames_dir)
        print(f"      ✓ {frames_dir} eliminado")
    
    elapsed = time.time() - start_time
    
    # Summary
    print("\n" + "=" * 80)
    print("  ✓✓✓ PROCESO COMPLETADO ✓✓✓")
    print("=" * 80)
    print(f"\n⏱️  Tiempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"💾 Video: {output_file}")
    print(f"📏 Tamaño: {output_file.stat().st_size / (1024*1024):.1f} MB")
    print(f"🎬 Duración: {duration:.1f}s @ {args.fps} FPS")
    print("\n" + "=" * 80)
    print("\n💡 Tips:")
    print("   • Reproduce con: vlc", str(output_file))
    print("   • Para más épocas: --end 200")
    print("   • Para video más fluido: --fps 5")
    print("   • Con GPU (más rápido): --gpu")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

