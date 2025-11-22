#!/usr/bin/env python3
"""
Crea videos a partir de frames pre-generados usando ffmpeg.
MUCHO más rápido que generar frames en tiempo real durante la animación.

Workflow óptimo:
1. python generate_frames.py --end 200 --workers 20  (rápido con paralelización)
2. python create_video_from_frames.py --fps 30       (súper rápido con ffmpeg)
"""

import sys
from pathlib import Path
import subprocess
import argparse

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import VISUALIZATIONS_DIR

# Import config from plot.py
from plot import subject, cond, band


def get_sorted_frames(frames_dir, pattern="*.png"):
    """Get all frames sorted by filename (assuming numeric naming)."""
    frames = list(frames_dir.glob(pattern))
    # Sort by filename (numeric)
    frames.sort(key=lambda x: int(x.stem))
    return frames


def create_video_ffmpeg(frames_dir, output_file, fps=30, preset="medium", crf=18, gpu=False):
    """
    Crea video usando ffmpeg (muy rápido).
    
    Parámetros:
    -----------
    frames_dir : Path
        Directorio con los frames (*.png)
    output_file : Path
        Archivo de salida (.mp4)
    fps : int
        Frames por segundo (default: 30)
    preset : str
        Preset de codificación: ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
        - ultrafast: más rápido, más grande
        - medium: balance (default)
        - slow: más lento, más pequeño
    crf : int
        Calidad: 0-51 (default: 18)
        - 0: sin pérdida (enorme)
        - 18: visualmente sin pérdida (recomendado)
        - 23: calidad por defecto
        - 28+: calidad baja
    gpu : bool
        Usar aceleración GPU con NVENC (requiere Nvidia GPU)
    """
    frames = get_sorted_frames(frames_dir)
    
    if len(frames) == 0:
        print(f"[ERROR] No se encontraron frames en {frames_dir}")
        return None
    
    print(f"[VIDEO] Encontrados {len(frames)} frames")
    print(f"[VIDEO] Duración estimada: {len(frames)/fps:.1f}s ({len(frames)/fps/60:.1f} min)")
    print()
    
    # Create a temporary file list for ffmpeg
    list_file = frames_dir / "frames_list.txt"
    with open(list_file, 'w') as f:
        for frame in frames:
            # ffmpeg concat demuxer format
            f.write(f"file '{frame.absolute()}'\n")
            f.write(f"duration {1/fps}\n")
    
    # Build ffmpeg command
    if gpu:
        # GPU encoding with NVENC (Nvidia)
        cmd = [
            'ffmpeg',
            '-y',  # overwrite output
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c:v', 'h264_nvenc',  # GPU encoder
            '-preset', 'fast',  # GPU preset
            '-rc', 'vbr',
            '-cq', str(crf),
            '-b:v', '0',
            '-pix_fmt', 'yuv420p',
            str(output_file)
        ]
    else:
        # CPU encoding with libx264 (universal)
        cmd = [
            'ffmpeg',
            '-y',  # overwrite output
            '-f', 'concat',
            '-safe', '0',
            '-i', str(list_file),
            '-c:v', 'libx264',
            '-preset', preset,
            '-crf', str(crf),
            '-pix_fmt', 'yuv420p',
            str(output_file)
        ]
    
    print(f"[VIDEO] Ejecutando ffmpeg...")
    print(f"[VIDEO] Encoder: {'GPU (NVENC)' if gpu else 'CPU (libx264)'}")
    print(f"[VIDEO] Preset: {preset if not gpu else 'fast (GPU)'}")
    print(f"[VIDEO] Calidad (CRF): {crf}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"[VIDEO] ✓ Video creado exitosamente!")
        
        # Clean up
        list_file.unlink()
        
        # Get file size
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"[VIDEO] Tamaño: {size_mb:.1f} MB")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] ffmpeg falló:")
        print(e.stderr)
        return None
    except FileNotFoundError:
        print("[ERROR] ffmpeg no está instalado!")
        print("Instalar con: sudo apt install ffmpeg")
        return None


def create_video_simple(frames_dir, output_file, fps=30, pattern="*.png"):
    """Crea video usando patrón simple de ffmpeg (alternativa más simple)."""
    
    frames = get_sorted_frames(frames_dir, pattern)
    
    if len(frames) == 0:
        print(f"[ERROR] No se encontraron frames en {frames_dir}")
        return None
    
    print(f"[VIDEO] Encontrados {len(frames)} frames")
    print(f"[VIDEO] Duración estimada: {len(frames)/fps:.1f}s")
    print()
    
    # Usar input pattern de ffmpeg
    # Asume que los frames están nombrados con números consecutivos
    first_frame = frames[0]
    
    # Detectar patrón de nombres (ej: 1001000.png → %07d.png)
    stem_len = len(first_frame.stem)
    input_pattern = frames_dir / f"%0{stem_len}d.png"
    
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', str(input_pattern),
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        str(output_file)
    ]
    
    print(f"[VIDEO] Ejecutando ffmpeg...")
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[VIDEO] ✓ Video creado exitosamente!")
        
        size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"[VIDEO] Tamaño: {size_mb:.1f} MB")
        
        return output_file
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Método simple falló, intentando con lista de archivos...")
        return create_video_ffmpeg(frames_dir, output_file, fps)
    except FileNotFoundError:
        print("[ERROR] ffmpeg no está instalado!")
        print("Instalar con: sudo apt install ffmpeg")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Crear videos a partir de frames pre-generados",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:

  # Workflow completo (más rápido):
  python generate_frames.py --end 200 --workers 20  # 2-3 min paralelo
  python create_video_from_frames.py --fps 30       # 10-20 seg con ffmpeg
  
  # Con GPU (si tienes Nvidia):
  python create_video_from_frames.py --fps 30 --gpu
  
  # Diferentes calidades:
  python create_video_from_frames.py --preset ultrafast --crf 23  # Rápido, baja calidad
  python create_video_from_frames.py --preset slow --crf 18       # Lento, alta calidad
  python create_video_from_frames.py --preset medium --crf 18     # Balance (default)

Configuración actual (desde plot.py):
  Sujeto:    {subject}
  Condición: {cond}
  Banda:     {band}
        """.format(subject=subject, cond=cond, band=band)
    )
    
    parser.add_argument("--frames-dir", type=str, default=None,
                       help="Directorio con frames (default: visualizations/plot)")
    parser.add_argument("--output", type=str, default=None,
                       help="Archivo de salida (default: auto-generado)")
    parser.add_argument("--fps", type=int, default=30,
                       help="Frames por segundo (default: 30)")
    parser.add_argument("--preset", type=str, default="medium",
                       choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                       help="Preset de codificación (default: medium)")
    parser.add_argument("--crf", type=int, default=18,
                       help="Calidad: 0-51, menor=mejor (default: 18)")
    parser.add_argument("--gpu", action="store_true",
                       help="Usar aceleración GPU con NVENC (requiere Nvidia)")
    
    args = parser.parse_args()
    
    # Setup directories
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
    else:
        frames_dir = VISUALIZATIONS_DIR / "plot"
    
    if not frames_dir.exists():
        print(f"[ERROR] Directorio no existe: {frames_dir}")
        print("[INFO] Primero genera frames con:")
        print("       python generate_frames.py --end 200 --workers 20")
        return
    
    # Setup output file
    if args.output:
        output_file = VISUALIZATIONS_DIR / "plot" / args.output
    else:
        output_file = VISUALIZATIONS_DIR / "plot" / f"{subject}_{cond}_{band}_video.mp4"
    
    print()
    print("="*70)
    print("  CREADOR DE VIDEOS DESDE FRAMES")
    print("="*70)
    print(f"  📁 Frames:     {frames_dir}")
    print(f"  💾 Output:     {output_file}")
    print()
    print(f"  📊 Config:")
    print(f"      Sujeto:    {subject}")
    print(f"      Condición: {cond}")
    print(f"      Banda:     {band}")
    print()
    print(f"  🎬 Video:")
    print(f"      FPS:       {args.fps}")
    print(f"      Preset:    {args.preset}")
    print(f"      CRF:       {args.crf} (calidad)")
    print(f"      GPU:       {'✓ NVENC' if args.gpu else '✗ CPU'}")
    print("="*70)
    print()
    
    # Create video
    import time
    start = time.time()
    
    result = create_video_ffmpeg(
        frames_dir=frames_dir,
        output_file=output_file,
        fps=args.fps,
        preset=args.preset,
        crf=args.crf,
        gpu=args.gpu
    )
    
    elapsed = time.time() - start
    
    if result:
        print()
        print("="*70)
        print("  ✓ COMPLETADO")
        print("="*70)
        print(f"  Tiempo:  {elapsed:.1f}s")
        print(f"  Output:  {result}")
        print("="*70)
        print()
        print("💡 TIPS:")
        print("   • Para más velocidad: --preset ultrafast")
        print("   • Para mejor calidad: --preset slow --crf 15")
        print("   • Con Nvidia GPU:     --gpu (mucho más rápido)")
        print()
    else:
        print()
        print("[ERROR] No se pudo crear el video")


if __name__ == "__main__":
    main()

