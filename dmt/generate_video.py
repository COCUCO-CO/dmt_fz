#!/usr/bin/env python3
"""
Script para generar videos de sincronización de redes cerebrales.
Usa las funciones modernas de plot.py con interpolación suave.
"""

import argparse
from plot import generate_simple_network_video, generate_smooth_video, generate_3d_brain_video

def main():
    parser = argparse.ArgumentParser(
        description="Generar videos de sincronización de redes cerebrales"
    )
    parser.add_argument(
        "--subject",
        type=str,
        default="S01",
        help="ID del sujeto (ej: S01, S02, etc.)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="DMT",
        choices=["DMT", "EC", "EO"],
        help="Condición experimental"
    )
    parser.add_argument(
        "--band",
        type=str,
        default="Alpha",
        choices=["Delta", "Theta", "Alpha", "Beta", "Gamma"],
        help="Banda de frecuencia"
    )
    parser.add_argument(
        "--view",
        type=str,
        default="graph",
        choices=["graph", "matrix", "oscillators", "all", "full", "3d"],
        help="Vista: graph (grafo), matrix (matriz), oscillators (osciladores), all (3 vistas), full (completo EEG+STC), 3d (cerebro 3D con sincronización)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Nombre del archivo de salida (default: auto-generado)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames por segundo (default: 30)"
    )
    parser.add_argument(
        "--interpolation",
        type=int,
        default=10,
        help="Factor de interpolación para suavizado (default: 10)"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Resolución (default: 120, usar 150-200 para mayor calidad)"
    )
    parser.add_argument(
        "--epoch-duration",
        type=float,
        default=2.0,
        help="Duración en segundos para mostrar cada época (default: 2.0)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80,
        help="Percentil threshold para visualización de conectividad (default: 80)"
    )
    parser.add_argument(
        "--rotate",
        action="store_true",
        help="Rotar cámara en video 3D (solo para --view 3d)"
    )
    parser.add_argument(
        "--no-edges",
        action="store_true",
        help="No mostrar conexiones en video 3D (solo para --view 3d)"
    )
    parser.add_argument(
        "--edge-threshold",
        type=float,
        default=90,
        help="Percentil threshold para mostrar conexiones en 3D (default: 90, solo para --view 3d)"
    )
    
    args = parser.parse_args()
    
    # Generate output filename if not provided
    if args.output is None:
        args.output = f"{args.subject}_{args.condition}_{args.band}_{args.view}.mp4"
    
    print("=" * 80)
    print("GENERADOR DE VIDEOS DE SINCRONIZACIÓN DE REDES")
    print("=" * 80)
    print(f"Sujeto:          {args.subject}")
    print(f"Condición:       {args.condition}")
    print(f"Banda:           {args.band}")
    print(f"Vista:           {args.view}")
    print(f"Salida:          {args.output}")
    print(f"FPS:             {args.fps}")
    print(f"Interpolación:   {args.interpolation}x")
    print(f"DPI:             {args.dpi}")
    print(f"Época duración:  {args.epoch_duration}s")
    print(f"Threshold:       {args.threshold}")
    print("=" * 80)
    print()
    
    # Calculate estimated video duration
    # Note: S01-DMT has ~180 valid epochs, each will be epoch_duration seconds
    estimated_duration = 180 * args.epoch_duration  # Rough estimate
    print(f"[INFO] Duración estimada del video: ~{estimated_duration:.0f}s ({estimated_duration/60:.1f} min)")
    print()
    
    # Generate video
    if args.view == "full":
        print("[INFO] Generando video completo (EEG + STC)...")
        output_path = generate_smooth_video(
            subject_id=args.subject,
            condition=args.condition,
            band=args.band,
            output_file=args.output,
            fps=args.fps,
            interpolation_factor=args.interpolation,
            dpi=args.dpi,
            epoch_duration=args.epoch_duration,
            threshold_stc=args.threshold
        )
    elif args.view == "3d":
        print("[INFO] Generando video 3D del cerebro con sincronización...")
        output_path = generate_3d_brain_video(
            subject_id=args.subject,
            condition=args.condition,
            band=args.band,
            output_file=args.output,
            fps=args.fps,
            interpolation_factor=args.interpolation,
            dpi=args.dpi,
            threshold_stc=args.threshold,
            epoch_duration=args.epoch_duration,
            rotate_camera=args.rotate,
            show_edges=not args.no_edges,
            edge_threshold=args.edge_threshold
        )
    else:
        print(f"[INFO] Generando video simple (vista: {args.view})...")
        output_path = generate_simple_network_video(
            subject_id=args.subject,
            condition=args.condition,
            band=args.band,
            output_file=args.output,
            fps=args.fps,
            interpolation_factor=args.interpolation,
            view=args.view,
            dpi=args.dpi,
            threshold_stc=args.threshold,
            epoch_duration=args.epoch_duration
        )
    
    print()
    print("=" * 80)
    print(f"✓ VIDEO GENERADO EXITOSAMENTE")
    print(f"Ubicación: {output_path}")
    print("=" * 80)
    
    return output_path


if __name__ == "__main__":
    main()

