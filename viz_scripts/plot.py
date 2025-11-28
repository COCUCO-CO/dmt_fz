#!/usr/bin/env python3
"""
Main entry point for EEG/Source synchronization visualizations.

This script provides a unified interface to generate various types of
visualizations from EEG and source space synchronization data.

Usage:
    python plot.py --subject S01 --condition DMT --band Alpha --mode all
    python plot.py -s S01 -c DMT -b Alpha -m advanced --max-epochs 10
    python plot.py -s S01 -c DMT -b Alpha -m video_smooth --fps 30

Available modes:
    - all: Full view with EEG and source space (original)
    - eeg: EEG channels only
    - stc: Source space only
    - advanced: Modern visualization with dark theme and glow effects
    - video_smooth: Smooth animated video of network dynamics
    - video_simple: Focused video showing one aspect
    - video_3d: 3D brain visualization with rotation
    - video_advanced: Advanced style video
"""

import argparse
from multiprocessing import Pool

# Import utilities and data loading
from plot_utils import load_subject_data, num_epochs

# Import frame generation functions
from plot_frames import plot_all, plot_eeg_only, plot_stc_only, plot_advanced

# Import video generation functions
from plot_videos import (
    generate_smooth_video,
    generate_simple_network_video,
    generate_3d_brain_video,
    generate_advanced_video
)

# Import global state module for accessing num_epochs after loading
import plot_utils


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate EEG/Source synchronization visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all frames for S01-DMT Alpha band
  python plot.py --subject S01 --condition DMT --band Alpha --mode all
  
  # Generate ADVANCED frames with glow effects and modern viz
  python plot.py --subject S01 --condition DMT --band Alpha --mode advanced
  
  # Generate only EEG frames with 8 workers
  python plot.py --subject S02 --condition EC --band Theta --mode eeg --workers 8
  
  # Generate only first 10 epochs
  python plot.py --subject S01 --condition DMT --band Alpha --epochs 0:10
  
  # Generate smooth video
  python plot.py --subject S01 --condition DMT --band Alpha --mode video_smooth --fps 30
  
  # Generate 3D brain video with rotation
  python plot.py --subject S01 --condition DMT --band Alpha --mode video_3d --rotate
  
  # Generate simple network video (graph view)
  python plot.py --subject S01 --condition DMT --band Alpha --mode video_simple --view graph
  
  # Generate advanced video with dark theme and glow effects
  python plot.py --subject S01 --condition DMT --band Alpha --mode video_advanced --max-epochs 10
  
  # Create video from existing frames in a folder
  python plot.py --mode frames_to_video --source advanced --fps 30
  python plot.py --mode frames_to_video --source all --fps 15
        """
    )
    
    # Basic parameters
    parser.add_argument("--subject", "-s", type=str, default="S01",
                        help="Subject ID (default: S01)")
    parser.add_argument("--condition", "-c", type=str, default="DMT",
                        choices=["DMT", "EC", "EO"],
                        help="Condition (default: DMT)")
    parser.add_argument("--band", "-b", type=str, default="Alpha",
                        choices=["Delta", "Theta", "Alpha", "Beta", "Gamma"],
                        help="Frequency band (default: Alpha)")
    
    # Mode selection
    parser.add_argument("--mode", "-m", type=str, default="all",
                        choices=["all", "eeg", "stc", "advanced", "video_smooth", "video_simple", "video_3d", "video_advanced", "frames_to_video"],
                        help="Visualization mode (default: all). Use 'frames_to_video' to create video from existing frames")
    
    # Frames to video parameters
    parser.add_argument("--source", type=str, default="advanced",
                        choices=["all", "eeg", "stc", "advanced"],
                        help="Source folder for frames_to_video mode (default: advanced)")
    
    # Processing parameters
    parser.add_argument("--workers", "-w", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--epochs", "-e", type=str, default="all",
                        help="Epochs to process: 'all' or 'start:end' (default: all)")
    parser.add_argument("--max-epochs", type=int, default=None,
                        help="Maximum number of epochs to process (default: all)")
    
    # Video parameters
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second for videos (default: 30)")
    parser.add_argument("--dpi", type=int, default=120,
                        help="DPI for video output (default: 120)")
    parser.add_argument("--view", type=str, default="graph",
                        choices=["graph", "matrix", "oscillators", "all"],
                        help="View for video_simple mode (default: graph)")
    parser.add_argument("--interpolation", type=int, default=10,
                        help="Interpolation factor for smooth videos (default: 10)")
    parser.add_argument("--rotate", action="store_true",
                        help="Enable camera rotation for 3D video")
    parser.add_argument("--no-edges", action="store_true",
                        help="Disable edges in 3D video")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output filename for videos (default: auto-generated)")
    
    # Output format
    parser.add_argument("--svg", action="store_true",
                        help="Save frames as SVG instead of PNG (vector graphics)")
    parser.add_argument("--quality", "-q", type=str, default="high",
                        choices=["high", "medium", "low"],
                        help="Image quality: high (~1MB), medium (~500KB), low (~250KB)")
    
    # Timeline style
    parser.add_argument("--only-points", action="store_true",
                        help="Show only points in Kuramoto timeline (no connecting lines)")
    
    return parser.parse_args()


def create_video_from_frames(source_folder="advanced", fps=30, output_file=None, 
                              subject="S01", condition="DMT", band="Alpha"):
    """Create a video from existing PNG frames in a folder using ffmpeg."""
    import subprocess
    from pathlib import Path
    import sys
    
    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
    from paths import VISUALIZATIONS_DIR, ensure_dir
    
    # New structure: mode/subject_condition_band/
    subfolder = f"{subject}_{condition}_{band}"
    frames_dir = VISUALIZATIONS_DIR / "plot" / source_folder / subfolder
    output_dir = ensure_dir(VISUALIZATIONS_DIR / "plot" / "videos")
    
    if not frames_dir.exists():
        print(f"[ERROR] Frames folder not found: {frames_dir}")
        print(f"[ERROR] Generate frames first with: python plot.py -m {source_folder} -s {subject} -c {condition} -b {band}")
        return None
    
    def get_sort_key(path):
        """Extract numeric part from filename for sorting."""
        stem = path.stem
        # Handle prefixed names like "adv_1001000", "eeg_1001000", "stc_1001000"
        if '_' in stem:
            parts = stem.split('_')
            try:
                return int(parts[-1])
            except ValueError:
                return 0
        # Handle plain numeric names like "1001000"
        try:
            return int(stem)
        except ValueError:
            return 0
    
    png_files = sorted(frames_dir.glob("*.png"), key=get_sort_key)
    
    if not png_files:
        print(f"[ERROR] No PNG files found in {frames_dir}")
        return None
    
    print(f"[FRAMES→VIDEO] Found {len(png_files)} frames in {frames_dir}")
    print(f"[FRAMES→VIDEO] FPS: {fps}, Duration: ~{len(png_files)/fps:.1f}s")
    
    if output_file is None:
        output_file = f"{subject}_{condition}_{band}_{source_folder}_{fps}fps.mp4"
    
    output_path = output_dir / output_file
    
    # Create file list for ffmpeg
    list_file = frames_dir / "frames_list.txt"
    with open(list_file, 'w') as f:
        for png in png_files:
            f.write(f"file '{png.name}'\n")
            f.write(f"duration {1/fps}\n")
    
    cmd = [
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', str(list_file), '-vf', f'fps={fps}',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
        str(output_path)
    ]
    
    print(f"[FRAMES→VIDEO] Running ffmpeg...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    list_file.unlink()
    
    if result.returncode == 0:
        file_size = output_path.stat().st_size / 1e6
        print(f"[FRAMES→VIDEO] ✓ Video saved: {output_path} ({file_size:.1f} MB)")
        return output_path
    else:
        print(f"[ERROR] ffmpeg failed: {result.stderr}")
        return None


def main():
    """Main entry point."""
    args = parse_args()
    
    print(f"[PLOT] Starting visualization")
    print(f"[PLOT] Subject: {args.subject}, Condition: {args.condition}, Band: {args.band}")
    print(f"[PLOT] Mode: {args.mode}, Workers: {args.workers}")
    
    # Load subject data with mode-specific output folder
    load_subject_data(args.subject, args.condition, args.band, mode=args.mode)
    
    # Determine epoch range
    if args.epochs == "all":
        epoch_range = range(plot_utils.num_epochs)
    else:
        try:
            start, end = map(int, args.epochs.split(":"))
            epoch_range = range(start, min(end, plot_utils.num_epochs))
        except ValueError:
            print(f"[ERROR] Invalid epochs format: {args.epochs}. Use 'all' or 'start:end'")
            return
    
    # Apply max_epochs limit
    if args.max_epochs is not None:
        epoch_range = range(epoch_range.start, min(epoch_range.start + args.max_epochs, epoch_range.stop))
    
    print(f"[PLOT] Processing epochs: {epoch_range.start} to {epoch_range.stop} ({len(epoch_range)} epochs)")
    
    # Set output format and quality
    fmt = "svg" if args.svg else "png"
    quality = args.quality
    if args.svg:
        print(f"[PLOT] Output format: SVG (vector graphics)")
    if quality != "high":
        print(f"[PLOT] Quality: {quality} (reduced file size)")
    
    # Execute based on mode
    if args.mode == "all":
        with Pool(args.workers) as p:
            p.starmap(plot_all, [(e, fmt, quality) for e in epoch_range])
            
    elif args.mode == "eeg":
        with Pool(args.workers) as p:
            p.starmap(plot_eeg_only, [(e, fmt, quality) for e in epoch_range])
            
    elif args.mode == "stc":
        with Pool(args.workers) as p:
            p.starmap(plot_stc_only, [(e, fmt, quality) for e in epoch_range])
    
    elif args.mode == "advanced":
        print(f"[PLOT] Using ADVANCED visualization with modern effects")
        if args.only_points:
            print(f"[PLOT] Timeline style: only points (no lines)")
        with Pool(args.workers) as p:
            p.starmap(plot_advanced, [(e, fmt, quality, args.only_points) for e in epoch_range])
            
    elif args.mode == "video_smooth":
        output = args.output or f"smooth_{args.subject}_{args.condition}_{args.band}.mp4"
        generate_smooth_video(
            subject_id=args.subject,
            condition=args.condition,
            band=args.band,
            output_file=output,
            fps=args.fps,
            interpolation_factor=args.interpolation,
            dpi=args.dpi,
            max_epochs=args.max_epochs
        )
        
    elif args.mode == "video_simple":
        output = args.output or f"simple_{args.view}_{args.subject}_{args.condition}_{args.band}.mp4"
        generate_simple_network_video(
            subject_id=args.subject,
            condition=args.condition,
            band=args.band,
            output_file=output,
            fps=args.fps,
            interpolation_factor=args.interpolation,
            view=args.view,
            dpi=args.dpi,
            max_epochs=args.max_epochs
        )
        
    elif args.mode == "video_3d":
        output = args.output or f"brain3d_{args.subject}_{args.condition}_{args.band}.mp4"
        generate_3d_brain_video(
            subject_id=args.subject,
            condition=args.condition,
            band=args.band,
            output_file=output,
            fps=args.fps,
            interpolation_factor=args.interpolation,
            dpi=args.dpi,
            rotate_camera=args.rotate,
            show_edges=not args.no_edges,
            max_epochs=args.max_epochs
        )
    
    elif args.mode == "video_advanced":
        output = args.output or f"advanced_{args.subject}_{args.condition}_{args.band}.mp4"
        generate_advanced_video(
            subject_id=args.subject,
            condition=args.condition,
            band=args.band,
            output_file=output,
            fps=args.fps,
            interpolation_factor=args.interpolation,
            dpi=args.dpi,
            max_epochs=args.max_epochs
        )
    
    elif args.mode == "frames_to_video":
        create_video_from_frames(
            source_folder=args.source,
            fps=args.fps,
            output_file=args.output,
            subject=args.subject,
            condition=args.condition,
            band=args.band
        )
    
    print(f"[PLOT] Done!")


if __name__ == '__main__':
    main()
