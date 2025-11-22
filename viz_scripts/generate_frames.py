#!/usr/bin/env python3
"""
Script simple para generar frames de visualización
Usa las funciones corregidas de plot.py
"""

import sys
from pathlib import Path

# Import from plot.py
from plot import (
    plot_all, 
    plot_eeg_only,
    plot_stc_only,
    subject_syncro, 
    subject_phases,
    rej,
    num_epochs,
    subject,
    cond,
    band
)

# Get output directory
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import VISUALIZATIONS_DIR
OUTPUT_DIR = VISUALIZATIONS_DIR / "plot"

def _process_single_epoch(args):
    """Función auxiliar para procesar una época (usada por multiprocessing)."""
    epoch, plot_func, samples, end_epoch = args
    try:
        plot_func(epoch, samples=samples)
        return (epoch, True, None)
    except Exception as e:
        import traceback
        return (epoch, False, traceback.format_exc())


def generate_frames(start_epoch=0, end_epoch=None, step=1, view="all", samples=None, workers=1):
    """
    Genera frames para un rango de épocas.
    
    Parámetros:
    -----------
    start_epoch : int
        Época inicial (default: 0)
    end_epoch : int
        Época final (default: todas las épocas disponibles)
    step : int
        Paso entre épocas (default: 1, todas las épocas)
    view : str
        Tipo de visualización: "all" (EEG+STC), "eeg" (solo EEG), "stc" (solo red cerebral)
    samples : list
        Muestras temporales a generar por época (default: [0, 400, 799])
    workers : int
        Número de procesos paralelos (default: 1 = secuencial, 20 = máximo recomendado)
    """
    if end_epoch is None:
        end_epoch = num_epochs
    
    if samples is None:
        samples = [0, 400, 799]
    
    # Seleccionar función de plotting
    view_functions = {
        "all": plot_all,
        "eeg": plot_eeg_only,
        "stc": plot_stc_only
    }
    
    if view not in view_functions:
        print(f"[ERROR] View '{view}' inválido. Opciones: all, eeg, stc")
        return
    
    plot_func = view_functions[view]
    view_name = {"all": "EEG+STC", "eeg": "EEG Only", "stc": "Source Space"}[view]
    
    # Lista de épocas a procesar
    epochs_to_process = list(range(start_epoch, end_epoch, step))
    total_epochs = len(epochs_to_process)
    
    print()
    print("="*70)
    print("  GENERANDO FRAMES")
    print("="*70)
    print(f"  📊 DATOS:")
    print(f"      Sujeto:    {subject}")
    print(f"      Condición: {cond}")
    print(f"      Banda:     {band}")
    print()
    print(f"  ⚙️  CONFIG:")
    print(f"      Vista:     {view_name}")
    print(f"      Épocas:    {start_epoch} → {end_epoch-1} (total: {total_epochs})")
    print(f"      Muestras:  {samples}")
    print(f"      Workers:   {workers} {'🚀 PARALELO' if workers > 1 else '🐌 SECUENCIAL'}")
    print()
    print(f"  💾 OUTPUT: {OUTPUT_DIR}")
    print()
    print(f"  ⚠️  Épocas rechazadas: {len(rej)}")
    if len(rej) > 0:
        print(f"      {rej[:10]}{'...' if len(rej) > 10 else ''}")
    print("="*70)
    print()
    
    if workers > 1:
        # Modo paralelo con multiprocessing
        from multiprocessing import Pool
        import time
        
        print(f"[PARALLEL] Iniciando pool de {workers} workers...")
        start_time = time.time()
        
        # Preparar argumentos para cada época
        job_args = [(epoch, plot_func, samples, end_epoch) for epoch in epochs_to_process]
        
        with Pool(workers) as pool:
            results = pool.map(_process_single_epoch, job_args)
        
        # Procesar resultados
        successful = 0
        failed = 0
        for epoch, success, error in results:
            if success:
                successful += 1
                print(f"[FRAMES] ✓ Época {epoch}/{end_epoch-1} completada")
            else:
                failed += 1
                print(f"[FRAMES] ✗ Error en época {epoch}:")
                print(error)
        
        elapsed = time.time() - start_time
        print()
        print("="*70)
        print(f"  ✓ PROCESO COMPLETADO")
        print("="*70)
        print(f"  Sujeto/Condición/Banda: {subject}/{cond}/{band}")
        print(f"  Tiempo total:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Exitosas:        {successful}/{total_epochs}")
        print(f"  Fallidas:        {failed}/{total_epochs}")
        print(f"  Velocidad:       {elapsed/total_epochs:.2f}s por época")
        print(f"  Output:          {OUTPUT_DIR}")
        print("="*70)
    
    else:
        # Modo secuencial (original)
        import time
        start_time = time.time()
        
        successful = 0
        failed = 0
        
        for epoch in epochs_to_process:
            if epoch in rej:
                print(f"[FRAMES] Época {epoch} marcada como rechazada (se genera igual)")
            
            epoch_start = time.time()
            try:
                plot_func(epoch, samples=samples)
                epoch_time = time.time() - epoch_start
                successful += 1
                print(f"[FRAMES] ✓ Época {epoch}/{end_epoch-1} | {subject}/{cond}/{band} | {epoch_time:.1f}s")
            except Exception as e:
                failed += 1
                print(f"[FRAMES] ✗ Error en época {epoch}: {e}")
                import traceback
                traceback.print_exc()
        
        elapsed = time.time() - start_time
        print()
        print("="*70)
        print(f"  ✓ PROCESO COMPLETADO")
        print("="*70)
        print(f"  Sujeto/Condición/Banda: {subject}/{cond}/{band}")
        print(f"  Tiempo total:    {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  Exitosas:        {successful}/{total_epochs}")
        print(f"  Fallidas:        {failed}/{total_epochs}")
        print(f"  Velocidad:       {elapsed/total_epochs:.2f}s por época")
        print(f"  Output:          {OUTPUT_DIR}")
        print("="*70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generar frames de visualización EEG/STC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Test rápido (3 épocas, secuencial)
  python generate_frames.py --test
  
  # Test rápido con 10 workers (PARALELO, mucho más rápido)
  python generate_frames.py --test --workers 10
  
  # Solo canales EEG (primeras 10 épocas)
  python generate_frames.py --view eeg --end 10
  
  # Rango completo con 20 workers (máxima velocidad)
  python generate_frames.py --view all --end 200 --workers 20
  
  # Solo red cerebral con paralelización
  python generate_frames.py --view stc --start 5 --end 15 --workers 15
  
  # Control de muestras temporales
  python generate_frames.py --view stc --samples 0 200 400 600 799 --workers 10

💡 TIPS:
  • --workers 1: secuencial (lento pero usa menos RAM)
  • --workers 10-20: paralelo (muy rápido, recomendado)
  • Más workers = más rápido, pero también más RAM
  • En CPU con 32 threads, usar hasta 20 workers es óptimo

NOTA: Sujeto, condición y banda se configuran en plot.py (líneas 369-376)
      Actualmente usando: {subject}, {cond}, {band}
        """.format(subject=subject, cond=cond, band=band)
    )
    
    parser.add_argument("--start", type=int, default=0, 
                       help="Época inicial (default: 0)")
    parser.add_argument("--end", type=int, default=None, 
                       help="Época final (default: todas las épocas)")
    parser.add_argument("--step", type=int, default=1, 
                       help="Paso entre épocas (default: 1)")
    parser.add_argument("--view", type=str, default="all", 
                       choices=["all", "eeg", "stc"],
                       help="Vista a generar: 'all' (EEG+STC completo), 'eeg' (solo canales EEG), 'stc' (solo red cerebral)")
    parser.add_argument("--samples", type=int, nargs="+", default=None,
                       help="Muestras temporales a generar por época (default: 0 400 799)")
    parser.add_argument("--workers", type=int, default=1,
                       help="Número de workers paralelos (default: 1, recomendado: 10-20 para acelerar)")
    parser.add_argument("--test", action="store_true", 
                       help="Modo test: genera solo las primeras 3 épocas con vista 'all'")
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"GENERADOR DE FRAMES - EEG/STC")
    print(f"{'='*70}")
    print(f"📁 Configuración actual (desde plot.py):")
    print(f"   • Sujeto:    {subject}")
    print(f"   • Condición: {cond}")
    print(f"   • Banda:     {band}")
    print(f"   • Total de épocas disponibles: {num_epochs}")
    print(f"   • Épocas rechazadas: {len(rej)} ({rej[:5]}{'...' if len(rej) > 5 else ''})")
    print(f"")
    print(f"⚙️  Parámetros de generación:")
    print(f"   • Vista:     {args.view}")
    print(f"   • Rango:     época {args.start} - {args.end if args.end else num_epochs-1}")
    print(f"   • Muestras:  {args.samples if args.samples else [0, 400, 799]}")
    print(f"")
    print(f"💾 Output: {OUTPUT_DIR}")
    print(f"{'='*70}\n")
    
    if args.test:
        print("[TEST MODE] Generando solo 3 épocas de prueba (vista completa)...\n")
        generate_frames(start_epoch=0, end_epoch=3, step=1, view="all", workers=args.workers)
    else:
        generate_frames(
            start_epoch=args.start,
            end_epoch=args.end,
            step=args.step,
            view=args.view,
            samples=args.samples,
            workers=args.workers
        )
        
    print(f"\n{'='*70}")
    print(f"✓ COMPLETADO")
    print(f"{'='*70}")
    print(f"")
    print(f"💡 Para cambiar sujeto/condición/banda:")
    print(f"   Editar viz_scripts/plot.py líneas 369-376:")
    print(f"   ")
    print(f"   band = \"Alpha\"        # Delta, Theta, Alpha, Beta, Gamma")
    print(f"   cond = \"DMT\"          # DMT, EC, EO")
    print(f"   subject = \"S01\"       # S01, S02, S03, etc.")
    print(f"")
    print(f"{'='*70}\n")

