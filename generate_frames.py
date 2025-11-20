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
    FRAMES_DIR,
    subject,
    cond,
    band
)

def generate_frames(start_epoch=0, end_epoch=None, step=1, view="all", samples=None):
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
    
    print(f"[FRAMES] Modo: {view_name}")
    print(f"[FRAMES] Sujeto: {subject}, Condición: {cond}, Banda: {band}")
    print(f"[FRAMES] Generando frames para épocas {start_epoch} a {end_epoch-1}")
    print(f"[FRAMES] Muestras por época: {samples}")
    print(f"[FRAMES] Output directory: {FRAMES_DIR}")
    print(f"[FRAMES] Épocas rechazadas: {rej}")
    print(f"[FRAMES] Total de épocas a procesar: {(end_epoch - start_epoch) // step}")
    
    for epoch in range(start_epoch, end_epoch, step):
        if epoch in rej:
            print(f"[FRAMES] Época {epoch} marcada como rechazada (se genera igual)")
        
        try:
            plot_func(epoch, samples=samples)
            print(f"[FRAMES] ✓ Época {epoch}/{end_epoch-1} completada")
        except Exception as e:
            print(f"[FRAMES] ✗ Error en época {epoch}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"[FRAMES] ✓ Proceso completado!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generar frames de visualización EEG/STC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Test rápido (3 épocas, ambos EEG y STC)
  python generate_frames.py --test
  
  # Solo canales EEG (primeras 10 épocas)
  python generate_frames.py --view eeg --end 10
  
  # Solo red cerebral completa (STC)
  python generate_frames.py --view stc --start 5 --end 15
  
  # Ambas vistas (modo completo)
  python generate_frames.py --view all --end 20
  
  # Control de muestras temporales
  python generate_frames.py --view stc --samples 0 200 400 600 799
        """
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
    parser.add_argument("--test", action="store_true", 
                       help="Modo test: genera solo las primeras 3 épocas con vista 'all'")
    
    args = parser.parse_args()
    
    if args.test:
        print("[TEST MODE] Generando solo 3 épocas de prueba (vista completa)...")
        generate_frames(start_epoch=0, end_epoch=3, step=1, view="all")
    else:
        print(f"\n{'='*60}")
        print(f"GENERADOR DE FRAMES - EEG/STC")
        print(f"{'='*60}\n")
        
        generate_frames(
            start_epoch=args.start,
            end_epoch=args.end,
            step=args.step,
            view=args.view,
            samples=args.samples
        )
        
        print(f"\n{'='*60}")
        print(f"Para configurar sujeto/condición/banda, editar líneas 365-376 de plot.py")
        print(f"{'='*60}\n")

