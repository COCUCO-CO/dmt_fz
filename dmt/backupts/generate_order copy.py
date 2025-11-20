#!/usr/bin/env python3
"""
Script para generar archivos order-*.pkl a partir de order_all-*.pkl

Este script:
1. Lee los archivos order_all-*.pkl (contienen DataFrames de fases filtradas por red)
2. Calcula el parámetro de orden de Kuramoto para cada DataFrame
3. Guarda los resultados como order-*.pkl (Series con valores 0-1)

Los archivos order-*.pkl tienen la estructura:
{
    'COND': {
        'Band': {
            'hemi': {
                'NET': [pd.Series, pd.Series, ...]  # Una Series por época
            }
        }
    }
}

Uso:
    python generate_order.py --workers 20 --conditions DMT EC EO
"""

import argparse
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from paths import RESULTS_DIR


def order_parameter(phase_df):
    """
    Calcula el parámetro de orden de Kuramoto a partir de un DataFrame de fases.
    
    Parameters
    ----------
    phase_df : pd.DataFrame
        DataFrame con fases (parcelas x samples)
    
    Returns
    -------
    pd.Series
        Serie temporal del parámetro de orden (valores entre 0 y 1)
    """
    phase_array = phase_df.values
    r = np.abs(np.exp(1j * phase_array).mean(axis=0))
    return pd.Series(r)


def _apply_pandas_compatibility_patches():
    """Apply monkey-patches for old pandas pickle compatibility."""
    import sys
    
    # Handle missing modules that were reorganized in pandas 2.0+
    try:
        import pandas.core.indexes.numeric
    except (ImportError, ModuleNotFoundError):
        import pandas.core.indexes as indexes
        sys.modules['pandas.core.indexes.numeric'] = indexes
    
    # Handle Int64Index, Float64Index, etc. that were removed in pandas 2.0
    try:
        from pandas import Int64Index
    except ImportError:
        from pandas import Index as Int64Index
        pd.Int64Index = Int64Index
        sys.modules['pandas'].Int64Index = Int64Index
    
    try:
        from pandas import Float64Index
    except ImportError:
        from pandas import Index as Float64Index
        pd.Float64Index = Float64Index
        sys.modules['pandas'].Float64Index = Float64Index
    
    try:
        from pandas import UInt64Index
    except ImportError:
        from pandas import Index as UInt64Index
        pd.UInt64Index = UInt64Index
        sys.modules['pandas'].UInt64Index = UInt64Index


def load_file(file_path):
    """Cargar archivo pickle con compatibilidad para versiones antiguas de pandas."""
    # Apply patches in this process (important for multiprocessing workers)
    _apply_pandas_compatibility_patches()
    
    try:
        with open(file_path, 'rb') as handle:
            return pickle.load(handle)
    except Exception as e:
        # Fallback: try pandas read_pickle which handles version mismatches better
        try:
            return pd.read_pickle(file_path)
        except Exception as e2:
            # Only print if both methods fail
            raise Exception(f"Failed to load {file_path}: pickle.load error: {e}, pd.read_pickle error: {e2}")


def save_file(data, folder, file_stem):
    """Guardar archivo pickle."""
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / f"{file_stem}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def process_order_all_file(file_path):
    """
    Procesa un archivo order_all-*.pkl y genera el correspondiente order-*.pkl
    
    Parameters
    ----------
    file_path : Path
        Ruta al archivo order_all-*.pkl
    
    Returns
    -------
    str
        Mensaje de éxito o error
    """
    try:
        file_path = Path(file_path)
        
        # Cargar datos
        order_all_data = load_file(file_path)
        
        # Extraer información del archivo
        cond = file_path.parent.name
        subject_id = file_path.name.replace("order_all-", "order-").replace(".pkl", "")
        
        # Estructura de salida
        order_data = {}
        order_data[cond] = {}
        
        # Obtener la estructura de datos
        cond_data = order_all_data.get(cond, {})
        
        if not cond_data:
            return f"[WARN] {file_path.name}: No data for condition {cond}"
        
        # Procesar cada banda
        for band, band_data in cond_data.items():
            order_data[cond][band] = {}
            
            # Procesar cada hemisferio
            for hemi, hemi_data in band_data.items():
                order_data[cond][band][hemi] = {}
                
                # Procesar cada red
                for net, epoch_list in hemi_data.items():
                    order_data[cond][band][hemi][net] = []
                    
                    # Calcular order parameter para cada época
                    for epoch_df in epoch_list:
                        if isinstance(epoch_df, pd.DataFrame) and not epoch_df.empty:
                            r_series = order_parameter(epoch_df)
                            order_data[cond][band][hemi][net].append(r_series)
                        else:
                            # Si no es un DataFrame válido, agregar una Series vacía
                            order_data[cond][band][hemi][net].append(pd.Series([]))
        
        # Guardar resultado
        output_folder = file_path.parent
        save_file(order_data, output_folder, subject_id)
        
        return f"[OK] {subject_id}.pkl"
    
    except Exception as e:
        return f"[ERROR] {file_path.name}: {str(e)}"


def process_condition(condition, workers):
    """
    Procesa todos los archivos order_all-*.pkl de una condición.
    
    Parameters
    ----------
    condition : str
        Condición a procesar (DMT, EC, EO)
    workers : int
        Número de procesos en paralelo
    """
    cond_dir = RESULTS_DIR / condition
    
    if not cond_dir.exists():
        print(f"[SKIP] Condición '{condition}': directorio no existe ({cond_dir})")
        return
    
    # Buscar archivos order_all-*.pkl
    order_all_files = sorted(cond_dir.glob("order_all-*.pkl"))
    
    if not order_all_files:
        print(f"[SKIP] Condición '{condition}': no se encontraron archivos order_all-*.pkl")
        return
    
    print(f"\n{'='*70}")
    print(f"[{condition}] Procesando {len(order_all_files)} archivos order_all-*.pkl")
    print(f"{'='*70}")
    
    if workers == 1:
        # Procesamiento secuencial
        for file_path in tqdm(order_all_files, desc=f"{condition}"):
            result = process_order_all_file(file_path)
            if result.startswith("[ERROR]") or result.startswith("[WARN]"):
                tqdm.write(result)
    else:
        # Procesamiento paralelo
        with Pool(processes=workers) as pool:
            results = list(tqdm(
                pool.imap(process_order_all_file, order_all_files),
                total=len(order_all_files),
                desc=f"{condition}"
            ))
            
            # Mostrar errores y warnings
            for result in results:
                if result.startswith("[ERROR]") or result.startswith("[WARN]"):
                    print(result)
    
    print(f"[{condition}] ✓ Completado")


def main():
    parser = argparse.ArgumentParser(
        description="Generar archivos order-*.pkl a partir de order_all-*.pkl"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Número de procesos en paralelo (por defecto: todos los núcleos disponibles)"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["DMT", "EC", "EO"],
        help="Condiciones a procesar (por defecto: DMT EC EO)"
    )
    
    args = parser.parse_args()
    
    # Determinar número de workers
    if args.workers is None or args.workers <= 0:
        workers = cpu_count()
    else:
        workers = args.workers
    
    print(f"\n{'='*70}")
    print(f"GENERACIÓN DE ARCHIVOS order-*.pkl")
    print(f"{'='*70}")
    print(f"Directorio base: {RESULTS_DIR}")
    print(f"Condiciones: {', '.join(args.conditions)}")
    print(f"Workers: {workers}")
    print(f"{'='*70}\n")
    
    # Procesar cada condición
    for condition in args.conditions:
        process_condition(condition, workers)
    
    print(f"\n{'='*70}")
    print(f"✓ PROCESO COMPLETADO")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

