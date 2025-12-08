#!/usr/bin/env python3
"""
clustering.py - Análisis de clustering sobre matrices de sincronización EEG

Este script realiza:
1. Carga datos de sincronización (syncro-*.pkl) de todas las condiciones
2. Extrae eigenvalores de las matrices de sincronización
3. Aplica PCA para reducción de dimensionalidad
4. Realiza clustering K-Means optimizando número de clusters
5. Calcula Silhouette Score para evaluar calidad
6. Genera heatmaps y visualizaciones
7. Opcionalmente construye modelos de Markov para transiciones

Uso:
    python clustering.py                           # Análisis completo
    python clustering.py --bands Alpha Theta       # Solo bandas específicas
    python clustering.py --max-k 10 --max-comps 8  # Limitar búsqueda
    python clustering.py --no-plots                # Sin visualizaciones
"""

import argparse
import os
import pickle
from copy import deepcopy
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Imports de sklearn
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import pairwise_distances, silhouette_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE

# Import de matplotlib
import matplotlib.pyplot as plt

# Paths del pipeline
from paths import RESULTS_DIR, SPECTRAL_DIR, ensure_dir


# =============================================================================
# Configuración
# =============================================================================

BAND_LIST = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
COND_LIST = ["DMT", "EC", "EO"]

# Directorio de salida para clustering
CLUSTERING_OUTPUT_DIR = RESULTS_DIR / "clustering_results"


# =============================================================================
# Funciones auxiliares
# =============================================================================

def log(message: str):
    """Imprime mensaje con prefijo."""
    print(f"[CLUSTERING] {message}")


def save_file(data, folder: Path, filename: str):
    """Guarda datos en archivo pickle."""
    folder = ensure_dir(folder)
    filepath = folder / f"{filename}.pkl"
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"Guardado: {filepath}")


def load_file(filepath: Path):
    """Carga archivo pickle."""
    with open(filepath, 'rb') as handle:
        return pickle.load(handle)


def cond_path(cond: str) -> Path:
    """Retorna path de la condición."""
    return RESULTS_DIR / cond


def reject_outliers(data: np.ndarray, m: float = 2.0) -> np.ndarray:
    """
    Retorna máscara booleana para filtrar outliers.
    
    Args:
        data: Array de datos
        m: Número de desviaciones estándar para el umbral
    
    Returns:
        Máscara booleana (True = mantener, False = outlier)
    """
    return np.abs(data - np.mean(data)) < m * np.std(data)


# =============================================================================
# Carga de datos
# =============================================================================

def _process_single_file(args: tuple) -> dict:
    """
    Procesa un archivo syncro-*.pkl (función para multiprocessing).
    
    Args:
        args: Tupla de (filepath, bands, splits_idx)
    
    Returns:
        Diccionario con eigenvalores y kuramoto por banda
    """
    filepath, bands, splits_idx = args
    
    result = {
        'eigen': {band: [] for band in bands},
        'kuramoto': {band: [] for band in bands},
        'error': None
    }
    
    try:
        with open(filepath, 'rb') as handle:
            data_dict = pickle.load(handle)
        
        for band in bands:
            # Obtener datos de Kuramoto
            kuramoto_data = data_dict.get("kuramoto_stc", {}).get(band, [])
            if kuramoto_data and len(kuramoto_data) > splits_idx:
                result['kuramoto'][band] = list(kuramoto_data[splits_idx])
            
            # Obtener matrices de sincronización y calcular eigenvalores
            syncro_data = data_dict.get("syncros_stc", {}).get(band, [])
            if syncro_data and len(syncro_data) > splits_idx:
                for epoch_mat in syncro_data[splits_idx]:
                    if isinstance(epoch_mat, np.ndarray) and epoch_mat.ndim == 2:
                        # np.linalg.eigh retorna (eigenvalues, eigenvectors)
                        # Usamos [0] para obtener los eigenvalues
                        eigenvalues = np.linalg.eigh(epoch_mat)[0]
                        result['eigen'][band].append(eigenvalues)
    
    except Exception as e:
        result['error'] = str(e)
    
    return result


def load_syncro_data(conditions: list[str] = None, bands: list[str] = None, 
                     splits_idx: int = 0, workers: int = None) -> tuple[dict, dict]:
    """
    Carga datos de sincronización de todos los sujetos y condiciones.
    
    Args:
        conditions: Lista de condiciones a cargar (default: todas)
        bands: Lista de bandas a cargar (default: todas)
        splits_idx: Índice del split temporal a usar (default: 0)
        workers: Número de workers para carga paralela (default: cpu_count)
    
    Returns:
        Tuple de (eigen_dict, kuramoto_dict) con estructura:
        {condición: {banda: [lista de eigenvalores/kuramoto]}}
    """
    if conditions is None:
        conditions = COND_LIST
    if bands is None:
        bands = BAND_LIST
    if workers is None or workers <= 0:
        workers = cpu_count()
    
    eigen_dict = {cond: {band: [] for band in bands} for cond in conditions}
    kuramoto_dict = {cond: {band: [] for band in bands} for cond in conditions}
    
    for cond in conditions:
        log(f"Cargando condición: {cond}")
        cond_dir = cond_path(cond)
        
        if not cond_dir.exists():
            log(f"[WARN] Directorio no existe: {cond_dir}")
            continue
        
        # Buscar archivos syncro-*.pkl
        syncro_files = sorted([
            cond_dir / f for f in os.listdir(cond_dir) 
            if f.startswith('syncro-') and f.endswith('.pkl')
        ])
        
        if not syncro_files:
            log(f"[WARN] No se encontraron archivos syncro-*.pkl en {cond_dir}")
            continue
        
        # Preparar argumentos para multiprocessing
        args_list = [(filepath, bands, splits_idx) for filepath in syncro_files]
        
        # Procesar en paralelo
        log(f"  Procesando {len(syncro_files)} archivos con {workers} workers...")
        
        if workers == 1:
            # Modo secuencial
            results = []
            for args in tqdm(args_list, desc=f"  {cond}"):
                results.append(_process_single_file(args))
        else:
            # Modo paralelo con imap_unordered (más eficiente)
            # chunksize reduce overhead de IPC
            chunksize = max(1, len(args_list) // (workers * 4))
            with Pool(processes=workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_process_single_file, args_list, chunksize=chunksize),
                    total=len(args_list),
                    desc=f"  {cond}"
                ))
        
        # Consolidar resultados
        errors = 0
        for result in results:
            if result['error']:
                errors += 1
                continue
            
            for band in bands:
                eigen_dict[cond][band].extend(result['eigen'][band])
                kuramoto_dict[cond][band].extend(result['kuramoto'][band])
        
        if errors > 0:
            log(f"  [WARN] {errors} archivos con errores")
    
    # Reportar estadísticas
    log("\nResumen de datos cargados:")
    for cond in conditions:
        for band in bands:
            n_eigen = len(eigen_dict[cond][band])
            if n_eigen > 0:
                log(f"  {cond}/{band}: {n_eigen} epochs")
    
    return eigen_dict, kuramoto_dict


# =============================================================================
# Análisis de Clustering
# =============================================================================

def find_optimal_clustering(data: np.ndarray, 
                            min_k: int = 2, max_k: int = 15,
                            min_comps: int = 2, max_comps: int = 10,
                            scaler_type: str = 'standard',
                            dim_reduction: str = 'pca',
                            cluster_method: str = 'kmeans',
                            distance_metric: str = 'euclidean',
                            use_top_eigenvalues: int = None) -> dict:
    """
    Busca la combinación óptima de componentes y número de clusters.
    
    Args:
        data: Matriz de eigenvalores (n_samples, n_features)
        min_k, max_k: Rango de número de clusters
        min_comps, max_comps: Rango de componentes para reducción
        scaler_type: 'standard', 'minmax', 'robust', o 'none'
        dim_reduction: 'pca', 'kernel_pca', o 'none'
        cluster_method: 'kmeans', 'gmm', 'hierarchical'
        distance_metric: 'euclidean', 'cosine', 'correlation'
        use_top_eigenvalues: Usar solo los N eigenvalores más grandes (None = todos)
    
    Returns:
        Diccionario con resultados del análisis
    """
    n_samples, n_features = data.shape
    
    # Filtrar eigenvalores si se especifica
    if use_top_eigenvalues is not None and use_top_eigenvalues < n_features:
        # Ordenar por magnitud y tomar los más grandes
        data = data[:, -use_top_eigenvalues:]
        n_features = use_top_eigenvalues
        log(f"    Usando top {use_top_eigenvalues} eigenvalores")
    
    # Aplicar escalado
    if scaler_type == 'standard':
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    elif scaler_type == 'minmax':
        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)
    elif scaler_type == 'robust':
        scaler = RobustScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        scaler = None
        data_scaled = data.copy()
    
    # Ajustar máximos si hay pocas muestras
    max_comps = min(max_comps, n_features, n_samples - 1)
    max_k = min(max_k, n_samples - 1)
    
    heatmap = np.zeros((max_comps - min_comps + 1, max_k - min_k + 1))
    best_score = -1
    best_params = {'n_components': min_comps, 'n_clusters': min_k}
    best_model = None
    best_reducer = None
    best_labels = None
    
    for n_comps in range(min_comps, max_comps + 1):
        # Reducción de dimensionalidad
        if dim_reduction == 'pca':
            reducer = PCA(n_components=n_comps)
            data_reduced = reducer.fit_transform(data_scaled)
        elif dim_reduction == 'kernel_pca':
            reducer = KernelPCA(n_components=n_comps, kernel='rbf')
            data_reduced = reducer.fit_transform(data_scaled)
        else:
            reducer = None
            data_reduced = data_scaled
        
        for n_clusters in range(min_k, max_k + 1):
            if n_clusters >= n_samples:
                continue
            
            # Clustering
            if cluster_method == 'kmeans':
                model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
                labels = model.fit_predict(data_reduced)
            elif cluster_method == 'gmm':
                model = GaussianMixture(n_components=n_clusters, random_state=42)
                labels = model.fit_predict(data_reduced)
            elif cluster_method == 'hierarchical':
                model = AgglomerativeClustering(n_clusters=n_clusters)
                labels = model.fit_predict(data_reduced)
            else:
                continue
            
            # Calcular silhouette
            try:
                if len(set(labels)) < 2:
                    score = 0
                else:
                    score = silhouette_score(data_reduced, labels, metric=distance_metric)
            except ValueError:
                score = 0
            
            heatmap[n_comps - min_comps, n_clusters - min_k] = score
            
            if score > best_score:
                best_score = score
                best_params = {
                    'n_components': n_comps, 
                    'n_clusters': n_clusters,
                    'scaler': scaler_type,
                    'dim_reduction': dim_reduction,
                    'cluster_method': cluster_method,
                    'distance_metric': distance_metric
                }
                best_model = deepcopy(model)
                best_reducer = deepcopy(reducer) if reducer else None
                best_labels = labels.copy()
    
    # Guardar datos reducidos para visualización
    best_data_reduced = None
    if best_reducer is not None and best_score > -1:
        if scaler is not None:
            best_data_reduced = best_reducer.transform(scaler.transform(data))
        else:
            best_data_reduced = best_reducer.transform(data)
    
    return {
        'heatmap': heatmap,
        'best_score': best_score,
        'best_params': best_params,
        'best_model': best_model,
        'best_kmeans': best_model,  # Compatibilidad
        'best_reducer': best_reducer,
        'best_pca': best_reducer,  # Compatibilidad
        'best_scaler': scaler,
        'best_labels': best_labels,
        'best_data_reduced': best_data_reduced,
        'min_k': min_k,
        'max_k': max_k,
        'min_comps': min_comps,
        'max_comps': max_comps,
        'actual_max_k': max_k,  # Valor ajustado
        'actual_max_comps': max_comps  # Valor ajustado
    }


def find_best_configuration(data: np.ndarray, 
                            min_k: int = 2, max_k: int = 10,
                            min_comps: int = 2, max_comps: int = 8) -> dict:
    """
    Prueba múltiples configuraciones para encontrar la mejor.
    
    Prueba combinaciones de:
    - Escaladores: standard, robust, minmax
    - Reducción: pca, kernel_pca
    - Clustering: kmeans, gmm, hierarchical
    - Métricas: euclidean, cosine
    """
    configurations = [
        {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'kmeans', 'distance_metric': 'euclidean'},
        {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'kmeans', 'distance_metric': 'cosine'},
        {'scaler_type': 'robust', 'dim_reduction': 'pca', 'cluster_method': 'kmeans', 'distance_metric': 'euclidean'},
        {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'gmm', 'distance_metric': 'euclidean'},
        {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'hierarchical', 'distance_metric': 'euclidean'},
        {'scaler_type': 'standard', 'dim_reduction': 'kernel_pca', 'cluster_method': 'kmeans', 'distance_metric': 'euclidean'},
        {'scaler_type': 'robust', 'dim_reduction': 'pca', 'cluster_method': 'gmm', 'distance_metric': 'cosine'},
    ]
    
    best_result = None
    best_score = -1
    all_results = []
    
    log("    Probando configuraciones...")
    for config in configurations:
        config_name = f"{config['scaler_type']}/{config['dim_reduction']}/{config['cluster_method']}/{config['distance_metric']}"
        
        result = find_optimal_clustering(
            data, min_k=min_k, max_k=max_k,
            min_comps=min_comps, max_comps=max_comps,
            **config
        )
        result['config_name'] = config_name
        all_results.append(result)
        
        if result['best_score'] > best_score:
            best_score = result['best_score']
            best_result = result
            log(f"      Nueva mejor: {config_name} -> {best_score:.4f}")
    
    best_result['all_results'] = all_results
    return best_result


# =============================================================================
# Búsqueda exhaustiva paralela
# =============================================================================

def generate_all_configurations(quick: bool = False) -> list[dict]:
    """
    Genera combinaciones de configuraciones.
    
    Args:
        quick: Si True, genera solo 12 configuraciones esenciales.
               Si False, genera las 144 combinaciones completas.
    """
    from itertools import product
    
    if quick:
        # 8 configuraciones RÁPIDAS (sin Kernel PCA ni Hierarchical que son lentos)
        configurations = [
            # KMeans - el más rápido y robusto
            {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'kmeans', 'distance_metric': 'euclidean', 'use_top_eigenvalues': None},
            {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'kmeans', 'distance_metric': 'cosine', 'use_top_eigenvalues': None},
            {'scaler_type': 'robust', 'dim_reduction': 'pca', 'cluster_method': 'kmeans', 'distance_metric': 'euclidean', 'use_top_eigenvalues': None},
            {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'kmeans', 'distance_metric': 'euclidean', 'use_top_eigenvalues': 50},
            
            # GMM - captura clusters elípticos (un poco más lento pero vale la pena)
            {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'gmm', 'distance_metric': 'euclidean', 'use_top_eigenvalues': None},
            {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'gmm', 'distance_metric': 'cosine', 'use_top_eigenvalues': None},
            {'scaler_type': 'robust', 'dim_reduction': 'pca', 'cluster_method': 'gmm', 'distance_metric': 'euclidean', 'use_top_eigenvalues': None},
            {'scaler_type': 'standard', 'dim_reduction': 'pca', 'cluster_method': 'gmm', 'distance_metric': 'euclidean', 'use_top_eigenvalues': 50},
            
            # NOTA: Kernel PCA y Hierarchical removidos porque son O(n²) o peor
            # y con 10k+ epochs tardan demasiado. Usar --full-search si se necesitan.
        ]
    else:
        # 144 configuraciones completas
        scalers = ['standard', 'robust', 'minmax']
        dim_reductions = ['pca', 'kernel_pca']
        cluster_methods = ['kmeans', 'gmm', 'hierarchical']
        distance_metrics = ['euclidean', 'cosine']
        top_eigenvalues_options = [None, 30, 50, 70]  # None = todos (102 eigenvalores)
        
        configurations = []
        for scaler, dim_red, cluster, distance, top_eigen in product(
            scalers, dim_reductions, cluster_methods, distance_metrics, top_eigenvalues_options
        ):
            config = {
                'scaler_type': scaler,
                'dim_reduction': dim_red,
                'cluster_method': cluster,
                'distance_metric': distance,
                'use_top_eigenvalues': top_eigen
            }
            configurations.append(config)
    
    # Agregar nombres a todas las configuraciones
    for config in configurations:
        eigen_str = f"top{config['use_top_eigenvalues']}" if config['use_top_eigenvalues'] else "all"
        config['name'] = f"{config['scaler_type']}_{config['dim_reduction']}_{config['cluster_method']}_{config['distance_metric']}_{eigen_str}"
    
    return configurations


def _run_single_config(args: tuple) -> dict:
    """
    Ejecuta una configuración específica (para multiprocessing).
    
    Args:
        args: (config, data, band, min_k, max_k, min_comps, max_comps)
    
    Returns:
        Diccionario con resultados
    """
    config, data, band, min_k, max_k, min_comps, max_comps = args
    
    try:
        result = find_optimal_clustering(
            data,
            min_k=min_k,
            max_k=max_k,
            min_comps=min_comps,
            max_comps=max_comps,
            scaler_type=config['scaler_type'],
            dim_reduction=config['dim_reduction'],
            cluster_method=config['cluster_method'],
            distance_metric=config['distance_metric'],
            use_top_eigenvalues=config['use_top_eigenvalues']
        )
        result['config'] = config
        result['config_name'] = config['name']
        result['band'] = band
        result['error'] = None
    except Exception as e:
        result = {
            'config': config,
            'config_name': config['name'],
            'band': band,
            'best_score': -1,
            'error': str(e)
        }
    
    return result


def run_full_search(eigen_dict: dict, kuramoto_dict: dict,
                   bands: list[str],
                   min_k: int, max_k: int,
                   min_comps: int, max_comps: int,
                   filter_outliers: bool,
                   outlier_threshold: float,
                   analysis_workers: int,
                   output_dir: Path,
                   show_plots: bool = False,
                   quick: bool = False,
                   max_epochs: int = None) -> dict:
    """
    Ejecuta búsqueda de configuraciones en paralelo.
    
    Args:
        eigen_dict: Datos de eigenvalores
        kuramoto_dict: Datos de Kuramoto
        bands: Bandas a analizar
        min_k, max_k: Rango de clusters
        min_comps, max_comps: Rango de componentes
        filter_outliers: Filtrar outliers
        outlier_threshold: Umbral de outliers
        analysis_workers: Workers para análisis paralelo
        output_dir: Directorio de salida
        show_plots: Mostrar plots interactivamente
        quick: Si True, usa solo 12 configuraciones esenciales
    
    Returns:
        Diccionario con todos los resultados
    """
    all_configurations = generate_all_configurations(quick=quick)
    log(f"Generadas {len(all_configurations)} configuraciones para probar")
    
    all_results = {}
    
    for band in bands:
        log(f"\n{'='*60}")
        log(f"BANDA: {band}")
        log(f"{'='*60}")
        
        # Preparar datos de la banda
        all_eigen = []
        all_kuramoto = []
        condition_labels = []
        
        for cond in COND_LIST:
            eigen_data = eigen_dict.get(cond, {}).get(band, [])
            kuramoto_data = kuramoto_dict.get(cond, {}).get(band, [])
            
            if eigen_data:
                all_eigen.extend(eigen_data)
                condition_labels.extend([cond] * len(eigen_data))
                
                for k in kuramoto_data:
                    if isinstance(k, np.ndarray):
                        all_kuramoto.append(np.mean(k))
                    else:
                        all_kuramoto.append(k)
        
        if len(all_eigen) < min_k + 1:
            log(f"[WARN] Insuficientes datos para {band}: {len(all_eigen)} epochs")
            continue
        
        X = np.asarray(all_eigen)
        condition_labels = np.array(condition_labels)
        
        # Filtrar outliers
        if filter_outliers and len(all_kuramoto) == len(all_eigen):
            kuramoto_array = np.asarray(all_kuramoto)
            mask = reject_outliers(kuramoto_array, m=outlier_threshold)
            X = X[mask]
            condition_labels = condition_labels[mask]
            log(f"Filtrado: {mask.sum()}/{len(mask)} epochs")
        
        # Submuestreo si hay demasiados epochs
        if max_epochs is not None and X.shape[0] > max_epochs:
            np.random.seed(42)  # Reproducibilidad
            indices = np.random.choice(X.shape[0], max_epochs, replace=False)
            X = X[indices]
            condition_labels = condition_labels[indices]
            log(f"Submuestreo: {max_epochs} epochs (de {len(indices) + (X.shape[0] - max_epochs)})")
        
        log(f"Datos: {X.shape[0]} epochs, {X.shape[1]} features")
        log(f"Ejecutando {len(all_configurations)} configuraciones con {analysis_workers} workers...")
        
        # Preparar argumentos para multiprocessing
        args_list = [
            (config, X, band, min_k, max_k, min_comps, max_comps)
            for config in all_configurations
        ]
        
        # Ejecutar análisis
        # NOTA: Limitamos workers porque multiprocessing copia datos a cada proceso
        effective_workers = min(analysis_workers, 4)  # Máximo 4 para evitar OOM
        
        if effective_workers == 1 or quick:
            # Modo secuencial: más lento pero usa menos memoria
            results = []
            for args in tqdm(args_list, desc=f"  {band}"):
                results.append(_run_single_config(args))
        else:
            with Pool(processes=effective_workers) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_run_single_config, args_list),
                    total=len(args_list),
                    desc=f"  {band}"
                ))
        
        # Ordenar por score
        results = sorted(results, key=lambda x: x.get('best_score', -1), reverse=True)
        
        # Guardar resultados de la banda
        band_output_dir = ensure_dir(output_dir / band)
        
        # Guardar resumen con campos expandidos
        summary = []
        for r in results:
            params = r.get('best_params', {})
            summary.append({
                'rank': len(summary) + 1,
                'config_name': r['config_name'],
                'best_score': r.get('best_score', -1),
                'n_clusters': params.get('n_clusters'),
                'n_components': params.get('n_components'),
                'scaler': params.get('scaler'),
                'dim_reduction': params.get('dim_reduction'),
                'cluster_method': params.get('cluster_method'),
                'distance_metric': params.get('distance_metric'),
                'error': r.get('error')
            })
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(band_output_dir / 'results_summary.csv', index=False)
        log(f"\nResumen guardado en: {band_output_dir / 'results_summary.csv'}")
        
        # Mostrar top 10
        log(f"\nTop 10 configuraciones para {band}:")
        log("-" * 70)
        for i, r in enumerate(results[:10]):
            score = r.get('best_score', -1)
            params = r.get('best_params', {})
            k = params.get('n_clusters', '?')
            comps = params.get('n_components', '?')
            log(f"  {i+1:2d}. {r['config_name']:<50} Score: {score:.4f} (k={k}, pca={comps})")
        
        # Guardar y plotear top 5
        log(f"\nGenerando plots y guardando datos para top 5...")
        for i, r in enumerate(results[:5]):
            config_dir = ensure_dir(band_output_dir / f"rank{i+1}_{r['config_name']}")
            
            # Guardar resultado completo (incluyendo labels para análisis posterior)
            result_to_save = {
                'config': r.get('config'),
                'best_score': r.get('best_score'),
                'best_params': r.get('best_params'),
                'heatmap': r.get('heatmap'),
                'min_k': r.get('min_k'),
                'max_k': r.get('max_k'),
                'min_comps': r.get('min_comps'),
                'max_comps': r.get('max_comps'),
                'best_labels': r.get('best_labels'),
                'condition_labels': condition_labels,
                'n_epochs': X.shape[0],
                'n_features': X.shape[1]
            }
            save_file(result_to_save, config_dir, 'clustering_result')
            
            # Generar plots
            r['band'] = band
            r['data'] = X
            r['condition_labels'] = condition_labels
            
            try:
                plot_silhouette_heatmap(r, output_dir=config_dir, show=show_plots)
                plot_pca_scatter(r, output_dir=config_dir, show=show_plots)
                if r.get('best_model') is not None:
                    plot_cluster_centers(r, output_dir=config_dir, show=show_plots)
                
                # Analizar composición de clusters (solo para rank1)
                if i == 0 and r.get('best_labels') is not None:
                    analyze_cluster_composition(
                        r['best_labels'], 
                        condition_labels,
                        band,
                        output_dir=config_dir,
                        show=show_plots
                    )
            except Exception as e:
                log(f"    [WARN] Error generando plots para {r['config_name']}: {e}")
        
        all_results[band] = {
            'results': results[:10],  # Solo guardar top 10 para ahorrar memoria
            'data_shape': X.shape,
            'n_configurations': len(all_configurations),
            'best_config': results[0] if results else None
        }
        
        # Liberar memoria
        del X, condition_labels, args_list, results
        import gc
        gc.collect()
        log(f"Memoria liberada para {band}")
    
    # Guardar resumen global
    global_summary = []
    for band, band_data in all_results.items():
        if band_data['best_config']:
            best = band_data['best_config']
            global_summary.append({
                'band': band,
                'best_config': best['config_name'],
                'best_score': best.get('best_score', -1),
                'n_clusters': best.get('best_params', {}).get('n_clusters'),
                'n_components': best.get('best_params', {}).get('n_components')
            })
    
    global_df = pd.DataFrame(global_summary)
    global_df.to_csv(output_dir / 'global_best_results.csv', index=False)
    
    log(f"\n{'='*60}")
    log("RESUMEN GLOBAL - MEJOR CONFIGURACIÓN POR BANDA")
    log(f"{'='*60}")
    for row in global_summary:
        log(f"  {row['band']:8s}: {row['best_config']:<45} Score: {row['best_score']:.4f}")
    
    return all_results


def run_clustering_analysis(eigen_dict: dict, kuramoto_dict: dict,
                            bands: list[str] = None,
                            min_k: int = 2, max_k: int = 15,
                            min_comps: int = 2, max_comps: int = 10,
                            filter_outliers: bool = True,
                            outlier_threshold: float = 2.0,
                            optimize: bool = False,
                            scaler_type: str = 'standard',
                            dim_reduction: str = 'pca',
                            cluster_method: str = 'kmeans',
                            distance_metric: str = 'euclidean',
                            use_top_eigenvalues: int = None) -> dict:
    """
    Ejecuta análisis de clustering para todas las bandas.
    
    Args:
        eigen_dict: Diccionario de eigenvalores por condición/banda
        kuramoto_dict: Diccionario de Kuramoto para filtrar outliers
        bands: Bandas a analizar (default: todas)
        min_k, max_k: Rango de clusters
        min_comps, max_comps: Rango de componentes PCA
        filter_outliers: Si filtrar outliers basado en Kuramoto
        outlier_threshold: Umbral para filtrado (desviaciones estándar)
        optimize: Si probar múltiples configuraciones automáticamente
        scaler_type: Tipo de escalador ('standard', 'minmax', 'robust', 'none')
        dim_reduction: Método de reducción ('pca', 'kernel_pca', 'none')
        cluster_method: Algoritmo de clustering ('kmeans', 'gmm', 'hierarchical')
        distance_metric: Métrica de distancia ('euclidean', 'cosine', 'correlation')
        use_top_eigenvalues: Usar solo los N eigenvalores más grandes
    
    Returns:
        Diccionario con resultados por banda
    """
    if bands is None:
        bands = BAND_LIST
    
    results = {}
    
    for band in bands:
        log(f"\nAnalizando banda: {band}")
        
        # Combinar datos de todas las condiciones
        all_eigen = []
        all_kuramoto = []
        condition_labels = []
        
        for cond in COND_LIST:
            eigen_data = eigen_dict.get(cond, {}).get(band, [])
            kuramoto_data = kuramoto_dict.get(cond, {}).get(band, [])
            
            if eigen_data:
                all_eigen.extend(eigen_data)
                condition_labels.extend([cond] * len(eigen_data))
                
                # Agregar Kuramoto (promediado por época)
                for k in kuramoto_data:
                    if isinstance(k, np.ndarray):
                        all_kuramoto.append(np.mean(k))
                    else:
                        all_kuramoto.append(k)
        
        if len(all_eigen) < min_k + 1:
            log(f"[WARN] Insuficientes datos para {band}: {len(all_eigen)} epochs")
            continue
        
        X = np.asarray(all_eigen)
        condition_labels = np.array(condition_labels)
        
        # Filtrar outliers si hay datos de Kuramoto
        if filter_outliers and len(all_kuramoto) == len(all_eigen):
            kuramoto_array = np.asarray(all_kuramoto)
            mask = reject_outliers(kuramoto_array, m=outlier_threshold)
            X = X[mask]
            condition_labels = condition_labels[mask]
            log(f"  Filtrado: {mask.sum()}/{len(mask)} epochs (outlier threshold={outlier_threshold}σ)")
        
        log(f"  Datos: {X.shape[0]} epochs, {X.shape[1]} features")
        
        # Buscar clustering óptimo
        if optimize:
            log(f"  Modo optimización: probando múltiples configuraciones...")
            result = find_best_configuration(
                X, min_k=min_k, max_k=max_k, 
                min_comps=min_comps, max_comps=max_comps
            )
        else:
            result = find_optimal_clustering(
                X, min_k=min_k, max_k=max_k, 
                min_comps=min_comps, max_comps=max_comps,
                scaler_type=scaler_type,
                dim_reduction=dim_reduction,
                cluster_method=cluster_method,
                distance_metric=distance_metric,
                use_top_eigenvalues=use_top_eigenvalues
            )
        
        result['data'] = X
        result['condition_labels'] = condition_labels
        result['band'] = band
        
        log(f"  Mejor configuración: {result['best_params']}")
        log(f"  Silhouette Score: {result['best_score']:.4f}")
        
        results[band] = result
    
    return results


# =============================================================================
# Visualización
# =============================================================================

def plot_silhouette_heatmap(result: dict, output_dir: Path = None, show: bool = True):
    """Genera heatmap de Silhouette scores."""
    band = result.get('band', 'Unknown')
    heatmap = result.get('heatmap')
    
    if heatmap is None:
        log(f"[WARN] No hay heatmap para {band}")
        return
    
    min_k = result.get('min_k', 2)
    max_k = result.get('actual_max_k', result.get('max_k', 15))
    min_comps = result.get('min_comps', 2)
    max_comps = result.get('actual_max_comps', result.get('max_comps', 10))
    best_params = result.get('best_params', {})
    best_score = result.get('best_score', 0)
    
    # Ajustar labels según tamaño real del heatmap
    n_rows, n_cols = heatmap.shape
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(heatmap, aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=ax, label='Silhouette Score')
    
    # Título con información de configuración
    config_info = best_params.get('cluster_method', 'kmeans')
    ax.set_title(f'Silhouette Score - Banda {band}\n'
                 f'Mejor: k={best_params.get("n_clusters", "?")}, '
                 f'PCA={best_params.get("n_components", "?")} '
                 f'(score={best_score:.3f}) [{config_info}]', fontsize=12)
    
    ax.set_xlabel('Número de Clusters (k)')
    ax.set_ylabel('Componentes PCA')
    
    # Labels correctos basados en tamaño real
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(range(min_k, min_k + n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(range(min_comps, min_comps + n_rows))
    
    # Anotar valores en el heatmap (solo si no es muy grande)
    if n_rows * n_cols <= 150:
        heatmap_max = heatmap.max() if heatmap.max() > 0 else 1
        for i in range(n_rows):
            for j in range(n_cols):
                value = heatmap[i, j]
                color = 'white' if value < 0.5 * heatmap_max else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                       color=color, fontsize=7)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = ensure_dir(output_dir)
        filepath = output_dir / f'silhouette_heatmap_{band}.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        log(f"Guardado: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_cluster_centers(result: dict, output_dir: Path = None, show: bool = True):
    """Visualiza los centros de los clusters (solo para KMeans y GMM con PCA estándar)."""
    band = result.get('band', 'Unknown')
    model = result.get('best_model') or result.get('best_kmeans')
    reducer = result.get('best_reducer') or result.get('best_pca')
    best_params = result.get('best_params', {})
    
    if model is None or reducer is None:
        log(f"[WARN] No hay modelo o reductor para plot de centros en {band}")
        return
    
    # Solo podemos obtener centros de KMeans y GMM
    # Y solo podemos hacer inverse_transform con PCA estándar
    cluster_method = best_params.get('cluster_method', 'kmeans')
    dim_reduction = best_params.get('dim_reduction', 'pca')
    
    if cluster_method == 'hierarchical':
        log(f"[INFO] Hierarchical clustering no tiene centros definidos, saltando plot para {band}")
        return
    
    if dim_reduction == 'kernel_pca':
        log(f"[INFO] Kernel PCA no soporta inverse_transform, saltando plot de centros para {band}")
        return
    
    # Obtener centros según el tipo de modelo
    try:
        if cluster_method == 'kmeans':
            centers = model.cluster_centers_
            n_clusters = model.n_clusters
        elif cluster_method == 'gmm':
            centers = model.means_
            n_clusters = model.n_components
        else:
            return
    except AttributeError:
        log(f"[WARN] No se pueden obtener centros del modelo para {band}")
        return
    
    # Transformar centros de vuelta al espacio original
    try:
        centers_original = reducer.inverse_transform(centers)
    except Exception as e:
        log(f"[WARN] No se puede hacer inverse_transform para {band}: {e}")
        return
    
    # Limitar a máximo 8 subplots para visualización clara
    n_clusters_show = min(n_clusters, 8)
    
    fig, axes = plt.subplots(1, n_clusters_show, figsize=(3.5 * n_clusters_show, 5))
    if n_clusters_show == 1:
        axes = [axes]
    
    scaler = MinMaxScaler()

    for i, ax in enumerate(axes):
        if i >= len(centers_original):
            break
        center = centers_original[i]
        
        # Normalizar para visualización
        center_norm = scaler.fit_transform(center.reshape(-1, 1)).flatten()
        colors = ['#e74c3c' if v > 0.5 else '#3498db' for v in center_norm]
        
        ax.barh(range(len(center)), center, color=colors, alpha=0.7)
        ax.set_title(f'Cluster {i+1}', fontsize=10)
        ax.set_xlabel('Valor', fontsize=9)
        if i == 0:
            ax.set_ylabel('ROI Index', fontsize=9)
        ax.tick_params(axis='both', labelsize=8)
    
    fig.suptitle(f'Centros de Clusters - Banda {band} ({cluster_method.upper()})', fontsize=12)
    plt.tight_layout()
    
    if output_dir:
        output_dir = ensure_dir(output_dir)
        filepath = output_dir / f'cluster_centers_{band}.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        log(f"Guardado: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_pca_scatter(result: dict, output_dir: Path = None, show: bool = True):
    """Scatter plot de los datos en espacio reducido con colores por cluster."""
    band = result.get('band', 'Unknown')
    labels = result.get('best_labels')
    data_reduced = result.get('best_data_reduced')
    best_params = result.get('best_params', {})
    best_score = result.get('best_score', 0)
    model = result.get('best_model') or result.get('best_kmeans')
    
    # Si no tenemos datos reducidos pre-calculados, intentar calcularlos
    if data_reduced is None:
        reducer = result.get('best_reducer') or result.get('best_pca')
        scaler = result.get('best_scaler')
        data = result.get('data')
        
        if reducer is None or data is None:
            log(f"[WARN] No hay datos para scatter plot de {band}")
            return
        
        try:
            if scaler is not None:
                data_reduced = reducer.transform(scaler.transform(data))
            else:
                data_reduced = reducer.transform(data)
        except Exception as e:
            log(f"[WARN] Error transformando datos para scatter plot de {band}: {e}")
            return
    
    # Si no tenemos labels, intentar predecirlos
    if labels is None and model is not None:
        try:
            labels = model.predict(data_reduced)
        except Exception:
            # Si falla predict (ej: AgglomerativeClustering), usamos fit_predict
            try:
                labels = model.fit_predict(data_reduced)
            except Exception as e:
                log(f"[WARN] No se pueden obtener labels para {band}: {e}")
                return
    
    if labels is None or data_reduced is None:
        log(f"[WARN] Datos insuficientes para scatter plot de {band}")
        return
    
    # Asegurar que tenemos al menos 2 dimensiones
    if data_reduced.shape[1] < 2:
        log(f"[WARN] Datos con menos de 2 dimensiones para scatter plot de {band}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot coloreado por cluster
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    cmap = plt.cm.get_cmap('tab10', max(n_clusters, 10))
    
    scatter = ax.scatter(data_reduced[:, 0], data_reduced[:, 1], 
                        c=labels, cmap=cmap, alpha=0.6, s=30)
    
    # Marcar centros si están disponibles
    cluster_method = best_params.get('cluster_method', 'kmeans')
    try:
        if cluster_method == 'kmeans' and hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], 
                      c='black', marker='X', s=200, edgecolors='white', linewidths=2,
                      label='Centros', zorder=5)
            ax.legend(loc='upper right')
        elif cluster_method == 'gmm' and hasattr(model, 'means_'):
            centers = model.means_
            ax.scatter(centers[:, 0], centers[:, 1], 
                      c='black', marker='X', s=200, edgecolors='white', linewidths=2,
                      label='Medias', zorder=5)
            ax.legend(loc='upper right')
    except Exception:
        pass  # Si falla, simplemente no mostramos centros
    
    dim_reduction = best_params.get('dim_reduction', 'pca')
    ax.set_xlabel(f'{dim_reduction.upper()} Componente 1')
    ax.set_ylabel(f'{dim_reduction.upper()} Componente 2')
    ax.set_title(f'Clustering - Banda {band}\n'
                 f'k={best_params.get("n_clusters", "?")}, '
                 f'{cluster_method.upper()}, '
                 f'Silhouette={best_score:.3f}')
    
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    
    if output_dir:
        output_dir = ensure_dir(output_dir)
        filepath = output_dir / f'pca_scatter_{band}.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        log(f"Guardado: {filepath}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


# =============================================================================
# Análisis de composición de clusters
# =============================================================================

def analyze_cluster_composition(labels: np.ndarray, condition_labels: np.ndarray,
                                 band: str, output_dir: Path = None, 
                                 show: bool = False) -> dict:
    """
    Analiza qué condiciones (DMT, EC, EO) componen cada cluster.
    
    Args:
        labels: Etiquetas de cluster para cada época
        condition_labels: Condición (DMT, EC, EO) de cada época
        band: Nombre de la banda
        output_dir: Directorio para guardar resultados
        show: Mostrar gráficos interactivamente
    
    Returns:
        Diccionario con estadísticas de composición
    """
    unique_clusters = np.unique(labels)
    unique_conditions = np.unique(condition_labels)
    n_clusters = len(unique_clusters)
    
    # Calcular composición de cada cluster
    composition = {}
    for cluster in unique_clusters:
        mask = labels == cluster
        cluster_conditions = condition_labels[mask]
        total = len(cluster_conditions)
        
        cluster_comp = {}
        for cond in unique_conditions:
            count = np.sum(cluster_conditions == cond)
            cluster_comp[cond] = {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
        
        composition[f'Cluster_{cluster}'] = {
            'total_epochs': int(total),
            'conditions': cluster_comp
        }
    
    # Calcular qué porcentaje de cada condición va a cada cluster
    condition_distribution = {}
    for cond in unique_conditions:
        cond_mask = condition_labels == cond
        cond_labels = labels[cond_mask]
        total_cond = len(cond_labels)
        
        cond_dist = {}
        for cluster in unique_clusters:
            count = np.sum(cond_labels == cluster)
            cond_dist[f'Cluster_{cluster}'] = {
                'count': int(count),
                'percentage': float(count / total_cond * 100)
            }
        
        condition_distribution[cond] = {
            'total_epochs': int(total_cond),
            'clusters': cond_dist
        }
    
    # Generar visualización
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Gráfico 1: Composición de cada cluster
    ax1 = axes[0]
    x = np.arange(n_clusters)
    width = 0.25
    colors = {'DMT': '#e74c3c', 'EC': '#3498db', 'EO': '#2ecc71'}
    
    for i, cond in enumerate(sorted(unique_conditions)):
        percentages = []
        for cluster in sorted(unique_clusters):
            pct = composition[f'Cluster_{cluster}']['conditions'].get(cond, {}).get('percentage', 0)
            percentages.append(pct)
        ax1.bar(x + i * width, percentages, width, label=cond, color=colors.get(cond, 'gray'))
    
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel('Porcentaje (%)')
    ax1.set_title(f'Composición de cada Cluster - Banda {band}\n(¿De qué condiciones está hecho cada cluster?)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([f'Cluster {c}' for c in sorted(unique_clusters)])
    ax1.legend(title='Condición')
    ax1.set_ylim(0, 100)
    
    # Agregar valores en las barras
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', fontsize=8)
    
    # Gráfico 2: Distribución de cada condición en clusters
    ax2 = axes[1]
    x2 = np.arange(len(unique_conditions))
    cluster_colors = plt.cm.Set2(np.linspace(0, 1, n_clusters))
    
    for i, cluster in enumerate(sorted(unique_clusters)):
        percentages = []
        for cond in sorted(unique_conditions):
            pct = condition_distribution.get(cond, {}).get('clusters', {}).get(f'Cluster_{cluster}', {}).get('percentage', 0)
            percentages.append(pct)
        ax2.bar(x2 + i * width, percentages, width, label=f'Cluster {cluster}', color=cluster_colors[i])
    
    ax2.set_xlabel('Condición')
    ax2.set_ylabel('Porcentaje (%)')
    ax2.set_title(f'Distribución por Condición - Banda {band}\n(¿A qué cluster va cada condición?)')
    ax2.set_xticks(x2 + width * (n_clusters - 1) / 2)
    ax2.set_xticklabels(sorted(unique_conditions))
    ax2.legend(title='Cluster')
    ax2.set_ylim(0, 100)
    
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%', fontsize=8)
    
    plt.tight_layout()
    
    if output_dir:
        output_dir = ensure_dir(output_dir)
        filepath = output_dir / f'cluster_composition_{band}.png'
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        log(f"Guardado: {filepath}")
        
        # Guardar también como CSV
        csv_data = []
        for cluster in sorted(unique_clusters):
            for cond in sorted(unique_conditions):
                csv_data.append({
                    'band': band,
                    'cluster': cluster,
                    'condition': cond,
                    'count': composition[f'Cluster_{cluster}']['conditions'].get(cond, {}).get('count', 0),
                    'pct_of_cluster': composition[f'Cluster_{cluster}']['conditions'].get(cond, {}).get('percentage', 0),
                    'pct_of_condition': condition_distribution.get(cond, {}).get('clusters', {}).get(f'Cluster_{cluster}', {}).get('percentage', 0)
                })
        
        csv_df = pd.DataFrame(csv_data)
        csv_df.to_csv(output_dir / f'cluster_composition_{band}.csv', index=False)
        log(f"Guardado: {output_dir}/cluster_composition_{band}.csv")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    # Imprimir resumen
    log(f"\n{'='*60}")
    log(f"COMPOSICIÓN DE CLUSTERS - Banda {band}")
    log(f"{'='*60}")
    
    for cluster in sorted(unique_clusters):
        log(f"\nCluster {cluster} ({composition[f'Cluster_{cluster}']['total_epochs']} epochs):")
        for cond in sorted(unique_conditions):
            pct = composition[f'Cluster_{cluster}']['conditions'].get(cond, {}).get('percentage', 0)
            count = composition[f'Cluster_{cluster}']['conditions'].get(cond, {}).get('count', 0)
            bar = '█' * int(pct / 5) + '░' * (20 - int(pct / 5))
            log(f"  {cond:4s}: {bar} {pct:5.1f}% ({count} epochs)")
    
    log(f"\n{'='*60}")
    log("INTERPRETACIÓN:")
    
    # Determinar qué cluster es "DMT-dominant"
    for cluster in sorted(unique_clusters):
        dmt_pct = composition[f'Cluster_{cluster}']['conditions'].get('DMT', {}).get('percentage', 0)
        ec_pct = composition[f'Cluster_{cluster}']['conditions'].get('EC', {}).get('percentage', 0)
        eo_pct = composition[f'Cluster_{cluster}']['conditions'].get('EO', {}).get('percentage', 0)
        
        if dmt_pct > 50:
            log(f"  Cluster {cluster}: Mayormente DMT ({dmt_pct:.1f}%) - Estado alterado")
        elif ec_pct + eo_pct > 70:
            log(f"  Cluster {cluster}: Mayormente Baseline ({ec_pct + eo_pct:.1f}% EC+EO) - Estado normal")
        else:
            log(f"  Cluster {cluster}: Mixto (DMT:{dmt_pct:.1f}%, EC:{ec_pct:.1f}%, EO:{eo_pct:.1f}%)")
    
    log(f"{'='*60}\n")
    
    return {
        'composition': composition,
        'condition_distribution': condition_distribution,
        'n_clusters': n_clusters,
        'band': band
    }


# =============================================================================
# Modelo de Markov (opcional)
# =============================================================================

def build_markov_model(sequence: list, threshold_percentile: float = None) -> dict:
    """
    Construye un modelo de Markov a partir de una secuencia de estados.
    
    Args:
        sequence: Lista de estados (etiquetas de clusters)
        threshold_percentile: Percentil para filtrar transiciones débiles
    
    Returns:
        Diccionario con modelo, estadísticas y matriz de adyacencia
    """
    from collections import Counter

    model = {}
    seq = list(sequence)
    
    for i in range(len(seq) - 1):
        current = seq[i]
        next_state = seq[i + 1]
        if current not in model:
            model[current] = []
        model[current].append(next_state)
    
    # Calcular probabilidades de transición
    model_probs = {}
    for state in model.keys():
        counts = Counter(model[state])
        total = sum(counts.values())
        model_probs[state] = {k: v / total for k, v in counts.items()}
    
    # Construir matriz de adyacencia
    states = sorted(set(seq))
    n_states = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}
    
    adj_matrix = np.zeros((n_states, n_states))
    for origin, destinations in model_probs.items():
        for dest, prob in destinations.items():
            if origin in state_to_idx and dest in state_to_idx:
                adj_matrix[state_to_idx[origin], state_to_idx[dest]] = prob
    
    if threshold_percentile is not None:
        threshold = np.percentile(adj_matrix[adj_matrix > 0], threshold_percentile)
        adj_matrix[adj_matrix < threshold] = 0
    
    return {
        'model': model,
        'probabilities': model_probs,
        'adjacency_matrix': adj_matrix,
        'states': states,
        'state_to_idx': state_to_idx
    }


# =============================================================================
# Predicción con Random Forest
# =============================================================================

def predict_experience(clustering_results: dict, output_dir: Path = None) -> dict:
    """
    Usa features de clustering para predecir experiencia subjetiva.
    
    Args:
        clustering_results: Resultados del análisis de clustering
        output_dir: Directorio para guardar resultados
    
    Returns:
        Diccionario con scores de predicción
    """
    # Cargar targets de experiencia
    target_file = SPECTRAL_DIR / "target.csv"
    labels_file = SPECTRAL_DIR / "target_labels.txt"
    
    if not target_file.exists() or not labels_file.exists():
        log(f"[WARN] Archivos de targets no encontrados en {SPECTRAL_DIR}")
        return {}
    
    targets = pd.read_csv(target_file, header=None)
    labels = pd.read_csv(labels_file, header=None)[0].tolist()
    
    log(f"\nPredicción de experiencia subjetiva")
    log(f"  Targets: {targets.shape[0]} sujetos, {len(labels)} variables")
    
    results = {}
    
    for band, result in clustering_results.items():
        if 'data' not in result or result['best_pca'] is None:
            continue
        
        data = result['data']
        pca = result['best_pca']
        kmeans = result['best_kmeans']
        
        # Reducir a nivel de sujeto (promedio de epochs por sujeto)
        # Nota: esto requiere mapeo sujeto->epochs que no tenemos aquí
        # Por ahora usamos los datos agregados
        
        X = pca.transform(data)
        
        # Solo si tenemos el mismo número de muestras que sujetos
        if X.shape[0] != targets.shape[0]:
            log(f"  [WARN] {band}: Dimensiones no coinciden ({X.shape[0]} vs {targets.shape[0]})")
            continue
        
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        band_results = {}
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        for i, label in enumerate(labels):
            y = targets.iloc[:, i].values
            try:
                scores = cross_val_score(reg, X_scaled, y, scoring='r2', cv=5)
                band_results[label] = {
                    'mean_r2': np.mean(scores),
                    'std_r2': np.std(scores)
                }
            except Exception as e:
                band_results[label] = {'error': str(e)}
        
        results[band] = band_results
        log(f"  {band}: Predicción completada")
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Análisis de clustering sobre matrices de sincronización EEG',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python clustering.py                           # Análisis completo
  python clustering.py --bands Alpha Theta       # Solo bandas específicas
  python clustering.py --max-k 10 --max-comps 8  # Limitar búsqueda
  python clustering.py --no-plots                # Sin visualizaciones
  python clustering.py --output-dir ./results    # Directorio de salida personalizado
        """
    )
    
    parser.add_argument(
        '--conditions', nargs='+', default=COND_LIST,
        help=f'Condiciones a analizar (default: {COND_LIST})'
    )
    parser.add_argument(
        '--bands', nargs='+', default=BAND_LIST,
        help=f'Bandas a analizar (default: {BAND_LIST})'
    )
    parser.add_argument(
        '--min-k', type=int, default=2,
        help='Número mínimo de clusters (default: 2)'
    )
    parser.add_argument(
        '--max-k', type=int, default=15,
        help='Número máximo de clusters (default: 15)'
    )
    parser.add_argument(
        '--min-comps', type=int, default=2,
        help='Mínimo de componentes PCA (default: 2)'
    )
    parser.add_argument(
        '--max-comps', type=int, default=10,
        help='Máximo de componentes PCA (default: 10)'
    )
    parser.add_argument(
        '--splits-idx', type=int, default=0,
        help='Índice del split temporal a usar (default: 0)'
    )
    parser.add_argument(
        '--no-filter-outliers', action='store_true',
        help='Desactivar filtrado de outliers'
    )
    parser.add_argument(
        '--outlier-threshold', type=float, default=2.0,
        help='Umbral para filtrado de outliers en desviaciones estándar (default: 2.0)'
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help='No generar visualizaciones'
    )
    parser.add_argument(
        '--show-plots', action='store_true',
        help='Mostrar plots interactivamente (default: solo guardar)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help=f'Directorio de salida (default: {CLUSTERING_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--predict', action='store_true',
        help='Ejecutar predicción de experiencia subjetiva'
    )
    parser.add_argument(
        '--workers', type=int, default=0,
        help='Número de workers para carga paralela (0 = todos los núcleos, 1 = secuencial)'
    )
    
    # Opciones de optimización
    parser.add_argument(
        '--optimize', action='store_true',
        help='Probar automáticamente múltiples configuraciones para mejor Silhouette'
    )
    parser.add_argument(
        '--scaler', type=str, default='standard',
        choices=['standard', 'minmax', 'robust', 'none'],
        help='Tipo de escalado de datos (default: standard)'
    )
    parser.add_argument(
        '--dim-reduction', type=str, default='pca',
        choices=['pca', 'kernel_pca', 'none'],
        help='Método de reducción de dimensionalidad (default: pca)'
    )
    parser.add_argument(
        '--cluster-method', type=str, default='kmeans',
        choices=['kmeans', 'gmm', 'hierarchical'],
        help='Algoritmo de clustering (default: kmeans)'
    )
    parser.add_argument(
        '--distance', type=str, default='euclidean',
        choices=['euclidean', 'cosine', 'correlation'],
        help='Métrica de distancia para Silhouette (default: euclidean)'
    )
    parser.add_argument(
        '--top-eigenvalues', type=int, default=None,
        help='Usar solo los N eigenvalores más grandes (default: todos)'
    )
    parser.add_argument(
        '--full-search', action='store_true',
        help='Búsqueda exhaustiva: prueba TODAS las 144 combinaciones en paralelo'
    )
    parser.add_argument(
        '--quick-search', action='store_true',
        help='Búsqueda rápida: prueba solo 12 configuraciones esenciales (~10x más rápido)'
    )
    parser.add_argument(
        '--load-workers', type=int, default=4,
        help='Workers para carga de archivos (I/O bound, default: 4)'
    )
    parser.add_argument(
        '--analysis-workers', type=int, default=4,
        help='Workers para análisis paralelo (default: 4, más causa problemas de memoria)'
    )
    parser.add_argument(
        '--low-memory', action='store_true',
        help='Modo bajo consumo de memoria: procesa secuencialmente y libera memoria agresivamente'
    )
    parser.add_argument(
        '--max-epochs', type=int, default=None,
        help='Limitar número de epochs por banda (submuestreo aleatorio). Útil para pruebas rápidas.'
    )
    parser.add_argument(
        '--analyze-composition', action='store_true',
        help='Analizar composición de clusters de resultados existentes (sin re-correr clustering)'
    )
    
    args = parser.parse_args()
    
    # Configurar directorio de salida
    output_dir = Path(args.output_dir) if args.output_dir else CLUSTERING_OUTPUT_DIR
    output_dir = ensure_dir(output_dir)
    
    # Modo análisis de composición (usa resultados existentes)
    if args.analyze_composition:
        log("=" * 60)
        log("ANÁLISIS DE COMPOSICIÓN DE CLUSTERS")
        log("=" * 60)
        log(f"Buscando resultados en: {output_dir}")
        
        for band in args.bands:
            band_dir = output_dir / band
            if not band_dir.exists():
                log(f"[WARN] No se encontró directorio para {band}")
                continue
            
            # Buscar el mejor resultado (rank1)
            rank1_dirs = list(band_dir.glob("rank1_*"))
            if not rank1_dirs:
                log(f"[WARN] No se encontró rank1 para {band}")
                continue
            
            result_file = rank1_dirs[0] / "clustering_result.pkl"
            if not result_file.exists():
                log(f"[WARN] No se encontró clustering_result.pkl para {band}")
                continue
            
            log(f"\n>>> Analizando {band}...")
            result = load_file(result_file)
            
            labels = result.get('best_labels')
            condition_labels = result.get('condition_labels')
            
            if labels is None or condition_labels is None:
                log(f"[WARN] {band}: No hay labels guardados. Re-corre --quick-search para generar.")
                continue
            
            analyze_cluster_composition(
                labels, 
                condition_labels, 
                band,
                output_dir=rank1_dirs[0],
                show=args.show_plots
            )
        
        log("\n" + "=" * 60)
        log("ANÁLISIS DE COMPOSICIÓN COMPLETADO")
        log("=" * 60)
        return
    
    # Configurar workers
    load_workers = args.load_workers if args.load_workers > 0 else 4
    analysis_workers = args.analysis_workers if args.analysis_workers > 0 else cpu_count()
    
    # Si se usa --workers, aplicar a carga (compatibilidad)
    if args.workers > 0:
        load_workers = args.workers
    
    # Modo full-search o quick-search
    if args.full_search or args.quick_search:
        quick_mode = args.quick_search
        n_configs = 12 if quick_mode else 144
        mode_name = "RÁPIDA (12 configs)" if quick_mode else "EXHAUSTIVA (144 configs)"
        
        log("=" * 60)
        log(f"BÚSQUEDA {mode_name} DE CLUSTERING EEG")
        log("=" * 60)
        log(f"Condiciones: {args.conditions}")
        log(f"Bandas: {args.bands}")
        log(f"Rango clusters: {args.min_k}-{args.max_k}")
        log(f"Rango PCA: {args.min_comps}-{args.max_comps}")
        log(f"Workers carga (I/O): {load_workers}")
        log(f"Workers análisis (CPU): {analysis_workers}")
        log(f"Directorio de salida: {output_dir}")
        log("=" * 60)
        
        # Cargar datos
        log("\n>>> PASO 1: Cargando datos de sincronización...")
        eigen_dict, kuramoto_dict = load_syncro_data(
            conditions=args.conditions,
            bands=args.bands,
            splits_idx=args.splits_idx,
            workers=load_workers
        )
        
        # Ejecutar búsqueda
        log(f"\n>>> PASO 2: Ejecutando búsqueda ({n_configs} configuraciones)...")
        all_results = run_full_search(
            eigen_dict, kuramoto_dict,
            bands=args.bands,
            min_k=args.min_k,
            max_k=args.max_k,
            min_comps=args.min_comps,
            max_comps=args.max_comps,
            filter_outliers=not args.no_filter_outliers,
            outlier_threshold=args.outlier_threshold,
            analysis_workers=analysis_workers,
            output_dir=output_dir,
            show_plots=args.show_plots,
            quick=quick_mode,
            max_epochs=args.max_epochs
        )
        
        log("\n" + "=" * 60)
        log(f"BÚSQUEDA {mode_name} COMPLETADA")
        log(f"Resultados guardados en: {output_dir}")
        log("=" * 60)
        return
    
    log("=" * 60)
    log("ANÁLISIS DE CLUSTERING EEG")
    log("=" * 60)
    log(f"Condiciones: {args.conditions}")
    log(f"Bandas: {args.bands}")
    log(f"Rango clusters: {args.min_k}-{args.max_k}")
    log(f"Rango PCA: {args.min_comps}-{args.max_comps}")
    log(f"Workers: {load_workers}")
    log(f"Directorio de salida: {output_dir}")
    log("=" * 60)
    
    # Cargar datos
    log("\n>>> PASO 1: Cargando datos de sincronización...")
    eigen_dict, kuramoto_dict = load_syncro_data(
        conditions=args.conditions,
        bands=args.bands,
        splits_idx=args.splits_idx,
        workers=load_workers
    )
    
    # Ejecutar clustering
    log("\n>>> PASO 2: Ejecutando análisis de clustering...")
    if args.optimize:
        log("Modo OPTIMIZACIÓN activado: probando múltiples configuraciones")
    
    results = run_clustering_analysis(
        eigen_dict, kuramoto_dict,
        bands=args.bands,
        min_k=args.min_k,
        max_k=args.max_k,
        min_comps=args.min_comps,
        max_comps=args.max_comps,
        filter_outliers=not args.no_filter_outliers,
        outlier_threshold=args.outlier_threshold,
        optimize=args.optimize,
        scaler_type=args.scaler,
        dim_reduction=args.dim_reduction,
        cluster_method=args.cluster_method,
        distance_metric=args.distance,
        use_top_eigenvalues=args.top_eigenvalues
    )
    
    # Guardar resultados
    log("\n>>> PASO 3: Guardando resultados...")
    
    # Guardar diccionarios procesados
    save_file(eigen_dict, output_dir, "eigen_all")
    save_file(kuramoto_dict, output_dir, "kuramoto_all")
    
    # Guardar resultados de clustering (sin objetos grandes)
    results_summary = {}
    for band, result in results.items():
        results_summary[band] = {
            'best_score': result['best_score'],
            'best_params': result['best_params'],
            'heatmap': result['heatmap'],
            'n_samples': result['data'].shape[0] if 'data' in result else 0
        }
    save_file(results_summary, output_dir, "clustering_summary")
    
    # Generar visualizaciones
    if not args.no_plots:
        log("\n>>> PASO 4: Generando visualizaciones...")
        plots_dir = ensure_dir(output_dir / "plots")
        
        for band, result in results.items():
            log(f"  Generando plots para banda {band}...")
            plot_silhouette_heatmap(result, output_dir=plots_dir, show=args.show_plots)
            plot_cluster_centers(result, output_dir=plots_dir, show=args.show_plots)
            plot_pca_scatter(result, output_dir=plots_dir, show=args.show_plots)
    
    # Predicción opcional
    if args.predict:
        log("\n>>> PASO 5: Predicción de experiencia...")
        prediction_results = predict_experience(results, output_dir=output_dir)
        if prediction_results:
            save_file(prediction_results, output_dir, "prediction_results")
    
    # Resumen final
    log("\n" + "=" * 60)
    log("RESUMEN DE RESULTADOS")
    log("=" * 60)
    
    for band, result in results.items():
        log(f"\n{band}:")
        log(f"  - Epochs analizados: {result['data'].shape[0]}")
        log(f"  - Mejor k: {result['best_params']['n_clusters']}")
        log(f"  - Mejor PCA componentes: {result['best_params']['n_components']}")
        log(f"  - Silhouette Score: {result['best_score']:.4f}")
    
    log("\n" + "=" * 60)
    log(f"Resultados guardados en: {output_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
