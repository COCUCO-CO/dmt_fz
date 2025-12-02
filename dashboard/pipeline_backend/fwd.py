#!/usr/bin/env python3
"""
Pipeline de fuentes EEG → métricas de sincronía.
Refactorizado para ser importable sin ejecutar código pesado.
"""

import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse_epochs
from mne.utils import set_log_level
from mne.filter import filter_data
from scipy.signal import hilbert
from tqdm import tqdm
import pickle
from itertools import combinations as comb
import argparse
import os
from multiprocessing import get_context, cpu_count
from pathlib import Path
import os.path as op
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
set_log_level("WARNING")

# Global state
WORKER_COUNT = 1
_INITIALIZED = False
_LABELS = None
_FS_DIR = None
_SUBJECTS_DIR = None
_SRC = None
_BEM = None


def log(message="", end="\n"):
    """Print log message with optional PID prefix for multiprocessing."""
    import sys
    prefix = f"[PID {os.getpid()}] " if WORKER_COUNT > 1 else ""
    print(f"{prefix}{message}", end=end, flush=True)
    sys.stdout.flush()


# =============================================================================
# Lazy initialization - only load heavy resources when needed
# =============================================================================

def _init_fsaverage():
    """Initialize fsaverage resources (lazy loading)."""
    global _INITIALIZED, _LABELS, _FS_DIR, _SUBJECTS_DIR, _SRC, _BEM
    
    if _INITIALIZED:
        return
    
    log("[SETUP] Inicializando fsaverage...")
    _FS_DIR = fetch_fsaverage(verbose=False)
    _SUBJECTS_DIR = op.dirname(_FS_DIR)
    _SRC = op.join(_FS_DIR, 'bem', 'fsaverage-ico-5-src.fif')
    _BEM = op.join(_FS_DIR, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')
    
    log("[SETUP] Cargando atlas de Schaefer (100 parcelas, 7 redes)")
    _LABELS = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', verbose=False)
    log(f"[SETUP] Atlas cargado: {len(_LABELS)} parcelas")
    
    _INITIALIZED = True


def get_labels():
    """Get labels, initializing if needed."""
    _init_fsaverage()
    return _LABELS


def get_src():
    """Get source space path, initializing if needed."""
    _init_fsaverage()
    return _SRC


def get_bem():
    """Get BEM path, initializing if needed."""
    _init_fsaverage()
    return _BEM


# =============================================================================
# File scanning
# =============================================================================

def scan_files(data_root):
    """Scan for .set files in the data directory."""
    eo_dir = data_root / "EO"
    ec_dir = data_root / "EC"
    dmt_dir = data_root / "DMT"
    
    dmt_files = sorted(dmt_dir.glob("*.set")) if dmt_dir.exists() else []
    ec_files = sorted(ec_dir.glob("*.set")) if ec_dir.exists() else []
    eo_files = sorted(eo_dir.glob("*.set")) if eo_dir.exists() else []
    
    return {
        "DMT": dmt_files,
        "EC": ec_files,
        "EO": eo_files,
    }


# =============================================================================
# EEG Processing Functions
# =============================================================================

freq_bands = {
    "Delta": [1, 4],
    "Theta": [4, 8],
    "Alpha": [8, 13],
    "Beta":  [13, 30],
    "Gamma": [30, 45]
}
band_list = list(freq_bands.keys())

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2


def _prepare_epochs(epochs):
    """Prepare epochs for source localization."""
    epochs = epochs.copy()
    epochs.pick('eeg')
    epochs.apply_baseline((None, None), verbose=False)
    epochs.set_montage('standard_1020')
    epochs.set_eeg_reference(projection=True, verbose=False)
    epochs.apply_proj()
    return epochs


def fwd_inv_stc(epochs, forward_n_jobs=None, preprocessed=False):
    """Compute forward solution, inverse operator, and source time courses."""
    _init_fsaverage()
    
    epochs_proc = epochs if preprocessed else _prepare_epochs(epochs)
    
    noise_cov = mne.compute_covariance(epochs_proc, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=False)
    fwd = mne.make_forward_solution(
        epochs_proc.info,
        trans='fsaverage',
        src=get_src(),
        bem=get_bem(),
        eeg=True,
        mindist=5.0,
        n_jobs=forward_n_jobs,
        verbose=False,
    )
    inverse_operator = make_inverse_operator(epochs_proc.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=False)
    
    stc = apply_inverse_epochs(
        epochs_proc,
        inverse_operator,
        lambda2,
        method=method,
        pick_ori=None,
        label=None,
        return_generator=True,
        verbose=False,
    )
    return stc


def filtered(signal, band, sfreq=500, n_jobs=None):
    """Apply bandpass filter."""
    l_freq = freq_bands[band][0]
    h_freq = freq_bands[band][1]
    return filter_data(
        data=signal,
        sfreq=sfreq,
        l_freq=l_freq,
        h_freq=h_freq,
        verbose=False,
        filter_length="auto",
        n_jobs=n_jobs,
    )


def diff_ang(theta1, theta2, full_p=2*np.pi, abso=True):
    """Calculate angular difference."""
    half_p = 0.5 * full_p
    fmod1 = np.fmod(theta2 - theta1 + half_p, full_p)
    fmod2 = np.fmod(fmod1 + full_p, full_p) - half_p
    return abs(fmod2) if abso else fmod2


def hilbert_transform(band_signals, trim=False):
    """Apply Hilbert transform to get envelope and phase."""
    signal_num = len(band_signals)
    samples = band_signals[0].shape[0]
    envelope_mat = np.zeros((signal_num, samples), dtype=np.float32)
    phase_mat = np.zeros((signal_num, samples), dtype=np.float32)

    for i, filtered_signal in enumerate(band_signals):
        analytic_signal = hilbert(filtered_signal)
        envelope_mat[i, :] = np.abs(analytic_signal)
        phase_mat[i, :] = np.angle(analytic_signal)
    
    if trim:
        envelope_mat = envelope_mat[:, trim:(samples-trim)]
        phase_mat = phase_mat[:, trim:(samples-trim)]
      
    return envelope_mat, phase_mat


def calculate_syncro(phase_mat):
    """Calculate synchronization matrix from phase data."""
    max_diff = np.pi * phase_mat.shape[1]
    size = phase_mat.shape[0]
    syncro_mat = np.zeros((size, size), dtype=np.float32)
    for i, j in comb(range(size), 2):
        signal1 = phase_mat[i, :]
        signal2 = phase_mat[j, :]
        value = 1 - (diff_ang(signal1, signal2).sum() / max_diff)
        syncro_mat[i, j] = value
        syncro_mat[j, i] = value
    np.fill_diagonal(syncro_mat, 1.0)
    return syncro_mat


def eeg_pre(epochs_data, epoch, num_channels=24):
    """Extract EEG signals for an epoch."""
    return [epochs_data[epoch, channel, :] for channel in range(num_channels)]


def stc_pre(epoch_data, labels):
    """Extract source time course signals for an epoch."""
    stc_signals = []
    for label in labels:
        try:
            label_data = epoch_data.in_label(label).data
            stc_signals.append(label_data.mean(axis=0))
        except:
            pass
    return stc_signals


euler_notation = np.vectorize(lambda x: np.exp(1j*x))

def order_parameter(phase):
    """Calculate Kuramoto order parameter."""
    r = np.abs(euler_notation(phase).mean(axis=0))
    return r.astype(np.float32)


# =============================================================================
# Main processing functions
# =============================================================================

markers_list = ["filtered_eeg", "phases_eeg", "amplitudes_eeg", "syncros_eeg", "kuramoto_eeg",
                "filtered_stc", "phases_stc", "amplitudes_stc", "syncros_stc", "kuramoto_stc"]


def do_the_math(file_name, condition, output_folder, forward_n_jobs=None, filter_n_jobs=None, max_epochs=None):
    """Process a single subject file."""
    _init_fsaverage()
    labels = get_labels()
    
    file_path = Path(file_name)
    subject_number = file_path.name.replace("_ICA_pruned.set", "").replace("_", "-")
    
    log(f"\n{'='*70}")
    log(f"[SUBJECT] {subject_number} ({condition})")
    log(f"{'='*70}")
    
    # Load epochs
    log("[1/4] Cargando épocas de EEGLAB")
    epochs_raw = mne.io.read_epochs_eeglab(str(file_name), montage_units='dm', verbose=False)
    epochs = _prepare_epochs(epochs_raw)
    epochs_data = epochs.get_data()
    num_epochs = epochs_data.shape[0]
    
    # Limit epochs if specified
    if max_epochs and max_epochs > 0 and max_epochs < num_epochs:
        log(f"        ↳ {num_epochs} épocas disponibles (limitando a {max_epochs})")
        num_epochs = max_epochs
        epochs_data = epochs_data[:max_epochs]
    else:
        log(f"        ↳ {num_epochs} épocas disponibles")
    
    subject_data = {marker: {band: [] for band in band_list} for marker in markers_list}
    
    log("[2/4] Calculando STC base y procesando bandas de frecuencia")
    stc_generator = fwd_inv_stc(epochs, forward_n_jobs=forward_n_jobs, preprocessed=True)
    stc_signals_by_epoch = []
    for epoch_idx in range(num_epochs):
        try:
            stc_epoch = next(stc_generator)
        except StopIteration:
            log(f"[WARN] El generador de STC finalizó antes de completar las {num_epochs} épocas EEG")
            break
        stc_signals_by_epoch.append(np.asarray(stc_pre(stc_epoch, labels)))
    
    num_stc_epochs = len(stc_signals_by_epoch)
    log(f"        ↳ Series STC generadas: {num_stc_epochs} (épocas EEG: {num_epochs})")
    if num_stc_epochs != num_epochs:
        log(f"[WARN] Mismatch entre STC ({num_stc_epochs}) y EEG ({num_epochs}); se usará el mínimo común")

    effective_epochs = min(num_epochs, num_stc_epochs)

    for band in band_list:
        log(f"        • Banda {band}: generando métricas")
        log("            ↳ Calculando Hilbert y sincronía (EEG/STC)")
        
        for epoch in tqdm(range(effective_epochs), desc=f"            Epochs ({band})", leave=False):
            # EEG
            signals_eeg = np.asarray(eeg_pre(epochs_data, epoch))
            filtered_eeg = filtered(signals_eeg, band, n_jobs=filter_n_jobs)
            envelope_eeg, phase_eeg = hilbert_transform(filtered_eeg, trim=100)
            syncro_eeg = calculate_syncro(phase_eeg)
            r_eeg = order_parameter(phase_eeg)
            
            subject_data["filtered_eeg"][band].append(filtered_eeg.astype(np.float32))
            subject_data["phases_eeg"][band].append(phase_eeg)
            subject_data["amplitudes_eeg"][band].append(envelope_eeg)
            subject_data["syncros_eeg"][band].append(syncro_eeg)
            subject_data["kuramoto_eeg"][band].append(r_eeg)
            
            # Sources
            signals_stc = stc_signals_by_epoch[epoch]
            filtered_stc = filtered(signals_stc, band, n_jobs=filter_n_jobs)
            amplitude_stc, phase_stc = hilbert_transform(filtered_stc, trim=100)
            syncro_stc = calculate_syncro(phase_stc)
            r_stc = order_parameter(phase_stc)
            
            subject_data["filtered_stc"][band].append(filtered_stc.astype(np.float32))
            subject_data["phases_stc"][band].append(phase_stc)
            subject_data["amplitudes_stc"][band].append(amplitude_stc)
            subject_data["syncros_stc"][band].append(syncro_stc)
            subject_data["kuramoto_stc"][band].append(r_stc)
    
    # Save results
    log("[3/4] Persistiendo resultados en disco")
    cond_folder = output_folder / condition
    cond_folder.mkdir(parents=True, exist_ok=True)
    fname = cond_folder / f"phases-{subject_number}.pkl"
    
    with open(fname, 'wb') as handle:
        pickle.dump(subject_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    log(f"[4/4] Pipeline completado: {subject_number} ({condition})")
    log(f"        ↳ Archivo generado: {fname}\n")
    
    return fname


def _init_worker(worker_total):
    """Initialize worker with worker count."""
    global WORKER_COUNT
    WORKER_COUNT = worker_total


def _process_subject(args):
    """Process a single subject (for multiprocessing)."""
    file_name, condition, output_folder, forward_n_jobs, filter_n_jobs, max_epochs = args
    return do_the_math(
        file_name,
        condition=condition,
        output_folder=Path(output_folder),
        forward_n_jobs=forward_n_jobs,
        filter_n_jobs=filter_n_jobs,
        max_epochs=max_epochs,
    )


def run_pipeline(
    data_root,
    output_folder,
    max_subjects_per_condition=None,
    conditions=("DMT", "EC", "EO"),
    jobs=0,
    workers=None,
    max_epochs=None,
):
    """
    Run the complete pipeline.
    
    Parameters
    ----------
    data_root : Path
        Directory containing EEG_CLEAN data with DMT/, EC/, EO/ subdirectories
    output_folder : Path
        Directory to save results
    max_subjects_per_condition : int or None
        Maximum subjects per condition. None or <= 0 means all.
    conditions : tuple
        Conditions to process
    jobs : int
        Total job budget. 0 means auto (all cores)
    workers : int or None
        Number of parallel workers
    max_epochs : int or None
        Maximum epochs per subject. None or <= 0 means all.
    """
    data_root = Path(data_root)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Parse parameters
    try:
        max_subjects_per_condition = int(max_subjects_per_condition) if max_subjects_per_condition else None
    except (TypeError, ValueError):
        max_subjects_per_condition = None
    if max_subjects_per_condition is not None and max_subjects_per_condition <= 0:
        max_subjects_per_condition = None

    try:
        jobs = int(jobs)
    except (TypeError, ValueError):
        jobs = 0
    if jobs <= 0:
        jobs = cpu_count()

    try:
        workers = int(workers) if workers else None
    except (TypeError, ValueError):
        workers = None
    if workers is not None and workers <= 0:
        workers = None

    try:
        max_epochs = int(max_epochs) if max_epochs else None
    except (TypeError, ValueError):
        max_epochs = None
    if max_epochs is not None and max_epochs <= 0:
        max_epochs = None

    # Scan files
    file_map = scan_files(data_root)
    log(f"[SETUP] Archivos DMT (.set): {len(file_map['DMT'])} detectados")
    log(f"[SETUP] Archivos EC (.set):  {len(file_map['EC'])} detectados")
    log(f"[SETUP] Archivos EO (.set):  {len(file_map['EO'])} detectados")
    
    # Build task list
    tasks = []
    for condition in conditions:
        files = file_map.get(condition, [])
        if not files:
            log(f"[SKIP] Sin archivos disponibles para la condición {condition}")
            continue

        to_process = files if max_subjects_per_condition is None else files[:max_subjects_per_condition]
        log(f"[RUN] {condition}: {len(to_process)} de {len(files)} sujetos serán procesados")

        for file_name in to_process:
            tasks.append((file_name, condition))

    if not tasks:
        log("[INFO] No hay sujetos para procesar con la configuración dada")
        return

    # Configure workers
    if workers is None:
        workers = min(jobs, len(tasks)) if len(tasks) else 1
    else:
        workers = min(workers, len(tasks)) if len(tasks) else 1
    workers = max(1, workers)

    global WORKER_COUNT
    WORKER_COUNT = workers

    forward_n_jobs = max(1, jobs // workers)
    filter_n_jobs = max(1, jobs // workers)

    epochs_info = f", max_epochs={max_epochs}" if max_epochs else ""
    log(f"[JOBS] total={jobs}, procesos={workers}, forward_n_jobs={forward_n_jobs}, filter_n_jobs={filter_n_jobs}{epochs_info}")
    log(f"[OUTPUT] {output_folder}")

    # Initialize fsaverage before multiprocessing (download if needed)
    _init_fsaverage()

    # Run pipeline
    if workers == 1:
        for file_name, condition in tasks:
            do_the_math(
                file_name,
                condition=condition,
                output_folder=output_folder,
                forward_n_jobs=forward_n_jobs,
                filter_n_jobs=filter_n_jobs,
                max_epochs=max_epochs,
            )
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(workers,),
        ) as pool:
            pool.map(
                _process_subject,
                [
                    (str(file_name), condition, str(output_folder), forward_n_jobs, filter_n_jobs, max_epochs)
                    for file_name, condition in tasks
                ],
            )
    
    # Save metadata
    save_metadata(output_folder, data_root)


def save_metadata(output_folder, data_root):
    """Save metadata files."""
    _init_fsaverage()
    labels = get_labels()
    
    log("[SETUP] Guardando metadata...")
    
    # Get channel info from first available file
    file_map = scan_files(data_root)
    sample_file = None
    for cond in ["DMT", "EC", "EO"]:
        if file_map[cond]:
            sample_file = file_map[cond][0]
            break
    
    if not sample_file:
        log("[WARN] No se encontraron archivos para extraer metadata de canales")
        return
    
    epochs_raw = mne.io.read_epochs_eeglab(str(sample_file), montage_units='dm', verbose=False)
    epochs = epochs_raw.set_montage('standard_1020')
    ch_names = epochs.get_montage().ch_names
    montage_coords_2d = np.array(list(epochs.get_montage()._get_ch_pos().values()))[:, :2]
    
    mapping = {i: ch_names[i] for i in range(min(24, len(ch_names)))}
    eeg_coords_2d = {ch_names[i]: montage_coords_2d[i, :] for i in range(min(24, len(ch_names)))}
    
    # Label processing
    replace_dict = {
        "7Networks_": "", "RH_": "RH ", "LH_": "LH ", "-rh": "", "-lh": "",
        "DorsAttn_": "DAN ", "Default_": "DMN ", "Limbic_": "LN ",
        "SalVentAttn_": "SVAN ", "SomMot_": "SMN ", "Vis_": "VN ", "Cont_": "FPN ",
        "Post_": "Posterior ", "Temp_": "Temporal ", "Par_": "Parietal ",
        "Cing_": "Cingulate ", "Med_": "Medial ", "PFC_": "PFC ",
        "PFCv_": "Prefrontal Ventral ", "PFCl_": "Lateral PFC ",
        "PFCmp_": "Medial PFC ", "PFCdPFCm_": "Prefrontal Dorsal Medial ",
        "OFC_": "Orbito-Frontal ", "pCun_": "Precuneus ",
        "FrOperIns_": "Frontal Operculum Insula ", "ParOper_": "Parietal Operculum ",
        "pCunPCC_": "Precuneus/Posterior Cingulate ", "PcunCing": "Precuneus Cingulate ",
        "TempOccPar_": "Tempro-Occipital-Parietal ", "TempPole_": "Temporal Pole ",
        "TempPar_": "Tempro-Parietal ", "FEF_": "Frontal Eye Fields ",
        "PrCv_": "Precentral Ventral ", "_": ""
    }
    
    def replacer(string, dictio):
        for k, v in dictio.items():
            string = string.replace(k, v)
        return string
    
    label_names = [replacer(label.name, replace_dict) for label in labels if not label.name.startswith('Background')]
    label_names_short = [replacer(label.name, {"7Networks_": "", "RH_": "", "LH_": "", "-rh": "", "-lh": "",
                                                "DorsAttn_": "", "Default_": "", "Limbic_": "",
                                                "SalVentAttn_": "", "SomMot_": "SMN", "Vis_": "VN",
                                                "Cont_": "", "_": ""})
                         for label in labels if not label.name.startswith('Background')]
    
    node_colors = [label.color for label in labels if not label.name.startswith('Background')]
    stc_coords_3d = np.asarray([label.pos.mean(axis=0) for label in labels if not label.name.startswith('Background')])
    
    # Save extra.pkl
    dump = [node_colors, label_names, label_names_short, stc_coords_3d, ch_names, mapping, eeg_coords_2d]
    output_folder = Path(output_folder)
    with open(output_folder / "extra.pkl", 'wb') as handle:
        pickle.dump(dump, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    log(f"[SETUP] Metadata guardada en {output_folder / 'extra.pkl'}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    from paths import EEG_CLEAN_DIR, RESULTS_DIR
    
    parser = argparse.ArgumentParser(description="Pipeline de fuentes EEG → métricas de sincronía.")
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Cantidad máxima de sujetos a procesar por condición. Use 0 para todos.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("DMT", "EC", "EO"),
        default=("DMT", "EC", "EO"),
        help="Condiciones a procesar.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Presupuesto total de jobs. Use 0 para auto.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Procesos en paralelo. Si no se pasa, se deriva de --jobs.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Máximo de epochs por sujeto. Use 0 para todos.",
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        data_root=EEG_CLEAN_DIR,
        output_folder=RESULTS_DIR,
        max_subjects_per_condition=args.max_subjects,
        conditions=args.conditions,
        jobs=args.jobs,
        workers=args.workers,
        max_epochs=args.max_epochs,
    )
