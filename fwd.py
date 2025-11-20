import numpy as np

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator
from mne.minimum_norm import apply_inverse_epochs
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

from paths import EEG_CLEAN_DIR, RESULTS_DIR, ensure_dir


#%%

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
set_log_level("WARNING")


def log(message="", end="\n"):
    prefix = f"[PID {os.getpid()}] " if WORKER_COUNT > 1 else ""
    tqdm.write(f"{prefix}{message}", end=end)


WORKER_COUNT = 1


#%%

# Directorios de datos preprocesados (.set)
data_root = EEG_CLEAN_DIR
eo_dir = data_root / "EO"
ec_dir = data_root / "EC"
dmt_dir = data_root / "DMT"

eyes_open_files_list = sorted(eo_dir.glob("*.set"))
eyes_closed_files_list = sorted(ec_dir.glob("*.set"))
dmt_files_list = sorted(dmt_dir.glob("*.set"))

log(f"[SETUP] Archivos DMT (.set): {len(dmt_files_list)} detectados")
log(f"[SETUP] Archivos EC (.set):  {len(eyes_closed_files_list)} detectados")
log(f"[SETUP] Archivos EO (.set):  {len(eyes_open_files_list)} detectados")

#%%

import os.path as op

verbose = False

# fsaverage

fs_dir = fetch_fsaverage(verbose=verbose) # Download fsaverage files
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # Source space
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif') # Boundary Element Method (BEM) for forward modeling

# Cargar parcelas de Schaefer (necesario para source localization)
log("[SETUP] Cargando atlas de Schaefer (100 parcelas, 7 redes)")
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', verbose=verbose)
log(f"[SETUP] Atlas cargado: {len(labels)} parcelas")

#%%

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

def _prepare_epochs(epochs):
    epochs = epochs.copy()
    epochs.pick('eeg')
    epochs.apply_baseline((None, None), verbose=verbose)
    epochs.set_montage('standard_1020')
    epochs.set_eeg_reference(projection=True, verbose=verbose)
    epochs.apply_proj()
    return epochs

# Define frequencies of interest
fmin, fmax = 0., 70.
bandwidth = 4.  # bandwidth of the windows in Hz

def fwd_inv_stc(epochs, forward_n_jobs=None, preprocessed=False):
    epochs_proc = epochs if preprocessed else _prepare_epochs(epochs)
    
    noise_cov = mne.compute_covariance(epochs_proc, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=verbose)
    fwd = mne.make_forward_solution(
        epochs_proc.info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=5.0,
        n_jobs=forward_n_jobs,
        verbose=verbose,
    )
    inverse_operator = make_inverse_operator(epochs_proc.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=verbose)
    
    stc = apply_inverse_epochs(
        epochs_proc,
        inverse_operator,
        lambda2,
        method=method,
        pick_ori=None,
        label=None,
        return_generator=True,
        verbose=verbose,
    )
    return stc


#%%

freq_bands = {
    "Delta": [1, 4],
    "Theta": [4, 8],
    "Alpha": [8, 13],
    "Beta":  [13, 30],
    "Gamma": [30, 45]
    }

band_list = list(freq_bands.keys())


def filtered(signal, band, sfreq=500, n_jobs=None):
  l_freq = freq_bands[band][0]
  h_freq = freq_bands[band][1]
  filtered_signal = filter_data(
      data=signal,
      sfreq=sfreq,
      l_freq=l_freq,
      h_freq=h_freq,
      verbose=False,
      filter_length="auto",
      n_jobs=n_jobs,
  )
  return filtered_signal


def diff_ang(theta1, theta2, full_p=2*np.pi, abso=True):
  half_p = 0.5 * full_p
  fmod1 = np.fmod(theta2 - theta1 + half_p, full_p)
  fmod2 = np.fmod(fmod1 + full_p, full_p) - half_p
  if abso==True:
    return abs(fmod2) #abs(np.fmod(np.fmod(theta2 - theta1 + half_p, full_p) + full_p, full_p) - half_p)
  else:
    return fmod2


def hilbert_transform(band_signals, trim=False):
  signal_num = len(band_signals)
  samples = band_signals[0].shape[0]
  envelope_mat = np.zeros((signal_num, samples), dtype=np.float32)
  phase_mat = np.zeros((signal_num, samples), dtype=np.float32)

  for i, filtered_signal in enumerate(band_signals):
    analytic_signal = hilbert(filtered_signal)
    envelope = np.abs(analytic_signal)
    inst_phase = np.angle(analytic_signal)

    envelope_mat[i,:] = envelope
    phase_mat[i,:] = inst_phase
  
  if trim != False:
    envelope_mat = envelope_mat[:,trim:(samples-trim)]
    phase_mat = phase_mat[:,trim:(samples-trim)]
      
  return envelope_mat, phase_mat


def calculate_syncro(phase_mat):
  max_diff = np.pi * phase_mat.shape[1]
  size = phase_mat.shape[0]
  syncro_mat = np.zeros((size,size), dtype=np.float32)
  for i, j in comb(range(size), 2):
        signal1 = phase_mat[i,:]
        signal2 = phase_mat[j,:]
        value = 1 - (diff_ang(signal1,signal2).sum()/max_diff)
        syncro_mat[i,j] = value
        syncro_mat[j,i] = value
  # Diagonal = 1 (una señal tiene sincronización perfecta consigo misma)
  np.fill_diagonal(syncro_mat, 1.0)
  return syncro_mat


def eeg_pre(epochs_data, epoch, num_channels=24):
    eeg_signals = []
    for channel in range(num_channels):
      signal = epochs_data[epoch, channel, :]
      eeg_signals.append(signal)
    return eeg_signals


def stc_pre(epoch_data, labels):
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
    r = np.abs(euler_notation(phase).mean(axis=0))
    return r.astype(np.float32)


#%%

# Carpeta de salida
output_folder = ensure_dir(RESULTS_DIR)
for cond in ["DMT", "EC", "EO"]:
    ensure_dir(output_folder / cond)

markers_list = ["filtered_eeg", "phases_eeg", "amplitudes_eeg", "syncros_eeg", "kuramoto_eeg",
                "filtered_stc", "phases_stc", "amplitudes_stc", "syncros_stc", "kuramoto_stc"]

band_dict = {band:[] for band in band_list}
subject_dict = {marker:band_dict for marker in markers_list}

def do_the_math(file_name, condition="DMT", forward_n_jobs=None, filter_n_jobs=None):
    # Extraer nombre del sujeto del archivo
    file_path = Path(file_name)
    subject_number = file_path.name.replace("_ICA_pruned.set", "").replace("_", "-")
    
    log(f"\n{'='*70}")
    log(f"[SUBJECT] {subject_number} ({condition})")
    log(f"{'='*70}")
    
    # Leer archivo .set de EEGLAB (ya preprocesado con ICA)
    log("[1/4] Cargando épocas de EEGLAB")
    epochs_raw = mne.io.read_epochs_eeglab(file_name, montage_units='dm', verbose=False)
    epochs = _prepare_epochs(epochs_raw)
    epochs_data = epochs.get_data()
    num_epochs = epochs_data.shape[0]
    log(f"        ↳ {num_epochs} épocas disponibles")
    
    subject_data = {marker: {band:[] for band in band_list} for marker in markers_list}
    
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
    
    # Guardar resultado
    log("[3/4] Persistiendo resultados en disco")
    fname = output_folder / condition / f"phases-{subject_number}.pkl"
    
    with open(fname, 'wb') as handle:
        pickle.dump(subject_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    log(f"[4/4] Pipeline completado: {subject_number} ({condition})")
    log(f"        ↳ Archivo generado: {fname}\n")


#%%


def _init_worker(worker_total):
    global WORKER_COUNT
    WORKER_COUNT = worker_total


def _process_subject(file_name, condition, forward_n_jobs, filter_n_jobs):
    return do_the_math(
        file_name,
        condition=condition,
        forward_n_jobs=forward_n_jobs,
        filter_n_jobs=filter_n_jobs,
    )


def run_pipeline(
    max_subjects_per_condition=1,
    conditions=("DMT", "EC", "EO"),
    jobs=1,
    workers=None,
):
    """
    Ejecuta el pipeline completo limitando la cantidad de sujetos por condición.

    Parameters
    ----------
    max_subjects_per_condition : int or None
        Número máximo de sujetos a procesar por condición. Si es None o <= 0, se procesan todos.
    conditions : iterable
        Condiciones a procesar (por defecto DMT, EC, EO).
    jobs : int
        Presupuesto total de “jobs” para distribuir entre procesos y threads. Usa 0 o negativo para auto = todos los cores.
    forward_n_jobs : int or None
        Override opcional para `make_forward_solution`. Si es None se deriva de `jobs`.
    filter_n_jobs : int or None
        Override opcional para `filter_data`. Si es None se deriva de `jobs`.
    workers : int or None
        Override opcional para procesos en paralelo (por sujeto). Si es None se deriva de `jobs`.
    """
    try:
        max_subjects_per_condition = int(max_subjects_per_condition)
    except (TypeError, ValueError):
        max_subjects_per_condition = 1

    if max_subjects_per_condition <= 0:
        max_subjects_per_condition = None

    try:
        jobs = int(jobs)
    except (TypeError, ValueError):
        jobs = 1
    if jobs <= 0:
        jobs = cpu_count()

    try:
        workers = None if workers is None else int(workers)
    except (TypeError, ValueError):
        workers = None
    if workers is not None and workers <= 0:
        workers = None

    condition_map = {
        "DMT": dmt_files_list,
        "EC": eyes_closed_files_list,
        "EO": eyes_open_files_list,
    }

    tasks = []

    for condition in conditions:
        files = condition_map.get(condition, [])
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

    if workers is None:
        workers = min(jobs, len(tasks)) if len(tasks) else 1
    else:
        workers = min(workers, len(tasks)) if len(tasks) else 1
    workers = max(1, workers)

    global WORKER_COUNT
    WORKER_COUNT = workers

    forward_n_jobs = max(1, jobs // workers)
    filter_n_jobs = max(1, jobs // workers)

    log(f"[JOBS] total={jobs}, procesos={workers}, forward_n_jobs={forward_n_jobs}, filter_n_jobs={filter_n_jobs}")

    if workers == 1:
        for file_name, condition in tasks:
            do_the_math(
                file_name,
                condition=condition,
                forward_n_jobs=forward_n_jobs,
                filter_n_jobs=filter_n_jobs,
            )
    else:
        ctx = get_context("spawn")
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(workers,),
        ) as pool:
            pool.starmap(
                _process_subject,
                [
                    (file_name, condition, forward_n_jobs, filter_n_jobs)
                    for file_name, condition in tasks
                ],
            )


def save_file(data, fname, folder):
    folder = ensure_dir(folder)
    with open(Path(folder) / fname, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_file(fname, folder):
    with open(Path(folder) / fname, 'rb') as handle:
        return pickle.load(handle)


#%%

dmt_epochs_raw = mne.io.read_epochs_eeglab(dmt_files_list[0], montage_units='dm', verbose=False)
epochs = dmt_epochs_raw.set_montage('standard_1020')
ch_names = epochs.get_montage().ch_names
montage_coords_2d = np.array(list(epochs.get_montage()._get_ch_pos().values()))[:,:2]

# trans = mne.coreg.estimate_head_mri_t('fsaverage', subjects_dir)
# epochs = epochs.get_montage().apply_trans(trans)

mapping = {i:ch_names[i] for i in range(24)}
eeg_coords_2d = {ch_names[i]:montage_coords_2d[i,:] for i in range(24)}

#%%

replace_dict = {"7Networks_":"",
"RH_": "RH ", "LH_": "LH ", "-rh":"", "-lh":"",

"DorsAttn_": "DAN ", #"Dorsal Attention ",
"Default_": "DMN ", #"Default ",
"Limbic_": "LN ", #"Limbic ",
"SalVentAttn_": "SVAN ", #"Salience/Ventral Attention ",
"SomMot_": "SMN ", #"Somatomotor ",
"Vis_": "VN ", #"Visual ",
"Cont_": "FPN ", #"Frontoparietal ",

"Post_":  "Posterior ",
"Temp_": "Temporal ",
"Par_": "Parietal ",

"Cing_": "Cingulate ",
"Med_": "Medial ",

"PFC_": "PFC ", #"Prefrontal Cortex ",
"PFCv_": "Prefrontal Ventral ",
"PFCl_": "Lateral PFC ", #"Lateral Prefrontal Cortex ",
"PFCmp_": "Medial PFC ", #"Medial Posterior Prefrontal Cortex ",
"PFCdPFCm_": "Prefrontal Dorsal Medial ",
"OFC_": "Orbito-Frontal ", #Cortex

"pCun_": "Precuneus ",
"FrOperIns_": "Frontal Operculum Insula ",
"ParOper_": "Parietal Operculum ",

"pCunPCC_": "Precuneus/Posterior Cingulate ", #Cortex
"PcunCing": "Precuneus Cingulate ",

"TempOccPar_": "Tempro-Occipital-Parietal ",
"TempPole_": "Temporal Pole ",
"TempPar_": "Tempro-Parietal ",

"FEF_": "Frontal Eye Fields ",
"PrCv_": "Precentral Ventral ",

"_":""}

replace_dict_short = {"7Networks_":"",
"RH_": "", "LH_": "", "-rh":"", "-lh":"",

"DorsAttn_": "", #"Dorsal Attention ",
"Default_": "", #"Default ",
"Limbic_": "", #"Limbic ",
"SalVentAttn_": "", #"Salience/Ventral Attention ",
"SomMot_": "SMN", #"Somatomotor ",
"Vis_": "VN", #"Visual ",
"Cont_": "", #"Frontoparietal ",

"_":""}

dict_networks = {
  "DAN": "Dorsal Attention Network (DAN)",
  "DMN": "Default Mode Network (DMN)",
  "LN": "Limbic Network (LN)",
  "SVA": "Salience/Ventral Attention Network (SVAN)",
  "SMN": "Somatomotor Network (SMN)",
  "VN": "Visual Network (VN)",
  "FPN": "Frontoparietal Network (FPN)"}

def replacer(string, dictio):
  for i in dictio.items():
      string = string.replace(i[0], i[1])
  return string

#%%

label_names = [replacer(label.name, replace_dict) for label in labels if not label.name.startswith('Background')]
label_names_short = [replacer(label.name, replace_dict_short) for label in labels if not label.name.startswith('Background')]

node_colors = [label.color for label in labels if not label.name.startswith('Background')]

stc_coords_3d = np.asarray([label.pos.mean(axis=0) for label in labels if not label.name.startswith('Background')])
stc_coords_2d = stc_coords_3d[:,:2]

# label_network = [x[:6] for x in label_names]

# labels = mne.read_labels_from_annot('fsaverage', parc='aparc', verbose=verbose)
# label_names = [label.name for label in labels if not label.name.startswith('unknown')]
# lh_labels = [name for name in label_names if name.endswith('lh')]
# rh_labels = [name for name in label_names if name.endswith('rh')]
# node_colors = [label.color for label in labels if not label.name.startswith('unknown')]
# stc_coords_2d = np.asarray([label.pos.mean(axis=0) for label in labels])[:,:2]

# sort_order = [(lh_labels+rh_labels).index(label) for label in label_names]
# label_names_sorted = [label_names[i] for i in sort_order]
# node_colors_sorted = [node_colors[i] for i in sort_order]

#%%

# # Chord plot order
# # We reorder the labels based on their location in the left hemi

# # Get the y-location of the label
# label_ypos_lh = []
# for name in lh_labels:
#     idx = label_names.index(name)
#     ypos = np.mean(labels[idx].pos[:, 1])
#     label_ypos_lh.append(ypos)

# # Reorder the labels based on their location
# left_labels = [label for (yp, label) in sorted(zip(label_ypos_lh, lh_labels))]

# # For the right hemi
# right_labels = [label[:-2]+'rh' for label in left_labels if label[:-2]+'rh' in rh_labels]

# # Save the plot order
# node_order = left_labels[::-1] + right_labels

#%%

# Guardar metadata (ejecutar solo una vez)
metadata_folder = RESULTS_DIR

dump = []
dump.append(node_colors)
dump.append(label_names)
dump.append(label_names_short)
dump.append(stc_coords_3d)
dump.append(ch_names)
dump.append(mapping)
dump.append(eeg_coords_2d)

save_file(dump, "extra.pkl", metadata_folder)
log(f"[SETUP] Metadata guardada en {metadata_folder / 'extra.pkl'}")

#%%

extras_path = data_root / 'extras.pickle'
file = open(extras_path, 'wb')
pickle.dump(node_colors, file)
pickle.dump(label_names, file)
pickle.dump(label_names_short, file)
#pickle.dump(node_order, file)
pickle.dump(stc_coords_3d, file)
pickle.dump(ch_names, file)
pickle.dump(mapping, file)
pickle.dump(eeg_coords_2d, file)
file.close()


#%%
# with open(folder+'subjects_phases_eeg.pickle', 'rb') as handle:
#     subjects_phases_eeg = pickle.load(handle)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de fuentes EEG → métricas de sincronía.")
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Cantidad máxima de sujetos a procesar por condición (1 por defecto). Use 0 o negativo para procesar todos.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=("DMT", "EC", "EO"),
        default=("DMT", "EC", "EO"),
        help="Condiciones a procesar (por defecto DMT, EC y EO).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Presupuesto total de jobs (procesos/hilos). 1 por defecto, use 0 o negativo para todos los cores.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Procesos en paralelo (por sujeto). Si no se pasa, se deriva de --jobs.",
    )
    args = parser.parse_args()
    run_pipeline(
        max_subjects_per_condition=args.max_subjects,
        conditions=args.conditions,
        jobs=args.jobs,
        workers=args.workers,
    )
