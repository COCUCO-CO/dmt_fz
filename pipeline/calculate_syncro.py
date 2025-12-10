

import pickle
import re
from pathlib import Path

from paths import RESULTS_DIR, ensure_dir


def save_file(data, folder, file):
    folder = ensure_dir(folder)
    with open(folder / f"{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(Path(file), 'rb') as handle:
        return pickle.load(handle)


def log(message: str):
    print(f"[SYNCRO] {message}")


def sanitize_label(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


def setup_condition(condition: str) -> bool:
    global CURRENT_CONDITION, SOURCE_FILE, subjects_phases, phases_eeg_all
    global phases_stc_all, subject_count, available_bands, subject_labels

    CURRENT_CONDITION = condition
    SOURCE_FILE = RESULTS_DIR / condition / f"subject_phases_{condition}.pkl"

    if not SOURCE_FILE.exists():
        log(f"[WARN] Source file not found for condition '{condition}': {SOURCE_FILE}")
        return False

    subjects_phases = load_file(SOURCE_FILE)
    phases_eeg_all = subjects_phases.get("phases_eeg", [])
    phases_stc_all = subjects_phases.get("phases_stc", [])
    subject_count = len(phases_eeg_all)

    if subject_count == 0:
        log(f"[WARN] No subjects found in {SOURCE_FILE}")
        return False

    if len(phases_stc_all) != subject_count:
        log(
            f"[WARN] EEG/STC subject count mismatch in {SOURCE_FILE}: "
            f"{subject_count} vs {len(phases_stc_all)}"
        )

    available_bands = sorted(phases_eeg_all[0].keys())
    subject_labels = subjects_phases.get(
        "subjects",
        [f"S{idx + 1:02d}-{condition}" for idx in range(subject_count)],
    )

    log(
        f"Loaded subject phases for condition '{condition}' from {SOURCE_FILE} | "
        f"subjects={subject_count}, bands={available_bands}"
    )
    return True


CURRENT_CONDITION: str | None = None
SOURCE_FILE: Path | None = None
subjects_phases: dict = {}
phases_eeg_all = []
phases_stc_all = []
subject_count = 0
available_bands: list[str] = []
subject_labels: list[str] = []


#%%


import numpy as np
from itertools import combinations as comb


def diff_ang(theta1, theta2, full_p=2*np.pi, abso=True):
  half_p = 0.5 * full_p
  fmod1 = np.fmod(theta2 - theta1 + half_p, full_p)
  fmod2 = np.fmod(fmod1 + full_p, full_p) - half_p
  if abso==True:
    return abs(fmod2) #abs(np.fmod(np.fmod(theta2 - theta1 + half_p, full_p) + full_p, full_p) - half_p)
  else:
    return fmod2


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


def order_parameter(phase):
    r = np.abs(np.exp(1j*phase).mean(axis=0))
    return r.astype(np.float32)


#%%

from tqdm import tqdm

from copy import deepcopy
import argparse
from multiprocessing import Pool, cpu_count

markers_list = ["syncros_eeg", "kuramoto_eeg", "syncros_stc", "kuramoto_stc"]
band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
band_dict = {band:[[] for i in range(10)] for band in deepcopy(band_list)}
subject_dict = {marker:deepcopy(band_dict) for marker in deepcopy(markers_list)}


def do_the_math(subj):
    subject_label = (
        subject_labels[subj] if subj < len(subject_labels) else f"S{subj + 1:02d}"
    )
    # Extract clean subject ID (e.g., "S01" from "S01-DMT.set" or "S01_DMT")
    # Match pattern like S01, S02, etc. at the beginning
    import re
    subject_match = re.match(r'^(S\d+)', subject_label, re.IGNORECASE)
    if subject_match:
        subject_id = subject_match.group(1).upper()  # e.g., "S01"
    else:
        # Fallback to sanitized label or default
        subject_id = sanitize_label(subject_label) or f"S{subj + 1:02d}"
    reference_band = "Alpha"
    phases_eeg_subject = phases_eeg_all[subj]
    phases_stc_subject = phases_stc_all[subj] if subj < len(phases_stc_all) else {}
    if phases_stc_subject and not isinstance(phases_stc_subject, dict):
        log(f"[WARN] Subject {subject_label}: unexpected STC structure ({type(phases_stc_subject)}). Skipping subject.")
        return
    reference_epochs = phases_eeg_subject.get(reference_band, [])

    if not reference_epochs:
        log(f"[WARN] Subject {subject_label}: band '{reference_band}' not found or empty. Skipping.")
        return

    num_epochs = len(reference_epochs)
    num_samples = reference_epochs[0].shape[1] if num_epochs else 0

    log(
        f"Subject {subject_label} ({subj + 1}/{subject_count}): "
        f"{num_epochs} epochs, {num_samples} samples/epoch (reference band '{reference_band}')"
    )

    subject_data = deepcopy(subject_dict)

    for band in band_list:
        phases_eeg = phases_eeg_subject.get(band)
        phases_stc = phases_stc_subject.get(band) if isinstance(phases_stc_subject, dict) else None

        if not phases_eeg or not phases_stc:
            log(f"[WARN] Subject {subject_label}: band '{band}' missing. Skipping band.")
            continue

        log(
            f"  - Band {band}: epochs={len(phases_eeg)}, "
            f"splits=2..11, samples_per_epoch={num_samples}"
        )

        effective_epochs = min(num_epochs, len(phases_eeg), len(phases_stc))
        if effective_epochs == 0:
            log(f"[WARN] Subject {subject_label}: band '{band}' has no epochs. Skipping band.")
            continue

        for splits in range(2, 12):
            ruler = np.linspace(0, num_samples, splits, dtype="int")
            desc = f"{subject_label} | {band} | splits={splits}"

            for epoch_idx in tqdm(range(effective_epochs), desc=desc, leave=False):
                for window_idx in range(len(ruler) - 1):
                    start = ruler[window_idx]
                    end = ruler[window_idx + 1]
                    if end - start < 2:
                        continue

                    phase_eeg_epoch = np.asarray(phases_eeg[epoch_idx])
                    phase_stc_epoch = np.asarray(phases_stc[epoch_idx])

                    if phase_eeg_epoch.ndim != 2 or phase_stc_epoch.ndim != 2:
                        continue

                    band_samples = min(
                        phase_eeg_epoch.shape[1],
                        phase_stc_epoch.shape[1],
                        num_samples,
                    )
                    window_end = min(end, band_samples)

                    if window_end - start < 2:
                        continue

                    phase_eeg = phase_eeg_epoch[:, start:window_end]
                    phase_stc = phase_stc_epoch[:, start:window_end]

                    # EEG
                    syncro_eeg = calculate_syncro(phase_eeg)
                    r_eeg = order_parameter(phase_eeg)

                    subject_data["syncros_eeg"][band][splits - 2].append(syncro_eeg)
                    subject_data["kuramoto_eeg"][band][splits - 2].append(r_eeg)

                    # Sources
                    syncro_stc = calculate_syncro(phase_stc)
                    r_stc = order_parameter(phase_stc)

                    subject_data["syncros_stc"][band][splits - 2].append(syncro_stc)
                    subject_data["kuramoto_stc"][band][splits - 2].append(r_stc)

    # Save with consistent naming: syncro-{subject_id}-{condition}.pkl
    output_folder = RESULTS_DIR / CURRENT_CONDITION
    output_file = f"syncro-{subject_id}-{CURRENT_CONDITION}"
    save_file(subject_data, output_folder, output_file)
    log(f"Saved metrics for {subject_label} → {output_folder / (output_file + '.pkl')}")

    
#%%

def process_condition(condition: str, workers: int):
    if not setup_condition(condition):
        return

    subjects = range(subject_count)

    if workers is None or workers <= 0:
        workers = cpu_count()

    log(
        f"Starting processing for condition '{condition}' "
        f"({subject_count} subjects) with workers={workers}"
    )

    if workers == 1:
        for subj in subjects:
            do_the_math(subj)
    else:
        with Pool(processes=workers) as pool:
            pool.map(do_the_math, subjects)


#%%


def main(workers: int | None = None, conditions: list[str] | None = None):
    if conditions is None or not conditions:
        conditions = ["DMT", "EC", "EO"]

    for condition in conditions:
        process_condition(condition, workers)


#%%

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calcular métricas de sincronía por sujeto.")
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Número de procesos en paralelo (1 por defecto, use 0 para todos los núcleos).",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["DMT", "EC", "EO"],
        help="Condiciones a procesar (por defecto DMT, EC, EO).",
    )
    args = parser.parse_args()
    main(workers=args.workers, conditions=args.conditions)