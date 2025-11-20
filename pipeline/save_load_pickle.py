import argparse
import pickle
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

from paths import RESULTS_DIR, ensure_dir


def save_file(data, file_stem):
    """Guardar `data` como pickle utilizando Path."""
    path = Path(file_stem)
    if path.suffix != ".pkl":
        path = path.with_suffix(".pkl")
    ensure_dir(path.parent)
    with open(path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_path):
    """Cargar pickle usando Path."""
    path = Path(file_path)
    with open(path, "rb") as handle:
        return pickle.load(handle)


def consolidate_condition(condition: str):
    cond_dir = RESULTS_DIR / condition
    if not cond_dir.exists():
        print(f"[PHASES] Condición '{condition}' sin carpeta en {cond_dir}.")
        return

    file_list = sorted(p for p in cond_dir.glob("phases-*.pkl"))
    if not file_list:
        print(f"[PHASES] Condición '{condition}' sin archivos phases-*.pkl en {cond_dir}.")
        return

    subjects_phases = {"phases_eeg": [], "phases_stc": [], "subjects": []}

    print(f"[PHASES] Condición '{condition}': {len(file_list)} archivos phases-*.pkl")
    for file in tqdm(file_list, desc=f"{condition} phases"):
        subject_data = load_file(file)
        subjects_phases["phases_eeg"].append(subject_data["phases_eeg"])
        subjects_phases["phases_stc"].append(subject_data["phases_stc"])
        subjects_phases["subjects"].append(file.stem.replace("phases-", ""))

    out_path = cond_dir / f"subject_phases_{condition}.pkl"
    save_file(subjects_phases, out_path)
    print(
        f"[PHASES] Guardado {out_path} "
        f"(sujetos={len(subjects_phases['phases_eeg'])}, "
        f"primero={subjects_phases['subjects'][0] if subjects_phases['subjects'] else 'n/a'})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidar phases-*.pkl por condición en subject_phases_<COND>.pkl"
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["DMT", "EC", "EO"],
        help="Condiciones a procesar (por defecto DMT, EC, EO).",
    )
    args = parser.parse_args()

    conditions = args.conditions
    workers = min(len(conditions), max(1, cpu_count()))
    with Pool(processes=workers) as pool:
        pool.map(consolidate_condition, conditions)