from pathlib import Path
import platform
import sys
import os

# Intentar cargar configuración personalizada (opcional)
try:
    from paths_config import CUSTOM_BASE_DIR
    if CUSTOM_BASE_DIR is not None:
        BASE_DIR = Path(CUSTOM_BASE_DIR).resolve()
        print(f"[PATHS] Using custom BASE_DIR from paths_config.py: {BASE_DIR}")
    else:
        CUSTOM_BASE_DIR = None
except ImportError:
    CUSTOM_BASE_DIR = None

# Si no hay configuración personalizada, usar detección automática
if CUSTOM_BASE_DIR is None:
    # BASE_DIR apunta al proyecto raíz (dmt_fz/) desde pipeline/
    # Subir 1 nivel: paths.py -> pipeline -> dmt_fz
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Información del sistema operativo
    system = platform.system()
    
    if system == "Windows":
        # En Windows, Path automáticamente usa backslashes
        print(f"[PATHS] Windows detected: Using {BASE_DIR}")
    elif system == "Linux":
        print(f"[PATHS] Linux detected: Using {BASE_DIR}")
    elif system == "Darwin":
        print(f"[PATHS] macOS detected: Using {BASE_DIR}")
    else:
        print(f"[PATHS] {system} detected: Using {BASE_DIR}")

# =============================================================================
# INPUT/OUTPUT DIRECTORY CONFIGURATION
# =============================================================================
# Check for environment variables (set by UI)
# This allows each pipeline run to have its own input/output directories

# INPUT: where to read EEG data from
_custom_input = os.environ.get('PIPELINE_INPUT_DIR')
if _custom_input:
    EEG_CLEAN_DIR = Path(_custom_input)
    print(f"[PATHS] Using custom input dir: {EEG_CLEAN_DIR}")
else:
    EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"

# OUTPUT: where pipeline results go
_custom_output = os.environ.get('PIPELINE_OUTPUT_DIR')
if _custom_output:
    PIPELINE_OUTPUT_BASE = Path(_custom_output)
    print(f"[PATHS] Using custom output dir: {PIPELINE_OUTPUT_BASE}")
else:
    PIPELINE_OUTPUT_BASE = None

# Configuración de directorios (Path maneja automáticamente / vs \ según el OS)
EEG_DIR = BASE_DIR / "EEG"

# RESULTS_DIR: where pipeline outputs go
if PIPELINE_OUTPUT_BASE:
    RESULTS_DIR = PIPELINE_OUTPUT_BASE
else:
    RESULTS_DIR = BASE_DIR / "fwd-inv-stc"

SPECTRAL_DIR = BASE_DIR / "spectral_sources"
EEGNET_DIR = BASE_DIR / "EEGNet"
RESULTS_PLOTS_DIR = BASE_DIR / "results_plots"
FRAMES_DIR = BASE_DIR / "frames"
VISUALIZATIONS_DIR = BASE_DIR / "visualizations"


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist and return the resolved Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Mostrar rutas al importar (solo si se ejecuta directamente)
if __name__ == "__main__":
    print("\n" + "="*70)
    print("CONFIGURACIÓN DE RUTAS DEL PIPELINE")
    print("="*70)
    print(f"Sistema Operativo: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"\nBASE_DIR:           {BASE_DIR}")
    print(f"EEG_CLEAN_DIR:      {EEG_CLEAN_DIR}")
    print(f"RESULTS_DIR:        {RESULTS_DIR}")
    print(f"SPECTRAL_DIR:       {SPECTRAL_DIR}")
    print(f"VISUALIZATIONS_DIR: {VISUALIZATIONS_DIR}")
    print("="*70)
    
    # Verificar existencia de directorios críticos
    print("\nVerificación de directorios:")
    for name, path in [
        ("BASE_DIR", BASE_DIR),
        ("EEG_CLEAN_DIR", EEG_CLEAN_DIR),
        ("RESULTS_DIR", RESULTS_DIR),
        ("SPECTRAL_DIR", SPECTRAL_DIR),
    ]:
        exists = "✓ Existe" if path.exists() else "✗ No existe"
        print(f"  {name:20s} {exists}")
    print("="*70 + "\n")
