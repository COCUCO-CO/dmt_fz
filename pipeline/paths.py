from pathlib import Path
import platform
import sys

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
    # BASE_DIR apunta al directorio padre (dmt/) desde pipeline/
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

# Configuración de directorios (Path maneja automáticamente / vs \ según el OS)
EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"
EEG_DIR = BASE_DIR / "EEG"
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
