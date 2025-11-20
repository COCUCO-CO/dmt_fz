# 🪟 Configuración para Windows

Esta guía te ayuda a configurar y ejecutar el pipeline en Windows.

## 📋 Requisitos previos

1. **Python 3.10+** instalado
2. **Anaconda/Miniconda** instalado
3. Ambiente conda `dmt_fz` configurado con las dependencias

## 🔧 Configuración inicial

### Opción 1: Usar la ubicación actual (recomendado)

Si tu proyecto está en una ubicación como:
- `C:\Users\TuUsuario\Documents\dmt_fz\dmt`
- `D:\Proyectos\dmt_fz\dmt`
- Etc.

**No necesitás hacer nada**, `paths.py` automáticamente detectará la ubicación.

### Opción 2: Configurar ruta personalizada

Si querés especificar una ruta diferente:

1. Copiar `paths_config.py.example` → `paths_config.py`
2. Editar `paths_config.py` y descomentar/modificar:
   ```python
   CUSTOM_BASE_DIR = Path("C:/Users/TuUsuario/Desktop/dmt_fz/dmt")
   ```
3. Guardar el archivo

## ▶️ Ejecutar el pipeline

### Método 1: Usando el script .bat (recomendado)

1. Abrir **Anaconda Prompt** o **Command Prompt**
2. Navegar al directorio pipeline:
   ```cmd
   cd C:\ruta\a\tu\proyecto\dmt\pipeline
   ```
3. Ejecutar:
   ```cmd
   run_pipeline.bat
   ```

### Método 2: Ejecutar scripts individuales

1. Abrir **Anaconda Prompt**
2. Activar ambiente:
   ```cmd
   conda activate dmt_fz
   ```
3. Navegar al directorio pipeline:
   ```cmd
   cd C:\ruta\a\tu\proyecto\dmt\pipeline
   ```
4. Ejecutar scripts uno por uno:
   ```cmd
   python fwd.py --jobs 0 --workers 7 --conditions DMT EC EO
   python save_load_pickle.py --conditions DMT EC EO
   python multi2pool2.py
   python generate_order.py --workers 20 --conditions DMT EC EO
   python plot_order.py --workers 20
   python pearson.py
   python clustering.py
   ```

## 🔍 Verificar configuración

Para verificar que las rutas están correctas:

```cmd
cd C:\ruta\a\tu\proyecto\dmt\pipeline
conda activate dmt_fz
python paths.py
```

Deberías ver algo como:
```
======================================================================
CONFIGURACIÓN DE RUTAS DEL PIPELINE
======================================================================
Sistema Operativo: Windows 10
Python: 3.10.x

BASE_DIR:        C:\ruta\a\tu\proyecto\dmt
EEG_CLEAN_DIR:   C:\ruta\a\tu\proyecto\dmt\EEG_CLEAN
RESULTS_DIR:     C:\ruta\a\tu\proyecto\dmt\fwd-inv-stc
SPECTRAL_DIR:    C:\ruta\a\tu\proyecto\dmt\spectral_sources
======================================================================

Verificación de directorios:
  BASE_DIR             ✓ Existe
  EEG_CLEAN_DIR        ✓ Existe
  RESULTS_DIR          ✓ Existe
  SPECTRAL_DIR         ✓ Existe
======================================================================
```

## ⚠️ Problemas comunes

### "conda no encontrado"

**Solución**: Usar **Anaconda Prompt** en lugar de Command Prompt normal.

### "No se pudo activar el ambiente dmt_fz"

**Solución**: Crear el ambiente primero:
```cmd
conda create -n dmt_fz python=3.10
conda activate dmt_fz
pip install mne numpy scipy matplotlib pandas scikit-learn sklearn-extra optuna tqdm pymatreader
```

### "FileNotFoundError: EEG_CLEAN no encontrado"

**Solución**: 
1. Verificar que los datos estén en la ubicación correcta
2. O configurar `paths_config.py` con la ruta correcta

### Los paths usan \ en lugar de /

**No hay problema**: Python y `pathlib.Path` manejan automáticamente `\` en Windows y `/` en Linux/Mac. Internamente todo funciona igual.

## 📊 Diferencias con Linux

| Aspecto | Linux | Windows |
|---------|-------|---------|
| Separador de paths | `/` | `\` (automático) |
| Script principal | `./run_pipeline.sh` | `run_pipeline.bat` |
| Activar conda | `conda activate dmt_fz` | `conda activate dmt_fz` (igual) |
| Multiprocessing | Funciona igual | Funciona igual |
| Performance | Similar | Similar |

## 🎯 Recomendaciones

1. **Usar SSD** si es posible (más rápido I/O)
2. **Cerrar programas pesados** durante la ejecución
3. **No suspender/hibernar** mientras corre el pipeline
4. **Tener 16+ GB RAM** (recomendado)
5. **150-200 GB de espacio libre** en disco

## 📝 Notas adicionales

- El script `.bat` es equivalente al `.sh` de Linux
- Todos los algoritmos funcionan igual en Windows y Linux
- Los archivos `.pkl` generados son compatibles entre sistemas operativos
- Podés copiar los resultados de Windows a Linux y viceversa sin problemas

