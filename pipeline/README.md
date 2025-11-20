# Pipeline de Análisis EEG

Esta carpeta contiene los scripts principales del pipeline de análisis de sincronización EEG.

## 📁 Archivos

### Scripts principales (en orden de ejecución):
1. **fwd.py** - Forward/Inverse solution + métricas básicas
2. **save_load_pickle.py** - Consolidación de datos por condición
3. **multi2pool2.py** - Filtrado por redes cerebrales
4. **calculate_syncro.py** - Análisis temporal con splits
5. **generate_order.py** - Cálculo de order parameter por red
6. **plot_order.py** - Visualización y gráficos
7. **pearson.py** - Correlaciones con cuestionarios
8. **clustering.py** - Identificación de estados cerebrales

### Archivos de configuración:
- **paths.py** - Configuración de rutas (apunta al directorio padre)
- **run_pipeline.sh** - Script para ejecutar el pipeline completo

## 🚀 Uso

### Ejecutar pipeline completo:

**Linux/Mac:**
```bash
cd /media/storage_hdd/dmt_fz/dmt/pipeline
./run_pipeline.sh
```

**Windows:**
```cmd
cd C:\ruta\a\dmt\pipeline
run_pipeline.bat
```

### Ejecutar scripts individuales:

```bash
cd /media/storage_hdd/dmt_fz/dmt/pipeline

# Paso 1: Procesamiento EEG (3-4 horas con 7 workers)
python fwd.py --jobs 0 --workers 7 --conditions DMT EC EO

# Paso 2a: Consolidar fases
python save_load_pickle.py --conditions DMT EC EO

# Paso 2b: Filtrar por redes (2-5 min)
python multi2pool2.py

# Paso 2c: Análisis temporal (10-15 min, opcional)
python calculate_syncro.py --workers 20 --conditions DMT EC EO

# Paso 2d: Order parameter (30 seg - 1 min)
python generate_order.py --workers 20 --conditions DMT EC EO

# Paso 3: Gráficos (5-10 min)
python plot_order.py --workers 20

# Paso 4: Correlaciones (3-5 min)
python pearson.py

# Paso 5: Clustering (2-4 horas, opcional)
python clustering.py
```

## 📂 Estructura de directorios

Los scripts en esta carpeta acceden a datos en el directorio padre:

```
dmt/                          (directorio padre)
├── pipeline/                 (esta carpeta)
│   ├── fwd.py
│   ├── multi2pool2.py
│   └── ...
├── EEG_CLEAN/               (datos de entrada)
├── fwd-inv-stc/             (resultados intermedios)
├── plot_order_results/      (gráficos)
├── pearson_results/         (correlaciones)
└── spectral_sources/        (datos de cuestionarios)
```

## 🔧 Configuración personalizada (opcional)

Si necesitás usar una ruta personalizada:

1. Copiar `paths_config.py.example` a `paths_config.py`
2. Editar y descomentar `CUSTOM_BASE_DIR` con tu ruta:
   ```python
   # Windows:
   CUSTOM_BASE_DIR = Path("D:/Proyectos/dmt_fz/dmt")
   
   # Linux/Mac:
   CUSTOM_BASE_DIR = Path("/home/usuario/proyectos/dmt_fz/dmt")
   ```
3. Los scripts automáticamente usarán esa ruta

## ⚠️ Notas importantes

1. **paths.py** detecta automáticamente el sistema operativo (Windows/Linux/Mac)
2. En Windows, usar `\` o `/` es indistinto (Path lo maneja automáticamente)
3. Los archivos generados se guardan en el directorio padre, NO en pipeline/
4. Asegurate de tener activado el ambiente conda: `conda activate dmt_fz`
5. El pipeline requiere ~150-200 GB de espacio en disco
6. Tiempo total estimado: 4-5 horas (sin clustering), 6-9 horas (con clustering)

## ✅ Correcciones aplicadas

Esta versión incluye las siguientes correcciones:

1. ✅ Diagonal de matriz de sincronización = 1.0 (en `fwd.py` y `calculate_syncro.py`)
2. ⚠️ **PENDIENTE**: Corregir `clustering.py` línea 67:
   ```python
   # CAMBIAR DE:
   eigen_dict[cond][band].append(np.diag(np.linalg.eigh(epoch)[1]))
   
   # A:
   eigen_dict[cond][band].append(np.linalg.eigh(epoch)[0])
   ```

## 📚 Documentación completa

Ver documentación en el directorio padre:
- `docs/README.md` - Descripción general del proyecto
- `docs/WORKFLOW.md` - Pipeline detallado paso a paso
- `docs/TECHNICAL_DETAILS.md` - Detalles matemáticos
- `docs/QUICKSTART.md` - Guía rápida

