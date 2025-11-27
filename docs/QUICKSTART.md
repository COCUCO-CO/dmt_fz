# Guía de Inicio Rápido

## 🚀 Para Empezar en 5 Minutos

### 1. ¿Qué hace este proyecto?
Analiza **sincronización de fases en EEG** durante experiencias psicodélicas (DMT) para correlacionar patrones cerebrales con experiencias subjetivas reportadas en cuestionarios.

### 2. Archivos más importantes

```
dmt_fz/
├── pipeline/                    # Procesamiento de datos
│   ├── fwd.py              ⭐ PASO 1: Pipeline principal (genera phases-*.pkl)
│   ├── multi2pool2.py      ⭐ PASO 2a: Filtra por redes (genera order_all-*.pkl)
│   ├── generate_order.py   ⭐ PASO 2b: Calcula order parameter (genera order-*.pkl)
│   ├── build_order_data.py ⭐ PASO 2c: Genera datos agregados
│   ├── pearson.py          📈 Correlaciones con cuestionarios
│   └── clustering.py       🎯 Identificación de estados
│
└── viz_scripts/                 # Visualización
    └── plot_order.py       📊 Histogramas y estadísticas de Kuramoto
```

### 3. Flujo básico

```
Datos EEG (.set)
    ↓
fwd.py → phases-*.pkl (fases + sincronización)
    ↓
multi2pool2.py → order_all-*.pkl (DataFrames de fases por red)
    ↓
generate_order.py → order-*.pkl (Order parameter por red)
    ↓
build_order_data.py → r_kuramoto_*.pkl (Datos agregados)
    ↓
    ├→ pearson.py → Correlaciones con experiencias
    └→ viz_scripts/plot_order.py → Gráficos finales
```

---

## 📖 Comandos Esenciales

### Setup Inicial (una sola vez)

```bash
# 1. Activar ambiente conda
conda activate dmt_fz

# 2. Verificar rutas
cd /media/storage_hdd/dmt_fz
python -c "from pipeline.paths import RESULTS_DIR, EEG_CLEAN_DIR; print(f'EEG: {EEG_CLEAN_DIR}\\nResults: {RESULTS_DIR}')"
```

### Pipeline Completo (29 sujetos)

```bash
cd /media/storage_hdd/dmt_fz

# PASO 1: Procesar EEG (3-4 horas)
python pipeline/fwd.py --jobs 0 --workers 7 --conditions DMT EC EO

# PASO 2a: Filtrar por redes (2-5 min)
python pipeline/multi2pool2.py

# PASO 2b: Calcular order parameter (30 seg - 1 min)
python pipeline/generate_order.py --workers 20 --conditions DMT EC EO

# PASO 2c: Generar datos agregados (2-5 min)
python pipeline/build_order_data.py --build-all

# PASO 3: Generar gráficos (5-10 min)
python viz_scripts/plot_order.py --workers 20

# PASO 4: Correlaciones (3-5 min)
python pipeline/pearson.py

# OPCIONAL: Clustering (2-4 horas)
python pipeline/clustering.py
```

### Pipeline Rápido (testing con 3 sujetos)

```bash
cd /media/storage_hdd/dmt_fz

# PASO 1: Procesar solo 3 sujetos (30-40 min)
python pipeline/fwd.py --max-subjects 3 --workers 3 --conditions DMT

# PASO 2a: Filtrar por redes (1 min)
python pipeline/multi2pool2.py

# PASO 2b: Calcular order parameter (10 seg)
python pipeline/generate_order.py --workers 10 --conditions DMT

# PASO 2c: Datos agregados
python pipeline/build_order_data.py --build-epochs-mean

# PASO 3: Gráficos (1-2 min)
python viz_scripts/plot_order.py --max-subjects 3

# PASO 4: Correlaciones
python pipeline/pearson.py
```

---

## 🔑 Conceptos Clave

### Parámetro de Orden de Kuramoto (r)
```
r = |promedio(exp(i·θⱼ))|

Interpretación:
- r = 0: Desincronizado (fases aleatorias)
- r = 1: Perfectamente sincronizado
- Usualmente: 0.1 - 0.6 en EEG
```

### Bandas de Frecuencia
```
Delta:  1-4 Hz    (sueño profundo)
Theta:  4-8 Hz    (meditación, creatividad)
Alpha:  8-13 Hz   (relajación, ojos cerrados)
Beta:   13-30 Hz  (atención, concentración)
Gamma:  30-45 Hz  (integración, consciencia)
```

### Redes Cerebrales (Schaefer Atlas)
```
FPN:  Frontoparietal       (control ejecutivo)
DMN:  Default Mode         (mente en reposo)
DAN:  Dorsal Attention     (atención dirigida)
SVA:  Salience/Ventral     (detección de relevancia)
LN:   Limbic               (emociones)
SMN:  Somatomotor          (movimiento)
VN:   Visual               (visión)
```

---

## 📁 Estructura de Archivos

### Entrada (NO incluidos en repo)
```
EEG_CLEAN/
├── DMT/S01-DMT_ICA_pruned.set    # Durante DMT
├── EC/S01-EC_ICA_pruned.set      # Baseline ojos cerrados
├── EO/S01-EO_ICA_pruned.set      # Baseline ojos abiertos
└── rejected_epochs.mat           # Épocas con artefactos
```

### Salida Intermedia
```
fwd-inv-stc/
├── phases-S01-DMT.pkl            # Fases, amplitudes, syncros (100-500 MB)
├── order_all-S01-DMT.pkl         # DataFrames de fases por red (50-200 MB)
├── order-S01-DMT.pkl             # Order parameter por red (50-200 MB)
├── r_kuramoto_nets_epochs_mean.pkl  # Datos agregados
├── r_kuramoto_nets_all_mean.pkl     # Datos agregados (pares de redes)
└── extra.pkl                     # Metadata (1 MB)
```

### Resultados Finales
```
plot_order_results/
├── kuramoto_epochs_summary_band_Alpha_*.png
├── hist_kuramoto_stc_Beta_*.png
└── ...

pearson_results/
├── correlation_coherence_DMT_vs_EC.png
└── ...
```

---

## 🛠️ Configuración Inicial

### 1. Instalar dependencias
```bash
conda create -n dmt_fz python=3.10
conda activate dmt_fz

pip install mne numpy scipy matplotlib pandas
pip install scikit-learn sklearn-extra
pip install optuna tqdm pymatreader
```

### 2. Verificar paths.py
```python
# pipeline/paths.py debe contener:
BASE_DIR = Path(__file__).resolve().parent.parent
EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"
```

### 3. Test rápido
```bash
# Procesar 1 sujeto
python pipeline/fwd.py --max-subjects 1 --conditions DMT

# Verificar output
ls -lh fwd-inv-stc/DMT/phases-*.pkl
```

---

## 🎯 Casos de Uso Comunes

### Ver evolución temporal del Kuramoto
```bash
python viz_scripts/plot_order.py
# → Genera gráficos de r(t) por época
```

### Comparar solo DMT vs Eyes Closed
```bash
# Modificar en viz_scripts/plot_order.py:
COND_LIST = ["DMT", "EC"]

python viz_scripts/plot_order.py
```

### Buscar correlaciones con experiencias
```bash
python pipeline/pearson.py
# → Matrices de correlación (7 redes × 5 bandas × 23 variables)
```

### Identificar estados cerebrales
```bash
python pipeline/clustering.py
# → Encuentra k-clusters óptimo con silueta
```

### Generar solo datos agregados, sin plots
```bash
python pipeline/build_order_data.py --build-all
# → Solo genera r_kuramoto_*.pkl
```

---

## ⚠️ Notas Importantes

### Diferencia entre archivos generados

**`order_all-*.pkl`** (generado por `multi2pool2.py`):
- Contiene: **DataFrames de fases filtradas** por red
- Estructura: `DataFrame(parcelas × samples)` con valores de fase (-π a π)
- Tamaño: ~50-200 MB por archivo
- Uso: Input para `generate_order.py`

**`order-*.pkl`** (generado por `generate_order.py`):
- Contiene: **Series de order parameter** calculado
- Estructura: `pd.Series([r_values])` con valores 0-1
- Tamaño: ~50-200 MB por archivo
- Uso: Input para `pearson.py` y análisis estadísticos

**`r_kuramoto_*.pkl`** (generado por `build_order_data.py`):
- Contiene: **Datos agregados** de Kuramoto
- Estructura: Diccionarios anidados por condición/banda/red
- Uso: Input para `viz_scripts/plot_order.py` (algunos plots)

### Sujetos Rechazados
```python
rejected_subjects = [2, 5, 8, 16, 23, 31]
# 29 sujetos válidos de 35 totales
```

### Épocas Rechazadas
- Archivo: `rejected_epochs.mat`
- Contiene épocas con artefactos por sujeto
- Se manejan automáticamente en `plot_order.py`

---

## 💾 Espacio en Disco

```
Input (.set):        ~1-2 GB por sujeto
phases-*.pkl:        ~100-500 MB por sujeto
order_all-*.pkl:     ~50-200 MB por sujeto
order-*.pkl:         ~50-200 MB por sujeto
Total estimado:      ~150-200 GB (29 sujetos × 3 condiciones)
```

---

## ⏱️ Tiempos de Ejecución

```
fwd.py:              ~30 min por sujeto (serial)
                     ~3-4 horas (29 sujetos, 7 workers)

multi2pool2.py:      ~2-5 min (29 sujetos, 20 cores)

generate_order.py:   ~30 seg - 1 min (29 sujetos, 20 cores)

build_order_data.py: ~2-5 min

plot_order.py:       ~5-10 min

pearson.py:          ~3-5 min

clustering.py:       ~2-4 horas (con grid search)
```

---

## 🆘 Problemas Comunes

### "ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'"
```bash
# Causa: Archivos .pkl guardados con pandas antiguo
# Solución: Ya implementado en load_file() - debería funcionar automáticamente
```

### "ufunc 'add' did not contain a loop with signature matching types"
```bash
# Causa: Datos no numéricos en arrays
# Solución: Ya implementado con flatten_and_extract_numbers()
```

### "ValueError: operands could not be broadcast together"
```bash
# Causa: Archivos order-*.pkl faltantes
# Solución: Ejecutar generate_order.py
python pipeline/generate_order.py --workers 20 --conditions DMT EC EO
```

### "FileNotFoundError: order-*.pkl"
```bash
# Causa: Pipeline incompleto
# Solución: Ejecutar en orden:
python pipeline/fwd.py                    # Genera phases-*.pkl
python pipeline/multi2pool2.py            # Genera order_all-*.pkl
python pipeline/generate_order.py         # Genera order-*.pkl
python pipeline/build_order_data.py       # Genera r_kuramoto_*.pkl
```

### "Memory Error"
```python
# Causa: Demasiadas épocas en memoria
# Solución: Reducir número de sujetos o usar menos workers
python pipeline/fwd.py --max-subjects 10 --workers 3
```

---

## 📊 Resultados Típicos

### Hallazgos Esperados

1. **Sincronización Aumentada en DMT**
   - Mayor en bandas bajas (Delta/Theta)
   - Más notorio en Default Mode Network

2. **Correlaciones Significativas**
   - Unity/Spiritual: ↑ sincronización DMN
   - Anxiety: ↓ sincronización FPN
   - Visual Imagery: ↑ sincronización VN

3. **Estados Cerebrales**
   - 3-5 estados discretos identificables
   - Transiciones más frecuentes en DMT vs baseline

---

## 📚 Dónde Leer Más

1. **WORKFLOW.md** → Flujo completo paso a paso con detalles técnicos
2. **README.md** → Visión general completa del proyecto
3. **TECHNICAL_DETAILS.md** → Matemática detallada

### Documentación Externa
- MNE-Python: https://mne.tools/stable/auto_tutorials/index.html
- Kuramoto Model: https://en.wikipedia.org/wiki/Kuramoto_model
- Schaefer Atlas: https://github.com/ThomasYeoLab/CBIG

---

## ✅ Checklist Rápido

Antes de empezar:
- [ ] Datos `.set` en EEG_CLEAN/
- [ ] Ambiente conda activado (`conda activate dmt_fz`)
- [ ] Dependencias instaladas
- [ ] paths.py configurado
- [ ] Espacio en disco suficiente (~200 GB)
- [ ] RAM suficiente (recomendado: 16 GB+)

Para análisis completo:
- [ ] `fwd.py` ejecutado → phases-*.pkl generados
- [ ] `multi2pool2.py` ejecutado → order_all-*.pkl generados
- [ ] `generate_order.py` ejecutado → order-*.pkl generados
- [ ] `build_order_data.py` ejecutado → r_kuramoto_*.pkl generados
- [ ] `plot_order.py` ejecutado → gráficos en plot_order_results/
- [ ] `pearson.py` ejecutado → correlaciones calculadas

---

## 🚀 Comando Todo-en-Uno

Para ejecutar el pipeline completo de una vez:

```bash
cd /media/storage_hdd/dmt_fz && \
conda activate dmt_fz && \
python pipeline/fwd.py --jobs 0 --workers 7 --conditions DMT EC EO && \
python pipeline/multi2pool2.py && \
python pipeline/generate_order.py --workers 20 --conditions DMT EC EO && \
python pipeline/build_order_data.py --build-all && \
python viz_scripts/plot_order.py --workers 20 && \
python pipeline/pearson.py && \
echo "✓ Pipeline completo ejecutado!"
```

**Tiempo total:** ~4-5 horas

---

**¡Listo para empezar! 🚀**

Ejecuta el comando todo-en-uno o sigue los pasos individuales en **WORKFLOW.md**.

---

**Última actualización:** 2025-11-27
