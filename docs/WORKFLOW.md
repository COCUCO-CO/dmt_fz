# Flujo de Trabajo Completo - Pipeline de Análisis EEG

## 📊 Diagrama de Flujo General

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATOS CRUDOS (.set files)                     │
│                  EEG_CLEAN/DMT/, EC/, EO/                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
        ┌───────▼──────┐ ┌───▼────┐ ┌─────▼──────┐
        │   DMT/       │ │  EC/   │ │   EO/      │
        │ *.set files  │ │ *.set  │ │  *.set     │
        └───────┬──────┘ └───┬────┘ └─────┬──────┘
                │            │            │
                └────────────┼────────────┘
                             │
                    ┌────────▼────────┐
                    │   fwd.py        │  ⭐ PASO 1: Pipeline Central
                    │                 │
                    │ • Forward Sol   │
                    │ • Inverse Sol   │
                    │ • Hilbert       │
                    │ • Sincronización│
                    │ • Kuramoto      │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
        ┌─────▼──────┐  ┌───▼────┐  ┌─────▼──────┐
        │  phases-   │  │ extra  │  │  (metadata)│
        │  *.pkl     │  │ .pkl   │  │            │
        └─────┬──────┘  └────────┘  └────────────┘
              │
    ┌─────────┼─────────────────────────────────────┐
    │         │                                     │
┌───▼────┐ ┌─▼────────┐ ┌──────────────┐ ┌────────▼─────────┐
│multi2  │ │calculate │ │save_load_    │ │ generate_order.py│
│pool2.py│ │_syncro.py│ │pickle.py     │ │  ⭐ PASO 2       │
│        │ │          │ │              │ │                  │
└───┬────┘ └─┬────────┘ └──────────────┘ └────────┬─────────┘
    │        │                                     │
┌───▼────────▼─────────────────────────────────────▼─────────┐
│              order_all-*.pkl  →  order-*.pkl               │
│         (DataFrames fases)    (Order parameter)            │
└───┬────────────────────────────────────────────────────────┘
    │
    ├──────────────────┬──────────────────┬──────────────────┐
    │                  │                  │                  │
┌───▼────┐      ┌──────▼──────┐   ┌─────▼────┐      ┌──────▼──────┐
│pearson │      │ plot_order  │   │clustering│      │ Otros       │
│  .py   │      │    .py      │   │   .py    │      │ análisis    │
│        │      │             │   │          │      │             │
└───┬────┘      └──────┬──────┘   └─────┬────┘      └──────┬──────┘
    │                  │                │                  │
┌───▼──────────────────▼────────────────▼──────────────────▼───┐
│                       RESULTADOS                              │
│                                                               │
│ • Correlaciones (pearson_results/)                           │
│ • Gráficos (plot_order_results/)                            │
│ • Clusters (clusters_*.pkl)                                  │
│ • Estadísticas                                               │
└───────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              ANÁLISIS DE MACHINE LEARNING (PARALELO)            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  spectral_sources/          EEGNet/                            │
│  ├── *.csv (PSD)           ├── *.h5 (modelos)                 │
│  ├── randomforest.py       ├── EEGNet CLF.py                  │
│  ├── DMT Reg.py           └── EEGNet REG.py                  │
│  └── DMT Autogluon.py                                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Detallado Paso a Paso

### FASE 0: Configuración Inicial (Una sola vez)

#### 0.1 Verificar Estructura de Directorios

```bash
# Verificar que paths.py apunta a las rutas correctas
cat dmt/paths.py
```

**Contenido esperado:**
```python
BASE_DIR = Path(__file__).resolve().parent
EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"
```

#### 0.2 Verificar Datos de Entrada

```bash
# Estructura esperada:
dmt/EEG_CLEAN/
├── DMT/
│   ├── S01-DMT_ICA_pruned.set
│   ├── S02-DMT_ICA_pruned.set
│   └── ...
├── EC/
│   ├── S01-EC_ICA_pruned.set
│   └── ...
├── EO/
│   ├── S01-EO_ICA_pruned.set
│   └── ...
└── rejected_epochs.mat
```

---

### FASE 1: Procesamiento Principal de EEG

#### 1.1 Forward/Inverse Solution + Métricas Básicas

```bash
cd /media/storage_hdd/dmt_fz/dmt
conda activate dmt_fz
python fwd.py --max-subjects 0 --jobs 0 --workers 7
```

**Parámetros:**
- `--max-subjects 0`: Procesar todos los sujetos (usar número positivo para limitar)
- `--jobs 0`: Usar todos los cores disponibles
- `--workers 7`: Número de procesos paralelos (ajustar según CPU)

**Input:**
```
EEG_CLEAN/
├── DMT/*.set
├── EC/*.set
└── EO/*.set
```

**Proceso:**
```python
for file in [dmt_files + eo_files + ec_files]:
    epochs = mne.io.read_epochs_eeglab(file)
    
    # 1. Source localization (dSPM)
    stc = fwd_inv_stc(epochs)
    
    for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
        for epoch in epochs:
            # 2. EEG processing
            signals_eeg = extract_eeg(epoch)
            filtered_eeg = filter_band(signals_eeg, band)
            phase_eeg, amp_eeg = hilbert_transform(filtered_eeg)
            syncro_eeg = calculate_syncro(phase_eeg)
            kuramoto_eeg = order_parameter(phase_eeg)
            
            # 3. Source processing
            signals_stc = extract_sources(stc[epoch])
            filtered_stc = filter_band(signals_stc, band)
            phase_stc, amp_stc = hilbert_transform(filtered_stc)
            syncro_stc = calculate_syncro(phase_stc)
            kuramoto_stc = order_parameter(phase_stc)
```

**Output:**
```
fwd-inv-stc/
├── DMT/
│   ├── phases-S01-DMT.pkl
│   ├── phases-S02-DMT.pkl
│   └── ...
├── EC/
│   ├── phases-S01-EC.pkl
│   └── ...
├── EO/
│   ├── phases-S01-EO.pkl
│   └── ...
└── extra.pkl  # Metadata (labels, coordinates, etc.)
```

**Estructura de `phases-*.pkl`:**
```python
{
    "filtered_eeg": {band: [epoch_arrays]},
    "phases_eeg": {band: [phase_matrices]},
    "amplitudes_eeg": {band: [amplitude_matrices]},
    "syncros_eeg": {band: [syncro_matrices]},
    "kuramoto_eeg": {band: [r_values]},
    
    "filtered_stc": {band: [epoch_arrays]},
    "phases_stc": {band: [phase_matrices]},
    "amplitudes_stc": {band: [amplitude_matrices]},
    "syncros_stc": {band: [syncro_matrices]},
    "kuramoto_stc": {band: [r_values]}
}
```

**Tiempo estimado:**
- Serial: ~15-20 horas (29 sujetos × 30-40 min)
- Con 7 workers: ~3-4 horas

---

#### 1.2 Consolidar Datos por Condición (Opcional)

```bash
python save_load_pickle.py --conditions DMT EC EO
```

**Input:** `phases-*.pkl` (individuales)

**Output:** `subject_phases_<COND>.pkl` (consolidado por condición)

**Utilidad:** Facilita análisis posteriores que necesitan todos los sujetos juntos

---

### FASE 2: Análisis por Redes Cerebrales

#### 2.1 Filtrar Fases por Redes

```bash
python multi2pool2.py
```

**Input:** `phases-*.pkl`

**Proceso:**
```python
# Redes cerebrales (Schaefer Atlas)
net_list = ["FPN", "DMN", "DAN", "LN ", "SVA", "SMN", "VN "]
hemi_list = ["RH", "LH", "both"]

for subject_file in all_phases_files:
    data = load_file(subject_file)
    
    for band in bands:
        for hemi in hemi_list:
            for net in net_list:
                # Filtrar parcelas de la red
                filtered_phases = network_filter(
                    phase_data, 
                    df_labels, 
                    hemi=hemi, 
                    net=net
                )
                # Guardar DataFrames filtrados
                result[cond][band][hemi][net].append(filtered_phases)
```

**Output:**
```
fwd-inv-stc/
├── DMT/
│   ├── order_all-S01-DMT.pkl
│   ├── order_all-S02-DMT.pkl
│   └── ...
├── EC/
│   ├── order_all-S01-EC.pkl
│   └── ...
└── EO/
    ├── order_all-S01-EO.pkl
    └── ...
```

**Estructura de `order_all-*.pkl`:**
```python
{
    "DMT": {
        "Delta": {
            "RH": {
                "FPN": [DataFrame_epoch1, DataFrame_epoch2, ...],  # parcelas × samples
                "DMN": [DataFrame_epoch1, ...],
                ...
            },
            "LH": {...},
            "both": {...}
        },
        "Theta": {...},
        ...
    }
}
```

**Tiempo:** ~2-5 min con 20 cores

---

#### 2.2 Calcular Order Parameter por Red ⭐ NUEVO

```bash
python generate_order.py --workers 20 --conditions DMT EC EO
```

**Input:** `order_all-*.pkl` (DataFrames de fases filtradas)

**Proceso:**
```python
for order_all_file in all_files:
    data = load_file(order_all_file)
    
    for band in bands:
        for hemi in hemis:
            for net in networks:
                for epoch_df in data[cond][band][hemi][net]:
                    # Calcular order parameter de Kuramoto
                    r_series = order_parameter(epoch_df)
                    # r_series es una pd.Series con valores 0-1
                    result[cond][band][hemi][net].append(r_series)
```

**Output:**
```
fwd-inv-stc/
├── DMT/
│   ├── order-S01-DMT.pkl
│   ├── order-S02-DMT.pkl
│   └── ...
├── EC/
│   ├── order-S01-EC.pkl
│   └── ...
└── EO/
    ├── order-S01-EO.pkl
    └── ...
```

**Estructura de `order-*.pkl`:**
```python
{
    "DMT": {
        "Delta": {
            "RH": {
                "FPN": [pd.Series([r_values]), ...],  # Una Series por época
                "DMN": [pd.Series([r_values]), ...],
                ...
            },
            "LH": {...},
            "both": {...}
        },
        "Theta": {...},
        ...
    }
}
```

**Diferencia clave:**
- `order_all-*.pkl`: DataFrames de fases (parcelas × samples)
- `order-*.pkl`: Series de order parameter (samples con valores 0-1)

**Tiempo:** ~30 segundos - 1 min con 20 cores

---

#### 2.3 Análisis Temporal con Splits (Opcional)

```bash
python calculate_syncro.py --workers 20 --conditions DMT EC EO
```

**Input:** `subject_phases_<COND>.pkl` (consolidado)

**Proceso:**
```python
for subject in subjects:
    for band in bands:
        for splits in range(2, 12):  # 2 a 11 ventanas temporales
            ruler = np.linspace(0, num_samples, splits)
            
            for i in range(len(ruler)-1):
                window_phase = phase[:, ruler[i]:ruler[i+1]]
                
                # Calcular métricas en ventana
                syncro = calculate_syncro(window_phase)
                kuramoto = order_parameter(window_phase)
```

**Output:** `syncro-*.pkl`

**Estructura:**
```python
{
    "syncros_eeg": {band: [[split_matrices]]},  # [splits][epochs]
    "kuramoto_eeg": {band: [[split_values]]},
    "syncros_stc": {band: [[split_matrices]]},
    "kuramoto_stc": {band: [[split_values]]}
}
```

**Utilidad:** Analizar dinámica temporal intra-época

**Tiempo:** ~10-15 min con 20 cores

---

### FASE 3: Análisis Estadístico y Visualización

#### 3.1 Generar Plots y Datos Agregados

```bash
python plot_order.py --workers 20
```

**Opciones:**
```bash
# Limitar sujetos (para testing)
python plot_order.py --max-subjects 5

# Solo generar datos, sin plots
python plot_order.py --no-plots

# Especificar workers
python plot_order.py --workers 10
```

**Input:**
- `syncro-*.pkl`
- `order_all-*.pkl`
- `rejected_epochs.mat`
- `extra.pkl`

**Proceso:**
1. **Carga metadata** (labels, épocas rechazadas)
2. **Construye catálogo de archivos** (evita escaneos repetidos)
3. **Genera plots** (si `--no-plots` no está activado):
   - Kuramoto summary (evolución temporal por sujeto)
   - Mean trajectories (promedios por condición)
   - Histogramas STC
   - Histogramas de fases por red
   - Histograma completo (7 redes × 5 bandas)
4. **Construye estructuras de datos**:
   - `r_kuramoto_nets_epochs_mean.pkl`
   - `r_kuramoto_nets_all_mean.pkl`

**Output:**
```
plot_order_results/
├── kuramoto_epochs_summary_band_Alpha_conditions_DMT_EO_EC.png
├── kuramoto_epochs_summary_band_Beta_conditions_DMT_EO_EC.png
├── kuramoto_gamma_summary_split_0_conditions_DMT_EO_EC.png
├── hist_kuramoto_stc_Alpha_conditions_DMT_EC.png
├── hist_kuramoto_pair_Alpha_both_DMN_conditions_DMT_EC.png
├── histogram_kuramoto_full_EC_vs_EO_hemi_both.png
└── ...

fwd-inv-stc/
├── r_kuramoto_nets_epochs_mean.pkl
└── r_kuramoto_nets_all_mean.pkl
```

**Tiempo:** ~5-10 min (con plots), ~2-3 min (sin plots)

---

#### 3.2 Correlaciones con Cuestionarios

```bash
python pearson.py
```

**Input:**
- `order-*.pkl` (o `r_kuramoto_nets_epochs_mean.pkl`)
- `spectral_sources/target.csv` (23 variables de cuestionarios)
- `spectral_sources/target_labels.txt`

**Proceso:**
```python
# 1. Preparar datos por sujeto
for cond in ["DMT", "EC"]:
    for band in bands:
        for net in networks:
            means_per_subject = []
            vars_per_subject = []
            
            for subject in subjects:
                epochs = load_order_data(subject, band, net)
                means_per_subject.append(np.mean(epochs))
                vars_per_subject.append(np.var(epochs))

# 2. Correlacionar con cuestionarios
for target_var in range(23):
    for band in bands:
        for net in networks:
            # Coherencia (mean)
            r_coh, p_coh = pearsonr(means_per_subject, target[:, target_var])
            
            # Metastabilidad (variance)
            r_meta, p_meta = pearsonr(vars_per_subject, target[:, target_var])

# 3. Corrección FDR
rejected, p_corrected = fdrcorrection(all_pvalues, alpha=0.05)

# 4. Visualizar
plot_correlation_heatmap(r_matrix, p_corrected)
```

**Output:**
```
pearson_results/
├── correlation_coherence_DMT_vs_EC.png
├── correlation_metastability_DMT_vs_EC.png
├── significant_correlations_FDR_corrected.csv
└── ...
```

**Tiempo:** ~3-5 min

---

#### 3.3 Clustering de Estados Cerebrales

```bash
python clustering.py
```

**Input:** `syncro-*.pkl`

**Proceso:**
```python
# 1. Extraer eigenvalores de matrices de sincronización
eigenvalues_list = []
for file in all_syncro_files:
    data = load_file(file)
    for epoch_syncro in data["syncros_stc"][band][split]:
        eigenvals = np.diag(np.linalg.eigh(epoch_syncro)[1])
        eigenvalues_list.append(eigenvals)

# 2. Optimización con Optuna
def objective(trial):
    n_comps = trial.suggest_int("n_comps", 2, 25)
    k_clusters = trial.suggest_int("k_clusters", 2, 25)
    
    # PCA
    pca = PCA(n_components=n_comps)
    data_reduced = pca.fit_transform(eigenvalues_array)
    
    # K-Medoids
    kmedoids = KMedoids(n_clusters=k_clusters)
    labels = kmedoids.fit_predict(data_reduced)
    
    # Validación
    silhouette = silhouette_score(data_reduced, labels)
    return silhouette

study = optuna.create_study(direction="maximize", sampler=GridSampler(search_space))
study.optimize(objective, n_jobs=20)

# 3. Aplicar mejor configuración
best_params = study.best_params
# ... clustering final ...
```

**Output:** `clusters_splits*.pkl`

```python
{
    "Delta": DataFrame([n_comps, k_clusters, silhouette_score]),
    "Theta": DataFrame(...),
    ...
}
```

**Tiempo:** ~2-4 horas (con grid search completo)

---

### FASE 4: Machine Learning (Independiente)

#### 4.1 Random Forest para Predicción

```bash
cd spectral_sources
python randomforest.py
```

**Input:**
- `theta.csv`, `alpha.csv`, etc. (PSD por parcela)
- `target.csv` (23 variables de cuestionarios)
- `target_labels.txt`

**Proceso:**
```python
# 1. Feature selection con correlación + FDR
for score_idx in range(23):
    for roi in range(90):
        r, p = pearsonr(psd[:, roi], target[:, score_idx])
    
    survived_rois = fdrcorrection(all_pvalues)

# 2. Cross-validation
for score in survived_features:
    X = psd[:, survived_features[score]]
    y = target[:, score]
    
    for iteration in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(...)
        
        rf = RandomForestRegressor(n_estimators=1000)
        rf.fit(X_train, y_train)
        
        r_score = pearsonr(y_test, rf.predict(X_test))[0]
        scores.append(r_score)
```

**Output:** Distribución de scores de predicción

**Tiempo:** ~30 min - 1 hora

---

#### 4.2 EEGNet Deep Learning

```bash
cd EEGNet
python EEGNet\ REG.py  # o EEGNet\ CLF.py
```

**Input:**
- Señales EEG crudas (epochs)
- `targets.csv`

**Output:**
- 23 modelos `.h5` (uno por variable)
- `score_history.csv`

**Tiempo:** ~2-4 horas (dependiendo de GPU)

---

## 📦 Resumen de Archivos Generados

### Archivos Intermedios (NO subir a git)

```
fwd-inv-stc/
├── phases-*.pkl              (~100-500 MB cada uno)
│   └── Contiene: fases, amplitudes, syncros, kuramoto (EEG y STC)
│
├── order_all-*.pkl           (~50-200 MB cada uno)
│   └── Contiene: DataFrames de fases filtradas por red
│
├── order-*.pkl               (~50-200 MB cada uno)
│   └── Contiene: Series de order parameter por red
│
├── syncro-*.pkl              (~200-800 MB cada uno)
│   └── Contiene: Métricas con splits temporales
│
├── r_kuramoto_nets_epochs_mean.pkl  (~10-50 MB)
│   └── Datos agregados para análisis
│
├── r_kuramoto_nets_all_mean.pkl     (~10-50 MB)
│   └── Datos agregados para análisis
│
└── extra.pkl                 (~1 MB, OK subirlo)
    └── Metadata: labels, coordinates, colors
```

### Archivos de Resultados

```
plot_order_results/
└── *.png                     (Gráficos de visualización)

pearson_results/
└── *.png, *.csv              (Correlaciones y matrices)

clustering_results/
└── clusters_*.pkl            (Estados identificados)
```

---

## ⚙️ Configuración para Nueva Máquina

### 1. Verificar paths.py

```python
# dmt/paths.py debe contener:
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"
SPECTRAL_DIR = BASE_DIR / "spectral_sources"
EEGNET_DIR = BASE_DIR / "EEGNet"
RESULTS_PLOTS_DIR = BASE_DIR / "results_plots"
FRAMES_DIR = BASE_DIR / "frames"
```

### 2. Instalar Dependencias

```bash
conda create -n dmt_fz python=3.10
conda activate dmt_fz

# Neurociencia
pip install mne
pip install pymatreader

# Procesamiento
pip install numpy scipy pandas
pip install matplotlib seaborn
pip install tqdm

# Machine Learning
pip install scikit-learn sklearn-extra
pip install optuna

# Deep Learning (opcional)
pip install tensorflow  # o pytorch
```

### 3. Test Pipeline

```bash
# Test 1: Verificar paths
python -c "from paths import RESULTS_DIR; print(RESULTS_DIR)"

# Test 2: Procesar 1 sujeto
python fwd.py --max-subjects 1 --conditions DMT

# Test 3: Verificar output
python -c "import pickle; print(pickle.load(open('fwd-inv-stc/DMT/phases-S01-DMT.pkl', 'rb')).keys())"
```

---

## 🐛 Troubleshooting Común

### Error: "ModuleNotFoundError: No module named 'pandas.core.indexes.numeric'"

**Causa:** Archivos `.pkl` guardados con pandas antiguo

**Solución:** Ya implementado en `load_file()` de `plot_order.py` y `generate_order.py`

### Error: "ufunc 'add' did not contain a loop with signature matching types"

**Causa:** Datos no numéricos en arrays

**Solución:** Ya implementado con `flatten_and_extract_numbers()` y `reject_outliers()`

### Error: "ValueError: operands could not be broadcast together"

**Causa:** Archivos `order-*.pkl` faltantes o vacíos

**Solución:** Ejecutar `python generate_order.py` para generarlos

---

## 📈 Checklist de Ejecución Completa

```
FASE 0: Configuración
[ ] paths.py configurado correctamente
[ ] Datos .set en EEG_CLEAN/
[ ] Dependencias instaladas

FASE 1: Procesamiento Principal
[ ] fwd.py ejecutado para todas las condiciones
    [ ] DMT: 29/29 archivos phases-*.pkl
    [ ] EC: 29/29 archivos
    [ ] EO: 29/29 archivos
[ ] extra.pkl generado

FASE 2: Análisis por Redes
[ ] multi2pool2.py completado
    [ ] order_all-*.pkl generados (87 archivos total)
[ ] generate_order.py completado ⭐ NUEVO
    [ ] order-*.pkl generados (87 archivos total)

FASE 3: Análisis Estadístico
[ ] plot_order.py ejecutado
    [ ] Gráficos generados en plot_order_results/
    [ ] r_kuramoto_*.pkl generados
[ ] pearson.py ejecutado
    [ ] Correlaciones calculadas
    [ ] Matrices generadas
[ ] clustering.py ejecutado (opcional)
    [ ] Estados identificados

FASE 4: Machine Learning (opcional)
[ ] Random Forest entrenado
[ ] EEGNet entrenado
```

---

## 🎯 Resumen Ejecutivo de Tiempos

**Para procesar datos completos (29 sujetos):**

```bash
1. python fwd.py --jobs 0 --workers 7          # ~3-4 horas
2. python multi2pool2.py                        # ~2-5 min
3. python generate_order.py --workers 20        # ~30 seg - 1 min ⭐ NUEVO
4. python plot_order.py --workers 20            # ~5-10 min
5. python pearson.py                            # ~3-5 min
6. python clustering.py (opcional)              # ~2-4 horas

Total: ~4-5 horas (sin clustering)
       ~6-9 horas (con clustering)
```

**Para análisis exploratorio rápido:**

```bash
# Procesar solo algunos sujetos
python fwd.py --max-subjects 3 --workers 3      # ~30-40 min
python multi2pool2.py                           # ~1 min
python generate_order.py --workers 10           # ~10 seg
python plot_order.py --max-subjects 3           # ~1-2 min

Total: ~35-45 min
```

---

## 📚 Documentación Adicional

- **QUICKSTART.md**: Guía rápida para empezar
- **README.md**: Visión general del proyecto
- **TECHNICAL_DETAILS.md**: Detalles matemáticos y teóricos

---

**Última actualización:** 2025-01-13
