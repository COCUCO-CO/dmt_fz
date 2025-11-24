# Proyecto de Investigación: Sincronización de Fases en EEG y Correlatos Subjetivos

## 📋 Descripción General

Este proyecto analiza la **sincronización de fases** en señales EEG mediante **solución inversa** (source localization) para identificar correlaciones con experiencias subjetivas durante estados alterados de consciencia inducidos por DMT y otras sustancias psicodélicas.

### Objetivos Principales
- Calcular sincronización de fases entre canales EEG y fuentes corticales
- Determinar el parámetro de orden de Kuramoto como medida de coherencia
- Correlacionar patrones de sincronización con reportes subjetivos (cuestionarios ASC, MEQ, NDE)
- Identificar estados cerebrales mediante clustering
- Predecir experiencias subjetivas usando machine learning

---

## 🏗️ Estructura del Proyecto

```
dmt_fz/
├── dmt/
│   ├── fwd.py                    ⭐ PIPELINE PRINCIPAL (Paso 1)
│   ├── multi2pool2.py            🔄 Filtrado por redes (Paso 2a)
│   ├── generate_order.py         📊 Order parameter (Paso 2b) [NUEVO]
│   ├── calculate_syncro.py       ⏱️ Análisis temporal con splits
│   ├── clustering.py             🎯 Identificación de estados
│   ├── pearson.py                📈 Correlaciones con cuestionarios
│   ├── plot_order.py             📊 Visualización y estadísticas
│   ├── kuramoto.py               🧮 Funciones de Kuramoto
│   ├── save_load_pickle.py       💾 Consolidación de datos
│   ├── paths.py                  🗂️ Configuración de rutas
│   ├── psd.py                    🌊 Power Spectral Density
│   ├── aligment.py               🔧 Setup inicial de fsaverage
│   │
│   ├── spectral_sources/         🎓 Machine Learning
│   │   ├── target.csv            📝 29 sujetos × 23 variables
│   │   ├── target_labels.txt     🏷️ Nombres de variables
│   │   ├── randomforest.py       🌲 Random Forest
│   │   ├── DMT Reg.py           
│   │   ├── DMT Autogluon.py     
│   │   └── EEGNet.py            
│   │
│   ├── EEGNet/                   🧠 Deep Learning
│   │   ├── EEGNet CLF.py        
│   │   ├── EEGNet REG.py        
│   │   ├── targets.csv          
│   │   └── models/               💾 23 modelos (.h5)
│   │
│   ├── docs/                     📚 Documentación
│   │   ├── QUICKSTART.md         🚀 Guía rápida
│   │   ├── WORKFLOW.md           📋 Pipeline detallado
│   │   ├── README.md             📖 Este archivo
│   │   └── TECHNICAL_DETAILS.md  🔬 Detalles técnicos
│   │
│   ├── EEG_CLEAN/                📁 Datos de entrada
│   │   ├── DMT/*.set
│   │   ├── EC/*.set
│   │   ├── EO/*.set
│   │   └── rejected_epochs.mat
│   │
│   ├── fwd-inv-stc/              💾 Datos procesados
│   │   ├── DMT/
│   │   │   ├── phases-*.pkl
│   │   │   ├── order_all-*.pkl
│   │   │   └── order-*.pkl
│   │   ├── EC/
│   │   ├── EO/
│   │   ├── r_kuramoto_*.pkl
│   │   └── extra.pkl
│   │
│   ├── plot_order_results/       📊 Gráficos
│   ├── pearson_results/          📈 Correlaciones
│   ├── Entrevistas/              💬 35 entrevistas
│   └── autocorrelacion_resultados/ 🔬 Análisis MD
│
└── README.md
```

---

## 🔬 Pipeline Principal de Análisis

### **Paso 1: fwd.py - Procesamiento Completo de EEG** ⭐

Este es el archivo central del proyecto. Realiza todo el procesamiento desde archivos `.set` hasta matrices de sincronización.

#### **Entrada**
```
EEG_CLEAN/
├── DMT/        # Datos durante experiencia con DMT
├── EC/         # Eyes Closed (baseline)
└── EO/         # Eyes Open (baseline)
```

#### **Procesamiento**

**1. Forward/Inverse Solution**
- Usa template `fsaverage` de MNE-Python
- Crea forward solution con BEM (Boundary Element Method)
- Aplica inverse operator con método dSPM (SNR=3)
- Localiza actividad en 100 parcelas de Schaefer (7 redes cerebrales)

**2. Filtrado por Bandas**
```python
freq_bands = {
    "Delta": [1, 4],      # Hz
    "Theta": [4, 8],
    "Alpha": [8, 13],
    "Beta":  [13, 30],
    "Gamma": [30, 45]
}
```

**3. Extracción de Fases**
- Aplica transformada de Hilbert a señales filtradas
- Extrae fase instantánea: `θ(t) = angle(hilbert(x(t)))`
- Calcula envolvente de amplitud

**4. Sincronización**
```python
def calculate_syncro(phase_mat):
    # Calcula matriz de sincronización NxN
    # Usa diferencias angulares normalizadas
    value = 1 - (diff_ang(signal1, signal2).sum() / max_diff)
```

**5. Parámetro de Kuramoto**
```python
def order_parameter(phase):
    # Coherencia global: r = |⟨e^(iθ)⟩|
    r = |mean(exp(1j * phase))|
    # r ∈ [0,1]: 0=incoherente, 1=sincronizado
```

#### **Salida**
```python
# phases-*.pkl contiene:
{
    "filtered_eeg": {band: [epochs]},
    "phases_eeg": {band: [matrices_fase]},
    "amplitudes_eeg": {band: [envolventes]},
    "syncros_eeg": {band: [matrices_NxN]},
    "kuramoto_eeg": {band: [valores_r]},
    # Lo mismo para "_stc" (fuentes)
}
```

**Comando:**
```bash
python fwd.py --jobs 0 --workers 7 --conditions DMT EC EO
```

**Tiempo:** ~3-4 horas (29 sujetos, 7 workers)

---

### **Paso 2a: multi2pool2.py - Análisis por Redes Cerebrales** 🔄

Procesa datos ya calculados filtrando por redes específicas.

#### **Redes Cerebrales Analizadas**
```python
net_list = [
    "FPN",  # Frontoparietal Network (control ejecutivo)
    "DMN",  # Default Mode Network (mente en reposo)
    "DAN",  # Dorsal Attention Network (atención dirigida)
    "LN ",  # Limbic Network (emociones)
    "SVA",  # Salience/Ventral Attention (detección de relevancia)
    "SMN",  # Somatomotor Network (movimiento)
    "VN "   # Visual Network (visión)
]

hemi_list = ["RH", "LH", "both"]  # Hemisferios
```

#### **Proceso**
- Lee archivos `phases-*.pkl` ya procesados
- Filtra parcelas por red y hemisferio
- Guarda DataFrames de fases filtradas
- Usa multiprocessing con 20 cores

#### **Salida**
```python
# order_all-*.pkl contiene:
{
    "DMT": {
        "Delta": {
            "RH": {
                "FPN": [DataFrame_epoch1, ...],  # parcelas × samples
                "DMN": [DataFrame_epoch1, ...],
                ...
            },
            "LH": {...},
            "both": {...}
        },
        ...
    }
}
```

**Comando:**
```bash
python multi2pool2.py
```

**Tiempo:** ~2-5 min (29 sujetos, 20 cores)

---

### **Paso 2b: generate_order.py - Calcular Order Parameter** ⭐ NUEVO

Calcula el parámetro de orden de Kuramoto a partir de las fases filtradas.

#### **Proceso**
```python
for order_all_file in all_files:
    data = load_file(order_all_file)
    
    for band in bands:
        for hemi in hemis:
            for net in networks:
                for epoch_df in data[cond][band][hemi][net]:
                    # Calcular order parameter
                    phase_array = epoch_df.values
                    r = np.abs(np.exp(1j * phase_array).mean(axis=0))
                    r_series = pd.Series(r)  # Valores 0-1
```

#### **Diferencia Clave**

**`order_all-*.pkl`** (input):
- DataFrames de fases: `parcelas × samples`
- Valores: fase en radianes (-π a π)

**`order-*.pkl`** (output):
- Series de order parameter: `samples`
- Valores: coherencia 0-1

#### **Salida**
```python
# order-*.pkl contiene:
{
    "DMT": {
        "Delta": {
            "RH": {
                "FPN": [pd.Series([r_values]), ...],  # Una Series por época
                "DMN": [pd.Series([r_values]), ...],
                ...
            },
            ...
        },
        ...
    }
}
```

**Comando:**
```bash
python generate_order.py --workers 20 --conditions DMT EC EO
```

**Tiempo:** ~30 seg - 1 min (29 sujetos, 20 cores)

---

### **Paso 3: calculate_syncro.py - Dinámica Temporal** ⏱️ (Opcional)

Analiza cómo cambia la sincronización en el tiempo.

#### **Método**
- Divide cada época en ventanas temporales (2-12 splits)
- Calcula sincronización y Kuramoto en cada ventana
- Permite análisis de fluctuaciones temporales

**Comando:**
```bash
python calculate_syncro.py --workers 20 --conditions DMT EC EO
```

**Tiempo:** ~10-15 min (29 sujetos, 20 cores)

---

### **Paso 4: clustering.py - Estados Cerebrales** 🎯

Identifica estados discretos de sincronización cerebral.

#### **Metodología**
1. **Eigenvalores**: Extrae eigenvalores de matrices de sincronización
2. **PCA**: Reduce dimensionalidad (componentes óptimas: 2-25)
3. **K-Medoids**: Clustering con métrica euclidiana
4. **Optimización**: Usa Optuna con GridSampler para encontrar mejor k
5. **Validación**: Coeficiente de silueta como métrica

**Comando:**
```bash
python clustering.py
```

**Tiempo:** ~2-4 horas (con grid search completo)

---

### **Paso 5: pearson.py - Correlación con Experiencia Subjetiva** 📈

Correlaciona medidas de sincronización con cuestionarios.

#### **Variables Dependientes (23 escalas)**
```
Cuestionarios:
├── ASC (Altered States of Consciousness)
│   ├── Unity, Spiritual, Blissful
│   ├── Insightfulness, Disembodiment
│   ├── Impaired, Anxiety
│   ├── Complex imagery, Elementary imagery
│   └── Audiovisual, Changed
│
├── MEQ (Mystical Experience Questionnaire)
│   ├── Mystical, Positive
│   ├── Transcendental, Ineffability
│   └── Awe
│
├── NDE (Near Death Experience)
│   ├── Cognition, Affect
│   ├── Paranormal, Transcendental
│
└── Post-Experiencia
    ├── Social, Fusion, Setting
```

#### **Análisis**
- Correlación de Pearson (r) entre sincronización y puntajes
- Corrección FDR (False Discovery Rate) para múltiples comparaciones
- Matrices de calor (red × banda × cuestionario)
- Análisis separados para:
  - **Coherencia**: media del parámetro r
  - **Metastabilidad**: varianza del parámetro r

**Comando:**
```bash
python pearson.py
```

**Tiempo:** ~3-5 min

---

### **Paso 6: plot_order.py - Visualización** 📊

Genera gráficos y análisis estadísticos.

#### **Funcionalidades**
- Evolución temporal del parámetro de Kuramoto por época
- Comparación DMT vs EC vs EO
- T-tests con corrección para múltiples comparaciones
- Distribuciones con ajuste gaussiano
- Maneja épocas rechazadas de `rejected_epochs.mat`

**Comandos:**
```bash
# Generar todos los plots
python plot_order.py --workers 20

# Solo generar datos, sin plots
python plot_order.py --no-plots

# Limitar sujetos (testing)
python plot_order.py --max-subjects 5
```

**Tiempo:** ~5-10 min (con plots), ~2-3 min (sin plots)

---

## 🧠 Machine Learning

### **spectral_sources/** 🎓

Predicción de experiencias subjetivas usando Power Spectral Density.

#### **randomforest.py**
- Random Forest Regressor (1000 árboles)
- Features: PSD en parcelas AAL90
- Targets: 23 variables de cuestionarios
- Cross-validation con 1000 splits aleatorios
- Selección de features con correlación + FDR

**Comando:**
```bash
cd spectral_sources
python randomforest.py
```

### **EEGNet/** 🧠

Deep learning con arquitectura EEGNet.

#### **Modelos Entrenados**
- 23 modelos (.h5) - uno por variable de cuestionario
- Clasificación y regresión
- Input: Señales EEG temporales
- Output: Predicción de escalas subjetivas

**Comando:**
```bash
cd EEGNet
python EEGNet\ REG.py  # o EEGNet\ CLF.py
```

### **machine_learning/clf/** 🕸️ ⭐ NUEVO

Graph Neural Networks (GAT) para clasificación de estados cerebrales.

#### **¿Por qué GNN?**
- ✅ Usa las matrices de sincronización ya calculadas por el pipeline
- ✅ Respeta la estructura de grafo natural del cerebro
- ✅ Incorpora parámetros de Kuramoto, coherence y metastability
- ✅ Proporciona attention weights interpretables
- ✅ Coherente con el análisis de sincronización de fases

#### **Arquitectura**
```
Época EEG → Grafo de Sincronización → GAT (4 layers, 8 heads) → Clasificador
```

#### **Features**
- **Nodos**: Canales EEG o parcelas cerebrales (100 ROIs Schaefer)
- **Aristas**: Valores de sincronización > threshold
- **Node features**: Estadísticas de fase/amplitud, complejidad temporal
- **Edge features**: Fuerza de sincronización
- **Graph features**: Kuramoto mean/std, métricas topológicas

#### **Entrenamiento Completo**
```bash
cd /media/storage_hdd/dmt_fz/machine_learning/clf
./run_pipeline.sh
```

#### **Componentes**
- `data/dataset_builder.py` - Construcción de grafos desde phases-*.pkl
- `models/gat_model.py` - Arquitectura GAT con edge attributes
- `train.py` - Entrenamiento con TensorBoard y early stopping
- `analysis/analyze_graphs.py` - Análisis estadístico DMT vs EC vs EO
- `analysis/visualize_attention.py` - Visualización de attention weights

#### **Resultados**
- Accuracy esperado: 70-85% (baseline aleatorio: 33%)
- Visualizaciones: confusion matrix, training curves, attention heatmaps
- Análisis estadístico de propiedades de grafos por condición
- Comparación por bandas de frecuencia

**Documentación completa:** `machine_learning/clf/README.md`

---

## 📦 Archivos Generados

### Archivos Intermedios (NO subir a git)

```
fwd-inv-stc/
├── phases-*.pkl              (~100-500 MB cada uno)
│   └── Fases, amplitudes, syncros, kuramoto (EEG y STC)
│
├── order_all-*.pkl           (~50-200 MB cada uno)
│   └── DataFrames de fases filtradas por red
│
├── order-*.pkl               (~50-200 MB cada uno) ⭐ NUEVO
│   └── Series de order parameter por red
│
├── syncro-*.pkl              (~200-800 MB cada uno)
│   └── Métricas con splits temporales
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

## 🔧 Dependencias Principales

```python
# Neurociencia
mne                    # Forward/inverse solution
mne.datasets          # fsaverage template

# Signal Processing
scipy                  # Hilbert transform, filtering
numpy                 # Operaciones numéricas

# Machine Learning
sklearn               # Random Forest, PCA, clustering
sklearn_extra         # K-Medoids
optuna                # Optimización de hiperparámetros
tensorflow/keras      # EEGNet

# Visualización
matplotlib
seaborn

# Otros
pandas                # Manipulación de datos
tqdm                  # Progress bars
pickle                # Serialización
pymatreader          # Leer archivos .mat
```

### Instalación

```bash
conda create -n dmt_fz python=3.10
conda activate dmt_fz

pip install mne numpy scipy matplotlib pandas
pip install scikit-learn sklearn-extra
pip install optuna tqdm pymatreader
```

---

## 🚀 Orden de Ejecución Recomendado

### Pipeline Completo

```bash
# 1. Procesamiento principal (3-4 horas)
python fwd.py --jobs 0 --workers 7 --conditions DMT EC EO

# 2a. Filtrado por redes (2-5 min)
python multi2pool2.py

# 2b. Calcular order parameter (30 seg - 1 min) ⭐ NUEVO
python generate_order.py --workers 20 --conditions DMT EC EO

# 3. Visualización (5-10 min)
python plot_order.py --workers 20

# 4. Correlaciones (3-5 min)
python pearson.py

# 5. Clustering (2-4 horas, opcional)
python clustering.py
```

### Análisis de ML (independiente)

```bash
cd spectral_sources
python randomforest.py

cd ../EEGNet
python EEGNet\ REG.py
```

---

## 📝 Notas Importantes

### Configuración de Rutas

El proyecto usa `paths.py` para gestionar rutas:

```python
# dmt/paths.py
BASE_DIR = Path(__file__).resolve().parent
EEG_CLEAN_DIR = BASE_DIR / "EEG_CLEAN"
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"
```

### Sujetos Rechazados

```python
rejected_subjects = [2, 5, 8, 16, 23, 31]
# 29 sujetos válidos de 35 totales
```

### Formato de Datos

- **Input**: Archivos EEGLAB (`.set`)
- **Intermediate**: Archivos pickle (`.pkl`)
- **Output**: Figuras PNG, matrices numpy, DataFrames pandas

---

## 📚 Referencias Teóricas

### Parámetro de Orden de Kuramoto

Mide coherencia global de osciladores acoplados:

```
r(t) = |1/N Σ exp(iθⱼ(t))|

donde:
- θⱼ(t): fase del oscilador j en tiempo t
- N: número de osciladores
- r ∈ [0,1]: 0=desincronizado, 1=perfectamente sincronizado
```

### Matriz de Sincronización

Sincronización par a par basada en diferencias de fase:

```
S[i,j] = 1 - (Σ|Δθ(t)|) / (π·T)

donde:
- Δθ(t): diferencia angular entre señales i,j
- T: duración temporal
```

### Redes de Schaefer

Parcelas del atlas Schaefer2018 (100 regiones, 7 redes):
- Basado en conectividad funcional intrínseca
- Validado con datos de Human Connectome Project

---

## 📖 Documentación Adicional

- **QUICKSTART.md**: Guía rápida para empezar en 5 minutos
- **WORKFLOW.md**: Pipeline detallado paso a paso
- **TECHNICAL_DETAILS.md**: Detalles matemáticos y teóricos

---

## 👥 Contacto y Contribuciones

Para preguntas sobre el proyecto, consultar con el equipo de investigación.

---

## 📄 Licencia

Este proyecto es parte de una investigación académica.

---

**Última actualización:** 2025-01-13
