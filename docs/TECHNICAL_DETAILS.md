# Documentación Técnica Detallada

## 📐 Fundamentos Matemáticos

### 1. Transformada de Hilbert y Fase Instantánea

La fase instantánea se extrae mediante la transformada de Hilbert:

```python
# Para una señal real x(t)
z(t) = x(t) + i·H[x(t)]  # Señal analítica

# Donde H[x(t)] es la transformada de Hilbert
H[x(t)] = (1/π) ∫ x(τ)/(t-τ) dτ

# Fase instantánea
θ(t) = arctan(H[x(t)] / x(t)) = angle(z(t))

# Envolvente (amplitud)
A(t) = |z(t)| = sqrt(x²(t) + H²[x(t)])
```

**Implementación:**
```python
def hilbert_transform(band_signals, trim=False):
    for filtered_signal in band_signals:
        analytic_signal = hilbert(filtered_signal)  # scipy.signal
        envelope = np.abs(analytic_signal)
        inst_phase = np.angle(analytic_signal)
```

---

### 2. Diferencia Angular Normalizada

Para calcular la diferencia entre dos fases circulares:

```python
def diff_ang(theta1, theta2, full_p=2*π, abso=True):
    """
    Calcula la distancia angular más corta entre dos fases
    Maneja correctamente el wrapping en 2π
    """
    half_p = 0.5 * full_p
    fmod1 = (theta2 - theta1 + half_p) % full_p
    fmod2 = (fmod1 + full_p) % full_p - half_p
    
    return abs(fmod2) if abso else fmod2
```

**Propiedades:**
- Resultado en [-π, π] (o [0, π] si absolute=True)
- Invariante ante rotaciones
- Distancia geodésica en círculo unitario

---

### 3. Matriz de Sincronización

Sincronización par a par entre N osciladores:

```
S ∈ ℝ^(N×N), S[i,j] ∈ [0,1]

S[i,j] = 1 - (1/(π·T)) Σₜ |Δθᵢⱼ(t)|

donde:
- T: número de muestras temporales
- Δθᵢⱼ(t): diferencia angular entre señales i y j
- max_diff = π·T (desincronización máxima)
```

**Interpretación:**
- S[i,j] = 1: Perfecta sincronización (fases idénticas)
- S[i,j] = 0: Desincronización total (fases aleatorias)
- Matriz simétrica: S[i,j] = S[j,i]
- Diagonal: S[i,i] = 0 (por definición)

**Implementación:**
```python
def calculate_syncro(phase_mat):
    # phase_mat: (N_channels, T_samples)
    max_diff = np.pi * phase_mat.shape[1]
    size = phase_mat.shape[0]
    syncro_mat = np.zeros((size, size))
    
    for i, j in combinations(range(size), 2):
        signal1 = phase_mat[i,:]
        signal2 = phase_mat[j,:]
        value = 1 - (diff_ang(signal1, signal2).sum() / max_diff)
        syncro_mat[i,j] = value
        syncro_mat[j,i] = value
    
    return syncro_mat
```

---

### 4. Parámetro de Orden de Kuramoto

Medida de coherencia global derivada del modelo de Kuramoto:

```
r(t) = |R(t)| donde R(t) = (1/N) Σⱼ exp(i·θⱼ(t))

Expandiendo:
r(t) = (1/N)|Σⱼ cos(θⱼ(t)) + i·sin(θⱼ(t))|
     = (1/N)√[(Σⱼ cos(θⱼ))² + (Σⱼ sin(θⱼ))²]
```

**Propiedades:**
- r ∈ [0, 1]
- r = 0: Fases uniformemente distribuidas (máxima entropía)
- r = 1: Todos los osciladores en fase
- Invariante ante rotaciones globales

**Implementación:**
```python
def order_parameter(phase):
    # phase: (N_channels, T_samples)
    euler_notation = np.exp(1j * phase)
    r = np.abs(euler_notation.mean(axis=0))
    return r  # shape: (T_samples,)
```

**Promedio temporal:**
```python
r_mean = order_parameter(phase).mean()  # Valor único por época
```

---

## 🧮 Solución Inversa (Inverse Problem)

### Forward Solution

Modelo lineal del problema directo:

```
M = G·J + ε

donde:
- M ∈ ℝ^(N_sensors × T): Señales medidas en sensores
- G ∈ ℝ^(N_sensors × N_sources): Matriz de ganancia (lead field)
- J ∈ ℝ^(N_sources × T): Corrientes en fuentes
- ε: Ruido
```

**Lead Field Matrix:**
- Calculada con Boundary Element Method (BEM)
- Modelo de 3 capas: cerebro, cráneo, piel
- Geometría de fsaverage (template MRI)

### Inverse Solution (dSPM)

Dynamic Statistical Parametric Mapping:

```
Ĵ = W·M

donde W es el operador inverso:

W = R·Gᵀ·(G·R·Gᵀ + λ²·C)⁻¹

Parámetros:
- R: Matriz de covarianza de fuentes
- C: Matriz de covarianza de ruido
- λ²: Parámetro de regularización (1/SNR²)
- SNR = 3 (típico para EEG)
```

**Normalización:**
```
dSPM normaliza por la desviación estándar:
J_normalized = Ĵ / √(diagonal(W·W·C))
```

**Implementación en fwd.py:**
```python
# SNR = 3
lambda2 = 1. / (3. ** 2)  # 0.111...

# Covarianza de ruido (baseline)
noise_cov = mne.compute_covariance(
    epochs, 
    tmax=0., 
    method=['shrunk', 'empirical']
)

# Operador inverso
inverse_operator = make_inverse_operator(
    epochs.info, 
    fwd, 
    noise_cov, 
    loose=0.2,    # Permite orientaciones no normales
    depth=0.8     # Compensación por profundidad
)

# Aplicación
stc = apply_inverse_epochs(
    epochs, 
    inverse_operator, 
    lambda2,
    method='dSPM'
)
```

---

## 🌐 Atlas de Schaefer

### Parcelas Corticales

**Schaefer2018_100Parcels_7Networks_order:**
- 100 regiones de interés (ROIs)
- 7 redes funcionales
- Resolución espacial: ~2-3 cm

### Estructura Jerárquica

```
Nomenclatura: [Hemisferio]_[Red]_[Región]

Ejemplos:
- RH_FPN_PFCdPFCm: Right Hemisphere, Frontoparietal, Prefrontal Cortex dorsal medial
- LH_DMN_pCunPCC: Left Hemisphere, Default Mode, Precuneus/Posterior Cingulate
```

### Redes Funcionales

| Red | Código | Descripción | N° Parcelas |
|-----|--------|-------------|-------------|
| Visual | VN | Procesamiento visual | ~10 |
| Somatomotor | SMN | Control motor y somatosensorial | ~15 |
| Dorsal Attention | DAN | Atención orientada | ~12 |
| Salience/Ventral Attention | SVA | Detección de salience | ~15 |
| Limbic | LN | Procesamiento emocional | ~8 |
| Frontoparietal | FPN | Control ejecutivo | ~20 |
| Default Mode | DMN | Red por defecto | ~20 |

### Filtrado por Redes

```python
# Crear DataFrame de etiquetas
df_labels = pd.DataFrame()
df_labels["label"] = [x[:6] for x in label_names]
df_labels["hemi"] = [x[:2] for x in df_labels["label"]]
df_labels["net"] = [x[3:] for x in df_labels["label"]]

# Filtrar parcelas
def network_filter(phase, df_labels, hemi="both", net="all"):
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    df = (df[df["hemi"]==hemi] if hemi != "both" else df)
    df = (df[df["net"]==net] if net != "all" else df)
    df = df.drop(columns=["hemi","label","net"])
    return df

# Ejemplo: Solo Default Mode Network, ambos hemisferios
dmn_phases = network_filter(phase_data, df_labels, hemi="both", net="DMN")
```

---

## 📊 Clustering de Estados

### Eigenvalores de Matrices de Sincronización

Las matrices de sincronización son simétricas semi-definidas positivas:

```
S = Sᵀ, S[i,j] ≥ 0

Descomposición espectral:
S = QΛQᵀ

donde:
- Q: Matriz de eigenvectores
- Λ: Matriz diagonal de eigenvalores λᵢ
- λ₁ ≥ λ₂ ≥ ... ≥ λₙ ≥ 0
```

**Interpretación:**
- Eigenvalores grandes → Modos dominantes de sincronización
- Distribución de eigenvalores caracteriza el patrón de conectividad

**Extracción:**
```python
eigenvalues, eigenvectors = np.linalg.eigh(syncro_matrix)
# Tomar diagonal de eigenvectores
diag_eigenvalues = np.diag(eigenvectors)
```

### Pipeline de Clustering

```
1. Feature Extraction
   ↓
   [Eigenvalues] → (N_epochs × N_sources)

2. Dimensionality Reduction
   ↓
   PCA(n_components=k) → (N_epochs × k)

3. Clustering
   ↓
   K-Medoids(n_clusters=c) → Labels

4. Validation
   ↓
   Silhouette Score
```

### Optimización con Optuna

```python
def objective(trial):
    n_comps = trial.suggest_int("n_comps", 2, 25)
    k_clusters = trial.suggest_int("k_clusters", 2, 25)
    
    # Reducción
    pca = PCA(n_components=n_comps)
    data = pca.fit_transform(eigenvalues_data)
    
    # Clustering
    kmedoids = KMedoids(n_clusters=k_clusters).fit(data)
    labels = kmedoids.labels_
    
    # Validación
    distances = pairwise_distances(data, metric="euclidean")
    silhouette = silhouette_score(distances, labels)
    
    return silhouette  # Maximizar

# Grid search
study = optuna.create_study(
    direction="maximize",
    sampler=GridSampler(search_space)
)
study.optimize(objective, n_jobs=20)
```

**Coeficiente de Silueta:**
```
s(i) = (b(i) - a(i)) / max{a(i), b(i)}

donde:
- a(i): distancia promedio intra-cluster
- b(i): distancia promedio al cluster más cercano
- s ∈ [-1, 1]: 1=bien clusterizado, -1=mal asignado
```

---

## 🔀 Análisis de Transiciones (Markov)

### Cadena de Markov de Primer Orden

```
P(Xₙ = sⱼ | Xₙ₋₁ = sᵢ) = Tᵢⱼ

Matriz de transición T:
Tᵢⱼ = #(transiciones i→j) / #(estado i)

Propiedades:
- Σⱼ Tᵢⱼ = 1  (filas suman 1)
- Tᵢⱼ ∈ [0,1]
```

**Implementación:**
```python
def markov_model(n, sequence, threshold=None):
    model = {}
    for i in range(len(sequence)-1):
        current_state = sequence[i]
        next_state = sequence[i+n]
        if current_state not in model:
            model[current_state] = []
        model[current_state].append(next_state)
    
    # Calcular probabilidades
    model_stats = {}
    for state in model.keys():
        counts = Counter(model[state])
        total = sum(counts.values())
        probabilities = {k: v/total for k, v in counts.items()}
        model_stats[state] = probabilities
    
    return model_stats
```

---

## 📈 Análisis de Correlación

### Correlación de Pearson

```
r = Σᵢ(xᵢ - x̄)(yᵢ - ȳ) / √[Σᵢ(xᵢ - x̄)² · Σᵢ(yᵢ - ȳ)²]

r ∈ [-1, 1]:
- r > 0: correlación positiva
- r < 0: correlación negativa
- |r| > 0.5: correlación moderada/fuerte
```

### Corrección FDR (Benjamini-Hochberg)

Problema de múltiples comparaciones:
- 7 redes × 5 bandas × 23 cuestionarios = 805 tests
- Inflación de tasa de falsos positivos

**Procedimiento FDR:**
```
1. Ordenar p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ

2. Encontrar máximo k tal que:
   pₖ ≤ (k/m)·α

3. Rechazar H₀ para todos i ≤ k

donde:
- m: número de tests
- α: nivel de significancia (ej: 0.05)
```

**Implementación:**
```python
from statsmodels.stats.multitest import fdrcorrection

# Array plano de p-values
p_values_flat = pvalues_matrix.flatten()

# Corrección
rejected, p_corrected = fdrcorrection(p_values_flat, alpha=0.05)

# Reshape a forma original
rejected_matrix = rejected.reshape(n_sources, n_bands)
```

### Visualización con Codificación de Color

```python
def r_to_color(r):
    # Significativo: Color plasma
    return cm.plasma(abs(r))

def to_gray(r):
    # No significativo: Escala de grises
    rgba = cm.twilight_shifted(abs(r))
    rgb = to_rgb(rgba)
    gray = color.rgb2gray(rgb)
    return cm.gray(gray)
```

---

## 🤖 Machine Learning

### Random Forest para Regresión

**Arquitectura:**
```
Ensemble de 1000 árboles de decisión

Feature Selection:
1. Correlación bivariada (r de Pearson)
2. Corrección FDR (α=0.05)
3. Threshold: ≥5 features sobreviven

Cross-Validation:
- 1000 iteraciones
- 80% train / 20% test
- Random splits
- Métrica: r de Pearson entre predicho y real
```

**Workflow:**
```python
for score_idx in range(23):
    # 1. Seleccionar features significativas
    X = psd_data[:, significant_regions[score_idx]]
    y = target_data[:, score_idx]
    
    if X.shape[1] >= 5:  # Mínimo 5 features
        for iteration in range(1000):
            # 2. Split aleatorio
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=iteration
            )
            
            # 3. Entrenar
            rf = RandomForestRegressor(n_estimators=1000)
            rf.fit(X_train, y_train)
            
            # 4. Evaluar
            y_pred = rf.predict(X_test)
            r, p = pearsonr(y_test, y_pred)
            scores.append(r)
```

### EEGNet (Deep Learning)

**Arquitectura Simplificada:**
```
Input: (batch, channels, timepoints, 1)
    ↓
Conv2D_temporal (kernel=(1, 64))
    ↓
DepthwiseConv2D_spatial (kernel=(channels, 1))
    ↓
SeparableConv2D
    ↓
AveragePooling
    ↓
Flatten
    ↓
Dense → Dropout → Output
```

**Características:**
- Diseñado específicamente para EEG/BCI
- Parámetros reducidos vs CNN convencional
- Captura características temporales y espaciales
- 23 modelos entrenados (uno por variable dependiente)

---

## 💾 Formato de Datos

### Estructura de Pickles

```python
# phases-S01-DMT.pkl
{
    # EEG (24 canales)
    "filtered_eeg": {
        "Delta": [array(24, T)],  # Lista de épocas
        "Theta": [...],
        ...
    },
    "phases_eeg": {
        "Delta": [array(24, T-200)],  # Trimmed
        ...
    },
    
    # Sources (100 parcelas)
    "filtered_stc": {
        "Delta": [array(100, T)],
        ...
    },
    "phases_stc": {
        "Delta": [array(100, T-200)],
        ...
    },
    
    # Matrices de sincronización
    "syncros_eeg": {
        "Delta": [array(24, 24)],
        ...
    },
    "syncros_stc": {
        "Delta": [array(100, 100)],
        ...
    },
    
    # Parámetros de Kuramoto
    "kuramoto_eeg": {
        "Delta": [array(T-200)],  # Serie temporal de r
        ...
    },
    "kuramoto_stc": {
        "Delta": [array(T-200)],
        ...
    }
}
```

### Dimensiones Típicas

```
N_subjects: 29 (de 35 total, 6 rechazados)
N_epochs: ~100-200 por sujeto (varía)
N_channels_eeg: 24
N_sources: 100 (parcelas Schaefer)
N_networks: 7
N_bands: 5
T_samples: ~1000 (2 segundos @ 500 Hz)
T_trimmed: ~800 (se quitan 100 muestras al inicio y final)
```

---

## ⚡ Optimización y Performance

### Multiprocessing

```python
from multiprocessing import Pool

if __name__ == '__main__':
    with Pool(20) as p:  # 20 procesos paralelos
        results = p.map(do_the_math, file_list)
```

**Speedup estimado:**
- 1 core: ~30 min por sujeto
- 20 cores: ~1.5 min por sujeto (x20 speedup)
- Total (29 sujetos): ~45 min vs 15 horas

### Memory Management

```python
# Usar generators para no mantener todo en memoria
stc = apply_inverse_epochs(
    epochs, 
    inverse_operator,
    return_generator=True  # ← Importante
)

for epoch_stc in stc:
    # Procesar uno a la vez
    process(epoch_stc)
```

---

## 🐛 Debugging y Validación

### Checks de Sanidad

```python
# 1. Rango de fases
assert np.all((phase >= -np.pi) & (phase <= np.pi))

# 2. Parámetro r en [0,1]
assert np.all((r >= 0) & (r <= 1))

# 3. Matriz simétrica
assert np.allclose(syncro_mat, syncro_mat.T)

# 4. Sin NaNs
assert not np.any(np.isnan(data))
```

### Visualización de Debug

```python
# Plot fase instantánea
plt.plot(inst_phase[channel, :])
plt.ylim(-np.pi, np.pi)
plt.axhline(0, color='k', linestyle='--')

# Plot parámetro r temporal
plt.plot(order_parameter(phase_data))
plt.ylim(0, 1)
plt.ylabel('Kuramoto r')
```

---

## 📚 Referencias

### Papers Clave

1. **Kuramoto Model:**
   - Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators.

2. **Source Localization:**
   - Dale et al. (2000). Dynamic statistical parametric mapping.
   - Gramfort et al. (2014). MNE software for processing MEG and EEG data.

3. **Schaefer Atlas:**
   - Schaefer et al. (2018). Local-global parcellation of the human cerebral cortex.

4. **EEGNet:**
   - Lawhern et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces.

### Software

- MNE-Python: https://mne.tools/
- Scikit-learn: https://scikit-learn.org/
- Optuna: https://optuna.org/

---


