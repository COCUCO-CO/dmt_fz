# Pipeline de Limpieza EEG - Guía de Implementación

## Resumen del Visualizador Actual

El visualizador en `/eeg_viewer` ya implementa:
- ✅ Carga de archivos (BDF, EDF, SET, FIF)
- ✅ Visualización de señales EEG con navegación temporal
- ✅ Filtro Notch (50/60 Hz)
- ✅ Filtro Bandpass
- ✅ Análisis FFT
- ✅ Transformada de Hilbert
- ✅ Selección de canales
- ✅ Generación y guardado de epochs

---

## Pasos de Limpieza EEG (Orden Estándar)

### **PASO 1: Carga y Visualización Inicial**
> *Ya implementado*

- Cargar archivo raw
- Mostrar información básica: sfreq, canales, duración
- Visualizar señal completa para inspección inicial

---

### **PASO 2: Filtrado Temporal**
> *Parcialmente implementado*

**2.1 Filtro High-Pass (recomendado: 0.1-1 Hz)**
- Elimina drift DC y fluctuaciones lentas
- Típicamente 0.1 Hz para ERP, 1 Hz para análisis de bandas
- **UI**: Slider + número editable

**2.2 Filtro Low-Pass (recomendado: 45-100 Hz)**
- Elimina ruido de alta frecuencia
- Típicamente 45 Hz si hay mucho ruido muscular
- **UI**: Slider + número editable

**2.3 Filtro Notch (50/60 Hz)**
> *Ya implementado*
- Elimina interferencia de línea eléctrica
- 50 Hz (Europa/Argentina) o 60 Hz (USA)

**Mejora sugerida**: Añadir presets:
- "Standard" (0.1-45 Hz + notch 50)
- "Alpha/Theta" (1-30 Hz + notch)
- "Custom"

---

### **PASO 3: Identificación de Canales Malos**

**Objetivo**: Detectar canales con problemas (flat, muy ruidosos, desconectados)

**Métodos de detección**:
1. **Flat channels**: Varianza < umbral (señal plana)
2. **Noisy channels**: Varianza > umbral o correlación baja con vecinos
3. **Bridged channels**: Correlación excesivamente alta entre canales adyacentes

**UI sugerida**:
```
┌─────────────────────────────────────────────────┐
│ // BAD CHANNELS                                 │
├─────────────────────────────────────────────────┤
│ [AUTO DETECT]  Threshold: [3.0] σ               │
│                                                 │
│ Detected: Fp1, F7, T3 (flat/noisy)              │
│                                                 │
│ ☑ Fp1 (flat - var: 0.02)                        │
│ ☑ F7  (noisy - var: 523.4)                      │
│ ☐ T3  (borderline - var: 89.2)                  │
│                                                 │
│ Manual add: [________] [+]                      │
│                                                 │
│ Action: ○ Mark only  ● Interpolate  ○ Exclude  │
│ [APPLY]                                         │
└─────────────────────────────────────────────────┘
```

**Acciones posibles**:
- **Mark only**: Solo marcarlos (se excluyen del promedio de re-referencia)
- **Interpolate**: Reconstruir usando canales vecinos (esférica/spline)
- **Exclude**: Eliminar del análisis

---

### **PASO 4: Re-Referenciación**

**Objetivo**: Establecer una referencia común para todos los canales

**Opciones comunes**:
1. **Average Reference**: Promedio de todos los canales (estándar para análisis de fuentes)
2. **Linked Mastoids (A1+A2)/2**: Común en estudios clínicos
3. **Single Mastoid (A1 o A2)**: Más simple
4. **Cz Reference**: Usado a veces para P300
5. **REST**: Referencia a infinito (más sofisticado)

**UI sugerida**:
```
┌─────────────────────────────────────────────────┐
│ // RE-REFERENCE                                 │
├─────────────────────────────────────────────────┤
│ Current ref: [Original/Unknown]                 │
│                                                 │
│ New reference:                                  │
│ ○ Average (all channels)                        │
│ ● Linked Mastoids (A1+A2)                       │
│ ○ Single electrode: [____▼]                     │
│ ○ Keep original                                 │
│                                                 │
│ ☑ Exclude bad channels from average             │
│                                                 │
│ [PREVIEW]  [APPLY]                              │
└─────────────────────────────────────────────────┘
```

---

### **PASO 5: Análisis ICA (Independent Component Analysis)**

**Objetivo**: Separar fuentes de señal para identificar y remover artefactos

**Artefactos típicos detectables con ICA**:
1. **Parpadeos (blinks)**: Componente frontal con picos claros
2. **Movimientos oculares**: Componente con gradiente frontal lateral
3. **ECG/Pulso**: Patrón cardíaco regular
4. **Ruido muscular (EMG)**: Componente con actividad de alta frecuencia
5. **Ruido de línea**: Componente con pico en 50/60 Hz

**Flujo de trabajo ICA**:
1. Calcular ICA (típicamente 20-25 componentes o n_channels - n_bad)
2. Visualizar componentes (topomap + timeseries)
3. Marcar componentes a excluir
4. Aplicar ICA (reconstruir señal sin componentes malos)

**UI sugerida**:
```
┌─────────────────────────────────────────────────────────────┐
│ // ICA DECOMPOSITION                                        │
├─────────────────────────────────────────────────────────────┤
│ Components: [20]  Method: [fastica▼]  [RUN ICA]             │
│ Status: ● Computing... (2:34 remaining)                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐     │
│  │ ICA000 │ │ ICA001 │ │ ICA002 │ │ ICA003 │ │ ICA004 │     │
│  │ (topo) │ │ (topo) │ │ (topo) │ │ (topo) │ │ (topo) │     │
│  │  ~~~   │ │  ~~~   │ │  ~~~   │ │  ~~~   │ │  ~~~   │     │
│  │ [BLINK]│ │ [ OK ] │ │ [EMG] │ │ [HEART]│ │ [ OK ] │     │
│  │   ☑    │ │   ☐    │ │   ☑    │ │   ☑    │ │   ☐    │     │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘     │
│                                                             │
│ Selected component: ICA000                                  │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ [Topomap grande]     [Timeseries del componente]        │ │
│ │                      [Power spectrum]                   │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [AUTO-DETECT BLINKS]  [AUTO-DETECT HEART]                   │
│                                                             │
│ Components to exclude: ICA000, ICA002, ICA003               │
│ [PREVIEW CLEANED]  [APPLY ICA]                              │
└─────────────────────────────────────────────────────────────┘
```

**Detección automática** (opcional pero muy útil):
- MNE tiene `find_bads_eog()` para blinks
- Puede usarse correlación con canal ECG para heartbeat
- ICLabel o similar para clasificación automática

---

### **PASO 6: Epoching (Segmentación)**
> *Ya implementado básico*

**Mejoras sugeridas**:
- Eventos/marcadores si están disponibles
- Overlapping configurable
- Baseline correction por epoch

---

### **PASO 7: Rechazo de Epochs**

**Objetivo**: Excluir epochs con artefactos residuales

**Criterios de rechazo**:
1. **Peak-to-peak amplitude**: Rechazar si amplitud > umbral (ej: 150 µV)
2. **Flat epochs**: Rechazar si varianza < umbral
3. **Gradient**: Rechazar si cambio muy abrupto

**UI sugerida**:
```
┌─────────────────────────────────────────────────┐
│ // EPOCH REJECTION                              │
├─────────────────────────────────────────────────┤
│ Thresholds:                                     │
│   Peak-to-peak: [150] µV                        │
│   Flat:         [0.5] µV                        │
│   Gradient:     [10] µV/ms                      │
│                                                 │
│ [AUTO DETECT]                                   │
│                                                 │
│ Epochs: 234 total                               │
│   ● Good: 198 (84.6%)                           │
│   ● Bad:  36  (15.4%)                           │
│                                                 │
│ Preview bad epochs: [◀] Epoch 12/36 [▶]         │
│ ┌─────────────────────────────────────────────┐ │
│ │        [Vista del epoch malo]               │ │
│ └─────────────────────────────────────────────┘ │
│ Reason: Peak-to-peak 234.5 µV (Fp1)             │
│                                                 │
│ [☐ Keep this epoch]                             │
│                                                 │
│ [APPLY REJECTION]  [SAVE CLEAN EPOCHS]          │
└─────────────────────────────────────────────────┘
```

---

### **PASO 8: Exportar/Guardar**

**Formatos de salida**:
- `.fif` (MNE native, preserva todo el preprocesamiento)
- `.set` (EEGLAB compatible)
- `.npy` (epochs como numpy array)
- `.csv` (metadata)

**Guardar también**:
- Log de preprocesamiento (qué pasos se aplicaron)
- Lista de canales malos
- Lista de epochs rechazados
- Componentes ICA excluidos

---

## Diseño UX/UI Recomendado

### **Principios de diseño**:

1. **Flujo lineal con steps visibles**
   - Sidebar o stepper mostrando en qué paso está
   - Permitir volver atrás sin perder trabajo

2. **Preview antes de aplicar**
   - Siempre mostrar antes/después
   - Botón "Preview" separado de "Apply"

3. **Undo/Historia**
   - Stack de operaciones aplicadas
   - Poder revertir al estado anterior

4. **Persistencia de sesión**
   - Auto-guardar progreso
   - Poder cerrar y continuar después

### **Layout sugerido**:

```
┌──────────────────────────────────────────────────────────────────────────┐
│  EEG_CLEANER v1.0                                    [UNDO] [SAVE] [EXIT]│
├────────────────┬─────────────────────────────────────────────────────────┤
│                │                                                         │
│ // PIPELINE    │  // MAIN VIEW                                           │
│                │  ┌───────────────────────────────────────────────────┐  │
│ ● 1. LOAD      │  │                                                   │  │
│ ○ 2. FILTER    │  │           [Visualización principal]               │  │
│ ○ 3. BAD CH    │  │           (EEG/ICA/Epochs según paso)              │  │
│ ○ 4. RE-REF    │  │                                                   │  │
│ ○ 5. ICA       │  │                                                   │  │
│ ○ 6. EPOCHS    │  └───────────────────────────────────────────────────┘  │
│ ○ 7. REJECT    │                                                         │
│ ○ 8. EXPORT    │  // CONTROLS (contextuales al paso actual)              │
│                │  ┌───────────────────────────────────────────────────┐  │
│ ────────────── │  │  [Controles específicos del paso activo]          │  │
│                │  │                                                   │  │
│ // HISTORY     │  │  [PREVIEW]           [APPLY]           [NEXT →]   │  │
│                │  └───────────────────────────────────────────────────┘  │
│ ↩ Filter applied│                                                        │
│ ↩ 3 bad ch marked│  // INFO BAR                                          │
│                │  File: S01_DMT.bdf | 512 Hz | 24 ch | Duration: 5:23   │
│                │                                                         │
└────────────────┴─────────────────────────────────────────────────────────┘
```

### **Atajos de teclado sugeridos**:

| Tecla | Acción |
|-------|--------|
| `Space` | Play/Pause reproducción |
| `←/→` | Navegar tiempo |
| `Ctrl+Z` | Undo |
| `Ctrl+S` | Guardar |
| `1-8` | Ir a paso N |
| `Enter` | Apply paso actual |
| `P` | Preview |
| `N` | Next step |

---

## Orden de Implementación Sugerido

### Prioridad Alta (Core functionality):
1. **Step 3: Bad Channels** - Detección automática + manual
2. **Step 4: Re-Reference** - Average y Linked Mastoids
3. **Step 5: ICA básico** - Calcular, visualizar, excluir manual
4. **Step 7: Epoch Rejection** - Por amplitud

### Prioridad Media (Mejoras importantes):
5. ICA auto-detect (blinks, heart)
6. Preview antes/después en todos los pasos
7. Historia/Undo
8. Presets de filtrado

### Prioridad Baja (Nice to have):
9. Interpolación de canales
10. ICLabel automático
11. REST reference
12. Batch processing

---

## Código de Ejemplo: Detección de Canales Malos

```python
import numpy as np
from scipy import stats

def detect_bad_channels(raw, std_threshold=3.0, flat_threshold=1e-7, 
                        corr_threshold=0.4):
    """
    Detectar canales malos automáticamente.
    
    Returns:
        dict: {'flat': [...], 'noisy': [...], 'uncorrelated': [...]}
    """
    data = raw.get_data(picks='eeg')
    ch_names = raw.ch_names
    
    bad = {'flat': [], 'noisy': [], 'uncorrelated': []}
    
    # Varianza por canal
    variances = np.var(data, axis=1)
    mean_var = np.mean(variances)
    std_var = np.std(variances)
    
    for i, (ch, var) in enumerate(zip(ch_names, variances)):
        # Flat channels
        if var < flat_threshold:
            bad['flat'].append(ch)
            continue
            
        # Noisy channels (varianza muy alta)
        z_score = (var - mean_var) / std_var
        if z_score > std_threshold:
            bad['noisy'].append(ch)
            continue
            
        # Correlación con vecinos (simplificado)
        correlations = []
        for j, other_data in enumerate(data):
            if i != j:
                r, _ = stats.pearsonr(data[i], other_data)
                correlations.append(abs(r))
        
        if np.mean(correlations) < corr_threshold:
            bad['uncorrelated'].append(ch)
    
    return bad
```

---

## Código de Ejemplo: ICA con MNE

```python
from mne.preprocessing import ICA

def compute_ica(raw, n_components=20, method='fastica', max_iter=500):
    """Calcular ICA."""
    # Filtrar high-pass a 1Hz para ICA (recomendado)
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None)
    
    ica = ICA(n_components=n_components, method=method, 
              max_iter=max_iter, random_state=42)
    ica.fit(raw_filtered)
    
    return ica

def plot_ica_components(ica, raw, n_cols=5):
    """Visualizar componentes ICA."""
    # Topomaps
    fig_topo = ica.plot_components(picks=range(ica.n_components_), 
                                    show=False)
    
    # Timeseries
    fig_sources = ica.plot_sources(raw, show=False)
    
    return fig_topo, fig_sources

def auto_detect_eog(ica, raw, ch_name='Fp1', threshold=3.0):
    """Detectar componentes de EOG automáticamente."""
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=ch_name,
                                                  threshold=threshold)
    return eog_indices, eog_scores
```

---

## Notas Técnicas

### Consideraciones para 250 Hz vs 500 Hz:
- **Filtro low-pass**: max = sfreq/2 (Nyquist)
  - 250 Hz → max lowpass = 125 Hz
  - 500 Hz → max lowpass = 250 Hz
- **ICA**: Puede requerir más epochs para 500 Hz (más samples)
- **Épocas**: Ajustar duración considerando sfreq

### Dispositivos Bluetooth vs Cableados:
- Bluetooth suele tener más artefactos de movimiento
- Considerar filtro high-pass más agresivo (1 Hz) para Bluetooth
- Mayor drift en Bluetooth → baseline correction más importante

### 24 canales:
- ICA recomendado: 15-20 componentes (dejar margen)
- Con pocos canales, ICA es más sensible a canales malos
- Siempre marcar canales malos ANTES de ICA

---

## Siguiente Paso

Una vez aprobado este documento, se procederá a implementar los pasos faltantes en el visualizador existente (`/eeg_viewer/main.py`), integrándolos con el código actual y manteniendo la estética "terminal/konsole" ya definida en `config.py`.


