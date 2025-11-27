# Spectral Sources - Datos para Machine Learning

Dataset de potencia espectral de fuentes EEG durante experiencia con DMT para predicción de experiencia subjetiva.

## Estructura del Dataset

### Features (Potencia Espectral)
Matrices de **29 sujetos × 90 regiones cerebrales** (atlas AAL90).

| Archivo | Banda | Condición |
|---------|-------|-----------|
| `alpha.csv` | Alpha (8-12 Hz) | Baseline (pre-DMT) |
| `beta.csv` | Beta (12-30 Hz) | Baseline |
| `delta.csv` | Delta (1-4 Hz) | Baseline |
| `theta.csv` | Theta (4-8 Hz) | Baseline |
| `gamma1.csv` | Gamma bajo (30-50 Hz) | Baseline |
| `gamma2.csv` | Gamma alto (50-100 Hz) | Baseline |
| `DMT_alpha.csv` | Alpha | Durante DMT |
| `DMT_beta.csv` | Beta | Durante DMT |
| `DMT_delta.csv` | Delta | Durante DMT |
| `DMT_theta.csv` | Theta | Durante DMT |
| `DMT_gamma1.csv` | Gamma bajo | Durante DMT |
| `DMT_gamma2.csv` | Gamma alto | Durante DMT |

### Targets (Variables a Predecir)
**`target.csv`**: 29 sujetos × 23 escalas de experiencia subjetiva.

| Cuestionario | Variables |
|--------------|-----------|
| **ASC** (Estados Alterados de Consciencia) | Unity, Spiritual, Blissful, Insightfulness, Disembodiment, Impaired, Anxiety, Complex imagery, Elementary imagery, Audiovisual, Changed |
| **NDE** (Experiencia Cercana a la Muerte) | Cognition, Affect, Paranormal, Transcendental |
| **MEQ** (Experiencia Mística) | Mystical, Positive, Transcendental, Ineffability, Awe |
| **Post** | Social, Fusion, Setting |

> Nota: Las primeras 11 columnas son ASC (0-100), luego 4 NDE, 5 MEQ y 3 Post (diversas escalas).

### Archivos Auxiliares
- **`AAL90.csv`**: Atlas AAL90 - 90 regiones con índice, nombre, abreviatura y coordenadas MNI (x,y,z)
- **`target_labels.txt`**: Nombres de las 23 variables target
- **`componentes.csv`**: Matriz de componentes/pesos ICA

## Uso para Machine Learning

### Carga de Datos
```python
import pandas as pd
import numpy as np

folder = "spectral_sources/"

# Cargar labels
aal90 = pd.read_csv(folder+"AAL90.csv", delimiter=";", index_col=0)
labels = pd.read_csv(folder+'target_labels.txt', header=None)[0].tolist()

# Cargar features y targets
X_baseline = pd.read_csv(folder+'theta.csv', names=aal90["Label"].tolist())
X_dmt = pd.read_csv(folder+'DMT_theta.csv', names=aal90["Label"].tolist())
y = pd.read_csv(folder+'target.csv', names=labels)
```

### Redes Cerebrales Predefinidas
```python
networks = {
    "FPN": [6,7,10,11,12,13,60,61,64,65],           # Frontoparietal
    "DMN": [22,23,30,31,34,35,64,65,66,67],         # Default Mode
    "Sensorimotor": [0,1,6,7,18,19,56,57,62,63,68,69],
    "Visual": [42,43,44,45,46,47,48,49,50,51,53],
    "Frontal": list(range(32)),
    "Parietal": [32,33,34,35,56,57,58,59,60,61,62,63,64,65,66,67,68,69],
    "Temporal": [36,37,38,39,78,79,80,81,82,83,84,85,86,87,88,89],
    "Subcortical": [40,41,70,71,72,73,74,75,76,77],
    "Occipital": list(range(42,56))
}
```

## Tareas de Predicción Sugeridas

| Tarea | Tipo | Descripción |
|-------|------|-------------|
| Regresión directa | Regresión | Predecir scores de cuestionarios a partir de potencia espectral |
| Clasificación binaria | Clasificación | Alta vs baja intensidad de experiencia (mediana split) |
| Multi-output | Regresión | Predecir múltiples dimensiones simultáneamente |
| Cambio pre-post | Regresión | Usar diferencia DMT-Baseline como features |

## Scripts Incluidos

| Script | Descripción |
|--------|-------------|
| `randomforest.py` | Correlación de Pearson + Random Forest con corrección FDR |
| `DMT Reg.py` | Random Forest explorando todas las bandas y redes |
| `EEGNet.py` | Deep learning (EEGNet) para clasificación/regresión |
| `DMT Autogluon.py` | AutoML con AutoGluon |

## Consideraciones

- **N pequeño**: Solo 29 sujetos, usar validación cruzada leave-one-out o k-fold
- **Alta dimensionalidad**: 90 features × 6 bandas = 540 features potenciales
- **Multicolinealidad**: Regiones cercanas están correlacionadas
- **Múltiples targets**: Considerar corrección por comparaciones múltiples

