# Experience Predictor

Multi-output regression model for predicting **23 subjective experience scores** from EEG data.

## Input Types

El modelo soporta **dos fuentes de datos** configurables:

| Input Type | Fuente | Datos | Dimensión |
|------------|--------|-------|-----------|
| `spectral` | `spectral_sources/` | Potencia espectral (CSVs) | 29 sujetos × 1620 features |
| `graphs` | `fwd-inv-stc/` | Grafos de sincronización (phases-*.pkl) | 29 sujetos × ~24 nodos × epochs |

### Spectral Mode
Usa potencia espectral en **90 regiones AAL** × **6 bandas**:
- Baseline (pre-DMT): alpha, beta, delta, theta, gamma1, gamma2
- Durante DMT: DMT_alpha, DMT_beta, etc.
- Diferencia: DMT - Baseline

### Graph Mode
Usa grafos de sincronización de `fwd-inv-stc/`:
- Nodos: 24 electrodos (o 100 parcelas si `use_stc: true`)
- Edges: Sincronización entre canales
- Features: Estadísticas de fase, amplitud, Kuramoto
- Agregación: Múltiples epochs → 1 grafo por sujeto

## Model Types

| Model | Descripción | Compatible con |
|-------|-------------|----------------|
| `mlp` | MLP simple | spectral (flat) |
| `gnn` | Graph Neural Network | spectral (con grafo AAL) o graphs |
| `hybrid` | GNN + features adicionales | graphs |

### GNN Architectures
- `gatv2`: Graph Attention Networks v2
- `gcn`: Graph Convolutional Networks
- `sage`: GraphSAGE
- `cheby`: Chebyshev spectral convolutions

## Usage

### Cambiar configuración

Editar `config/config.yaml`:

```yaml
# Elegir fuente de datos
input_type: "spectral"  # o "graphs"

# Elegir modelo
model:
  type: "mlp"  # "mlp", "gnn", o "hybrid"
  gnn:
    conv_type: "gatv2"  # "gatv2", "gcn", "sage", "cheby"
```

### Ejemplos de configuración

**1. MLP con datos espectrales (rápido, baseline):**
```yaml
input_type: "spectral"
model:
  type: "mlp"
```

**2. GNN sobre regiones AAL:**
```yaml
input_type: "spectral"
spectral:
  build_graph: true  # Construir grafo de 90 regiones
model:
  type: "gnn"
  gnn:
    conv_type: "gatv2"
```

**3. GNN con grafos de sincronización:**
```yaml
input_type: "graphs"
model:
  type: "gnn"
  gnn:
    conv_type: "cheby"  # Chebyshev convolutions
```

### Ejecutar

```bash
cd machine_learning/experience_predictor
python train.py --config config/config.yaml
```

## Output

```
output/
├── cv_results.json      # Métricas de cross-validation
└── training.log         # Log de entrenamiento
```

## Métricas

- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error
- **R²**: Coefficient of determination
- **Pearson r**: Correlación por target (promedio)

## Estructura de Carpetas

```
experience_predictor/
├── config/
│   └── config.yaml       # Configuración completa
├── data/
│   ├── __init__.py
│   ├── loader.py         # Carga datos espectrales
│   ├── dataset.py        # Dataset PyTorch
│   └── graph_loader.py   # Carga grafos de fwd-inv-stc
├── models/
│   ├── __init__.py
│   └── regressor.py      # MLP, GNN, Hybrid
├── train.py              # Script principal
├── run.sh
├── requirements.txt
└── README.md
```

## Targets (23 experiencias subjetivas)

| Categoría | Escalas |
|-----------|---------|
| **ASC** (11) | Unity, Spiritual, Blissful, Insightfulness, Disembodiment, Impaired, Anxiety, Complex imagery, Elementary imagery, Audiovisual, Changed |
| **NDE** (4) | Cognition, Affect, Paranormal, Transcendental |
| **MEQ** (5) | Mystical, Positive, Transcendental, Ineffability, Awe |
| **Post** (3) | Social, Fusion, Setting |

## Notas

- Los reportes de experiencia subjetiva **solo aplican a DMT** (no EC/EO)
- Con `input_type: "graphs"`, solo se cargan datos de la condición DMT
- N=29 sujetos: se usa k-fold CV (5 folds por defecto)
