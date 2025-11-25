# GAT Classifier para Estados Cerebrales EEG

Pipeline de clasificación de estados cerebrales (DMT vs EC vs EO) usando **Graph Attention Networks** sobre matrices de sincronización EEG.

## Justificación Científica

Este enfoque usa GAT porque:
- Respeta la **estructura de grafo** de la conectividad cerebral
- Aprovecha las **matrices de sincronización** ya calculadas en el pipeline
- Los **attention weights** permiten interpretar qué conexiones discriminan entre condiciones
- Integra **parámetros de Kuramoto** como features globales

## Estructura del Proyecto

```
clf/
├── config/config.yaml        # Configuración (editar aquí)
├── data/dataset_builder.py   # Construcción de grafos desde phases-*.pkl
├── models/gat_model.py       # Arquitectura GAT
├── analysis/
│   ├── analyze_graphs.py     # Análisis estadístico DMT vs EC
│   └── visualize_attention.py
├── tests/                    # Suite de validación de datasets
│   ├── test_dataset.py       # Tests completos (pytest compatible)
│   └── validate_before_train.py  # Validación rápida pre-training
├── train.py                  # Entrenamiento principal
├── train_per_band.py         # Entrenamiento por banda (recomendado)
├── ensemble_bands.py         # Combinar predicciones de múltiples bandas
├── evaluate.py               # Evaluación del modelo
├── verify_dataset.py         # Verificar integridad del dataset y labels
└── hyperparam_search.py      # Búsqueda de hiperparámetros (random search)
```

## Setup

```bash
# Crear entorno
conda create -n gat_clf python=3.10 && conda activate gat_clf

# PyTorch con CUDA (ajustar versión según tu GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip install torch-geometric pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Dependencias
pip install -r requirements.txt

# Verificar
python -c "import torch, torch_geometric; print('OK')"
```

## Uso Rápido

### Opción 1: Test rápido con banda Alpha (30-60 min)
```bash
./quick_train_alpha.sh
cat output_alpha/test_results.json
```

### Opción 2: Entrenamiento completo por banda (recomendado)
```bash
python train_per_band.py
cat band_comparison_results.json
```

### Opción 3: Ensemble (después de entrenar todas las bandas)
```bash
python ensemble_bands.py --method average
```

### Opción 4: Validar dataset antes de entrenar
```bash
# Validación completa (recomendado)
python tests/validate_before_train.py --config config/config.yaml

# O análisis detallado con visualizaciones
python verify_dataset.py --config config/config.yaml
```
Genera: distribución de labels, separabilidad de clases (silhouette score), detección de data leakage.

### Opción 5: Búsqueda de hiperparámetros
```bash
# 20 experimentos con random search
python hyperparam_search.py --n_experiments 20 --bands Alpha --max_epochs 100

# Ver resultados
tensorboard --logdir=hyperparam_search/
cat hyperparam_search/search_*/summary.json
```

### Monitorear entrenamiento
```bash
tensorboard --logdir=runs_alpha
# O todas las bandas juntas:
tensorboard --logdir_spec=Delta:runs_delta,Theta:runs_theta,Alpha:runs_alpha,Beta:runs_beta,Gamma:runs_gamma
```

### TensorBoard: Qué se loguea
- **Scalars**: Loss, Accuracy (train/val), Learning rate
- **Attention**: Distribuciones y heatmaps de attention weights por capa
- **Embeddings**: Visualización t-SNE/UMAP de 3 tipos:
  - `graph_GAT_embedding`: Después de capas GAT + pooling
  - `graph_GAT_features_embedding`: + features de grafo
  - `graph_GAT_MLP_embedding`: Logits finales (antes de softmax)

## Pipeline de Datos

```
phases-*.pkl (tu pipeline)
    ↓
Matriz de sincronización (NxN) → Aristas del grafo (pesos = sync)
    +
Fases/amplitudes (NxT) → Features de nodo (stats temporales)
    +
Kuramoto order parameter → Features globales (coherencia, metastabilidad)
    ↓
Grafo PyTorch Geometric → GAT → Clasificación [DMT, EC, EO]
```

## Configuración Esencial

Editar `config/config.yaml`:

```yaml
data:
  use_stc: true              # true: 100 parcelas Schaefer | false: 24 electrodos
  bands: ["Alpha"]           # Bandas a usar (o todas: Delta, Theta, Alpha, Beta, Gamma)
  graph:
    fully_connected: true    # true: todas las conexiones | false: solo > threshold
    edge_threshold: 0.3      # Umbral si fully_connected: false

model:
  architecture:
    conv_type: "gatv2"       # "gatv2" (con attention) o "cheby" (espectral)
    hidden_dim: 128
    num_gat_layers: 4
    num_attention_heads: 8
    dropout: 0.4

training:
  num_epochs: 300
  batch_size: 32
  learning_rate: 0.001
  early_stopping:
    patience: 50
```

## Por qué Entrenar por Banda

**Problema del multi-banda mezclado:** Si mezclas grafos de diferentes bandas en un dataset, el modelo no sabe de qué banda viene cada grafo. Un grafo Alpha-DMT puede parecerse más a Alpha-EC que a Delta-DMT, confundiendo patrones de banda con patrones de condición.

**Solución:** Entrenar modelos separados por banda permite:
- Comparar qué banda es más informativa para clasificar DMT
- Resultados interpretables y publicables
- Ensemble final que combina información de todas las bandas

## Resultados Esperados

| Banda | Accuracy Esperado | Motivo |
|-------|-------------------|--------|
| Alpha | 75-85% | Ritmo dominante en reposo, fuertemente modulado por psicodélicos |
| Theta | 70-80% | Memoria, navegación, estados alterados |
| Beta  | 70-80% | Atención, procesamiento activo |
| Delta | 65-75% | Estados profundos |
| Gamma | 60-70% | Alta variabilidad |

**Ensemble:** 80-90% (mejora +2-5% sobre mejor banda individual)

## Troubleshooting

| Problema | Solución |
|----------|----------|
| "No graphs were created" | Verificar `phases_dir` en config |
| "CUDA out of memory" | Reducir `batch_size` a 16 o 8 |
| Accuracy ~33% (random) | Verificar datos, reducir learning_rate a 0.0001 |
| "Dataset cache exists" | Usar `--force-rebuild` |

## Comparación con Otros Métodos

| Método | Input | Ventaja | Cuándo usar |
|--------|-------|---------|-------------|
| EEGNet | EEG crudo | End-to-end | Máximo poder predictivo |
| Random Forest | PSD features | Rápido, interpretable | Baseline |
| **GAT** | Grafos sync | Usa tu pipeline, attention interpretable | Leveraging análisis de sync |

## Tests y Validación

El proyecto incluye una suite de tests para validar el dataset **antes** de entrenar, evitando errores comunes.

### Estructura de Tests

```
tests/
├── __init__.py
├── test_dataset.py          # Suite completa de validaciones
└── validate_before_train.py # Script rápido pre-entrenamiento
```

### Qué se valida

| Test | Qué verifica | Por qué importa |
|------|--------------|-----------------|
| `check_labels_valid` | Labels en rango [0, N-1] | Labels fuera de rango causan errores de training |
| `check_class_distribution` | Distribución similar entre splits | Diferencias >30% indican problemas de split |
| `check_data_leakage` | Ningún sujeto en múltiples splits | Data leakage causa overfitting artificial |
| `check_feature_integrity` | No NaN/Inf en features | Valores inválidos corrompen el modelo |
| `check_graph_structure` | Shapes correctas, grafos no vacíos | Grafos malformados causan errores |

### Cómo correr los tests

```bash
# 1. Validación rápida (antes de entrenar)
python tests/validate_before_train.py --config config/config.yaml

# 2. Validación desde cache existente
python tests/test_dataset.py --cache data/cache/dataset_*.pkl --classes DMT EC

# 3. Con pytest (para CI/CD)
pytest tests/test_dataset.py -v

# 4. Tests específicos
pytest tests/test_dataset.py::test_no_data_leakage -v
pytest tests/test_dataset.py::test_class_distribution_balanced -v
```

### Validación automática en training

La validación se ejecuta **automáticamente** al inicio de `train.py`. Si hay errores críticos, el entrenamiento no comienza.

Para desactivar (no recomendado):
```yaml
# En config.yaml
strict_validation: false
```

### Output de ejemplo

```
======================================================================
DATASET VALIDATION
======================================================================

--- INFO ---
✓ train: All 6700 labels valid (range [0, 1])
✓ val: All 1300 labels valid (range [0, 1])
✓ test: All 1400 labels valid (range [0, 1])
✓ train: DMT: 3500 (52.2%), EC: 3200 (47.8%)
✓ val: DMT: 680 (52.3%), EC: 620 (47.7%)
✓ test: DMT: 730 (52.1%), EC: 670 (47.9%)
✓ No subject overlap between train and val (15 train, 3 val)
✓ No subject overlap between train and test (15 train, 4 test)
✓ No NaN values

======================================================================
VALIDATION PASSED ✓
======================================================================
```

### Errores comunes detectados

| Error | Causa | Solución |
|-------|-------|----------|
| "DATA LEAKAGE: subjects in both train and val" | Split no agrupa por sujeto | Activar `group_by_subject: true` |
| "train/val ratio differs by >30%" | Split no estratificado | El fix ya está aplicado, borrar cache |
| "NaN in node features" | Datos corruptos en phases-*.pkl | Revisar pipeline de preprocesamiento |
| "Invalid label 2, expected [0, 1]" | Más clases de las esperadas | Verificar `conditions` en config |

## Referencias

- Veličković et al. (2018). Graph Attention Networks. ICLR.
- Brody et al. (2021). How Attentive are Graph Attention Networks? (GATv2)
- Defferrard et al. (2016). Chebyshev Convolutions on Graphs.
