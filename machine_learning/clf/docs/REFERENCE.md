# Referencia Técnica

## Arquitectura del Modelo

```
Input Graph
    ↓
[Edge Encoder] (edge attributes → hidden dim)
    ↓
[GAT Layers 1-4] (multi-head attention + residual + batch norm + dropout)
    ↓
[Global Pooling] (attention / mean+max / Set2Set)
    ↓
[Concatenate graph features] (Kuramoto, topology)
    ↓
[MLP Classifier] (256 → 128 → 3 clases)
    ↓
Output: [DMT, EC, EO]
```

## Tipos de Convolución

### GATv2Conv (default)
```yaml
conv_type: "gatv2"
num_attention_heads: 8
use_edge_attr: true
```
- **Usa:** Attention mechanism para ponderar vecinos
- **Ventaja:** Interpretable (attention weights), usa edge attributes (sync values)
- **Desventaja:** Más parámetros, riesgo de overfitting en datasets pequeños

### ChebConv
```yaml
conv_type: "cheby"
cheby_k: 3
```
- **Usa:** Convolución espectral basada en Laplaciano del grafo
- **Ventaja:** Menos parámetros, más robusto, usa estructura global
- **Desventaja:** No usa edge attributes, menos interpretable

## Configuración de Grafos

### Nodos
- `use_stc: true` → 100 parcelas Schaefer (source space)
- `use_stc: false` → 24 canales EEG (electrode space)

### Aristas
- `fully_connected: true` → Todas las conexiones posibles (N×(N-1))
- `fully_connected: false` → Solo donde sync > `edge_threshold`

### Features de Nodo (extraídas de phases-*.pkl)
- Phase mean/std
- Amplitude mean/std  
- Temporal complexity (entropy, kurtosis, CV, range)

### Features de Grafo
- Kuramoto mean/std (coherencia/metastabilidad)
- Global sync mean/std
- Density, mean degree

## Parámetros de Entrenamiento

```yaml
training:
  num_epochs: 300           # Máximo de épocas
  batch_size: 32            # Ajustar según memoria GPU
  learning_rate: 0.001      # Reducir si no converge
  optimizer: "adam"         # adam, adamw, sgd
  
  scheduler:
    type: "reduce_on_plateau"
    patience: 20
    factor: 0.5
  
  early_stopping:
    patience: 50            # Stop si no mejora en N épocas
  
  gradient_clip: 1.0        # Clipping de gradientes
  label_smoothing: 0.1      # Regularización
```

## Ensemble Methods

```bash
# Promedio de probabilidades
python ensemble_bands.py --method average

# Promedio ponderado por accuracy de validación
python ensemble_bands.py --method weighted

# Votación mayoritaria
python ensemble_bands.py --method voting

# Máxima confianza
python ensemble_bands.py --method max

# Solo bandas específicas
python ensemble_bands.py --bands Alpha Theta Beta
```

## Análisis Post-Entrenamiento

```bash
# Estadísticas de grafos
python analysis/analyze_graphs.py

# Visualizar attention weights
python analysis/visualize_attention.py --checkpoint checkpoints_alpha/best_model.pt

# Evaluación detallada
python evaluate.py --checkpoint checkpoints/best_model.pt
```

## Uso Programático

```python
import torch
from models import BrainStateGAT

# Cargar modelo
checkpoint = torch.load('checkpoints/best_model.pt')
model = BrainStateGAT(**checkpoint['config']['model'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predecir
with torch.no_grad():
    output = model(graph_data)
    prediction = output.argmax(dim=1)
    
# Extraer attention weights
attention_weights = model.get_attention_weights(graph_data)

# Extraer embeddings intermedios
embeddings = model.get_embeddings(graph_data)
# embeddings['graph_embeddings'] → Pre-MLP embedding
# embeddings['logits'] → Post-MLP embedding
```

## Embeddings Extraídos

El modelo genera 3 tipos de embeddings guardados en `output_{band}/embeddings/`:

| Embedding | Shape | Descripción |
|-----------|-------|-------------|
| `graph_GAT_embedding` | (N, 1024) | Representación después de GAT + pooling |
| `graph_GAT_features_embedding` | (N, 1032) | + features de grafo (Kuramoto, etc.) |
| `graph_GAT_MLP_embedding` | (N, 3) | Logits finales (antes de softmax) |

### Uso para análisis
```python
import pickle

# Cargar embeddings
with open('output_alpha/embeddings/test_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)

embeddings = data['graph_GAT_embedding']  # (N, 1024)
labels = data['labels']                    # (N,)
conditions = data['conditions']            # ['DMT', 'EC', 'EO', ...]
subjects = data['subjects']                # ['S01', 'S02', ...]

# Visualizar con t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
emb_2d = tsne.fit_transform(embeddings)
```

## Matrices de Atención

Las matrices de atención se guardan con **nombres de electrodos** en los ejes:
- Formato: `NOMBRE-INDICE` (ej: `Fp1-0`, `Cz-12`, `O2-23`)
- Los 24 electrodos se cargan automáticamente de `extra.pkl`

```python
from utils import load_electrode_names, format_electrode_labels

ch_names = load_electrode_names()  # ['Fp1', 'Fp2', 'F3', ...]
labels = format_electrode_labels(ch_names)  # ['Fp1-0', 'Fp2-1', 'F3-2', ...]
```

### Atención por clase
```python
import pickle

with open('output_alpha/attention/attention_per_class.pkl', 'rb') as f:
    att = pickle.load(f)

# att['DMT'][0] = Matriz de atención promedio de DMT, capa 0
# att['EC'][0] = Matriz de atención promedio de EC, capa 0
diff = att['DMT'][0] - att['EC'][0]  # Diferencia DMT vs EC
```

## Requerimientos de Hardware

| Config | RAM | GPU VRAM | Tiempo (all bands) |
|--------|-----|----------|-------------------|
| Mínimo | 8GB | CPU only | ~4-6h |
| Recomendado | 16GB | 8GB+ | ~30-60min |
| Óptimo | 32GB | 16GB+ | ~20-30min |

## Outputs Generados

```
checkpoints_{band}/best_model.pt         # Modelo entrenado
output_{band}/test_results.json          # Métricas de test
output_{band}/confusion_matrix.png       # Matriz de confusión
output_{band}/training_curves.png        # Curvas de loss/accuracy
output_{band}/embeddings/                # Embeddings para análisis
    ├── train_embeddings.pkl
    └── test_embeddings.pkl
output_{band}/attention/                 # Matrices de atención
    ├── attention_distributions.png
    ├── attention_layer_1_average.png    # Con nombres de electrodos
    ├── attention_layer_2_average.png
    └── attention_per_class.pkl          # Atención promedio por clase
runs_{band}/                             # TensorBoard logs
band_comparison_results.json             # Comparación entre bandas
output/analysis/                         # Análisis estadístico
output/verification/                     # Verificación del dataset
    ├── class_separability_pca.png
    └── graph_features_all.csv
```

## Verificación del Dataset

Antes de entrenar, verifica la integridad del dataset:

```bash
python verify_dataset.py --config config/config.yaml
```

Genera:
- Distribución de labels (train/val/test)
- Estadísticas de features por clase
- Detección de data leakage (sujetos compartidos entre splits)
- Score de separabilidad de clases (silhouette score)
- Plot PCA de separabilidad

## Búsqueda de Hiperparámetros

Random search sobre los hiperparámetros más importantes:

```bash
python hyperparam_search.py --n_experiments 20 --bands Alpha --max_epochs 100
```

### Hiperparámetros en la búsqueda

| Categoría | Parámetro | Rango |
|-----------|-----------|-------|
| **Arquitectura** | hidden_dim | [64, 128, 256, 512] |
| | num_gat_layers | [2, 3, 4] |
| | num_attention_heads | [4, 8, 16] |
| | dropout | 0.1 - 0.6 |
| | concat_heads | [True, False] |
| **Pooling** | method | [mean, max, add, attention] |
| **MLP** | hidden_dims | [[256,128], [512,256], ...] |
| | dropout | 0.2 - 0.7 |
| **Training** | learning_rate | 1e-5 - 1e-2 (log) |
| | weight_decay | 1e-6 - 1e-3 (log) |
| | batch_size | [16, 32, 64, 128] |
| **Grafo** | fully_connected | [True, False] |
| | edge_threshold | 0.1 - 0.5 |

### Resultados
```bash
# Ver summary
cat hyperparam_search/search_*/summary.json

# TensorBoard (todos los experimentos)
tensorboard --logdir=hyperparam_search/
```

