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
```

## Requerimientos de Hardware

| Config | RAM | GPU VRAM | Tiempo (all bands) |
|--------|-----|----------|-------------------|
| Mínimo | 8GB | CPU only | ~4-6h |
| Recomendado | 16GB | 8GB+ | ~30-60min |
| Óptimo | 32GB | 16GB+ | ~20-30min |

## Outputs Generados

```
checkpoints_{band}/best_model.pt    # Modelo entrenado
output_{band}/test_results.json     # Métricas de test
output_{band}/confusion_matrix.png  # Matriz de confusión
output_{band}/training_curves.png   # Curvas de loss/accuracy
runs_{band}/                        # TensorBoard logs
band_comparison_results.json        # Comparación entre bandas
output/analysis/                    # Análisis estadístico
```

