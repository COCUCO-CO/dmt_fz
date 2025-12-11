# BrainStateVAE: Variational Autoencoder for EEG Synchronization Graphs

Variational Autoencoder con Graph Attention Networks para análisis de estados cerebrales en EEG.

## Estructura

```
autoencoder/
├── config/
│   └── config.yaml          # Configuración del modelo y entrenamiento
├── models/
│   ├── __init__.py
│   └── vae_model.py          # VAE con encoder GAT
├── data/
│   └── __init__.py           # Reutiliza dataset del clasificador
├── utils/
│   ├── __init__.py
│   ├── logger.py             # Logging
│   └── visualization.py      # Visualizaciones (latent space, reconstructions)
├── train.py                  # Script de entrenamiento
├── hyperparam_search.py      # Búsqueda de hiperparámetros
├── extract_activations.py    # Extractor de activaciones GAT para clustering
├── requirements.txt
└── README.md
```

## Características

### Modelo VAE
- **Encoder**: Múltiples capas GATv2Conv con skip connections
- **Espacio Latente**: Variacional (μ, log_var) con reparametrización
- **Decoder**: Reconstruye features de nodos y pesos de edges
- **Loss**: L = L_recon + β · KL(q(z|x) || p(z))

### Funcionalidades
- β-VAE con annealing (linear, cosine, cyclical)
- Free bits para prevenir colapso del posterior
- Extracción de activaciones de capas GAT intermedias
- Compatibilidad con clustering.py del pipeline

### Logging
- TensorBoard: losses, latent space, reconstructions, attention weights
- Checkpoints periódicos
- Early stopping

## Uso Rápido

### 1. Entrenar el VAE

```bash
cd /media/storage_hdd/dmt_fz/machine_learning/autoencoder

# Entrenamiento básico
python train.py --config config/config.yaml

# Con rebuild de dataset
python train.py --config config/config.yaml --force-rebuild
```

### 2. Búsqueda de Hiperparámetros

```bash
# Búsqueda rápida (5 experimentos, configuraciones reducidas)
python hyperparam_search.py --n_experiments 5 --quick --max_epochs 50

# Búsqueda completa
python hyperparam_search.py --n_experiments 20 --max_epochs 100

# Ver resultados en TensorBoard
tensorboard --logdir=hyperparam_search/
```

### 3. Extraer Activaciones GAT para Clustering

```bash
# Extraer activaciones del mejor modelo
python extract_activations.py --checkpoint checkpoints/best_model.pt

# Con opciones adicionales
python extract_activations.py \
    --checkpoint checkpoints/best_model.pt \
    --per-subject \
    --save-attention \
    --output activations/

# Resultado: archivos .pkl compatibles con clustering.py
```

### 4. Correr Clustering en Activaciones

```bash
# Usar activaciones del espacio latente (mu)
cd /media/storage_hdd/dmt_fz/pipeline

python clustering.py \
    --input ../machine_learning/autoencoder/activations/vae_activations_mu.pkl \
    --quick-search

# O usar activaciones de una capa GAT específica
python clustering.py \
    --input ../machine_learning/autoencoder/activations/vae_activations_encoder_layer_1.pkl \
    --quick-search
```

## Configuración Importante

### Espacio Latente (`model.latent.dim`)
- Valores típicos: 32, 64, 128
- Mayor dimensión = más capacidad pero más difícil de visualizar

### β-VAE (`loss.kl.weight`)
- β < 1: Más énfasis en reconstrucción
- β = 1: VAE estándar
- β > 1: Mayor regularización, latent space más estructurado

### Annealing
- Gradualmente aumenta β durante entrenamiento
- Ayuda a evitar colapso del posterior
- Tipos: linear, cosine, cyclical

## Ejemplo de Flujo Completo

```bash
# 1. Entrenar VAE
python train.py --config config/config.yaml

# 2. Extraer activaciones de capas GAT
python extract_activations.py \
    --checkpoint checkpoints/best_model.pt \
    --per-subject \
    --save-attention

# 3. Visualizar en TensorBoard
tensorboard --logdir=runs/

# 4. Correr clustering en activaciones
cd ../pipeline
python clustering.py \
    --input ../machine_learning/autoencoder/activations/vae_activations_mu.pkl \
    --bands Alpha Theta \
    --quick-search
```

## Estructura de Salida

Después de extraer activaciones:

```
activations/
├── vae_activations_pooled.pkl    # Representación pooled (antes del latent)
├── vae_activations_mu.pkl        # Media del espacio latente
├── vae_activations_z.pkl         # Muestras del espacio latente
├── vae_activations_encoder_layer_0.pkl  # Activaciones capa GAT 0
├── vae_activations_encoder_layer_1.pkl  # Activaciones capa GAT 1
├── extraction_metadata.pkl       # Metadata de la extracción
├── per_subject/                  # Activaciones por sujeto
│   ├── S01_activations.pkl
│   ├── S02_activations.pkl
│   └── ...
└── attention_weights/            # Pesos de atención
    ├── attention_layer_0.pkl
    └── ...
```

## Formato Compatible con clustering.py

Los archivos `vae_activations_*.pkl` tienen la estructura:

```python
{
    'DMT': {
        'Alpha': [array_epoch1, array_epoch2, ...],
        'Beta': [...],
        ...
    },
    'EC': {...},
    'EO': {...}
}
```

Esto es idéntico a `eigen_all.pkl` pero con activaciones GAT en lugar de eigenvalores.









