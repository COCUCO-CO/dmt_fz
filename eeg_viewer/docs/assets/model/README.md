# Assets - Model Page

Imágenes y GIFs necesarios para documentar la página Model (`/model`).

## Archivos Requeridos

### Screenshots (PNG)

| Archivo | Descripción | Resolución Sugerida |
|---------|-------------|---------------------|
| `overview.png` | Vista general de la página | 1920x1080 |
| `dataset_scanner.png` | Panel de escaneo de dataset | 400x400 |
| `dataset_graph.png` | Info de dataset tipo graph | 400x300 |
| `dataset_image.png` | Info de dataset tipo image | 400x300 |
| `model_config.png` | Configuración del modelo | 400x500 |
| `training_controls.png` | Botones Train/Stop | 400x100 |
| `architecture_viz.png` | Diagrama de arquitectura | 800x200 |
| `reconstruction.png` | Original vs Reconstruido | 800x400 |
| `console_training.png` | Log de entrenamiento | 800x300 |

### GIFs Animados

| Archivo | Descripción | Duración Sugerida |
|---------|-------------|-------------------|
| `dataset_scan.gif` | Escaneo de dataset en acción | 5-10s |
| `training_metrics.gif` | Métricas actualizándose durante training | 15-20s |
| `architecture_update.gif` | Cambio de parámetros y actualización | 5-10s |

## Secciones de la Página

### Dataset Panel
- Selector de ruta
- Botón de escaneo
- Información detectada
- Selector EEG/STC
- Checkboxes de bandas
- Slider de subsample

### Model Config Panel
- Tipo de modelo (VAE Graph, VAE Image, AE)
- Arquitectura (Latent, Hidden, GAT layers, Heads)
- Decoder (GAT layers, Hidden dims)
- Training (Epochs, Batch, LR, Optimizer)
- Regularization (Patience, Grad clip)
- Loss (KL weight, Node/Edge weights)
- Workers

### Tabs de Visualización

#### ARCH Tab
- Diagrama visual de la arquitectura
- INPUT → ENCODER → LATENT → DECODER → OUTPUT
- Estadísticas resumen

#### METRICS Tab
- Gráficos de pérdida en tiempo real
- Total Loss, Recon Loss, KL Loss
- Estadísticas: Best Val, Current Epoch, Train Loss

#### RECON Tab
- Slider de época
- Heatmaps: Original, Reconstructed, Difference
- Métricas MSE/MAE

#### CONSOLE Tab
- Log en tiempo real
- Colores por tipo de mensaje
- Botón Clear

## Cómo Capturar

### Para Training Metrics
1. Inicia un entrenamiento con datos de ejemplo
2. Captura las primeras 10-20 épocas
3. Muestra cómo se actualizan los gráficos

### Para Dataset Scanner
1. Apunta a un directorio con datos de phases
2. Click en Scan Dataset
3. Captura la información que aparece

### Para Architecture
1. Ajusta los parámetros de arquitectura
2. Click en Refresh
3. Muestra cómo cambia el diagrama





