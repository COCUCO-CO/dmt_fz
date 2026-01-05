# Assets - Analysis Page

Imágenes y GIFs necesarios para documentar la página Analysis (`/analysis`).

## Archivos Requeridos

### Screenshots (PNG)

| Archivo | Descripción | Resolución Sugerida |
|---------|-------------|---------------------|
| `overview.png` | Vista general de la página | 1920x1080 |
| `model_loading.png` | Panel de carga de modelo | 400x300 |
| `model_info.png` | Información del modelo cargado | 400x200 |
| `latent_2d.png` | Espacio latente 2D (PCA) | 600x500 |
| `latent_3d.png` | Espacio latente 3D | 600x500 |
| `layer_activations.png` | Activaciones por capa | 800x400 |
| `attention_matrix.png` | Matriz de atención | 500x500 |
| `attention_topography.png` | Pesos de atención en topografía | 500x500 |
| `sample_selector.png` | Selector de muestra | 400x200 |

### GIFs Animados

| Archivo | Descripción | Duración Sugerida |
|---------|-------------|-------------------|
| `latent_pca.gif` | Interacción con el espacio latente | 10-15s |
| `layer_exploration.gif` | Navegación por capas | 10-15s |
| `attention_visualization.gif` | Visualización de pesos de atención | 10-15s |
| `sample_playback.gif` | Paso de muestra por la red | 10-15s |

## Secciones de la Página

### Model Loading
- Selector de checkpoint
- Info del modelo cargado
- Arquitectura resumida
- Mejor loss de validación

### Latent Space Tab
- Selector de método (PCA, t-SNE)
- Gráfico interactivo 2D/3D
- Puntos coloreados por condición
- Varianza explicada

### Activations Tab
- Selector de capa
- Selector de muestra
- Histograma de activaciones
- Heatmap de activaciones

### Attention Tab
- Selector de cabeza de atención
- Matriz de atención
- Visualización topográfica
- Grafo de conexiones

## Requisitos Previos

Para capturar estas imágenes necesitas:

1. **Un modelo entrenado**: Ejecuta un entrenamiento en la página Model primero
2. **Datos de test**: El modelo debe haber guardado resultados de test
3. **Checkpoint guardado**: En `cache/autoencoder/checkpoints/`

## Cómo Capturar

### Para Latent Space
1. Carga un modelo entrenado
2. Procesa el dataset de test
3. Cambia entre PCA y t-SNE
4. Rota la visualización 3D

### Para Attention Weights
1. Usa un modelo con capas GAT
2. Selecciona una muestra
3. Explora las diferentes cabezas de atención
4. Muestra matriz y topografía

### Para Layer Activations
1. Pasa una muestra por el modelo
2. Selecciona diferentes capas
3. Muestra cómo cambian las activaciones




