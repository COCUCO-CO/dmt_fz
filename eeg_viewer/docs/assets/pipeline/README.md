# Assets - Pipeline Page

Imágenes y GIFs necesarios para documentar la página Pipeline (`/pipeline`).

## Archivos Requeridos

### Screenshots (PNG)

| Archivo | Descripción | Resolución Sugerida |
|---------|-------------|---------------------|
| `overview.png` | Vista general de la página | 1920x1080 |
| `io_config.png` | Panel de configuración I/O | 400x200 |
| `global_params.png` | Parámetros globales | 400x250 |
| `step_1_fwd.png` | Step 1: Forward Model | 400x200 |
| `step_2_consolidate.png` | Step 2: Consolidate | 400x200 |
| `step_3_network.png` | Step 3: Network Filter | 400x200 |
| `step_4_syncro.png` | Step 4: Synchronization | 400x200 |
| `step_5_order.png` | Step 5: Order Parameter | 400x200 |
| `step_6_aggregate.png` | Step 6: Aggregate | 400x200 |
| `step_7_pearson.png` | Step 7: Pearson Correlation | 400x200 |
| `step_8_clustering.png` | Step 8: Clustering | 400x200 |
| `console.png` | Consola de salida | 800x200 |
| `file_browser_output.png` | Browser de archivos de salida | 800x150 |

### GIFs Animados

| Archivo | Descripción | Duración Sugerida |
|---------|-------------|-------------------|
| `visualization_panel.gif` | Cambio entre visualizaciones de pasos | 10-15s |
| `run_step.gif` | Ejecución de un paso | 10-15s |
| `animation_generator.gif` | Generación de animación cerebral | 15-20s |
| `console_output.gif` | Log en tiempo real | 10s |

## Fases del Pipeline

### Fase 1: Source Localization
- Step 1: Forward Model (`fwd.py`)
- Step 2: Consolidate (`save_load_pickle.py`)

### Fase 2: Network Analysis
- Step 3: Network Filter (`multi2pool2.py`)
- Step 4: Synchronization (`calculate_syncro.py`)
- Step 5: Order Parameter (`generate_order.py`)
- Step 6: Aggregate (`build_order_data.py`)

### Fase 3: Statistical Analysis
- Step 7: Pearson (`pearson.py`)
- Step 8: Clustering (`clustering.py`)

## Visualizaciones por Paso

Cada paso tiene visualizaciones específicas que deben capturarse:

| Step | Visualización |
|------|---------------|
| 1 | Modelo forward, topografía |
| 2 | Estadísticas de datos consolidados |
| 3 | Fases extraídas por banda |
| 4 | Mapas de sincronización |
| 5 | Series temporales de orden |
| 6 | Distribuciones agregadas |
| 7 | Matrices de correlación de Pearson |
| 8 | Resultados de clustering |

## Cómo Capturar

Para el Animation Generator:
1. Selecciona un sujeto con datos procesados
2. Elige una condición y banda
3. Inicia la generación
4. Captura el progreso y el resultado

Para la consola:
1. Ejecuta un paso que tenga salida abundante
2. Captura el scroll del log en tiempo real

