# Assets - Pipeline Page

Imágenes para documentar la página Source Localization (`/pipeline`).

## Archivos Disponibles

| Archivo | Descripción |
|---------|-------------|
| `overview.png` | Vista general con grafo cerebral 3D, panel de pasos, consola ejecutando |
| `step_1_source_localization.png` | Visualización Hilbert 2D (envolvente, fase) y 3D Phase Space |
| `step_1_source_localization_results.png` | Console log con resultado completo de Steps 1-2 |
| `step_1_source_localization_results_files.png` | Panel FILES mostrando archivos phases-*.pkl generados |
| `step_2_filtrar_redes.png` | Step 2: 7 redes funcionales (Schaefer) con pie chart |
| `step_3_syncro.png` | Step 3: Matrices PLV Alpha para DMT/EC/EO con histogramas |
| `step_3_syncro_animation.png` | Panel de animaciones expandido con video player |
| `step_4_kuramoto.png` | Step 4: Osciladores Kuramoto, R=0.570, R(t), barras por banda |

## Estructura del Pipeline

### Fase 1: Procesamiento Inicial
- **Step 1:** Localización de Fuentes (`fwd.py`)
- **Step 2:** Consolidar Datos (`save_load_pickle.py`)

### Fase 2: Análisis de Conectividad
- **Step 3:** Filtrar por Redes (`multi2pool2.py`)
- **Step 4:** Sincronización (`calculate_syncro.py`)
- **Step 5:** Coherencia Global (`generate_order.py`)
- **Step 6:** Agregar Métricas (`build_order_data.py`)

### Fase 3: Análisis Avanzado
- **Step 7:** Correlaciones (`pearson.py`)
- **Step 8:** Estados Cerebrales (`clustering.py`)

## Visualizaciones por Tab

| Tab | Step | Contenido | Imagen |
|-----|------|-----------|--------|
| 1 | Step 1 | Localización de Fuentes | `step_1_source_localization.png` |
| 2 | Step 3 | Filtrar por Redes | `step_2_filtrar_redes.png` |
| 3 | Step 4 | Sincronización | `step_3_syncro.png` |
| 4 | Step 5 | Coherencia Global | `step_4_kuramoto.png` |
| 5 | Step 7 | Correlaciones | (pendiente) |
| 6 | Step 8 | Estados Cerebrales | (pendiente) |

**Nota:** Steps 2 y 6 son pasos de consolidación sin visualización propia.
