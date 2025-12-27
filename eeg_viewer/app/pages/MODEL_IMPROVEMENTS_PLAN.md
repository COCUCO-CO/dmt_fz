# Plan de Mejoras - Sección Model

## Resumen de Cambios

### 1. Configuración del Decoder (GAT Layers)
**Ubicación**: Panel izquierdo, sección "Architecture"

Agregar opciones para configurar el decoder:
- `Decoder GAT layers`: 0-4 (0 = MLP decoder, >0 = GAT decoder)
- `Decoder hidden dims`: configuración de dimensiones ocultas

```
Encoder                          Decoder
┌─────────────────┐              ┌─────────────────┐
│ GAT layers: 3   │              │ GAT layers: 0-4 │ ← NUEVO
│ Hidden: 64      │     →        │ Hidden: [256,128]│
│ Heads: 4        │   Latent     │                 │
└─────────────────┘     64       └─────────────────┘
```

### 2. Visualización de Arquitectura del Modelo
**Ubicación**: Panel derecho, nueva pestaña "ARCH" o arriba de CONSOLE

Diagrama ASCII/simple mostrando:
```
┌──────────────────────────────────────────────────────────────┐
│                    VAE Architecture                           │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│   Input          Encoder           Latent        Decoder     │
│  ┌─────┐       ┌─────────┐        ┌─────┐      ┌─────────┐  │
│  │24 ch│  →    │GAT×3    │   →    │ 64  │  →   │MLP      │  │
│  │     │       │64 hidden│        │ dim │      │256→128  │  │
│  │     │       │4 heads  │        │     │      │         │  │
│  └─────┘       └─────────┘        └─────┘      └─────────┘  │
│                                                               │
│   Nodes: 24 (EEG) | Features: 5 bands × stats               │
│   Total params: ~125K | Trainable: ~125K                    │
└──────────────────────────────────────────────────────────────┘
```

### 3. Preview de Datos de Entrada
**Ubicación**: Panel izquierdo, debajo de "Scan Dataset" (después de escanear)

Mostrar un ejemplo de los datos:

**Para Grafos (phases):**
```
┌─────────────────────────────────────┐
│ 📊 Data Preview                     │
├─────────────────────────────────────┤
│ Sample: phases-S01.pkl              │
│                                     │
│ ┌─────────────────────────────┐     │
│ │    [Matriz de conectividad] │     │
│ │    24×24 heatmap pequeño    │     │
│ └─────────────────────────────┘     │
│                                     │
│ Nodes: 24 (electrodes)              │
│ Features/node: 15                   │
│   • Phase stats: 5                  │
│   • Amplitude stats: 5              │
│   • Temporal: 5                     │
│ Edges: ~276 (threshold 0.3)         │
│ Labels: DMT, EC, EO                 │
└─────────────────────────────────────┘
```

**Para Imágenes:**
```
┌─────────────────────────────────────┐
│ 📊 Data Preview                     │
├─────────────────────────────────────┤
│ Sample: image_001.png               │
│                                     │
│ ┌─────────────────────────────┐     │
│ │    [Thumbnail de imagen]    │     │
│ │         64×64               │     │
│ └─────────────────────────────┘     │
│                                     │
│ Size: 224×224                       │
│ Channels: 3 (RGB)                   │
│ Classes: cat, dog, bird             │
└─────────────────────────────────────┘
```

### 4. Información de Features
Mostrar claramente qué features se extraen:
- Para grafos: node features, edge features
- Para imágenes: dimensiones, canales
- Para time series: sampling rate, channels

## Estructura de Archivos

```
app/pages/
├── model.py                    # Modificar
└── model/
    ├── __init__.py
    ├── architecture_viz.py     # NUEVO: Visualización de arquitectura
    ├── data_preview.py         # NUEVO: Preview de datos
    └── config_panels.py        # NUEVO: Paneles de configuración (opcional)
```

## Cambios en model.py

### Panel Izquierdo (Configuración)
1. ✅ Dataset panel (existente)
2. ➕ **Data Preview panel** (NUEVO - aparece después de scan)
3. ✅ Model Config panel (modificar para agregar decoder config)
4. ✅ Training panel (existente)

### Panel Derecho (Visualización)
1. ➕ **Architecture tab** (NUEVO) - diagrama del modelo
2. ✅ Metrics tab (existente)
3. ✅ Recon tab (existente)  
4. ✅ Console tab (existente)

## Tests a Crear

```python
# tests/test_model_ui_improvements.py

class TestDecoderConfig:
    def test_decoder_gat_layers_range()
    def test_decoder_config_updates_model()
    
class TestArchitectureViz:
    def test_generates_ascii_diagram()
    def test_shows_correct_dimensions()
    def test_updates_on_config_change()
    
class TestDataPreview:
    def test_shows_preview_after_scan()
    def test_graph_preview_shows_matrix()
    def test_image_preview_shows_thumbnail()
    def test_shows_feature_info()
```

## Orden de Implementación

1. **Tests primero** - Crear tests para validar funcionalidad
2. **Decoder config** - Agregar campos de configuración
3. **Architecture viz** - Crear componente de visualización
4. **Data preview** - Crear panel de preview
5. **Integración** - Conectar todo sin romper training

## Ideas Adicionales

1. **Estimación de parámetros**: Mostrar número estimado de parámetros del modelo
2. **Estimación de tiempo**: "Estimated training time: ~15 min"
3. **Recomendaciones**: "⚠️ High latent dim for small dataset"
4. **Export config**: Botón para exportar configuración a YAML
5. **Load config**: Cargar configuración previa





