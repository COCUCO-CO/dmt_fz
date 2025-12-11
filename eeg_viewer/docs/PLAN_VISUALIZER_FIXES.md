# Plan de Corrección: Visualizadores del Pipeline

## Problemas Identificados

### 1. Step 2 - Error de Plotly
- **Error**: `titlefont` es propiedad deprecada, debe usar `title=dict(font=...)`
- **Decisión**: Eliminar este visualizador (no aporta valor científico, solo agrupa archivos)

### 2. Step 4 - Matriz PLV muy grande y mal diseñada
- **Problemas**:
  - Matriz demasiado grande para el visualizador
  - Selector incluye banda pero debería ser solo sujeto + condiciones
  - No permite ver múltiples condiciones lado a lado
- **Solución**:
  - Selector de sujeto solamente
  - Checkboxes para condiciones (DMT, EC, EO)
  - Si múltiples: mostrar matrices lado a lado (hasta 3 en una fila)
  - Stats y distribución debajo de cada matriz
  - Tamaño compacto que quepa en pantalla

### 3. Step 5 - No muestra datos Kuramoto
- **Problema**: No encuentra archivos `order-*.pkl`
- **Causa probable**: Los archivos están en subdirectorios con nombres diferentes
- **Solución**: Revisar patrones de búsqueda y formato de datos

### 4. Step 6 - Eliminar
- **Razón**: Duplica información de Step 2 sin valor científico adicional
- **Acción**: Remover visualizador del registry

### 5. Animation Generator - Verificar visibilidad
- **Estado**: El componente existe y está en page.py
- **Acción**: Verificar que se muestra correctamente (expansion "ANIMACIONES")

---

## Plan de Implementación

### Fase 1: Eliminar visualizadores innecesarios
1. ✅ Eliminar Step2Visualizer del registry
2. ✅ Eliminar Step6Visualizer del registry
3. ✅ Actualizar __init__.py de visualizers

### Fase 2: Rediseñar Step 4 (PLV Matrices)
1. ✅ Cambiar selector: solo sujeto
2. ✅ Agregar checkboxes de condiciones
3. ✅ Lógica para mostrar múltiples condiciones lado a lado
4. ✅ Reducir tamaño de matrices (height=250px max)
5. ✅ Stats y histograma debajo de cada matriz

### Fase 3: Arreglar Step 5 (Kuramoto)
1. ✅ Revisar file_patterns para incluir más formatos
2. ✅ Debug de carga de datos
3. ✅ Verificar estructura del archivo order-*.pkl

### Fase 4: Tests
1. ✅ Test para verificar que Step 2 y 6 no existen en registry
2. ✅ Test para Step 4 con múltiples condiciones
3. ✅ Test para Step 5 carga de datos

---

## Nuevo Diseño Step 4

```
┌─────────────────────────────────────────────────────────────┐
│ Step 4 Sincronización                                       │
├─────────────────────────────────────────────────────────────┤
│ Sujeto: [S01 ▼]   ☑ DMT  ☑ EC  ☐ EO    [Static][Interactive]│
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐                         │
│  │   DMT PLV    │  │    EC PLV    │                         │
│  │  (heatmap)   │  │  (heatmap)   │                         │
│  │   250x250    │  │   250x250    │                         │
│  └──────────────┘  └──────────────┘                         │
│  ┌──────────────┐  ┌──────────────┐                         │
│  │ Stats + Hist │  │ Stats + Hist │                         │
│  └──────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
```

---

## Archivos a Modificar

| Archivo | Acción |
|---------|--------|
| `visualizers/__init__.py` | Remover Step2 y Step6 del registry |
| `visualizers/step_4_viz.py` | Rediseñar completamente |
| `visualizers/step_5_viz.py` | Arreglar carga de datos |
| `tests/pipeline/test_visualizer_consistency.py` | Actualizar tests |

---

## Orden de Ejecución

1. ✅ Actualizar registry (remover 2 y 6)
2. ✅ Reescribir step_4_viz.py
3. ✅ Arreglar step_5_viz.py  
4. ✅ Actualizar tests
5. ✅ Verificar todos los tests pasan (744 passed)

## Estado Final

**744 tests pasando**

### Cambios realizados:
- Step 2 y 6 removidos del registry (no aportan valor científico)
- Step 4 rediseñado con checkboxes de condiciones (DMT, EC, EO)
- Step 4 muestra matrices lado a lado cuando se seleccionan múltiples
- Step 4 heatmaps compactos (height=220px)
- Step 5 muestra debug info cuando no encuentra datos
- AnimationGenerator visible en expansion "ANIMACIONES"

