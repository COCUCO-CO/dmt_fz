# Plan de Corrección: Tamaños de Gráficos y Loading Spinner

## Problemas Identificados

### 1. Step 2 y Step 3 - Gráficos muy grandes y pixelados
- **Causa**: Uso de Matplotlib con `figsize` grande que se escala mal
- **Efecto**: Gráficos gigantes, pixelados y que no caben en pantalla

### 2. Step 6 - Información duplicada
- **Causa**: Muestra gráfico de barras "Archivos por Condición" similar a Step 2
- **Efecto**: Redundancia visual, no aporta valor diferencial

### 3. No hay indicador de carga
- **Causa**: Los visualizadores pueden tardar en renderizar
- **Efecto**: Usuario no sabe si la UI está respondiendo

---

## Soluciones Propuestas

### Solución 1: Migrar Step 2 y Step 3 a Plotly

**Beneficios**:
- Tamaños responsivos con `height=300` fijo
- Mejor calidad visual (vectorial, no raster)
- Consistencia con otros visualizadores (Step 1, Step 4)

**Implementación**:
```python
# Antes (Matplotlib)
fig, ax = self.create_matplotlib_figure(figsize=(4, 2.5))

# Después (Plotly)
import plotly.graph_objects as go
fig = go.Figure(...)
fig.update_layout(height=300, template='plotly_dark')
ui.plotly(fig).classes('w-full')
```

### Solución 2: Simplificar Step 6

**Cambio**: Remover gráfico de barras redundante, solo mostrar:
- Lista de archivos agregados con tamaños
- Mensaje informativo sobre qué contienen
- Sin gráfico (evita duplicación con Step 2)

### Solución 3: Agregar Loading Spinner

**Ubicación**: `VisualizationPanel._render_visualizer()`

**Flujo**:
1. Usuario hace clic en step [n]
2. Se muestra spinner centrado
3. Se carga el visualizador
4. Spinner se oculta y se muestra contenido

**Implementación**:
```python
async def _render_visualizer(self) -> None:
    self._container.clear()
    
    # Show loading spinner
    with self._container:
        with ui.row().classes('w-full justify-center items-center p-8'):
            ui.spinner('dots', size='lg', color='primary')
            ui.label('Cargando...').style('color: #888;')
    
    # Small delay to show spinner
    await asyncio.sleep(0.05)
    
    # Clear and render actual content
    self._container.clear()
    # ... render visualizer
```

---

## Tests a Implementar

### Test 1: Altura consistente de gráficos
```python
def test_step_2_uses_plotly_with_fixed_height():
    """Step 2 chart should use Plotly with height=300."""
    
def test_step_3_uses_plotly_with_fixed_height():
    """Step 3 chart should use Plotly with height=300."""
```

### Test 2: Step 6 no tiene gráfico redundante
```python
def test_step_6_no_bar_chart():
    """Step 6 should not have a bar chart (avoid duplication)."""
```

### Test 3: Loading spinner presente
```python
def test_visualization_panel_has_loading_state():
    """Panel should show loading spinner during render."""
```

---

## Orden de Implementación

1. ✅ Crear tests para los requisitos
2. ✅ Agregar loading spinner al VisualizationPanel
3. ✅ Migrar Step 2 a Plotly
4. ✅ Migrar Step 3 a Plotly  
5. ✅ Simplificar Step 6
6. ✅ Ejecutar tests y verificar

---

## Métricas de Éxito

- [x] Gráficos de Step 2 y 3 usan Plotly con height fijo (200-220px)
- [x] Step 6 no muestra gráfico redundante
- [x] Loading spinner aparece al cambiar de step
- [x] Todos los tests pasan (744 tests ✓)

## Cambios Implementados

### Step 2 (Consolidar Datos)
- Migrado de Matplotlib a Plotly
- `height=200` para gráfico compacto
- `max-width: 350px` para evitar desbordamiento
- Usa `template='plotly_dark'` para tema oscuro

### Step 3 (Filtrar por Redes)
- Migrado de Matplotlib a Plotly
- `height=220` para pie chart compacto
- `max-width: 280px` para evitar desbordamiento
- Donut chart con `hole=0.3`

### Step 6 (Agregar Métricas)
- Removido gráfico de barras redundante
- Solo muestra información de archivos
- Grid responsive con CSS Grid
- Mensaje informativo sobre uso de archivos

### VisualizationPanel
- Agregado loading spinner al cambiar de step
- Spinner muestra "Cargando visualización..."
- Usa `ui.timer(0.05, ...)` para mostrar spinner antes del render
- Lógica de render movida a `_do_render_visualizer()`

