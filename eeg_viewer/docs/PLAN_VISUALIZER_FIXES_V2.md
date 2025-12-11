# Plan de Corrección V2: Visualizadores del Pipeline

## Problemas Pendientes

### 1. Steps 2 y 6 - Eliminar completamente del selector
- **Problema**: Aparecen como botones 2 y 6 pero dicen "no visualizer"
- **Solución**: No mostrar los botones 2 y 6 en el selector de steps

### 2. Step 4 - Selector de sujeto no funciona
- **Problema**: Cambiar sujeto no actualiza la visualización
- **Solución**: Revisar evento `_on_subject_change` y cache

### 3. Step 4 - Baja definición de matriz PLV
- **Problema**: La imagen y texto no tienen buena definición
- **Solución**: Aumentar DPI de matplotlib, mejorar fonts en Plotly

### 4. Step 5 - No muestra datos Kuramoto
- **Problema**: No encuentra archivos de orden Kuramoto
- **Datos disponibles**:
  - `/media/storage_hdd/dmt_fz/fwd-inv-stc/kuramoto_all.pkl` - R(t) por banda/condición (5166 épocas x valores)
  - `/media/storage_hdd/dmt_fz/fwd-inv-stc/r_kuramoto_nets_all_mean.pkl` - R promedio por red (7 redes)
  - `/media/storage_hdd/dmt_fz/fwd-inv-stc/DMT/order_all-S*.pkl` - Datos por sujeto
- **Solución**: Rediseñar Step 5 para mostrar:
  - Osciladores de Kuramoto (animación/estático tipo Wikipedia)
  - R promedio por banda (barra)
  - R promedio por las 7 redes cerebrales
  - Soporte para datos STC y EEG

---

## Estructura de Datos Kuramoto

### kuramoto_all.pkl
```
{
  'DMT': {
    'Delta': [array1, array2, ...],  # 5166 arrays de R(t)
    'Theta': [...],
    'Alpha': [...],
    'Beta': [...],
    'Gamma': [...]
  },
  'EC': {...},
  'EO': {...}
}
```

### r_kuramoto_nets_all_mean.pkl
```
{
  'DMT': {
    'Delta': {
      'RH': {'FPN': {...}, 'DMN': {...}, ...},  # 7 redes
      'LH': {...},
      'both': {'FPN': {...}, 'DMN': {...}, 'DAN': {...}, 'LN ': {...}, 'SVA': {...}, 'SMN': {...}, 'VN ': {...}}
    },
    'Theta': {...},
    ...
  },
  'EC': {...},
  'EO': {...}
}
```

### 7 Redes Cerebrales (Schaefer)
- FPN: Frontoparietal Network (Control ejecutivo)
- DMN: Default Mode Network (Reposo)
- DAN: Dorsal Attention Network (Atención)
- LN: Limbic Network (Emociones)
- SVA: Salience/Ventral Attention (Estímulos)
- SMN: Somatomotor Network (Motor)
- VN: Visual Network (Visual)

---

## Plan de Implementación

### Fase 1: Ocultar steps 2 y 6 del selector
1. Modificar `visualization_panel.py` → `_render_header()`
2. No renderizar botones para steps 2 y 6

### Fase 2: Arreglar selector de sujeto en Step 4
1. Revisar `step_4_viz.py` → `_on_subject_change()`
2. Limpiar cache correctamente
3. Forzar re-render

### Fase 3: Mejorar definición Step 4
1. Aumentar altura Plotly a 280px
2. Mejorar fonts (size 9 → 11)
3. Mejorar colorbar

### Fase 4: Rediseñar Step 5 Kuramoto
1. Nueva clase `Step5Visualizer` con:
   - Gráfico de osciladores (círculo polar con puntos)
   - Timeline R(t) por banda
   - Bar chart R promedio por red (7 redes)
   - Selector de condición y banda
2. Cargar datos desde `/media/storage_hdd/dmt_fz/fwd-inv-stc/`

---

## Nuevo Diseño Step 5

```
┌─────────────────────────────────────────────────────────────┐
│ Step 5 Coherencia Global (Kuramoto)                         │
├─────────────────────────────────────────────────────────────┤
│ Condición: [DMT ▼]  Banda: [Alpha ▼]  Época: [_0_]          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌────────────────────────────────┐   │
│  │  OSCILADORES     │  │  R(t) Timeline                 │   │
│  │    (polar)       │  │  ~~~~~~~~~~~~~~~^~~~           │   │
│  │     ● ●          │  │       Mean: 0.65               │   │
│  │   ●     ●        │  └────────────────────────────────┘   │
│  │    ●  ●          │                                       │
│  │      →R=0.7      │  ┌────────────────────────────────┐   │
│  └──────────────────┘  │  R promedio por RED            │   │
│                        │  FPN ████████ 0.72             │   │
│  ┌──────────────────┐  │  DMN ██████ 0.58               │   │
│  │ R por BANDA      │  │  DAN ████████ 0.71             │   │
│  │ δ ████ 0.45      │  │  ...                           │   │
│  │ θ █████ 0.52     │  └────────────────────────────────┘   │
│  │ α ███████ 0.68   │                                       │
│  │ β ████ 0.42      │                                       │
│  │ γ ███ 0.35       │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Archivos a Modificar

| Archivo | Cambio |
|---------|--------|
| `visualization_panel.py` | Ocultar botones 2 y 6 |
| `step_4_viz.py` | Arreglar selector sujeto + mejor definición |
| `step_5_viz.py` | Reescribir completamente |
| `test_visualizer_consistency.py` | Actualizar tests |

---

## Tests Necesarios

1. ✅ `test_step_selector_skips_2_and_6` - Verificar que no aparecen
2. ✅ `test_step_4_has_condition_checkboxes` - Verificar checkboxes condiciones  
3. ✅ `test_step_5_has_oscillator_plot` - Verificar osciladores Kuramoto
4. ✅ `test_step_5_has_network_comparison` - Verificar las 7 redes
5. ✅ `test_step_5_has_band_comparison` - Verificar comparación por banda

---

## Estado Final

**748 tests pasando**

### Cambios implementados:

1. **Steps 2 y 6 ocultos**: Solo aparecen botones [1][3][4][5][7][8]
2. **Step 4 selector de sujeto arreglado**: Limpia cache y datos al cambiar
3. **Step 4 mejor definición**: height=280px, fonts más grandes, colorbar mejorado
4. **Step 5 completamente rediseñado**:
   - Carga datos desde `/media/storage_hdd/dmt_fz/fwd-inv-stc/`
   - Osciladores Kuramoto (gráfico polar tipo Wikipedia)
   - Timeline R(t) con mean destacado
   - Comparación R por banda (bar chart)
   - Comparación R por red cerebral (7 redes Schaefer)

