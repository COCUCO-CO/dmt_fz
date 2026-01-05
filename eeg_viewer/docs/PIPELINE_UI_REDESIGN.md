# Pipeline UI Redesign - Documento de Diseño

## 📋 Resumen Ejecutivo

Este documento describe las mejoras de UX/UI para la sección Pipeline del EEG Viewer, manteniendo 100% de compatibilidad con la funcionalidad existente.

**Objetivo**: Hacer que cualquier usuario (incluso sin conocimiento técnico) pueda entender y usar el pipeline correctamente.

---

## 🔒 Lo que DEBE preservarse (No romper)

### Funcionalidad Core
| Componente | Archivo | Qué hace | Tests que lo validan |
|------------|---------|----------|---------------------|
| `run_pipeline_step()` | `runner.py` | Ejecuta scripts como subprocesos | `test_pipeline_runner.py` |
| `stop_current_process()` | `runner.py` | Detiene procesos con SIGTERM/SIGKILL | `test_pipeline_stop.py` |
| `pipeline_log()` | `utils.py` | Logging en consola | `test_pipeline_utils.py` |
| Estado en PS | `global_state.py` | Persistencia de parámetros | `test_pipeline_params_persistence.py` |
| Indicador running | `running_indicator.py` | Indicador naranja global | `test_pipeline_running_indicator.py` |

### Estado que persiste (PS - PipelineState)
```python
# Estos valores DEBEN seguir persistiendo al cambiar de pestaña:
PS.running              # Si hay un proceso corriendo
PS.current_step         # Nombre del step actual
PS.running_task_name    # Nombre para indicador
PS.start_time          # Para calcular tiempo transcurrido
PS.current_process     # Referencia al proceso (para STOP)
PS.log_history         # Historial de logs
PS.selected_run        # Directorio de output seleccionado

# Parámetros globales
PS.max_subjects, PS.max_epochs, PS.conditions, PS.workers, PS.jobs

# Parámetros de clustering
PS.min_k, PS.max_k, PS.min_comps, PS.max_comps, PS.bands, PS.clustering_search_mode

# Input/Output
PS.input_dir
```

### Argumentos de scripts (NO modificar)
Los scripts del pipeline (`fwd.py`, `clustering.py`, etc.) esperan argumentos específicos. Los métodos `build_args()` de cada step DEBEN seguir generando los mismos argumentos:

```python
# Ejemplo: fwd.py espera estos args exactos
['--max-subjects', '5', '--conditions', 'DMT', 'EC', '--jobs', '0', '--workers', '7', '--max-epochs', '0']
```

### Tabs del panel derecho
- CONSOLE: Log de salida + STOP + CLEAR
- FILES: Browser de archivos del run
- SYSTEM: Monitor CPU/RAM/GPU
- VISUALIZE: Visualización de datos

---

## 🎨 Mejoras Propuestas

### 1. Estructura de Fases (Visual)

**Antes**: 8 steps en lista plana
**Después**: 3 fases colapsables

```
┌─────────────────────────────────────────────────────────────┐
│ 📦 FASE 1: PROCESAMIENTO INICIAL                    [▼]     │
├─────────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 🧠 1. Localización de Fuentes                          │ │
│ │    Transforma EEG de sensores → 100 regiones cerebrales│ │
│ │    ⏱️ ~3-4h (menos con max_epochs)                      │ │
│ │    [▶ EJECUTAR]                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 📁 2. Consolidar Datos                                  │ │
│ │    Agrupa phases-*.pkl en subject_phases_{cond}.pkl    │ │
│ │    ⏱️ ~1-2 min                                          │ │
│ │    [▶ EJECUTAR]                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 🔗 FASE 2: ANÁLISIS DE CONECTIVIDAD                 [▼]     │
├─────────────────────────────────────────────────────────────┤
│  ... Steps 3-6 ...                                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 📊 FASE 3: ANÁLISIS AVANZADO (Opcional)             [▼]     │
├─────────────────────────────────────────────────────────────┤
│  ... Steps 7-8 ...                                          │
└─────────────────────────────────────────────────────────────┘
```

### 2. Nomenclatura Clara

| Step | Nombre Técnico Actual | Nuevo Nombre | Descripción para Usuario |
|------|----------------------|--------------|--------------------------|
| 1 | SOURCE_LOCALIZATION | **Localización de Fuentes** | "Convierte las señales de los electrodos en actividad de 100 regiones cerebrales usando un modelo anatómico" |
| 2 | CONSOLIDATE_PHASES | **Consolidar Datos** | "Combina los archivos individuales de cada sujeto en archivos agrupados por condición" |
| 3 | NETWORK_FILTERING | **Filtrar por Redes** | "Separa la actividad cerebral según las 7 redes funcionales (Default Mode, Visual, etc.)" |
| 4 | SYNCHRONIZATION | **Sincronización** | "Calcula cuánto se sincronizan las regiones cerebrales entre sí" |
| 5 | ORDER_PARAMETER | **Coherencia Global** | "Mide el orden/coherencia global del cerebro usando el parámetro de Kuramoto" |
| 6 | AGGREGATE_ORDER | **Agregar Métricas** | "Combina las métricas de coherencia de todos los sujetos" |
| 7 | CORRELATIONS | **Correlaciones (Pearson)** | "Analiza correlaciones entre condiciones experimentales" |
| 8 | CLUSTERING | **Estados Cerebrales** | "Identifica estados cerebrales recurrentes mediante clustering" |

### 3. Botones más descriptivos

**Antes**:
```
[RUN fwd.py]
```

**Después**:
```
[▶ Ejecutar Localización]
```

### 4. Indicadores de Estado

Cada step mostrará su estado actual:
- ⚪ **Pendiente**: No ejecutado aún
- 🟡 **En progreso**: Ejecutándose ahora
- ✅ **Completado**: Archivos de salida encontrados
- ⚠️ **Requiere**: Indica qué steps previos necesita

### 5. Información de Output

**Antes**:
```
→ run_*/phases-{subj}.pkl
```

**Después**:
```
📁 Genera: phases-S01.pkl, phases-S02.pkl... (1 por sujeto)
```

### 6. Ayuda Contextual

Botón `?` que muestra tooltip/modal con:
- Qué hace el step en detalle
- Qué archivos de entrada necesita
- Qué archivos genera
- Tiempo estimado según configuración actual
- Tips y consideraciones

---

## 🏗️ Plan de Implementación

### Fase A: Preparación (Sin cambios visibles)
1. Agregar nuevos campos a los steps (display_name, description, phase, etc.)
2. Crear componente `PhaseContainer` reutilizable
3. Crear componente `StepCard` mejorado
4. **NO modificar** la lógica de `build_args()` ni `run_pipeline_step()`

### Fase B: Migración Gradual
1. Actualizar `step_1_fwd.py` con nuevo formato
2. Verificar que tests pasan
3. Repetir para steps 2-8
4. Actualizar `page.py` para usar PhaseContainers

### Fase C: Indicadores de Estado
1. Agregar función `check_step_status()` que verifica archivos de output
2. Mostrar indicadores visuales
3. NO bloquear ejecución (solo informativo)

---

## 📁 Archivos a Modificar

### Nuevos archivos a crear:
```
app/pages/pipeline/
├── components/
│   ├── phase_container.py    # Contenedor colapsable de fase
│   └── step_card.py          # Card mejorada para cada step
└── constants/
    └── step_info.py          # Nombres, descripciones, tiempos
```

### Archivos a modificar:
```
app/pages/pipeline/
├── steps/
│   ├── base.py               # Agregar campos: phase, display_name, user_description
│   ├── step_1_fwd.py         # Actualizar metadata
│   ├── step_2_consolidate.py # Actualizar metadata
│   ├── step_3_network.py     # Actualizar metadata
│   ├── step_4_syncro.py      # Actualizar metadata
│   ├── step_5_order.py       # Actualizar metadata
│   ├── step_6_aggregate.py   # Actualizar metadata
│   ├── step_7_pearson.py     # Actualizar metadata
│   └── step_8_clustering.py  # Actualizar metadata
└── page.py                   # Usar nuevo layout con fases
```

### Archivos que NO se tocan:
```
app/pages/pipeline/
├── runner.py                 # ❌ NO TOCAR - lógica de ejecución
├── utils.py                  # ❌ NO TOCAR - funciones core
├── config.py                 # ❌ NO TOCAR - constantes del sistema
└── tabs/
    ├── console.py            # ❌ NO TOCAR - funciona correctamente
    ├── files.py              # ❌ NO TOCAR
    ├── system.py             # ❌ NO TOCAR
    └── visualize.py          # ❌ NO TOCAR
```

---

## ✅ Validación

### Tests existentes que DEBEN seguir pasando:
```bash
pytest tests/pipeline/ -v
# Actualmente: 548 passed
```

### Tests específicos críticos:
- `test_pipeline_runner.py` - Ejecución de scripts
- `test_pipeline_stop.py` - Funcionalidad STOP
- `test_pipeline_running_indicator.py` - Indicador naranja
- `test_pipeline_params_persistence.py` - Persistencia de parámetros
- `test_pipeline_clustering_params.py` - Params de clustering
- `test_pipeline_steps.py` - build_args() de cada step

### Nuevos tests a agregar:
- `test_step_metadata.py` - Verificar que todos los steps tienen metadata completa
- `test_phase_grouping.py` - Verificar agrupación correcta

---

## 🎯 Criterios de Éxito

1. ✅ Todos los 548+ tests existentes pasan
2. ✅ Los scripts generan los mismos argumentos
3. ✅ El estado persiste al cambiar de pestaña
4. ✅ STOP funciona correctamente
5. ✅ El indicador naranja funciona
6. ✅ Un usuario nuevo entiende qué hace cada paso sin leer código

---

## 📅 Orden de Implementación

1. **Crear constantes de metadata** (`step_info.py`)
2. **Actualizar `BasePipelineStep`** con nuevos campos
3. **Actualizar cada step** con su metadata
4. **Crear `PhaseContainer`** component
5. **Crear `StepCard`** mejorado
6. **Actualizar `page.py`** para nuevo layout
7. **Ejecutar todos los tests**
8. **Verificación visual manual**

---

## 💬 Notas Importantes

### Principio fundamental:
> **La UI puede cambiar completamente, pero `build_args()` y `run_pipeline_step()` NO se tocan.**

### Rollback:
Si algo falla, el archivo `pipeline_old.py` sigue disponible como backup.

### Compatibilidad:
Los usuarios que ya tienen runs existentes podrán seguir usándolos sin problemas.








