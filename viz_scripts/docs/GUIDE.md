# 📘 Guía Completa de Visualización de Sincronización Cerebral

Esta guía explica **en detalle** qué hace cada script, cómo funciona técnicamente y por qué es útil para el análisis neurocientífico.

---

## 📚 Índice

1. [Conceptos Fundamentales](#conceptos-fundamentales)
2. [Scripts de Generación de Frames/Videos](#scripts-de-generación-de-framesvideos)
3. [Scripts de Análisis Estadístico](#scripts-de-análisis-estadístico)
4. [Scripts de Visualización Especializada](#scripts-de-visualización-especializada)
5. [Librería Base](#librería-base)
6. [Workflow Completo](#workflow-completo)
7. [FAQ Técnicas](#faq-técnicas)

---

## 🧠 Conceptos Fundamentales

### ¿Qué estamos visualizando?

Este proyecto visualiza la **sincronización de fase** entre diferentes regiones cerebrales durante estados alterados de consciencia (DMT) vs. estados basales (ojos cerrados/abiertos).

### Conceptos Clave

#### 1. **Parámetro de Orden de Kuramoto**
```
Definición: Mide la sincronización global de un sistema de osciladores
Rango: 0 (desincronizado) → 1 (perfectamente sincronizado)

Interpretación neuronal:
- Kuramoto alto → Muchas regiones oscilando en fase (estado ordenado)
- Kuramoto bajo → Regiones independientes (estado desordenado)
```

**Hipótesis DMT:** Se espera que DMT reduzca la sincronización global (entropía aumentada), a diferencia de estados como el sueño profundo que aumentan sincronización.

#### 2. **Sincronización por Pares (Phase Locking Value - PLV)**
```
Definición: Sincronización entre dos regiones específicas
Rango: 0 (sin sincronía) → 1 (fase bloqueada)

Visualización: 
- Matriz: muestra PLV entre todos los pares de regiones
- Grafo: nodos = regiones, aristas = conexiones fuertes
- Osciladores polares: fase de cada región en círculo
```

#### 3. **Bandas de Frecuencia**
```
Delta (0.5-4 Hz):   Sueño profundo, procesos inconscientes
Theta (4-8 Hz):     Memoria, creatividad, estados meditativos
Alpha (8-13 Hz):    Relajación, integración cortical
Beta (13-30 Hz):    Atención, pensamiento activo
Gamma (30-100 Hz):  Binding perceptual, consciencia
```

#### 4. **Niveles de Análisis**

**EEG (Electroencefalografía):**
- 32 electrodos en el cuero cabelludo
- Medición directa de actividad eléctrica
- Espacialmente difusa (muchas fuentes mezcladas)

**STC (Source-Time Course) / Fuentes:**
- 68 regiones cerebrales (parcelas del atlas Desikan-Killiany)
- Obtenido por "source reconstruction" (beamforming)
- Espacialmente específico pero con incertidumbre matemática

---

## 🎬 Scripts de Generación de Frames/Videos

### 1. `generate_frames.py` ⭐ **MÁS USADO**

#### ¿Qué hace?
Genera **imágenes estáticas (PNG)** de la sincronización cerebral en momentos específicos. Cada imagen muestra:
- Matriz de sincronización (mapa de calor)
- Grafo de red cerebral (nodos + aristas)
- Osciladores de fase (círculo polar)
- Serie temporal del parámetro de Kuramoto

#### ¿Cómo lo hace?
```python
Proceso:
1. Lee datos de sincronización pre-calculados (syncros_eeg, syncros_stc)
2. Para cada época (intervalo de 2 segundos):
   - Toma 3 snapshots temporales: inicio (0), medio (400), final (799)
   - Genera figura matplotlib con 6-9 subplots
   - Guarda como PNG en visualizations/plot/
3. Con --workers N: procesa N épocas en paralelo

Paralelización:
- Usa multiprocessing.Pool
- Cada worker es un proceso independiente
- No hay dependencias entre épocas → perfecto para paralelizar
```

#### ¿Por qué es útil?
- **Análisis exploratorio:** Identificar épocas con sincronización alta/baja
- **Control de calidad:** Detectar artefactos o datos anómalos
- **Figuras para papers:** Exportar momentos representativos
- **Base para videos:** Los frames se pueden unir después

#### Opciones Clave
```bash
--view all    # EEG + STC completo (6 gráficos)
--view eeg    # Solo canales EEG (3 gráficos)
--view stc    # Solo fuentes cerebrales (3 gráficos)
--workers 20  # Paralelización (20 épocas a la vez)
--samples 0 400 799  # Momentos dentro de cada época
```

#### Ejemplo de Output
```
Archivo: 1005000400.png
Nombre decodificado:
  - 1005: época 5 (base 1000 + epoch_number)
  - 000400: sample 400 (punto temporal)
  
Título en imagen:
  "SUJETO: S01 | CONDICIÓN: DMT | BANDA: Alpha | Época 5/200 | Sample 400"
```

---

### 2. `create_video_from_frames.py` ⭐ **RECOMENDADO PARA VIDEOS**

#### ¿Qué hace?
Toma los frames generados por `generate_frames.py` y los une en un **video MP4** usando ffmpeg.

#### ¿Cómo lo hace?
```python
Proceso:
1. Escanea directorio visualizations/plot/ buscando *.png
2. Ordena frames numéricamente (1001000000.png, 1002000000.png, ...)
3. Crea lista temporal para ffmpeg (frames_list.txt)
4. Ejecuta ffmpeg con encoder:
   - CPU: libx264 (universal, lento)
   - GPU: h264_nvenc (Nvidia, 5-10x más rápido)
5. Guarda video con codec H.264 (compatible con todo)

ffmpeg hace:
- Compresión temporal (solo guarda diferencias entre frames)
- Compresión espacial (DCT + cuantización)
- Resultado: 200 frames × 5MB = 1GB → video de 50-100MB
```

#### ¿Por qué es MUCHO más rápido que generate_video.py?
```
generate_video.py (VIEJO - LENTO):
├─ Genera cada frame en tiempo real durante animación
├─ matplotlib.animation.FuncAnimation es lento
├─ No paralelizable (animación es secuencial)
└─ 200 épocas × 10s/época = 30-60 minutos

create_video_from_frames.py (NUEVO - RÁPIDO):
├─ Lee frames ya generados del disco
├─ ffmpeg está ultra-optimizado en C
├─ Usa SIMD, multithreading, GPU encoding
└─ 10-20 segundos total (o 5-10s con GPU)

Speedup: 100-300x más rápido
```

#### ¿Por qué es útil?
- **Observar evolución temporal:** Ver cómo cambia la sincronización
- **Presentaciones:** Videos son más engaging que imágenes estáticas
- **Detección de patrones:** Oscilaciones, transiciones de estado
- **Compartir resultados:** MP4 es universal

#### Opciones Clave
```bash
--fps 30         # Frames por segundo (más alto = más suave)
--preset medium  # Velocidad de encoding (ultrafast/medium/slow)
--crf 18         # Calidad (menor = mejor, 18 = visualmente perfecto)
--gpu            # Usar Nvidia NVENC (requiere GPU compatible)
```

#### Parámetros de Calidad
```
Preset (velocidad):
  ultrafast → 5 seg, 200 MB (menos compresión)
  medium    → 15 seg, 80 MB  (balance - default)
  slow      → 60 seg, 50 MB  (más compresión)

CRF (calidad visual):
  15 → casi sin pérdida, archivos grandes
  18 → visualmente perfecto (default, recomendado)
  23 → calidad estándar de ffmpeg
  28 → se nota compresión, archivos pequeños
```

---

### 3. `generate_all.sh` 🚀 **SCRIPT TODO-EN-UNO**

#### ¿Qué hace?
Ejecuta automáticamente el workflow completo: generar frames → crear video.

#### ¿Cómo lo hace?
```bash
Secuencia:
1. Muestra configuración actual (subject/cond/band)
2. Estima tiempos de procesamiento
3. Pregunta confirmación
4. Ejecuta generate_frames.py con paralelización
5. Ejecuta create_video_from_frames.py
6. Mide tiempos y reporta estadísticas

Ventajas:
- Un solo comando para todo
- Manejo de errores automático
- Reportes de tiempo detallados
- Confirma antes de iniciar
```

#### ¿Por qué es útil?
- **Simplicidad:** No recordar comandos complejos
- **Consistencia:** Siempre usa configuración óptima
- **Productividad:** Lanzar y dejar corriendo
- **Batch processing:** Procesar múltiples sujetos/condiciones

#### Uso
```bash
# Procesamiento completo (200 épocas)
./generate_all.sh

# Test rápido (10 épocas)
./generate_all.sh --test

# Con GPU para máxima velocidad
./generate_all.sh --gpu --workers 20

# Personalizado
./generate_all.sh --epochs 100 --workers 15 --fps 60
```

---

### 4. `generate_video.py` ⚠️ **LEGACY - NO RECOMENDADO**

#### ¿Qué hace?
Genera videos directamente usando matplotlib.animation (método antiguo).

#### ¿Por qué NO usarlo?
```
Problemas:
1. Genera frames en tiempo real (lento)
2. No paralelizable
3. Alta carga de CPU (matplotlib no optimizado)
4. 30-60 minutos para video completo

Razón de existencia:
- Código legacy de antes de optimizar
- Mantenido por compatibilidad
- Puede tener funciones únicas (--view matrix, etc.)
```

#### Cuándo usarlo (raro)
- Necesitas una vista específica no disponible en generate_frames
- Debugging de funciones de plotting
- Generar video muy corto (1-2 épocas)

---

## 📊 Scripts de Análisis Estadístico

### 5. `visualize_results.py` 📈

#### ¿Qué hace?
Análisis estadístico **comparativo** del parámetro de Kuramoto entre:
- Sujetos (S01, S02, ..., S15)
- Condiciones (DMT, EC, EO)
- Bandas de frecuencia (Delta, Theta, Alpha, Beta, Gamma)
- Niveles (EEG vs. Sources)

#### ¿Cómo lo hace?
```python
Proceso:
1. Lee TODOS los datos de sincronización (todos los sujetos)
   Ubicación: results/syncros/
   
2. Extrae parámetro de Kuramoto para cada época:
   kuramoto[subject][condition][band][level] → array de épocas
   
3. Calcula estadísticas:
   - Media ± desviación estándar
   - Mediana, cuartiles
   - Outliers (método IQR)
   
4. Genera visualizaciones:
   a) Serie temporal promedio ± IC95% (todas las bandas)
   b) Boxplots comparativos (DMT vs EC vs EO)
   c) Heatmaps banda × condición
   
5. Exporta CSV con estadísticas descriptivas
   Formato: subject, condition, band, level, mean, std, median, q25, q75
```

#### ¿Por qué es útil? (Ciencia)

**Pregunta principal:** ¿DMT altera la sincronización cerebral de forma diferente a estados basales?

**Hipótesis testeable:**
```
H0: Kuramoto_DMT = Kuramoto_EC = Kuramoto_EO
H1: Kuramoto_DMT ≠ Kuramoto_EC

Predicción basada en teoría:
- DMT → desincronización (Kuramoto ↓) en Alpha/Beta
- DMT → hipersincronización (Kuramoto ↑) en Theta/Delta?
- Diferencias entre EEG (global) vs Sources (local)
```

**Aplicaciones:**
1. **Papers:** Figuras comparativas con barras de error
2. **Detectar efectos:** Qué bandas son más sensibles a DMT
3. **Variabilidad inter-sujeto:** Algunos responden más que otros
4. **Control de calidad:** Verificar que EC y EO son diferentes

#### Output Explicado

**`kuramoto_eeg_by_band.png`:**
- Serie temporal: Eje X = épocas (~400s totales), Eje Y = Kuramoto
- Línea = media entre sujetos
- Banda sombreada = IC 95%
- Permite ver: estabilidad temporal, tendencias, variabilidad

**`kuramoto_comparison_boxplot.png`:**
- Cada caja = distribución de Kuramoto para una condición
- Box = Q1-Q3 (50% central), Línea = mediana
- Whiskers = rango (sin outliers), Puntos = outliers
- Comparación visual directa: DMT vs EC vs EO

**`summary_statistics.csv`:**
```csv
subject,condition,band,level,mean,std,median,q25,q75
S01,DMT,Alpha,eeg,0.456,0.089,0.461,0.401,0.512
S01,DMT,Alpha,sources,0.382,0.102,0.378,0.321,0.441
...
```
Útil para: análisis estadístico en R/Python, tablas en papers, meta-análisis.

---

## 🔬 Scripts de Visualización Especializada

### 6. `visualize_hilbert_improved.py` 🌊

#### ¿Qué hace?
Visualización **3D interactiva** de la transformada de Hilbert (fase y amplitud instantáneas) usando Plotly.

#### ¿Cómo funciona matemáticamente?

```python
Transformada de Hilbert:
1. Señal original: x(t) = A(t) * cos(φ(t))
   Ejemplo: EEG de un canal en banda Alpha
   
2. Señal analítica: z(t) = x(t) + i*H[x(t)]
   H[x(t)] = transformada de Hilbert (convolución con 1/πt)
   
3. Extracción:
   A(t) = |z(t)|           → Amplitud instantánea (envolvente)
   φ(t) = angle(z(t))      → Fase instantánea (ángulo)
   ω(t) = dφ(t)/dt         → Frecuencia instantánea

Visualización 3D:
   Eje X: Tiempo
   Eje Y: Amplitud A(t)
   Eje Z: Fase φ(t)
   Color: puede ser frecuencia, amplitud, o época
```

#### ¿Qué muestra en la práctica?

```
Interpretación visual:
- Altura (Y): Qué tan fuerte es la oscilación
- Altura (Z): En qué punto del ciclo está (0-2π)
- Espiral: Oscilación regular (en banda)
- Caos: Oscilación irregular (fuera de banda)

Aplicaciones:
1. Verificar que el filtrado de banda funciona bien
   → Debe verse oscilación clara en esa frecuencia
   
2. Ver variaciones de amplitud (arousal, atención)
   → Altura Y cambia con el tiempo
   
3. Detectar transiciones de fase (phase resets)
   → Saltos bruscos en Z
```

#### ¿Por qué Plotly y no Matplotlib?

```
Plotly (este script):
✓ Interactivo en navegador (rotar, zoom, pan)
✓ HTML standalone (compartir fácilmente)
✓ Hover info (ver valores exactos)
✓ Renderizado hardware-accelerated

Matplotlib:
✓ Figuras estáticas para papers
✗ No interactivo (o muy lento con %matplotlib widget)
```

#### Uso Científico

**No es una visualización "bonita", es una herramienta diagnóstica:**

```
Preguntas que responde:
1. ¿El filtro de banda está funcionando?
   → Ver frecuencia instantánea consistente
   
2. ¿Hay artifacts de pre-procesamiento?
   → Saltos bruscos, discontinuidades
   
3. ¿La amplitud modula con eventos?
   → Cambios en envolvente A(t)
   
4. ¿Hay acoplamiento fase-amplitud (PAC)?
   → Amplitud alta ocurre en fases específicas
```

---

### 7. `visualize_brain_3d.py` 🧠 **MAPAS 3D EN MOVIMIENTO**

#### ¿Qué hace?
Crea **videos 3D del cerebro** con colores que representan sincronización local, mapeados sobre la superficie cortical anatómica.

#### ¿Cómo funciona? (EXPLICACIÓN DETALLADA)

##### Paso 1: Datos de Entrada
```python
Input:
- Sincronización por parcel (68 regiones): [0.2, 0.5, 0.8, ...]
  Cada valor = sincronización promedio de esa región con el resto
  
- Coordenadas 3D de cada parcel: [(x1,y1,z1), (x2,y2,z2), ...]
  Ubicación espacial del centro de cada región
  
- Malla del cerebro (brain mesh): 
  Archivo: fsaverage surface (FreeSurfer)
  Contenido: ~150,000 vértices + triángulos (geometría 3D)
```

##### Paso 2: Interpolación (El Truco Clave)

```python
Problema:
- Tenemos 68 valores discretos (uno por parcel)
- Necesitamos ~150,000 valores (uno por vértice de la malla)
- ¿Cómo "rellenar" los espacios?

Solución: Inverse Distance Weighting (IDW)

Para cada vértice v del cerebro:
1. Calcular distancia a cada uno de los 68 parcels:
   d[i] = ||v - parcel_center[i]||  (distancia euclidiana 3D)
   
2. Calcular pesos (cercano = más influencia):
   w[i] = 1 / (d[i]^power + epsilon)
   power = 2 típicamente (inverso cuadrático)
   epsilon = pequeño número para evitar división por cero
   
3. Promedio ponderado:
   color[v] = Σ(w[i] * syncro[i]) / Σ(w[i])
   
Resultado:
- Vértices cercanos a parcel A toman el color de A
- Vértices entre A y B mezclan ambos colores suavemente
- Gradiente suave, visualmente agradable
```

**Analogía:** Es como interpolar temperatura en un mapa meteorológico. Tienes estaciones en 68 ciudades, pero quieres colorear TODO el país. IDW hace eso.

##### Paso 3: Renderizado 3D (PyVista)

```python
PyVista:
- Librería de visualización científica 3D (basada en VTK)
- Pensada para volúmenes, mallas, campos escalares
- Mucho más rápida que Matplotlib 3D

Proceso por frame:
1. Crear plotter (cámara, iluminación)
2. Agregar mesh cerebral con colores interpolados:
   plotter.add_mesh(
       brain_mesh,
       scalars=interpolated_colors,  # de IDW
       cmap='coolwarm',               # azul=baja, rojo=alta sincro
       lighting=True                   # efectos 3D realistas
   )
3. Renderizar a imagen
4. Guardar frame

Para video:
- Repetir para cada época (timepoint)
- Opcionalmente: rotar cámara (efecto dramático)
```

##### Paso 4: Ensamblaje del Video

```python
Frames individuales → Video MP4
1. Guardar frames temporales en temp_color_frames/
2. Llamar ffmpeg (igual que create_video_from_frames.py)
3. Eliminar frames temporales
4. Output: video final
```

#### ¿Por qué es LENTO?

```
Bottlenecks:
1. Interpolación IDW:
   - Para cada uno de los 150,000 vértices:
     - Calcular distancia a 68 parcels: 68 × sqrt() = lento
     - Total: 150k × 68 = 10 millones de operaciones
   - Por época: ~500-1000 frames
   - Total: 10M × 1000 = 10 mil millones de operaciones
   
2. Renderizado PyVista:
   - Crear mesh, aplicar iluminación, renderizar
   - ~1-2 segundos por frame
   - 1000 frames × 1.5s = 25 minutos
   
Optimizaciones parciales:
- Cache de distancias (se calculan una vez)
- Batch processing donde se puede
- Aún así: 10-30 min para video completo

Por eso: SOLO usar para figuras finales de papers
```

#### ¿Por qué es útil científicamente?

```
Visualización anatómica:
- Permite ver QUÉ REGIONES son más sincrónicas
- Intuición espacial: lóbulo frontal vs occipital, etc.
- Identificar gradientes cortico-subcorticales

Aplicaciones:
1. Papers/presentaciones: impresionante visualmente
2. Hipótesis espaciales: ¿DMT afecta más corteza prefrontal?
3. Comparación con fMRI: sincronización vs activación
4. Divulgación: público general entiende "cerebro de colores"

Limitaciones:
- Source reconstruction tiene incertidumbre (~1-2cm)
- Interpolación es cosmética (no añade información)
- Lento de generar (solo para figuras finales)
```

#### Parámetros Importantes

```bash
--fps 2           # Frames por segundo (bajo para explorar)
--start 0 --end 10  # Solo 10 épocas (test)
--rotate          # Rotar cámara (cinematográfico)
--no-edges        # No mostrar conexiones (solo colores)
```

---

### 8. `plots3d.py` 🔬

#### ¿Qué hace?
Setup básico de visualización 3D con Plotly para exploración interactiva rápida.

#### ¿Cómo lo hace?
```python
Funciones principales:
1. plot_3d_brain_network():
   - Nodos = parcels (esferas en coordenadas 3D)
   - Aristas = conexiones sobre threshold
   - Interactivo: rotar, zoom en navegador
   
2. plot_3d_brain_surface():
   - Carga surface mesh simplificada
   - Mapea colores por parcel (sin interpolación)
   - Más rápido que visualize_brain_3d.py

Diferencia con visualize_brain_3d.py:
- plots3d.py: exploración rápida, muestra en pantalla
- visualize_brain_3d.py: renderizado profesional, exporta video
```

#### ¿Cuándo usarlo?
- Exploración interactiva durante análisis
- Testear visualizaciones antes de renderizar video
- Verificar coordenadas de parcels
- No guarda archivos (solo fig.show())

---

### 8. `visualize_hilbert_video.py` 🌊 **VIDEO DE EVOLUCIÓN HILBERT**

#### ¿Qué hace?
Genera un **video** mostrando cómo evoluciona la transformada de Hilbert en el espacio 3D **época a época**. Es la versión temporal/dinámica de `visualize_hilbert_improved.py`.

#### ¿Cómo funciona?

```python
Proceso:
1. Para cada época (ej: 0-50):
   a) Extrae fase y amplitud instantáneas (Hilbert)
   b) Construye señal analítica z(t) = A(t) * exp(i*φ(t))
   c) Proyecta en 3D: (tiempo, Re(z), Im(z))
   d) Genera frame Plotly con trayectoria 3D
   e) Renderiza a PNG (1920×1080)

2. Une frames con ffmpeg:
   - Input: hilbert_0000.png, hilbert_0001.png, ...
   - Output: video MP4 @ 2-5 FPS
   - Codec: H.264 (universal)

Diferencias con visualize_hilbert_improved.py:
- Improved: 1 época, interactivo HTML
- Video: múltiples épocas, video MP4
```

#### ¿Qué puedes ver en el video?

```
Interpretación de cambios temporales:

1. Amplitud (altura Y):
   - Aumenta → región más activa (arousal, atención)
   - Disminuye → región menos activa (habituación)
   - Constante → estado estable
   
2. Fase (ángulo en plano complejo):
   - Rotación suave → oscilación coherente en banda
   - Saltos bruscos → phase reset (evento cognitivo)
   - Deriva lenta → fuera de banda central
   
3. Forma de la trayectoria:
   - Espiral regular → oscilación pura (sinusoide)
   - Espiral irregular → modulación de amplitud
   - Caótica → mezcla de frecuencias (mala filtración)

Comparaciones científicas:
- DMT temprano vs tardío: ¿cambia amplitud?
- DMT vs EC: ¿más o menos estable?
- Por banda: Theta suele ser más estable que Gamma
```

#### Ejemplo de Hallazgo Científico

```
Hipótesis: DMT aumenta variabilidad de fase en Alpha

Evidencia en video:
1. EC (ojos cerrados): 
   - Trayectoria muy regular
   - Amplitud constante
   - Rotación suave y predecible
   
2. DMT (5-10 min post-administración):
   - Trayectoria más errática
   - Amplitud fluctuante
   - Saltos de fase frecuentes
   
3. DMT (20-30 min post):
   - Retorno gradual a regularidad
   - Amplitud disminuye
   
Interpretación: DMT desestabiliza oscilaciones Alpha, 
sugiriendo reducción de coherencia cortical. Compatible 
con "entropía aumentada" y desintegración de redes.
```

#### ¿Por qué es útil científicamente?

**Ventajas sobre visualización estática:**
1. **Contexto temporal:** Una época aislada no muestra tendencias
2. **Detección de transiciones:** Ver cuándo ocurren cambios
3. **Periodicidades lentas:** Modulaciones de 0.1-1 Hz visibles
4. **Efectos de droga:** Onset, peak, offset son evidentes

**Aplicaciones concretas:**
```
1. Caracterizar time-course de DMT:
   - ¿Cuándo es máximo el efecto?
   - ¿Es abrupto o gradual?
   - ¿Hay rebote post-pico?
   
2. Comparar individuos:
   - Responders vs non-responders
   - Variabilidad inter-sujeto
   
3. Validar preprocesamiento:
   - ¿Hay artifacts periódicos?
   - ¿El filtro de banda funciona bien?
   
4. Papers/presentaciones:
   - Video > imagen estática
   - Material suplementario
```

#### Parámetros Importantes

```bash
--fps 2        # Lento: apreciar cada época (recomendado para análisis)
--fps 5        # Fluido: ver evolución temporal suave
--fps 10       # Rápido: overview general

--start 0 --end 50    # Rango de épocas (50 = ~100 segundos de EEG)

--gpu          # Usar Nvidia NVENC (encoding 5x más rápido)

--no-projections  # Desactivar proyecciones ortogonales (más limpio)

--keep-frames  # No eliminar frames PNG (para debug/reutilización)
```

#### Performance

```
50 épocas:
- Generación de frames: 2-4 minutos
- Encoding ffmpeg: 5-10 segundos (GPU) o 10-20 seg (CPU)
- Total: ~3-5 minutos

200 épocas:
- Generación: 10-15 minutos
- Encoding: 10-20 segundos
- Total: ~12-18 minutos

Bottleneck: Plotly renderizado (no paralelizable fácilmente)
```

#### Comparación con otros Videos

| Script | Contenido | Duración típica | Uso principal |
|--------|-----------|-----------------|---------------|
| `create_video_from_frames.py` | Red cerebral completa | 3-6 min | Sincronización global |
| `visualize_brain_3d.py` | Anatomía 3D coloreada | 3-6 min | Localización espacial |
| **`visualize_hilbert_video.py`** | **Señal analítica 3D** | **20-60 seg** | **Dinámica fase/amplitud** |

**Complementariedad:**
- `create_video_from_frames`: QUÉ regiones sincronizan
- `visualize_brain_3d`: DÓNDE en el cerebro
- `visualize_hilbert_video`: CÓMO evoluciona la oscilación

---

## 📚 Librería Base

### 9. `plot.py` 🏗️ **LIBRERÍA CENTRAL**

#### ¿Qué es?
Es la **librería de funciones** que todos los otros scripts importan. NO se ejecuta directamente (excepto para debug).

#### Funciones Principales

##### `plot_kuramoto_order()`
```python
Qué: Grafica serie temporal del parámetro de Kuramoto
Input: kuramoto_mat (épocas × 1), índice de época actual
Output: Línea temporal con punto destacado en época actual

Detalles:
- Marca épocas rechazadas (artefactos) en gris
- Añade línea horizontal en media
- Permite ver contexto temporal (esta época vs resto)
```

##### `plot_mat()`
```python
Qué: Heatmap de matriz de sincronización
Input: matriz NxN de PLV (phase locking values)
Output: Imagen con colores (azul=baja, rojo=alta sincro)

Detalles:
- Threshold: oculta conexiones débiles (< percentil X)
- Orden personalizable (clustering, por redes, etc.)
- Anotaciones opcionales con nombres de regiones
```

##### `plot_graph()`
```python
Qué: Grafo de red cerebral (nodos + aristas)
Input: matriz de sincronización, coordenadas 2D
Output: Visualización estilo "grafo" networkx

Detalles:
- Nodos posicionados según coordenadas anatómicas
- Aristas = conexiones sobre threshold
- Grosor de arista proporcional a PLV
- Colores por red funcional (DMN, FPN, etc.)
```

##### `plot_osc()`
```python
Qué: Diagrama polar de osciladores (fases)
Input: matriz de fases (regiones × tiempo), sample específico
Output: Círculo con líneas desde centro (cada región)

Detalles:
- Ángulo = fase en ese momento (0-2π)
- Largo de línea = constante (todas iguales)
- Color = por red funcional
- Jitter: añade ruido visual para separar líneas superpuestas
- Flecha gruesa = vector de orden (sincronización resultante)

Interpretación:
- Líneas juntas → alta sincronización
- Líneas dispersas → baja sincronización
- Largo de flecha central = parámetro de Kuramoto
```

##### `plot_all()`, `plot_eeg_only()`, `plot_stc_only()`
```python
Qué: Funciones de alto nivel que combinan todos los plots
Input: índice de época, samples a graficar
Output: Figura completa multi-panel

plot_all():
  Panel superior: EEG (3 gráficos)
  Panel inferior: Sources (3 gráficos)
  
plot_eeg_only():
  Solo los 3 gráficos de EEG
  
plot_stc_only():
  Solo los 3 gráficos de Sources

Estas son las funciones que generate_frames.py llama.
```

#### Configuración Global (Líneas 369-376)

```python
# IMPORTANTE: Cambiar aquí para cambiar todos los scripts
band = "Alpha"        # Banda a visualizar
cond = "DMT"          # Condición experimental
subject = "S01"       # Sujeto a procesar

# Todos los scripts importan estas variables:
from plot import subject, cond, band
```

**¿Por qué centralizado?**
- Cambias una vez → afecta todos los scripts
- Consistencia garantizada
- Menos errores (no olvidar cambiar en algún lugar)

#### ¿Por qué NO ejecutar plot.py directamente?

```python
plot.py es una LIBRERÍA, no un SCRIPT:
- Define funciones, no las ejecuta
- Necesitas llamar las funciones desde otro script
- Similar a: numpy.py, matplotlib.py (no los ejecutas)

Para testing rápido:
- Tiene un bloque if __name__ == "__main__" al final
- Genera algunas épocas de prueba
- Pero normalmente usas generate_frames.py en su lugar
```

---

## 🔄 Workflow Completo

### Workflow Típico de Análisis

```bash
# ============================================================================
# FASE 1: EXPLORACIÓN INICIAL
# ============================================================================

# 1. Ver qué datos hay disponibles
ls ../results/syncros/

# 2. Verificar configuración
python3 -c "from plot import subject, cond, band; print(f'{subject}, {cond}, {band}')"

# 3. Test rápido (3 épocas, visual check)
python generate_frames.py --test --workers 10
# Output: ../visualizations/plot/*.png
# Verificar: ¿Se ven bien? ¿Hay artifacts?


# ============================================================================
# FASE 2: GENERACIÓN COMPLETA
# ============================================================================

# 4. Generar todos los frames (200 épocas)
python generate_frames.py --end 200 --workers 20
# Tiempo: 2-3 minutos con 20 workers
# Output: 200 épocas × 3 samples = 600 imágenes

# 5. Crear video
python create_video_from_frames.py --fps 30 --gpu
# Tiempo: 5-10 segundos con GPU, 10-20 seg con CPU
# Output: S01_DMT_Alpha_video.mp4


# ============================================================================
# FASE 3: ANÁLISIS ESTADÍSTICO
# ============================================================================

# 6. Análisis comparativo entre condiciones
python visualize_results.py
# Lee TODOS los sujetos/condiciones
# Output: 
#   - kuramoto_eeg_by_band.png
#   - kuramoto_comparison_boxplot.png
#   - summary_statistics.csv

# 7. Inspeccionar CSV con estadísticas
head ../visualizations/visualize_results/summary_statistics.csv


# ============================================================================
# FASE 4: VISUALIZACIONES ESPECIALIZADAS (OPCIONAL)
# ============================================================================

# 8. Transformada de Hilbert interactiva
python visualize_hilbert_improved.py
# Output: HTML interactivo, abrir en navegador

# 9. Video 3D del cerebro (LENTO - solo para figuras finales)
python visualize_brain_3d.py --start 0 --end 5 --fps 2
# Tiempo: 10-15 minutos para 5 épocas
# Output: video 3D con interpolación anatómica


# ============================================================================
# FASE 5: ITERAR SOBRE OTROS SUJETOS/CONDICIONES
# ============================================================================

# 10. Cambiar configuración para otro sujeto
vim plot.py +370
# Cambiar: subject = "S02", cond = "EC", band = "Theta"

# 11. Repetir pasos 4-9

# O automatizar:
for SUBJ in S01 S02 S03; do
    for COND in DMT EC EO; do
        for BAND in Alpha Beta Theta; do
            # Actualizar plot.py
            sed -i "370s/.*/band = \"$BAND\"/" plot.py
            sed -i "371s/.*/cond = \"$COND\"/" plot.py
            sed -i "372s/.*/subject = \"$SUBJ\"/" plot.py
            
            # Generar
            ./generate_all.sh --epochs 200 --workers 20
        done
    done
done
```

---

## ❓ FAQ Técnicas

### ¿Por qué 3 samples por época (0, 400, 799)?

```
Época = 2 segundos = 1000 samples (a 500 Hz)

Sample 0:   inicio (t=0.0s)
Sample 400: medio (t=0.8s)  
Sample 799: final (t=1.598s)

Razón: Balance entre detalle temporal y número de archivos
- Más samples → más detalle pero más espacio en disco
- 3 samples es suficiente para capturar dinámica intra-época
- Para videos suaves: puedes usar más (--samples 0 100 200 ... 900)
```

### ¿Por qué la sincronización varía tanto entre samples?

```
Es NORMAL y esperado:

Sincronización no es estática:
- Fluctúa naturalmente en escalas de 100-500ms
- Refleja micro-estados cerebrales transitorios
- Similar a "parpadeo" de redes funcionales en fMRI

Alta variabilidad puede indicar:
1. Estado exploratorio (baja meta-estabilidad)
2. Transiciones entre atractores
3. Alta entropía neural (¿efecto de DMT?)

Baja variabilidad:
1. Estado estable, meta-estable
2. Alta predictibilidad
3. Baja entropía (¿sueño profundo?)
```

### ¿Qué threshold usar para visualización?

```
Threshold = percentil de la matriz de sincronización

Ejemplo con threshold=80:
- Solo muestra aristas con PLV > percentil 80
- Es decir, el 20% de conexiones más fuertes

Recomendaciones:
- Exploración: 70-80 (muestra estructura principal)
- Papers: 90-95 (solo conexiones muy fuertes, menos clutter)
- Análisis cuantitativo: NO uses threshold (usa matriz completa)

IMPORTANTE: Threshold es solo VISUAL, no afecta el cálculo de Kuramoto
```

### ¿EEG o Sources (STC)?

```
EEG (electrodos en cuero cabelludo):
✓ Medición directa (no requiere suposiciones)
✓ Menos procesamiento = menos artefactos introducidos
✗ Mezcla de múltiples fuentes (volume conduction)
✗ No sabemos QUÉ región cerebral específica

Sources (reconstrucción de fuentes):
✓ Localización anatómica específica
✓ Menos contaminación por volume conduction
✗ Requiere suposiciones (modelo de cabeza, algoritmo)
✗ Incertidumbre espacial (~1-2cm)
✗ "Leakage" entre regiones cercanas

Recomendación científica:
1. Analiza AMBOS
2. Si encuentras efecto en ambos → robusto
3. Si solo en uno → interpretar con cuidado
4. EEG para efectos globales, Sources para efectos espaciales
```

### ¿Por qué algunos scripts son tan lentos?

```
Jerarquía de velocidad:

RÁPIDO (segundos):
- visualize_results.py: solo lee datos, cálculos simples
- create_video_from_frames.py: ffmpeg está mega-optimizado
- plots3d.py: Plotly es eficiente

MEDIO (minutos con paralelización):
- generate_frames.py: matplotlib es lento, pero paralelo ayuda
  Sin workers: 20-30 min
  Con 20 workers: 2-3 min

LENTO (10-30 minutos):
- visualize_brain_3d.py: interpolación IDW + PyVista rendering
  Unavoidable, es mucho cómputo

EXTREMADAMENTE LENTO (30-60+ min):
- generate_video.py: matplotlib.animation sin paralelizar
  → Por eso creamos la alternativa (frames + ffmpeg)
```

### ¿Se puede usar GPU para acelerar?

```
Sí, pero limitado:

GPU aceleración disponible en:
✓ create_video_from_frames.py --gpu
  Usa NVENC (Nvidia encoder hardware)
  Speedup: 2-5x
  
✗ generate_frames.py
  Matplotlib usa CPU (numpy/scipy)
  GPU no ayudaría significativamente
  Mejor solución: paralelización CPU (--workers)
  
✗ visualize_brain_3d.py
  PyVista puede usar GPU para rendering
  Pero no está configurado aquí
  Speedup potencial: 1.5-2x (no vale la pena configurar)

Conclusión: --gpu en video encoding sí ayuda, lo demás no vale la pena
```

### ¿Cómo interpretar el parámetro de Kuramoto?

```
Valores típicos:
0.0 - 0.2:  Desincronización completa (ruido, artefacto, o estado caótico)
0.2 - 0.4:  Baja sincronización (normal en algunas bandas/condiciones)
0.4 - 0.6:  Sincronización moderada (típico en Alpha/Theta)
0.6 - 0.8:  Alta sincronización (sueño, meditación, o artefacto)
0.8 - 1.0:  Sincronización extrema (patológico o artefacto técnico)

Comparaciones importantes:
1. DMT vs EC: ¿Aumenta o disminuye sincronización?
2. Bandas: Theta suele tener más sincro que Gamma
3. Temporal: ¿Estable o fluctuante?
4. EEG vs Sources: ¿Consistente?

Cuidado con artefactos:
- Sincro extremadamente alta (>0.9) puede ser artefacto
- Sincro extremadamente baja (<0.1) puede ser ruido
- Cambios bruscos entre épocas → revisar datos crudos
```

### ¿Qué significa "época rechazada"?

```
Durante pre-procesamiento (anterior a visualización):
1. Se detectan artefactos automáticamente:
   - Movimientos oculares (EOG)
   - Parpadeos
   - Movimientos musculares (EMG)
   - Saltos de voltaje
   
2. Épocas con artefactos → marcadas en rejected_epochs.mat

3. En visualización:
   - Se generan DE TODAS FORMAS (para continuidad temporal)
   - Se marcan en gris en la serie temporal de Kuramoto
   - Etiqueta "[FRAMES] Época X rechazada (se genera igual)"

Razón: Para videos, necesitas secuencia completa sin huecos
Para análisis estadístico: SÍ se excluyen las rechazadas
```

---

## 📖 Referencias Conceptuales

### Papers Fundamentales

1. **Parámetro de Kuramoto:**
   - Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence.
   - Acebrón et al. (2005). The Kuramoto model: A simple paradigm for synchronization.

2. **Sincronización de Fase en EEG:**
   - Lachaux et al. (1999). Measuring phase synchrony in brain signals.
   - Varela et al. (2001). The brainweb: Phase synchronization and large-scale integration.

3. **DMT y Neuroimagen:**
   - Carhart-Harris et al. (2016). Neural correlates of the LSD experience.
   - Timmermann et al. (2019). DMT models the near-death experience.

### Recursos Técnicos

- **MNE-Python:** https://mne.tools (análisis EEG/MEG)
- **PyVista:** https://docs.pyvista.org (visualización 3D científica)
- **NetworkX:** https://networkx.org (análisis de grafos)
- **ffmpeg:** https://ffmpeg.org (procesamiento de video)

---

## 🎓 Resumen Para Paper (Methods Section)

```
Visualizations were generated using custom Python scripts. 

Synchronization matrices (phase locking values) were visualized as:
(1) heatmaps with network-ordered parcellations, 
(2) graph representations with anatomically-positioned nodes, and 
(3) polar plots of instantaneous phases. 

The Kuramoto order parameter time series was overlaid to show global 
synchronization dynamics. For anatomical visualization, parcel-level 
synchronization values were interpolated onto a cortical surface mesh 
using inverse distance weighting (power=2) and rendered with PyVista.

Statistical comparisons across conditions (DMT/EC/EO) and frequency bands 
were performed on the Kuramoto parameter, with results visualized as 
time series (mean ± 95% CI) and boxplots. Videos were generated by 
parallel frame rendering (n=20 workers) followed by H.264 encoding 
(ffmpeg, CRF=18, 30fps).
```

---

## 🚀 Quick Start (TL;DR)

```bash
# 1. Test rápido (ver que funciona)
python generate_frames.py --test --workers 10

# 2. Generar todo (frames + video)
./generate_all.sh --gpu

# 3. Análisis estadístico
python visualize_results.py

# 4. (Opcional) Hilbert 3D interactivo
python visualize_hilbert_improved.py

# 5. (Opcional) Video 3D anatómico (LENTO)
python visualize_brain_3d.py --start 0 --end 5
```

---

**Última actualización:** Noviembre 2025
**Autor:** Pipeline de visualización para análisis DMT-EEG
**Licencia:** Uso académico

¿Preguntas? Revisa el código, está comentado en detalle. 🧠✨

