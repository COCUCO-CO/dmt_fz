# 📊 Visualization Scripts

Scripts para visualización de datos de sincronización EEG/STC.

> 📘 **¿Primera vez aquí?** Lee la [**Guía Completa (GUIDE.md)**](GUIDE.md) para entender **QUÉ hace cada script, CÓMO funciona técnicamente y POR QUÉ es útil científicamente.**

> 📖 Este README es un **manual de referencia rápida**. Para explicaciones detalladas, ejemplos y conceptos científicos → [GUIDE.md](GUIDE.md)

---

## 🚀 Quick Start

```bash
cd viz_scripts

# 1. Ver configuración actual (sujeto, condición, banda)
./config_viz.sh

# 2. Test rápido (3 épocas, secuencial)
python generate_frames.py --test

# 3. Test rápido con paralelización (10x más rápido)
python generate_frames.py --test --workers 10

# 4. Generar rango completo con máxima velocidad
python generate_frames.py --end 200 --workers 20
```

---

## 📂 Scripts Disponibles

### 🎨 **Generación de Frames/Videos**

#### `generate_frames.py` ⭐ **MÁS USADO**
Genera frames estáticos (PNG) de sincronización EEG/STC.

```bash
# Secuencial (lento)
python generate_frames.py --start 0 --end 50

# Paralelo con 20 workers (RÁPIDO - recomendado)
python generate_frames.py --start 0 --end 50 --workers 20

# Opciones de vista
--view all   # EEG + STC (completo, default)
--view eeg   # Solo canales EEG
--view stc   # Solo red cerebral (sources)

# Control de muestras temporales
--samples 0 200 400 600 799  # 5 frames por época
```

**Performance:**
- Sin workers (secuencial): ~5-10s por época
- Con 20 workers: ~0.5-1s por época (10x más rápido)
- 200 épocas: ~2-3 min con paralelización vs 20-30 min secuencial

#### `create_video_from_frames.py` ⭐ **RECOMENDADO PARA VIDEOS**
Crea videos desde frames pre-generados usando ffmpeg (SÚPER RÁPIDO).

```bash
# Workflow óptimo (10-20x más rápido):
# 1. Generar frames con paralelización
python generate_frames.py --end 200 --workers 20  # 2-3 min

# 2. Unir frames con ffmpeg
python create_video_from_frames.py --fps 30       # 10-20 seg!

# Con GPU Nvidia (aún más rápido)
python create_video_from_frames.py --fps 30 --gpu  # 5-10 seg!
```

**Performance:**
- Método antiguo (`generate_video.py`): 30-60 min ❌
- Método nuevo (frames + ffmpeg): 2-3 min ✅
- Con GPU: ~2 min ⚡

#### `generate_video.py` ⚠️ **LENTO - NO RECOMENDADO**
Genera frames en tiempo real (muy lento, solo para legacy).

```bash
# NO recomendado - muy lento
python generate_video.py --subject S01 --condition DMT --band Alpha --view graph
```

#### `plot.py`
Librería principal con todas las funciones de plotting.
- **No se ejecuta directamente** (usa los wrappers arriba)
- Configurar sujeto/condición/banda en líneas 369-376

---

### 📈 **Análisis Estadístico**

#### `visualize_results.py`
Análisis comparativo entre sujetos y condiciones.

```bash
python visualize_results.py
```

**Output:**
- `kuramoto_eeg_by_band.png` - Comparación Kuramoto EEG
- `kuramoto_sources_by_band.png` - Comparación Sources
- `kuramoto_comparison_boxplot.png` - Boxplots DMT vs EC vs EO
- `summary_statistics.csv` - Estadísticas descriptivas

---

### 🧠 **Visualización Hilbert**

#### `visualize_hilbert_improved.py`
Visualización 3D interactiva de la transformada de Hilbert (época única).

```bash
python visualize_hilbert_improved.py
```

**Output:**
- HTML interactivo (rotar/zoom en navegador)
- PNG de alta resolución

#### `visualize_hilbert_video.py` ⭐ **NUEVO - EVOLUCIÓN TEMPORAL**
Video 3D mostrando cómo evoluciona la señal analítica época a época.

```bash
# Test rápido (10 épocas, 2 FPS)
python visualize_hilbert_video.py --end 10 --fps 2

# Video completo (50 épocas, fluido)
python visualize_hilbert_video.py --end 50 --fps 5 --gpu

# Sin proyecciones (más limpio)
python visualize_hilbert_video.py --end 50 --no-projections
```

**Aplicaciones científicas:**
- Ver efectos temporales de DMT (onset, peak, offset)
- Detectar transiciones de estado cerebral
- Identificar patrones recurrentes de amplitud/fase
- Comparar estabilidad DMT vs EC/EO

---

### 🌐 **Visualización 3D**

#### `visualize_brain_3d.py`
Video 3D con colores de sincronización interpolados sobre superficie cerebral.

```bash
python visualize_brain_3d.py --start 0 --end 10 --fps 2
```

**Nota:** Muy lento (interpolación IDW), solo para figuras finales.

#### `plots3d.py`
Setup básico de Plotly 3D (solo muestra interactivo, no guarda).

---

## ⚙️ Configuración

### Cambiar Sujeto/Condición/Banda

**Opción 1: Script helper (recomendado)**
```bash
./config_viz.sh --set
```

**Opción 2: Editar manualmente**
```bash
vim plot.py +370

# Cambiar estas líneas:
band = "Alpha"        # Delta, Theta, Alpha, Beta, Gamma
cond = "DMT"          # DMT, EC, EO
subject = "S01"       # S01, S02, S03, etc.
```

---

## 📁 Output

Todos los outputs se guardan en `visualizations/`:

```
visualizations/
├── plot/                           # Frames y videos (generate_frames.py)
│   ├── 1001000000.png             # Frames individuales
│   └── *.mp4                       # Videos
├── visualize_results/              # Análisis estadístico
│   ├── kuramoto_*.png
│   └── summary_statistics.csv
├── visualize_hilbert_improved/     # Hilbert interactivo
│   └── hilbert_plotly_*.html
└── visualize_brain_3d/             # Videos 3D
    └── *.mp4
```

---

## 🔧 Troubleshooting

### "No module named 'plot'"
```bash
# Asegúrate de ejecutar desde viz_scripts/
cd viz_scripts
python generate_frames.py --test
```

### Multiprocessing no funciona
```bash
# Verificar que matplotlib use backend no-interactivo
export MPLBACKEND=Agg
python generate_frames.py --test --workers 10
```

### Cambió sujeto/banda pero sigue usando el anterior
```bash
# Forzar recarga de plot.py
rm __pycache__/plot.*.pyc
python generate_frames.py --test
```

### "Época rechazada"
Es normal, las épocas rechazadas **se generan igual** (para tener secuencia completa).

---

## 💡 Tips

### Para análisis rápido:
```bash
# Generar solo primeras 10 épocas de cada vista
python generate_frames.py --view eeg --end 10 --workers 10
python generate_frames.py --view stc --end 10 --workers 10
```

### Para videos finales:
```bash
# Workflow óptimo (RÁPIDO):
# 1. Generar frames en paralelo
python generate_frames.py --end 200 --workers 20  # 2-3 min

# 2. Crear video con ffmpeg
python create_video_from_frames.py --fps 30       # 10-20 seg

# O con GPU (si tienes Nvidia)
python create_video_from_frames.py --fps 30 --gpu # 5-10 seg
```

### Para papers:
```bash
# Análisis estadístico
python visualize_results.py

# Hilbert interactivo
python visualize_hilbert_improved.py

# 3D con interpolación (lento pero bonito)
python visualize_brain_3d.py --end 5 --fps 2
```

---

## 📚 Documentación Adicional

- **`GUIDE.md`** - 📘 **Guía completa explicativa** (¡LÉELO PRIMERO!)
  - Qué hace cada script en detalle
  - Cómo funciona técnicamente
  - Por qué es útil científicamente
  - Explicación de conceptos (Kuramoto, PLV, IDW, etc.)
  - FAQ técnicas
- `3D_VISUALIZATION_GUIDE.md` - Guía técnica de visualización 3D
- `../visualizations/README.md` - Estructura de outputs

---

## 🎯 Workflow Típico

```bash
# 1. Configurar
./config_viz.sh --set
# Sujeto: S01
# Condición: DMT
# Banda: Alpha

# 2. Test rápido (verifica que todo funciona)
python generate_frames.py --test --workers 10

# 3. Generar frames (paralelo - RÁPIDO)
python generate_frames.py --end 200 --workers 20  # 2-3 min

# 4. Crear video (ffmpeg - MUY RÁPIDO)
python create_video_from_frames.py --fps 30       # 10-20 seg
# O con GPU: --gpu (5-10 seg)

# 5. Análisis estadístico
python visualize_results.py

# 6. Visualización Hilbert
python visualize_hilbert_improved.py

# Outputs en: ../visualizations/
```

---

**Última actualización:** Se agregó soporte para multiprocessing (--workers) para acelerar generación de frames hasta 10-20x.

