# 📑 Índice de Documentación - viz_scripts

## 🎯 ¿Qué estás buscando?

### 🚀 "Quiero empezar rápido, generar visualizaciones YA"
→ **[README.md](README.md)** - Quick Start y comandos rápidos

```bash
cd viz_scripts
python generate_frames.py --test --workers 10
./generate_all.sh
```

---

### 📖 "Quiero ENTENDER qué hace cada script y cómo funciona"
→ **[GUIDE.md](GUIDE.md)** - Guía completa explicativa (⭐ RECOMENDADO)

**Contenido:**
- 🧠 Conceptos fundamentales (Kuramoto, PLV, sincronización)
- 🔍 Cada script explicado en detalle:
  - ¿Qué hace?
  - ¿Cómo funciona técnicamente?
  - ¿Por qué es útil científicamente?
- 🎨 Explicación detallada de **mapas 3D en movimiento** (IDW, PyVista)
- ❓ FAQ técnicas
- 🔄 Workflow completo de análisis

**Lee esto si:**
- Es tu primera vez con el código
- Necesitas entender la ciencia detrás de las visualizaciones
- Quieres saber por qué hay tantos scripts
- No entiendes qué es "Kuramoto", "PLV", "IDW", etc.
- Vas a escribir un paper y necesitas describir los métodos

---

### 🧊 "Necesito generar videos 3D del cerebro"
→ **[3D_VISUALIZATION_GUIDE.md](3D_VISUALIZATION_GUIDE.md)** - Guía técnica de 3D

**Contenido:**
- Uso de `visualize_brain_3d.py`
- Parámetros de PyVista
- Optimización de rendering
- Troubleshooting de videos 3D

---

### 📂 "¿Dónde están mis archivos generados?"
→ **[../visualizations/README.md](../visualizations/README.md)** - Estructura de outputs

```
visualizations/
├── plot/                    # Frames y videos (generate_frames.py)
├── visualize_results/       # Análisis estadístico
├── visualize_hilbert_improved/  # Hilbert 3D
└── visualize_brain_3d/      # Videos 3D del cerebro
```

---

## 📊 Referencia Rápida de Scripts

| Script | Función Principal | Velocidad | Cuándo Usar |
|--------|-------------------|-----------|-------------|
| `generate_frames.py` ⭐ | Frames estáticos PNG | Rápido (con `--workers`) | **Uso diario, análisis** |
| `create_video_from_frames.py` ⭐ | Unir frames → video | Muy rápido (ffmpeg) | **Después de generate_frames** |
| `generate_all.sh` 🚀 | Todo automático | Rápido | **Modo automático** |
| `visualize_results.py` | Análisis estadístico | Rápido | **Comparaciones entre condiciones** |
| `visualize_hilbert_improved.py` | Hilbert 3D interactivo (1 época) | Rápido | **Exploración fase/amplitud** |
| `visualize_hilbert_video.py` ⭐ | Video Hilbert evolución temporal | Medio | **Ver cambios época a época** |
| `visualize_brain_3d.py` | Videos 3D anatómicos | LENTO ⚠️ | **Figuras finales para papers** |
| `plot.py` | Librería de funciones | N/A | **No ejecutar directamente** |
| `plots3d.py` | Exploración 3D rápida | Rápido | **Debugging, exploración** |
| `generate_video.py` | Video (método viejo) | MUY LENTO ❌ | **NO usar, legacy** |

---

## 🎓 Para Escribir un Paper

### Methods Section
Ver [GUIDE.md - Resumen Para Paper](GUIDE.md#resumen-para-paper-methods-section)

### Figuras Recomendadas

1. **Fig 1: Sincronización global**
   ```bash
   python visualize_results.py
   # Output: kuramoto_comparison_boxplot.png
   ```

2. **Fig 2: Ejemplo de red cerebral**
   ```bash
   python generate_frames.py --test --view stc
   # Elegir frame representativo de visualizations/plot/
   ```

3. **Fig 3: Evolución temporal Hilbert**
   ```bash
   python visualize_hilbert_video.py --end 50 --fps 2
   # Video muestra dinámica fase/amplitud
   ```

4. **Fig 4: Anatomía 3D** (solo si hay espacio/tiempo)
   ```bash
   python visualize_brain_3d.py --start 0 --end 5 --fps 2
   # Extraer frame del video con ffmpeg
   ```

5. **Supplementary: Videos**
   ```bash
   ./generate_all.sh  # Red cerebral completa
   python visualize_hilbert_video.py --end 100  # Hilbert temporal
   # Upload como material suplementario
   ```

---

## 🔧 Troubleshooting

### "No sé por dónde empezar"
1. Lee [GUIDE.md](GUIDE.md) sección "Conceptos Fundamentales"
2. Ejecuta `python generate_frames.py --test --workers 10`
3. Abre las imágenes en `../visualizations/plot/`
4. Lee [GUIDE.md](GUIDE.md) sección del script que usaste

### "Los scripts son muy lentos"
- ¿Usaste `--workers 20`? → [GUIDE.md - FAQ: Por qué algunos scripts son lentos](GUIDE.md#por-qué-algunos-scripts-son-tan-lentos)
- ¿Tienes GPU Nvidia? → Usa `--gpu` en `create_video_from_frames.py`
- ¿Estás usando `generate_video.py`? → Cámbialo por `generate_all.sh`

### "No entiendo qué estoy visualizando"
→ Lee [GUIDE.md - Conceptos Fundamentales](GUIDE.md#conceptos-fundamentales)
- Explica Kuramoto, PLV, bandas de frecuencia, EEG vs Sources

### "¿Por qué hay tantos scripts?"
→ Lee [GUIDE.md - Introducción](GUIDE.md)
- Cada script tiene un propósito específico
- Hay redundancia histórica (versiones viejas mantenidas por compatibilidad)

### "Los mapas 3D no tienen sentido"
→ Lee [GUIDE.md - visualize_brain_3d.py](GUIDE.md#7-visualize_brain_3dpy-🧠-mapas-3d-en-movimiento)
- Explica interpolación IDW paso a paso
- Analogía con mapas meteorológicos
- Limitaciones científicas

---

## 🎯 Casos de Uso Comunes

### Caso 1: "Generar figuras para una presentación"
```bash
# Análisis comparativo
python visualize_results.py

# Frame representativo de red cerebral
python generate_frames.py --test --workers 10 --view stc

# Video corto para mostrar
./generate_all.sh --test --gpu
```
**Output:** Boxplots + frame bonito + video de 20 segundos

---

### Caso 2: "Analizar un nuevo sujeto"
```bash
# 1. Cambiar configuración
vim plot.py +370
# subject = "S05", cond = "DMT", band = "Alpha"

# 2. Generar todo
./generate_all.sh --gpu

# 3. Ver resultados
eog ../visualizations/plot/*.png
vlc ../visualizations/plot/S05_DMT_Alpha_video.mp4
```

---

### Caso 3: "Comparar DMT vs EC para todos los sujetos"
```bash
# Análisis estadístico grupal
python visualize_results.py

# Ver CSV con estadísticas
libreoffice ../visualizations/visualize_results/summary_statistics.csv
```

---

### Caso 4: "Paper - necesito todas las figuras"
```bash
# Loop sobre todas las condiciones relevantes
for COND in DMT EC; do
    for BAND in Theta Alpha Beta; do
        # Actualizar config
        sed -i "370s/.*/band = \"$BAND\"/" plot.py
        sed -i "371s/.*/cond = \"$COND\"/" plot.py
        
        # Generar
        python generate_frames.py --end 50 --workers 20
        python create_video_from_frames.py --fps 30
    done
done

# Análisis estadístico
python visualize_results.py

# Videos 3D (solo DMT Alpha como ejemplo)
sed -i "370s/.*/band = \"Alpha\"/" plot.py
sed -i "371s/.*/cond = \"DMT\"/" plot.py
python visualize_brain_3d.py --start 0 --end 10 --fps 2
```

**Output:** Decenas de videos + figuras estadísticas + video 3D anatómico

---

## 🌟 Recomendaciones del Desarrollador

1. **Siempre empieza con `--test`**
   - Verifica que todo funciona antes de procesar 200 épocas
   
2. **Usa `--workers` siempre que puedas**
   - `--workers 20` acelera 10-20x
   
3. **Lee GUIDE.md al menos una vez**
   - Te ahorrará horas de confusión
   
4. **No uses `generate_video.py`**
   - Está ahí por legacy, usa `generate_all.sh` en su lugar
   
5. **`visualize_brain_3d.py` es lento, úsalo solo para figuras finales**
   - 10-30 minutos de rendering, solo vale la pena para papers
   
6. **Guarda configuraciones en un script**
   - Anota qué sujeto/condición/banda generaste cada vez
   - Ejemplo: `echo "S01 DMT Alpha $(date)" >> log.txt`

---

## 📞 Contacto / Contribuciones

Si encuentras bugs, tienes preguntas o quieres contribuir:
1. Revisa primero [GUIDE.md](GUIDE.md) y este índice
2. Busca en el código (está comentado)
3. Crea un issue con detalles

---

**Última actualización:** Noviembre 2025  
**Mantenedor:** Pipeline DMT-EEG  
**Licencia:** Uso académico

---

## 🗺️ Mapa Mental

```
viz_scripts/
│
├─ 📖 DOCUMENTACIÓN
│  ├─ INDEX.md (este archivo) ← Empieza aquí
│  ├─ GUIDE.md ⭐ ← Explicaciones detalladas
│  ├─ README.md ← Referencia rápida
│  └─ 3D_VISUALIZATION_GUIDE.md ← Guía técnica 3D
│
├─ 🎬 GENERACIÓN (uso diario)
│  ├─ generate_frames.py ⭐ ← Genera frames PNG
│  ├─ create_video_from_frames.py ⭐ ← Frames → video
│  └─ generate_all.sh 🚀 ← Automático
│
├─ 📊 ANÁLISIS
│  └─ visualize_results.py ← Estadísticas comparativas
│
├─ 🔬 ESPECIALIZADO
│  ├─ visualize_hilbert_improved.py ← Hilbert 3D (1 época)
│  ├─ visualize_hilbert_video.py ⭐ ← Hilbert video temporal
│  ├─ visualize_brain_3d.py ← Videos 3D anatómicos (lento)
│  └─ plots3d.py ← Exploración rápida
│
├─ 🏗️ LIBRERÍA
│  └─ plot.py ← Funciones base (no ejecutar)
│
└─ ❌ LEGACY (no usar)
   └─ generate_video.py ← Método viejo (lento)
```

---

¡Feliz visualización! 🧠✨

