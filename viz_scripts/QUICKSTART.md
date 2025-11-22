# 🚀 Quick Start - Visualización de Sincronización Cerebral

## ⚡ Primeros 5 Minutos

```bash
cd /media/storage_hdd/dmt_fz/viz_scripts

# 1. Test rápido - verifica que todo funciona
python generate_frames.py --test --workers 10

# 2. Ver imágenes generadas
eog ../visualizations/plot/*.png

# 3. Crear video de prueba
python create_video_from_frames.py --fps 30

# 4. Ver video
vlc ../visualizations/plot/*.mp4
```

**¿Funcionó?** ✅ Continúa abajo  
**¿Errores?** ❌ Ve a [Troubleshooting](#troubleshooting)

---

## 📋 Casos de Uso Comunes

### Caso 1: "Quiero generar figuras para una presentación" (10 min)

```bash
# Frames de sincronización completa (EEG + Sources)
python generate_frames.py --end 50 --workers 20

# Video fluido
python create_video_from_frames.py --fps 30 --gpu

# Análisis estadístico (comparar DMT vs EC vs EO)
python visualize_results.py

# Hilbert video (evolución temporal)
python visualize_hilbert_video.py --end 30 --fps 3
```

**Output:**
- `visualizations/plot/` - 150 imágenes PNG + video MP4
- `visualizations/visualize_results/` - gráficos comparativos + CSV
- `visualizations/visualize_hilbert_video/` - video 3D evolución temporal

---

### Caso 2: "Analizar nuevo sujeto/condición" (15 min)

```bash
# 1. Cambiar configuración (editar líneas 370-372)
vim plot.py
# subject = "S02"
# cond = "EC"  
# band = "Theta"

# 2. Generar todo automáticamente
./generate_all.sh --gpu

# 3. Ver resultados
eog ../visualizations/plot/*.png
vlc ../visualizations/plot/S02_EC_Theta_video.mp4
```

---

### Caso 3: "Preparar material para paper" (30-60 min)

```bash
# Figuras principales
python visualize_results.py  # Estadísticas comparativas

# Videos para supplementary material
for COND in DMT EC; do
    for BAND in Alpha Beta Theta; do
        # Actualizar config
        sed -i "370s/.*/band = \"$BAND\"/" plot.py
        sed -i "371s/.*/cond = \"$COND\"/" plot.py
        
        # Generar
        python generate_frames.py --end 100 --workers 20
        python create_video_from_frames.py --fps 30 --gpu
        
        # Hilbert video
        python visualize_hilbert_video.py --end 50 --fps 3 --gpu
    done
done

# Video 3D anatómico (LENTO - solo para figura destacada)
python visualize_brain_3d.py --start 0 --end 10 --fps 2
```

**Output:** 6 videos de red + 6 videos Hilbert + 1 video 3D anatómico

---

### Caso 4: "Exploración interactiva rápida" (5 min)

```bash
# Hilbert 3D interactivo (abrir en navegador)
python visualize_hilbert_improved.py
firefox ../visualizations/visualize_hilbert_improved/*.html

# Exploración 3D con Plotly
python plots3d.py  # Muestra en pantalla
```

---

## 🎯 Comandos por Objetivo

### Objetivo: **Sincronización Global**
```bash
python generate_frames.py --view all --end 100 --workers 20
python create_video_from_frames.py --fps 30
```
→ Muestra EEG + Sources con parámetro de Kuramoto

---

### Objetivo: **Comparación Estadística**
```bash
python visualize_results.py
libreoffice ../visualizations/visualize_results/summary_statistics.csv
```
→ Boxplots DMT vs EC vs EO, tablas estadísticas

---

### Objetivo: **Evolución Temporal de Fase**
```bash
python visualize_hilbert_video.py --end 50 --fps 3
```
→ Video 3D mostrando cómo cambia fase/amplitud época a época

---

### Objetivo: **Localización Anatómica**
```bash
python visualize_brain_3d.py --start 0 --end 10 --fps 2
```
→ Video 3D con colores sobre superficie cerebral (LENTO)

---

## 📊 Comparación de Scripts

| ¿Qué quieres ver? | Script | Tiempo | Output |
|-------------------|--------|--------|--------|
| Red cerebral completa | `generate_frames.py + create_video` | 3-5 min | Video MP4 |
| Estadísticas comparativas | `visualize_results.py` | 1 min | PNG + CSV |
| Evolución fase/amplitud | `visualize_hilbert_video.py` | 5-10 min | Video MP4 |
| Anatomía 3D coloreada | `visualize_brain_3d.py` | 15-30 min | Video MP4 |
| Exploración interactiva | `visualize_hilbert_improved.py` | 30 seg | HTML |

---

## ⚙️ Configuración Rápida

### Cambiar Sujeto/Condición/Banda

```bash
# Editar plot.py líneas 370-372
vim plot.py +370

# O usar sed:
sed -i '370s/.*/band = "Theta"/' plot.py
sed -i '371s/.*/cond = "DMT"/' plot.py
sed -i '372s/.*/subject = "S01"/' plot.py

# Verificar cambio
grep -A3 "frames_path = " plot.py
```

### Ver Configuración Actual

```bash
python3 -c "from plot import subject, cond, band; print(f'{subject} | {cond} | {band}')"
```

---

## 🎬 Ejemplos de Comandos Completos

### Ejemplo 1: Video corto de calidad (para reunión)
```bash
python generate_frames.py --start 0 --end 20 --workers 10 --view all
python create_video_from_frames.py --fps 30 --preset fast --gpu
```
**Output:** Video de ~40 segundos, generado en ~2 minutos

---

### Ejemplo 2: Video largo para análisis detallado
```bash
python generate_frames.py --end 200 --workers 20 --view all
python create_video_from_frames.py --fps 10 --preset medium
```
**Output:** Video de ~20 segundos (10 FPS), todas las épocas

---

### Ejemplo 3: Múltiples vistas en paralelo
```bash
# Terminal 1
python generate_frames.py --view eeg --end 100 --workers 10

# Terminal 2
python generate_frames.py --view stc --end 100 --workers 10

# Después, crear videos
python create_video_from_frames.py --fps 30
```

---

### Ejemplo 4: Hilbert con diferentes ROIs
```bash
# ROI automática (Default/Control network)
python visualize_hilbert_video.py --end 50 --fps 3

# ROI específica (ver índices en extra.pkl)
python visualize_hilbert_video.py --roi 10 --end 50 --fps 3
python visualize_hilbert_video.py --roi 25 --end 50 --fps 3
```

---

## 🛠️ Troubleshooting

### "ImportError: No module named X"
```bash
# Instalar dependencias
pip install numpy scipy matplotlib plotly tqdm kaleido

# Para videos
sudo apt install ffmpeg
```

### "Cannot connect to X server"
```bash
# Usar backend no-interactivo
export MPLBACKEND=Agg
python generate_frames.py --test --workers 10
```

### "ffmpeg not found"
```bash
sudo apt update
sudo apt install ffmpeg
```

### "Frames generados pero video no se crea"
```bash
# Verificar frames
ls -lh ../visualizations/plot/*.png

# Crear video manualmente
cd ../visualizations/plot
ffmpeg -framerate 30 -pattern_type glob -i '*.png' -c:v libx264 -crf 18 -pix_fmt yuv420p output.mp4
```

### "Script muy lento"
```bash
# ¿Estás usando --workers?
python generate_frames.py --end 50 --workers 20  # SÍ ✓

# ¿Tienes GPU Nvidia?
python create_video_from_frames.py --gpu  # 5x más rápido

# ¿Usas generate_video.py? (no recomendado)
# Cambia a: generate_all.sh
```

### "Las imágenes no muestran sujeto/condición/banda"
```bash
# Verificar que usas la versión actualizada de plot.py
grep "SUJETO:" viz_scripts/plot.py

# Si no aparece, el código no está actualizado
# Los títulos deberían verse como:
# "SUJETO: S01 | CONDICIÓN: DMT | BANDA: Alpha | ..."
```

---

## 📖 Documentación Completa

- **[INDEX.md](INDEX.md)** - Navegar toda la documentación
- **[GUIDE.md](GUIDE.md)** - Explicaciones detalladas de cada script
- **[README.md](README.md)** - Referencia técnica rápida
- **[3D_VISUALIZATION_GUIDE.md](3D_VISUALIZATION_GUIDE.md)** - Guía de visualización 3D

---

## 💡 Tips Pro

1. **Siempre empieza con `--test`** para verificar que funciona
2. **Usa `--workers 20`** para máxima velocidad (CPU con 32 threads)
3. **`--gpu` en videos** acelera 5-10x (si tienes Nvidia)
4. **Guarda configuraciones** en un log para recordar qué generaste
5. **Frames primero, video después** - más flexible que generate_video.py

---

## 🎓 Workflow Recomendado

```bash
# DÍA 1: Exploración inicial
python generate_frames.py --test --workers 10  # Verificar datos
python visualize_results.py                     # Overview estadístico
python visualize_hilbert_improved.py            # Ejemplo interactivo

# DÍA 2: Generar material completo
./generate_all.sh --gpu --epochs 200            # Videos completos
python visualize_hilbert_video.py --end 100     # Hilbert temporal

# DÍA 3: Material para paper (si es necesario)
python visualize_brain_3d.py --end 10           # 3D anatómico (lento)

# Después: analizar outputs
eog ../visualizations/plot/*.png
vlc ../visualizations/plot/*.mp4
libreoffice ../visualizations/visualize_results/*.csv
```

---

**¿Preguntas?** Lee [GUIDE.md](GUIDE.md) para explicaciones detalladas de cada script.

**¿Bugs?** Revisa el código - está bien comentado.

**¡Éxito en tu análisis! 🧠✨**

