# ⚡ Tips de Rendimiento para Visualización 3D

## 🐌 ¿Por qué el video original es tan lento?

El script `visualize_brain_3d.py` con configuración por defecto genera:

- **60 frames por época** (2 segundos × 30 FPS)
- **327,684 vértices** de superficie cortical
- **100 esferas 3D** (parcelas) con iluminación especular
- **~500-1000 líneas 3D** (edges de conectividad)
- **Renderizado offscreen** de alta calidad

**Tiempo estimado:** 5-10 minutos por época = **2-3 horas para 20 épocas**

---

## 🚀 Soluciones Rápidas

### 1. Usar visualize_brain_3d_fast.py (MÁS RÁPIDO)

```bash
python visualize_brain_3d_fast.py --end 5
```

**Optimizaciones:**
- ❌ Sin superficie cortical
- ⬇️ 10 FPS en lugar de 30
- ⬇️ 5 frames por época en lugar de 60
- ❌ Sin edges
- ⬇️ Esferas de baja resolución (8×8 en lugar de 16×16)
- ⬇️ Resolución 1280×720 en lugar de 1920×1080

**Resultado:** ~30 segundos por época = **2.5 minutos para 5 épocas**

---

### 2. Modificar visualize_brain_3d.py

#### Opción A: Sin superficie cortical

```bash
python visualize_brain_3d.py --mode video --no-surface --end 20
```

**Ahorra:** ~40% del tiempo

#### Opción B: Sin edges

```bash
python visualize_brain_3d.py --mode video --no-edges --end 20
```

**Ahorra:** ~30% del tiempo

#### Opción C: Ambos

```bash
python visualize_brain_3d.py --mode video --no-surface --no-edges --end 20
```

**Ahorra:** ~60% del tiempo

#### Opción D: Cambiar FPS y duración en código

Editar `visualize_brain_3d.py` línea 400:

```python
# Antes (60 frames por época)
frames_per_epoch = int(fps * 2)  # 2 segundos por época

# Después (15 frames por época)
frames_per_epoch = int(fps * 0.5)  # 0.5 segundos por época
```

**Ahorra:** 75% del tiempo

---

### 3. Usar Plotly para exploración rápida

```bash
python visualize_brain_3d_plotly.py --mode animated --end 20
```

**Ventajas:**
- ⚡ Instantáneo (no renderiza frames)
- 🖱️ Interactivo en navegador
- 📊 Slider temporal
- 💾 Archivo HTML pequeño (~500KB)

**Desventaja:** No genera video MP4

---

## 📊 Comparación de Tiempos

| Método | Épocas | Tiempo | Calidad | Uso |
|--------|--------|--------|---------|-----|
| `visualize_brain_3d.py` (full) | 20 | **2-3 horas** | ⭐⭐⭐⭐⭐ | Publicación |
| `visualize_brain_3d.py` --no-surface | 20 | **1.5 horas** | ⭐⭐⭐⭐ | Presentación |
| `visualize_brain_3d_fast.py` | 20 | **10 minutos** | ⭐⭐⭐ | Preview |
| `visualize_brain_3d_fast.py` | 5 | **2.5 minutos** | ⭐⭐⭐ | Testing |
| Snapshots (4 imágenes) | 1 | **30 segundos** | ⭐⭐⭐⭐⭐ | Figuras |
| Plotly animated | 20 | **10 segundos** | ⭐⭐⭐⭐ | Exploración |

---

## 🎯 Workflow Recomendado

### Para Desarrollo/Testing:
```bash
# 1. Test rápido (2-3 minutos)
python visualize_brain_3d_fast.py --end 5

# 2. Revisar video generado
# 3. Si se ve bien, continuar con calidad alta
```

### Para Análisis:
```bash
# Usar Plotly interactivo
python visualize_brain_3d_plotly.py --mode animated --end 50
# Abrir en navegador y explorar
```

### Para Presentaciones:
```bash
# Snapshots de épocas clave
python visualize_brain_3d.py --mode snapshot --epoch 5
python visualize_brain_3d.py --mode snapshot --epoch 15
python visualize_brain_3d.py --mode snapshot --epoch 25
```

### Para Publicaciones:
```bash
# Video de alta calidad (dejar corriendo overnight)
python visualize_brain_3d.py --mode video --start 0 --end 100 --fps 30
```

---

## 💻 Optimizaciones Adicionales

### 1. Usar GPU para renderizado

Si tienes GPU NVIDIA:

```bash
pip install vtk[cuda]
```

**Mejora:** 2-3x más rápido

### 2. Reducir resolución de esferas

En `visualize_brain_3d.py`, línea con `pv.Sphere()`:

```python
# Alta resolución (lento)
sphere = pv.Sphere(radius=size, center=pos)  # Default: 16x16

# Baja resolución (rápido)
sphere = pv.Sphere(radius=size, center=pos, phi_resolution=8, theta_resolution=8)
```

**Mejora:** 50% más rápido

### 3. Procesar en chunks

En lugar de 100 épocas de una vez:

```bash
# Chunk 1
python visualize_brain_3d.py --start 0 --end 25 --output video_part1.mp4

# Chunk 2
python visualize_brain_3d.py --start 25 --end 50 --output video_part2.mp4

# Chunk 3
python visualize_brain_3d.py --start 50 --end 75 --output video_part3.mp4

# Chunk 4
python visualize_brain_3d.py --start 75 --end 100 --output video_part4.mp4

# Combinar con ffmpeg
ffmpeg -f concat -i list.txt -c copy video_final.mp4
```

### 4. Usar resolución más baja

Editar línea 343:

```python
# Alta resolución
plotter = viz.create_plotter(window_size=(1920, 1080), off_screen=True)

# Baja resolución (2x más rápido)
plotter = viz.create_plotter(window_size=(1280, 720), off_screen=True)
```

---

## 🔍 Debugging: ¿Por qué está tan lento?

### Revisar uso de recursos:

```bash
# Terminal 1: Ejecutar script
python visualize_brain_3d.py --mode video --end 5

# Terminal 2: Monitorear
htop  # Ver CPU/RAM
nvidia-smi  # Ver GPU (si tienes)
```

### Indicadores de problemas:

- **CPU al 100%:** Normal durante renderizado
- **RAM > 8GB:** Considera reducir épocas
- **Disk I/O alto:** Normal al guardar frames
- **Sin output por > 5 minutos:** Puede estar trabado, ver logs

### Si se queda trabado:

1. **Ctrl+C** para cancelar
2. Verificar `frames/temp_3d_frames/` - ¿se están generando imágenes?
3. Abrir una imagen para verificar que se renderiza correctamente
4. Si no se generan frames, revisar permisos de escritura

---

## 📝 Resumen de Comandos

```bash
# Testing (más rápido)
python visualize_brain_3d_fast.py --end 5

# Exploración (instantáneo)
python visualize_brain_3d_plotly.py --mode animated --end 20

# Snapshots (30 seg)
python visualize_brain_3d.py --mode snapshot --epoch 5

# Video calidad media (rápido)
python visualize_brain_3d.py --mode video --no-surface --no-edges --end 20

# Video calidad alta (lento)
python visualize_brain_3d.py --mode video --end 20
```

---

## 🆘 Si está muy lento

**Opción 1:** Usa `visualize_brain_3d_fast.py` siempre

**Opción 2:** Genera snapshots en lugar de video

**Opción 3:** Usa Plotly para exploración interactiva

**Opción 4:** Reduce épocas a procesar (5-10 en lugar de 50-100)

**Opción 5:** Procesa overnight y deja corriendo

---

**Recuerda:** El renderizado 3D de calidad es inherentemente lento. Para trabajo diario, usa las versiones rápidas. Para figuras finales, usa la versión completa overnight.


