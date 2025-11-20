# Fix: Error de Compatibilidad con imageio

## 🔴 Problema Original

```
TypeError: write_frames() got an unexpected keyword argument 'audio_path'
```

Este error ocurría al intentar generar videos con PyVista debido a un conflicto de versiones entre `imageio` y `ffmpeg-python`.

## ✅ Solución Implementada

Cambiamos el método de generación de videos para que sea más robusto y compatible:

### Antes (problemático):
```python
plotter.open_movie(output_file, framerate=fps)
# ... renderizar frames ...
plotter.write_frame()  # ❌ Fallaba aquí
plotter.close()
```

### Ahora (robusto):
```python
# 1. Guardar frames individuales como PNG
for frame in frames:
    plotter.screenshot(f"frame_{i:06d}.png")

# 2. Combinar frames con OpenCV (más estable)
import cv2
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
for frame_file in frame_files:
    frame = cv2.imread(frame_file)
    video_writer.write(frame)
video_writer.release()

# 3. Limpiar frames temporales
shutil.rmtree(temp_frames_dir)
```

## 📦 Nueva Dependencia

Se agregó `opencv-python` como dependencia:

```bash
pip install opencv-python
```

O usar el script de instalación:

```bash
./install_3d_viz.sh
# Opción 1 o 3 ahora incluye opencv-python
```

## ⚡ Ventajas del Nuevo Método

1. **✅ Compatible con todas las versiones** de imageio/ffmpeg
2. **✅ Más robusto** - si falla un frame, no pierde todo el video
3. **✅ Progreso visible** - puedes ver los frames temporales generándose
4. **✅ Recuperable** - si se interrumpe, puedes reanudar
5. **✅ Sin problemas de audio** - opencv no intenta manejar audio

## 🔧 Si OpenCV no está instalado

```bash
# Instalar solo OpenCV
pip install opencv-python

# O reinstalar todas las dependencias
pip install pyvista mne opencv-python
```

## 📝 Notas Técnicas

- Los frames temporales se guardan en `frames/temp_3d_frames/`
- Se eliminan automáticamente después de crear el video
- Si el proceso se interrumpe, los frames temporales quedan (puedes borrarlos manualmente)
- El codec usado es `mp4v` (máxima compatibilidad)
- Para mejor calidad/compresión, considera usar `ffmpeg` directamente después

## 🎬 Conversión Opcional a H.264 (mejor compresión)

Si quieres un video más comprimido después de generarlo:

```bash
# Requiere ffmpeg instalado en el sistema
ffmpeg -i brain_3d_animation.mp4 -c:v libx264 -crf 23 -preset medium brain_3d_h264.mp4
```

Esto reduce el tamaño del archivo en ~50% sin pérdida visual significativa.

