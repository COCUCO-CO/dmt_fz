# 📊 Visualizations Directory

Esta carpeta contiene **todos los outputs** generados por los scripts de `viz_scripts/`, organizados por script para mayor claridad.

## 📁 Estructura

```
visualizations/
├── plot/                              # Outputs de plot.py
│   ├── *.png                         # Frames individuales (EEG+STC)
│   └── *.mp4                         # Videos de sincronización
│
├── visualize_hilbert_improved/       # Outputs de visualize_hilbert_improved.py
│   ├── hilbert_plotly_*.html         # Visualización interactiva 3D
│   └── hilbert_plotly_*.png          # Capturas de alta resolución
│
├── visualize_results/                # Outputs de visualize_results.py
│   ├── kuramoto_eeg_by_band.png      # Comparación Kuramoto EEG
│   ├── kuramoto_sources_by_band.png  # Comparación Kuramoto Sources
│   ├── kuramoto_temporal_*.png       # Evolución temporal
│   ├── kuramoto_comparison_boxplot.png # Boxplots comparativos
│   └── summary_statistics.csv        # Estadísticas descriptivas
│
└── visualize_brain_3d/               # Outputs de visualize_brain_3d.py
    ├── *.mp4                         # Videos 3D con colores interpolados
    └── color_frames_*/               # Frames individuales (backup)
```

## 🎯 ¿Qué Script Genera Qué?

| Script | Carpeta Output | Contenido |
|--------|---------------|-----------|
| `plot.py` | `plot/` | Frames y videos de sincronización 2D (EEG+STC) |
| `visualize_hilbert_improved.py` | `visualize_hilbert_improved/` | Señal analítica 3D interactiva (Plotly) |
| `visualize_results.py` | `visualize_results/` | Análisis estadístico grupal (comparaciones) |
| `visualize_brain_3d.py` | `visualize_brain_3d/` | Videos 3D con interpolación IDW |
| `plots3d.py` | *(solo interactivo)* | No guarda archivos (solo muestra en navegador) |
| `generate_frames.py` | `plot/` | Usa funciones de plot.py |
| `generate_video.py` | `plot/` | Usa funciones de plot.py |

## 🚀 Uso Rápido

```bash
# Generar frames 2D (se guardan en visualizations/plot/)
cd viz_scripts
python generate_frames.py --test

# Generar análisis estadístico (se guarda en visualizations/visualize_results/)
python visualize_results.py

# Generar visualización Hilbert interactiva (se guarda en visualizations/visualize_hilbert_improved/)
python visualize_hilbert_improved.py

# Generar video 3D con interpolación (se guarda en visualizations/visualize_brain_3d/)
python visualize_brain_3d.py --start 0 --end 10 --fps 2
```

## 🔄 Migración desde Versión Anterior

Si tenías outputs en carpetas antiguas:
- `frames/` → ahora en `visualizations/plot/`
- `results_plots/` → ahora dividido según el script que lo generó

**Nota:** Las carpetas antiguas `frames/` y `results_plots/` todavía existen para compatibilidad con scripts legacy. Puedes eliminarlas una vez migrado todo.

## 📋 Limpieza

Para limpiar outputs antiguos:

```bash
# Ver tamaño de cada carpeta
du -h --max-depth=1 visualizations/

# Limpiar outputs de un script específico
rm -rf visualizations/plot/*.png
rm -rf visualizations/visualize_brain_3d/*.mp4

# Limpiar todo (¡cuidado!)
rm -rf visualizations/*/
```

## ℹ️ Información Adicional

- Los scripts crean automáticamente las subcarpetas si no existen
- Los nombres de archivos siguen el patrón: `{tipo}_{sujeto}_{condición}_{banda}.{ext}`
- Para visualización 3D interactiva, ver `3D_VISUALIZATION_GUIDE.md` en `viz_scripts/`

---

**Última actualización:** Esta estructura se implementó para organizar mejor los outputs y evitar mezclar resultados de diferentes scripts.

