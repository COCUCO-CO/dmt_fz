# Índice de Scripts de Visualización

## Script Principal

### `plot.py`
Entry point único para todas las visualizaciones.

```bash
python plot.py -s S01 -c DMT -b Alpha -m advanced --max-epochs 10
```

**Modos:**
- `all` - Vista completa EEG + STC
- `eeg` - Solo EEG
- `stc` - Solo Source Space
- `advanced` - Tema oscuro con glow
- `video_smooth` - Video completo
- `video_simple` - Video de una vista
- `video_3d` - Cerebro 3D
- `video_advanced` - Video oscuro

---

## Módulos Internos

### `plot_utils.py`
Funciones compartidas y estado global.
- `load_file()`, `save_file()` - I/O
- `plot_mat()` - Matriz de conectividad
- `plot_graph()` - Grafo de red
- `plot_kuramoto_order()` - Timeline de r
- `plot_osc()` - Círculo de fases
- `load_subject_data()` - Carga datos y configura estado

### `plot_frames.py`
Generación de frames estáticos.
- `plot_all()` - Frame completo
- `plot_eeg_only()` - Solo EEG
- `plot_stc_only()` - Solo STC
- `plot_advanced()` - Estilo moderno

### `plot_videos.py`
Generación de videos.
- `generate_smooth_video()` - Video suave interpolado
- `generate_simple_network_video()` - Una vista animada
- `generate_3d_brain_video()` - Cerebro 3D
- `generate_advanced_video()` - Estilo oscuro

---

## Scripts Auxiliares

| Script | Función |
|--------|---------|
| `visualize_results.py` | Estadísticas comparativas |
| `visualize_hilbert_improved.py` | Hilbert 3D interactivo |
| `visualize_hilbert_video.py` | Video evolución Hilbert |
| `visualize_brain_3d.py` | Cerebro 3D interpolado (lento) |
| `create_video_from_frames.py` | Une frames con ffmpeg |

---

## Estructura de Archivos

```
viz_scripts/
├── plot.py              # ← Ejecutar este
├── plot_utils.py        # Utilidades compartidas
├── plot_frames.py       # Generación frames
├── plot_videos.py       # Generación videos
├── README.md            # Documentación principal
├── QUICKSTART.md        # Inicio rápido
└── INDEX.md             # Este archivo
```

## Outputs

```
visualizations/plot/
├── all/
├── eeg/
├── stc/
├── advanced/
├── video_smooth/
├── video_simple/
├── video_3d/
└── video_advanced/
```
