# Visualization Scripts

Visualización de sincronización EEG/Source Space para análisis de estados alterados de consciencia.

## Scripts Disponibles

| Script | Descripción | Input requerido |
|--------|-------------|-----------------|
| `plot.py` | Entry point para frames y videos | `phases-*.pkl`, `syncro-*.pkl` |
| `plot_order.py` | Histogramas y estadísticas de Kuramoto | `order-*.pkl`, `syncro-*.pkl` |
| `visualize_brain_3d.py` | Cerebro 3D interpolado | `phases-*.pkl` |
| `visualize_results.py` | Estadísticas comparativas | `order-*.pkl` |

---

## plot.py - Frames y Videos

```bash
cd viz_scripts

# Frames básicos
python plot.py -s S01 -c DMT -b Alpha -m all --max-epochs 10

# Frames avanzados (tema oscuro, glow effects)
python plot.py -s S01 -c DMT -b Alpha -m advanced --max-epochs 10

# Frames en SVG (vector)
python plot.py -s S01 -c DMT -b Alpha -m advanced --max-epochs 5 --svg

# Frames ligeros (1/4 del tamaño)
python plot.py -s S01 -c DMT -b Alpha -m advanced --max-epochs 10 -q low

# Frames medianos (1/2 del tamaño)
python plot.py -s S01 -c DMT -b Alpha -m all --max-epochs 10 -q medium

# Video
python plot.py -s S01 -c DMT -b Alpha -m video_advanced --max-epochs 5

# Crear video desde frames existentes
python plot.py -m frames_to_video --source advanced --fps 30
```

### Modos

| Modo | Descripción | Output |
|------|-------------|--------|
| `all` | EEG + Source Space completo | Frames |
| `eeg` | Solo EEG (24 electrodos) | Frames |
| `stc` | Solo Source Space (100 parcelas) | Frames |
| `advanced` | Tema oscuro, efectos glow | Frames |
| `video_smooth` | Video animado completo | MP4 |
| `video_simple` | Video de una vista | MP4 |
| `video_3d` | Cerebro 3D con rotación | MP4 |
| `video_advanced` | Video estilo oscuro | MP4 |
| `frames_to_video` | Une frames con ffmpeg | MP4 |

### Argumentos

```
-s, --subject       Sujeto (S01, S02, ...) [default: S01]
-c, --condition     Condición (DMT, EC, EO) [default: DMT]
-b, --band          Banda (Delta, Theta, Alpha, Beta, Gamma) [default: Alpha]
-m, --mode          Modo de visualización [default: all]

-w, --workers       Workers paralelos [default: 16]
--max-epochs        Límite de epochs [default: todos]
--svg               Guardar frames como SVG (vector)
-q, --quality       Calidad: high (~1MB), medium (~500KB), low (~250KB)

--fps               FPS para videos [default: 30]
--interpolation     Suavizado en videos [default: 10]
--source            Carpeta origen para frames_to_video
-o, --output        Nombre archivo salida
```

---

## plot_order.py - Estadísticas de Kuramoto

Genera histogramas y análisis estadísticos del parámetro de orden de Kuramoto.

```bash
cd viz_scripts

# Generar todos los plots
python plot_order.py

# Limitar sujetos (para testing)
python plot_order.py --max-subjects 5

# Controlar paralelismo
python plot_order.py --workers 8
```

### Argumentos

```
--max-subjects N    Límite de sujetos por condición [default: todos]
--workers N         Workers paralelos [default: cpu_count()]
```

### Outputs (en `plot_order_results/`)

| Tipo de plot | Descripción |
|--------------|-------------|
| `kuramoto_epochs_summary_*.png` | Trayectorias temporales por sujeto |
| `kuramoto_gamma_summary_*.png` | Promedios por condición |
| `hist_kuramoto_stc_*.png` | Histogramas por banda |
| `hist_kuramoto_pair_*.png` | Histogramas por red/hemisferio |
| `histogram_kuramoto_full_*.png` | Grid comparativo con tests FDR |

### Requisitos

Requiere archivos generados por el pipeline:
- `syncro-*.pkl` (de `fwd.py`)
- `order-*.pkl` (de `generate_order.py`)
- `r_kuramoto_nets_epochs_mean.pkl` (opcional, de `build_order_data.py`)

---

## Outputs

Cada script guarda en su carpeta:

```
visualizations/plot/
├── all/            # Frames modo all
├── advanced/       # Frames modo advanced
├── video_smooth/   # Videos
└── videos/         # Videos desde frames

plot_order_results/
├── kuramoto_epochs_summary_*.png
├── hist_kuramoto_*.png
└── histogram_kuramoto_full_*.png
```

---

## Arquitectura

```
viz_scripts/
├── plot.py              # Entry point frames/videos
├── plot_order.py        # Estadísticas de Kuramoto
├── plot_utils.py        # Funciones compartidas
├── plot_frames.py       # Generación de frames
├── plot_videos.py       # Generación de videos
├── visualize_brain_3d.py
├── visualize_results.py
└── docs/                # Documentación extendida
```

---

## Lo que muestra cada frame

| Componente | Qué muestra | Interpretación |
|------------|-------------|----------------|
| 📈 Timeline Kuramoto | r(t) promedio por época | Alto = sincronización global |
| 🟥 Matriz | PLV entre canales/parcelas | Colores = fuerza de conexión |
| 🕸️ Grafo | Red de conectividad | Aristas = PLV > umbral |
| ⭕ Círculo de fases | Posición angular de osciladores | Agrupados = sincronizados |
| 🏹 Flecha dorada | Vector medio de fases | Longitud = r, dirección = fase media |

---

## Scripts adicionales

| Script | Uso |
|--------|-----|
| `visualize_results.py` | Estadísticas comparativas |
| `visualize_brain_3d.py` | Cerebro 3D interpolado |
| `create_video_from_frames.py` | ffmpeg wrapper (legacy) |
| `generate_frames.py` | Generación batch de frames |
| `generate_video.py` | Generación batch de videos |

---

## Troubleshooting

```bash
# Ejecutar siempre desde viz_scripts
cd viz_scripts

# Limpiar cache
rm -rf __pycache__

# Ver ayuda
python plot.py --help
python plot_order.py --help
```

Ver `docs/` para documentación técnica detallada.
