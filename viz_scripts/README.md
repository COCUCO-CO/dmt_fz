# Visualization Scripts

Visualización de sincronización EEG/Source Space para análisis de estados alterados de consciencia.

## Uso

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

## Modos

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

## Argumentos

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

## Outputs

Cada modo guarda en su carpeta:
```
visualizations/plot/
├── all/            # Frames modo all
├── advanced/       # Frames modo advanced
├── video_smooth/   # Videos
└── videos/         # Videos desde frames
```

## Arquitectura

```
viz_scripts/
├── plot.py          # Entry point (ejecutar este)
├── plot_utils.py    # Funciones compartidas
├── plot_frames.py   # Generación de frames
├── plot_videos.py   # Generación de videos
└── docs/            # Documentación extendida
```

## Lo que muestra cada frame

| Componente | Qué muestra | Interpretación |
|------------|-------------|----------------|
| Timeline Kuramoto | r(t) promedio por época | Alto = sincronización global |
| Matriz | PLV entre canales/parcelas | Colores = fuerza de conexión |
| Grafo | Red de conectividad | Aristas = PLV > umbral |
| Círculo de fases | Posición angular de osciladores | Agrupados = sincronizados |
| Flecha dorada | Vector medio de fases | Longitud = r, dirección = fase media |

## Scripts adicionales

| Script | Uso |
|--------|-----|
| `visualize_results.py` | Estadísticas comparativas |
| `visualize_brain_3d.py` | Cerebro 3D interpolado |
| `create_video_from_frames.py` | ffmpeg wrapper (legacy) |

## Troubleshooting

```bash
# Ejecutar siempre desde viz_scripts
cd viz_scripts

# Limpiar cache
rm -rf __pycache__

# Ver ayuda
python plot.py --help
```

Ver `docs/` para documentación técnica detallada.
