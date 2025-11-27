# Quick Start

## Generar Frames

```bash
cd viz_scripts

# Test rápido (3 epochs)
python plot.py -m all --max-epochs 3

# Vista avanzada (tema oscuro, glow)
python plot.py -m advanced --max-epochs 5

# Cambiar sujeto/condición/banda
python plot.py -s S02 -c EC -b Theta -m all --max-epochs 10
```

## Generar Videos

```bash
# Video avanzado (recomendado)
python plot.py -m video_advanced --max-epochs 10

# Video 3D con rotación
python plot.py -m video_3d --max-epochs 5 --rotate

# Video simple (solo grafo)
python plot.py -m video_simple --view graph --max-epochs 10
```

## Modos

| Modo | Descripción |
|------|-------------|
| `all` | EEG + Source Space completo |
| `eeg` | Solo EEG |
| `stc` | Solo Source Space |
| `advanced` | Tema oscuro con efectos |
| `video_smooth` | Video animado completo |
| `video_simple` | Una vista animada |
| `video_3d` | Cerebro 3D rotando |
| `video_advanced` | Video estilo oscuro |

## Outputs

```
visualizations/plot/{mode}/
```

Cada modo tiene su carpeta para no mezclar outputs.

## Ayuda

```bash
python plot.py --help
```
