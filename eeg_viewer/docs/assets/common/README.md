# Assets - Common

Imágenes y GIFs para elementos compartidos entre todas las páginas.

## Archivos Requeridos

### Screenshots (PNG)

| Archivo | Descripción | Resolución Sugerida |
|---------|-------------|---------------------|
| `logo.png` | Logo de la aplicación (opcional) | 200x200 |
| `header.png` | Header global con monitor de sistema | 1200x60 |
| `navigation.png` | Barra de navegación entre páginas | 600x50 |
| `system_monitor.png` | Monitor CPU/RAM/GPU | 300x50 |
| `running_indicator.png` | Indicador de proceso en ejecución | 200x40 |
| `theme_dark.png` | Ejemplo del tema oscuro | 800x400 |

### GIFs Animados

| Archivo | Descripción | Duración Sugerida |
|---------|-------------|-------------------|
| `navigation.gif` | Cambio entre páginas | 5-10s |
| `system_monitor.gif` | Monitor actualizándose | 5s |

## Elementos del Header

### System Monitor
Muestra en tiempo real:
- **CPU**: Porcentaje de uso
- **RAM**: Memoria utilizada / Total
- **GPU**: Uso y memoria VRAM (si disponible)

### Running Indicator
- Estado: Idle / Running
- Nombre del proceso en ejecución
- Animación de actividad

### Navigation Bar
Botones para cambiar entre páginas:
- VIEWER (/)
- CLEANER (/cleaner)
- PIPELINE (/pipeline)
- MODEL (/model)
- ANALYSIS (/analysis)

## Theme

El tema de la aplicación es estilo terminal:

| Elemento | Color |
|----------|-------|
| Background | `#0a0a0a` |
| Card | `#111111` |
| Border | `#1e1e1e` |
| Primary (green) | `#00ff88` |
| Secondary (cyan) | `#00d4ff` |
| Warning (yellow) | `#ffcc00` |
| Error (red) | `#ff5555` |
| Text | `#c8c8c8` |
| Text dim | `#666666` |

## Cómo Capturar

### Para el Header
1. Abre cualquier página
2. Captura solo la parte superior
3. Incluye el sistema monitor

### Para Navigation
1. Pasa el mouse sobre los botones
2. Captura con un botón resaltado
3. Muestra el estilo activo

### Para Logo
Si se desea crear un logo:
- Fondo transparente
- Colores del tema (verde terminal)
- Simple y reconocible

