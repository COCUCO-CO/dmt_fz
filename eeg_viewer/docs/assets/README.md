# Assets para Documentación

Este directorio contiene todos los assets visuales (imágenes, GIFs, videos) usados en la documentación de EEG Viewer.

## Estructura

```
assets/
├── README.md           # Este archivo
├── common/             # Elementos compartidos (header, navegación, logo)
├── viewer/             # Página Viewer (/)
├── cleaner/            # Página Cleaner (/cleaner)
├── pipeline/           # Página Pipeline (/pipeline)
├── model/              # Página Model (/model)
└── analysis/           # Página Analysis (/analysis)
```

## Resumen de Assets Necesarios

### Por Página

| Página | Screenshots | GIFs | Total |
|--------|-------------|------|-------|
| Common | 6 | 2 | 8 |
| Viewer | 9 | 5 | 14 |
| Cleaner | 10 | 4 | 14 |
| Pipeline | 13 | 4 | 17 |
| Model | 9 | 3 | 12 |
| Analysis | 9 | 4 | 13 |
| **Total** | **56** | **22** | **78** |

### Prioridad de Captura

**Alta Prioridad** (esenciales para entender la interfaz):
1. `viewer/overview.png` - Vista principal
2. `viewer/file_browser.gif` - Navegación de archivos
3. `viewer/compare_mode.gif` - Comparación dual
4. `cleaner/overview.png` - Vista del cleaner
5. `cleaner/step_5_ica.gif` - ICA (más complejo)
6. `pipeline/overview.png` - Vista del pipeline
7. `model/training_metrics.gif` - Métricas en tiempo real
8. `common/header.png` - Header global

**Media Prioridad** (mejoran la documentación):
- Screenshots de cada paso del cleaner
- Screenshots de cada step del pipeline
- Visualizaciones específicas

**Baja Prioridad** (opcionales/decorativos):
- Logo personalizado
- Variaciones de tema
- Capturas redundantes

## Guía de Captura

### Herramientas Recomendadas

#### Screenshots
- **Linux**: Flameshot, gnome-screenshot, Spectacle
- **Windows**: Snipping Tool, ShareX, Greenshot
- **macOS**: ⌘+⇧+4, CleanShot X

#### GIFs
- **Linux**: Peek, Kazam → gifski, SimpleScreenRecorder
- **Windows**: ScreenToGif, ShareX
- **macOS**: GIPHY Capture, Kap

#### Optimización
- **PNG**: `optipng -o5 imagen.png`
- **GIF**: `gifsicle -O3 --colors 256 input.gif -o output.gif`

### Especificaciones

#### Screenshots (PNG)
- Formato: PNG (preferido) o JPEG (para fotos)
- Resolución: Mínimo 800px de ancho
- Tamaño máximo: 500KB por imagen
- Sin información personal visible

#### GIFs Animados
- FPS: 15-20
- Colores: 256 máximo
- Duración: 5-20 segundos
- Tamaño máximo: 5MB por GIF
- Loop: Infinito

### Naming Convention

```
{tipo}_{descripcion}.{ext}

Ejemplos:
- overview.png
- step_1_load.png
- file_browser.gif
- training_metrics.gif
```

- Usar `snake_case`
- Nombres descriptivos
- Sin espacios ni caracteres especiales

## Proceso de Documentación

### Paso 1: Preparar el Entorno
1. Iniciar la aplicación: `./run.sh`
2. Preparar datos de ejemplo
3. Configurar resolución del navegador

### Paso 2: Capturar Screenshots
1. Navegar a la sección
2. Configurar estado deseado
3. Capturar pantalla
4. Recortar y optimizar

### Paso 3: Grabar GIFs
1. Preparar la secuencia de acciones
2. Iniciar grabación
3. Ejecutar las acciones lentamente
4. Detener y optimizar

### Paso 4: Organizar
1. Nombrar correctamente
2. Mover a la carpeta correspondiente
3. Verificar que aparece en el markdown

## Contribución

Al agregar nuevos assets:

1. Seguir las convenciones de nombre
2. Optimizar antes de commit
3. Actualizar el README de la carpeta
4. Verificar que los links en INTERFACE_GUIDE.md funcionan

## Checklist

- [ ] common/logo.png (opcional)
- [ ] common/header.png
- [ ] common/navigation.png
- [ ] viewer/overview.png
- [ ] viewer/file_browser.gif
- [ ] viewer/eeg_traces.gif
- [ ] viewer/compare_mode.gif
- [ ] cleaner/overview.png
- [ ] cleaner/step_5_ica.gif
- [ ] pipeline/overview.png
- [ ] pipeline/visualization_panel.gif
- [ ] model/overview.png
- [ ] model/training_metrics.gif
- [ ] analysis/overview.png
- [ ] analysis/latent_pca.gif

*(Lista completa en cada subcarpeta)*

