# Guía de Visualización 3D del Cerebro 🧠

## 🎯 Dos Opciones Avanzadas

### 1. **PyVista** - Videos 3D de Alta Calidad
- ✅ Renderizado fotorrealista
- ✅ Videos MP4 suaves con rotación
- ✅ Superficie cortical completa (fsaverage)
- ✅ Iluminación profesional
- ⚠️ Requiere más recursos (GPU recomendada)

### 2. **Plotly** - Visualización Web Interactiva
- ✅ Interactivo 100% en navegador
- ✅ Rotación, zoom, hover con el mouse
- ✅ Slider temporal para animar épocas
- ✅ No requiere GPU, funciona en cualquier PC
- ⚠️ Sin superficie cortical (solo parcelas)

---

## 🚀 Instalación

### Opción 1: PyVista (recomendado para videos)

```bash
pip install pyvista mne
```

**Opcional (para superficie fsaverage):**
```bash
# MNE descargará automáticamente fsaverage la primera vez
# Requiere ~200MB de espacio
```

**Para mejorar rendimiento (opcional):**
```bash
# Si tienes GPU NVIDIA
pip install vtk[cuda]
```

### Opción 2: Plotly (recomendado para exploración)

```bash
pip install plotly kaleido
```

---

## 📊 Uso de PyVista (Videos)

### ⚡ Test RÁPIDO (recomendado primero):
```bash
python visualize_brain_3d_fast.py --end 5
```

Genera video de prueba en ~1-2 minutos:
- Sin superficie cortical (más rápido)
- 10 FPS (suficiente para preview)
- 5 frames por época
- Solo parcelas (sin edges)

### Test rápido con snapshots:
```bash
python visualize_brain_3d.py --mode snapshot --epoch 5
```

Genera 4 capturas desde diferentes ángulos:
- Isométrico
- Frontal
- Lateral
- Superior

### Generar video completo (LENTO - puede tomar horas):
```bash
python visualize_brain_3d.py --mode video --start 0 --end 20 --fps 30
```

⚠️ **Advertencia:** Con superficie cerebral + 30 FPS + edges, puede tomar 5-10 minutos POR ÉPOCA!

### Personalizar video:
```bash
python visualize_brain_3d.py \
    --mode video \
    --start 0 \
    --end 50 \
    --fps 60 \
    --output mi_cerebro_dmt.mp4 \
    --no-edges  # Sin mostrar conexiones (más limpio)
```

### Sin superficie cortical (más rápido):
```bash
python visualize_brain_3d.py --mode video --no-surface --end 30
```

---

## 🌐 Uso de Plotly (Interactivo)

### Visualización estática de una época:
```bash
python visualize_brain_3d_plotly.py --mode static --epoch 5
```

Abre `brain_3d_interactive.html` en tu navegador y:
- 🖱️ Click + Arrastrar = Rotar
- 🖱️ Scroll = Zoom
- 🖱️ Hover = Ver detalles de parcela

### Con conexiones entre parcelas:
```bash
python visualize_brain_3d_plotly.py --mode static --epoch 5 --edges
```

### Animación con slider temporal:
```bash
python visualize_brain_3d_plotly.py --mode animated --start 0 --end 20
```

Genera HTML con:
- ▶️ Botón Play/Pause
- 🎚️ Slider para navegar épocas
- 🔄 Sincronización dinámica en tiempo real

---

## 🎨 Características de la Visualización

### PyVista incluye:

1. **Superficie Cortical (opcional)**
   - Modelo 3D real del cerebro (fsaverage)
   - Semi-transparente para ver parcelas internas
   - ~80,000 vértices, ~160,000 caras

2. **Parcelas Cerebrales (100)**
   - Esferas 3D con iluminación especular
   - Tamaño proporcional a sincronización
   - Colores: Colormap Plasma (azul→amarillo→rojo)

3. **Conexiones Funcionales**
   - Solo conexiones fuertes (percentil 85+)
   - Opacidad proporcional a fuerza
   - Grosor ajustable

4. **Cámara Dinámica**
   - Rotación 360° suave
   - Elevación oscilante (efecto "breathing")
   - Zoom ajustable

5. **Metadata en Pantalla**
   - Sujeto, Condición, Banda
   - Época actual / Total
   - Order Parameter de Kuramoto (r)

### Plotly incluye:

1. **Parcelas Interactivas**
   - Hover muestra nombre y sincronización
   - Click para destacar
   - Colorbar con escala

2. **Navegación Libre**
   - Rotación ilimitada
   - Zoom preciso
   - Pan (desplazamiento)

3. **Slider Temporal (modo animated)**
   - Navega entre épocas instantáneamente
   - Play automático configurable
   - Labels claros de época

---

## 📁 Salida

Todos los outputs se organizan en `visualizations/` por script:

### plot.py (frames estáticos y videos):
```
visualizations/plot/
├── 1001000000.png                   # Frames individuales
├── 1001000400.png
├── network_dynamics.mp4             # Videos generados
└── network_alpha_smooth.mp4
```

### visualize_hilbert_improved.py:
```
visualizations/visualize_hilbert_improved/
├── hilbert_plotly_S01-DMT_Alpha_epoch1.html
└── hilbert_plotly_S01-DMT_Alpha_epoch1.png
```

### visualize_results.py:
```
visualizations/visualize_results/
├── kuramoto_eeg_by_band.png
├── kuramoto_sources_by_band.png
├── kuramoto_temporal_S01_Alpha.png
├── kuramoto_comparison_boxplot.png
└── summary_statistics.csv
```

### visualize_brain_3d.py:
```
visualizations/visualize_brain_3d/
├── brain_3d_multiview.mp4
└── color_frames_S01_DMT_Alpha/      # Si no se pudo crear video
```

---

## ⚙️ Parámetros Avanzados (PyVista)

### En el código (`visualize_brain_3d.py`):

```python
generate_3d_brain_video(
    # Básico
    subject_id="S01",
    condition="DMT",
    band_name="Alpha",
    start_epoch=0,
    end_epoch=50,
    
    # Video
    fps=30,                    # Frames por segundo (30-60 recomendado)
    output_file="video.mp4",
    
    # Cámara
    rotation_speed=2.0,        # Grados por frame (más = más rápido)
    
    # Visualización
    show_brain_surface=True,   # Mostrar corteza
    show_edges=True,           # Mostrar conexiones
    edge_threshold=85,         # Percentil para conexiones (85-95)
    
    # Parcelas
    parcel_base_size=0.003,    # Tamaño base (metros en espacio MNI)
    parcel_scale_factor=3.0,   # Multiplicador por sincronización
    
    # Avanzado
    use_fsaverage=True         # False = solo parcelas (más rápido)
)
```

---

## 💡 Tips y Mejores Prácticas

### PyVista:

1. **Primera ejecución es lenta** (descarga fsaverage ~200MB)
   - Ejecuciones posteriores son rápidas

2. **Para videos largos:**
   - Usar `--no-surface` (50% más rápido)
   - Reducir `fps` a 24-30
   - Procesar en chunks de 50 épocas

3. **Calidad vs Velocidad:**
   - Calidad máxima: fps=60, surface=True, edges=True
   - Balance: fps=30, surface=True, edges=False
   - Rápido: fps=24, surface=False, edges=False

4. **Memoria:**
   - Cada época usa ~100MB
   - Con 8GB RAM: máximo 50-60 épocas por video
   - Solución: generar videos en chunks

### Plotly:

1. **Navegador recomendado:** Chrome o Firefox (mejor rendimiento WebGL)

2. **Para presentaciones:**
   - Usar `--mode static` (carga más rápida)
   - Sin edges (visualización más limpia)

3. **Para análisis:**
   - Usar `--mode animated` (slider temporal)
   - Con edges (ver conectividad)

4. **Compartir:**
   - Los HTML son autocontenidos (100-500KB)
   - Funcionan offline
   - Compatible con móviles

---

## 🔬 Interpretación Científica

### Colores (Plasma colormap):

- 🔵 **Azul/Morado**: Baja sincronización local
- 🟡 **Amarillo**: Sincronización moderada
- 🔴 **Rojo**: Alta sincronización local

### Tamaño de Parcelas:

- Pequeñas: Menor conectividad con otras regiones
- Grandes: Mayor conectividad (hub)

### Conexiones (edges):

- Solo se muestran las más fuertes (top 15% por defecto)
- Grosor/opacidad = Fuerza de sincronización

### Order Parameter (r):

- r ≈ 0: Desincronizado (ruido)
- r ≈ 0.5: Sincronización parcial
- r ≈ 1: Sincronización completa (poco realista en cerebro)

---

## 🎓 Ejemplos de Uso Científico

### 1. Comparar condiciones:
```bash
# DMT
python visualize_brain_3d.py --mode video --end 30 --output dmt_brain.mp4

# Cambiar en plot.py: cond = "EC"
# EC (baseline)
python visualize_brain_3d.py --mode video --end 30 --output ec_brain.mp4
```

### 2. Explorar banda específica:
```bash
# Cambiar en plot.py: band = "Gamma"
python visualize_brain_3d.py --mode video --end 20 --output gamma_fast.mp4
```

### 3. Análisis de hub regions:
```bash
# Generar con edges para ver hubs
python visualize_brain_3d_plotly.py --mode static --epoch 10 --edges

# En el navegador: hover sobre parcelas grandes con muchas conexiones
```

### 4. Series temporales:
```bash
# Video largo para ver evolución temporal
python visualize_brain_3d.py --mode video --start 0 --end 200 --fps 30
```

---

## 🐛 Troubleshooting

### PyVista no renderiza:
```bash
# Verificar instalación
python -c "import pyvista; pyvista.Report()"

# Si falla, reinstalar
pip uninstall pyvista vtk
pip install pyvista
```

### "No module named 'mne'":
```bash
pip install mne
```

### Video corrupto:
- Verificar espacio en disco
- Cerrar otros programas pesados
- Reducir resolución en código (window_size=(1280, 720))

### Plotly no se ve en navegador:
- Verificar que el archivo HTML se generó
- Abrir con Chrome/Firefox (no IE)
- Deshabilitar bloqueadores de JavaScript

### Muy lento:
- Usar `--no-edges` (3x más rápido)
- Reducir epochs procesadas
- Cerrar otras aplicaciones

---

## 📚 Referencias

- **PyVista**: https://docs.pyvista.org/
- **Plotly**: https://plotly.com/python/3d-scatter-plots/
- **MNE-Python**: https://mne.tools/stable/auto_tutorials/index.html
- **Fsaverage**: https://surfer.nmr.mgh.harvard.edu/fswiki/FsAverage

---

## 🚧 Próximas Mejoras

Ideas para implementar:

- [ ] Animación con morphing suave entre épocas
- [ ] Múltiples cerebros en subplots (comparar condiciones)
- [ ] Overlay de activación estadística (t-maps)
- [ ] Export a Blender para renders hiper-realistas
- [ ] VR/AR export para visualización inmersiva
- [ ] Time-frequency decomposition en 3D
- [ ] Network metrics overlay (betweenness, degree)

---

¿Preguntas? Ver código fuente comentado en:
- `visualize_brain_3d.py` (PyVista)
- `visualize_brain_3d_plotly.py` (Plotly)

