# Guía de Generación de Frames - EEG/STC

## 🎯 Funcionalidades

El sistema permite generar frames de visualización con **3 modos**:

### Modos de visualización:

1. **`all`** (Completo): Muestra EEG y red cerebral (STC) juntos
   - Vista original con ambos paneles
   - Figura de 24x20 pulgadas
   - 6 subplots (3 para EEG + 3 para STC + 2 timelines Kuramoto)

2. **`eeg`** (Solo canales EEG): Muestra únicamente los electrodos del EEG
   - Vista compacta de 24x10 pulgadas
   - 4 subplots (matriz, grafo, osciladores, timeline)
   - Ideal para enfocarse en actividad superficial

3. **`stc`** (Solo red cerebral): Muestra únicamente la red de 100 parcelas cerebrales
   - Vista compacta de 24x10 pulgadas
   - 4 subplots (matriz, grafo, osciladores, timeline)
   - Ideal para enfocarse en fuentes corticales

---

## 🚀 Uso Rápido

### Test básico (3 épocas, vista completa):
```bash
python generate_frames.py --test
```

### Solo canales EEG:
```bash
python generate_frames.py --view eeg --end 10
```

### Solo red cerebral (para animación 3D después):
```bash
python generate_frames.py --view stc --end 20
```

### Vista completa (ambos):
```bash
python generate_frames.py --view all --end 15
```

---

## 📋 Parámetros Completos

```bash
python generate_frames.py [opciones]

Opciones:
  --start N       Época inicial (default: 0)
  --end N         Época final (default: todas las épocas disponibles)
  --step N        Paso entre épocas (default: 1, genera todas)
  --view MODE     Modo de visualización: 'all', 'eeg', 'stc' (default: all)
  --samples N...  Muestras temporales por época (default: 0 400 799)
  --test          Modo test rápido (3 épocas)
  -h, --help      Mostrar ayuda completa
```

---

## 📊 Ejemplos Avanzados

### Generar cada 2 épocas (más rápido):
```bash
python generate_frames.py --view stc --step 2 --end 50
```

### Control fino de muestras temporales:
```bash
python generate_frames.py --view stc --samples 0 200 400 600 799 --end 10
```

### Rango específico de épocas:
```bash
python generate_frames.py --view eeg --start 10 --end 20
```

---

## ⚙️ Configuración

Antes de ejecutar, verifica la configuración en `plot.py` (líneas 365-376):

```python
subject = "S01"    # Sujeto a procesar
cond = "DMT"       # Condición: "DMT", "EC", "EO"
band = "Alpha"     # Banda: "Delta", "Theta", "Alpha", "Beta", "Gamma"
```

---

## 📁 Salida

Los frames se guardan en: `/media/storage_hdd/dmt_fz/dmt/frames/`

### Nomenclatura de archivos:

- **Vista completa (`all`)**: `1002000.png`, `1002400.png`, ... 
  - Formato: `(1000+epoch)*1000+sample`
  
- **Solo EEG (`eeg`)**: `eeg_1002000.png`, `eeg_1002400.png`, ...
  - Formato: `eeg_(1000+epoch)*1000+sample`
  
- **Solo STC (`stc`)**: `stc_1002000.png`, `stc_1002400.png`, ...
  - Formato: `stc_(1000+epoch)*1000+sample`

### Por época se generan 3 frames (por defecto):
- Sample 0 (inicio de época)
- Sample 400 (mitad de época) 
- Sample 799 (final de época)

---

## 🧠 Próximos Pasos: Visualización 3D

Los frames `stc_*` están diseñados para ser base de animaciones 3D:

```python
# Usar stc_coords_3d de plot.py para posiciones 3D de parcelas
# Combinar con sincronización temporal para animación cerebro 3D
```

Ideas para implementar:
- Rotación 360° del cerebro
- Colores basados en sincronización
- Edges dinámicos según umbral de conectividad
- Interpolación suave entre frames

---

## 🔧 Troubleshooting

### Error: "No module named 'plot'"
```bash
cd /media/storage_hdd/dmt_fz/dmt
python generate_frames.py --test
```

### Error: épocas no disponibles
- Verifica que existan archivos `syncro-*.pkl` y `phases-*.pkl` en `fwd-inv-stc/{cond}/`

### Frames se ven mal
- Ajusta `threshold` (líneas 378-380 en `plot.py`)
- Para EEG: `eeg_threshold = 60`
- Para STC: `stc_threshold = 80`

---

## 💡 Tips

1. **Empieza con `--test`** para verificar que todo funcione
2. **Usa `--view stc`** si solo necesitas la red cerebral (más rápido)
3. **Ajusta `--samples`** si necesitas más/menos resolución temporal
4. **Usa `--step 2`** o mayor para pruebas rápidas
5. Los frames ocupan ~500KB cada uno, planifica el espacio en disco

---

## 📞 Soporte

Para más información sobre las funciones internas:
- `plot_eeg_only()`: Línea 388 de `plot.py`
- `plot_stc_only()`: Línea 431 de `plot.py`  
- `plot_all()`: Línea 470 de `plot.py`

