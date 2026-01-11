# Dataset Scanner Module

## 📋 Overview

El módulo `dataset_scanner.py` es un escáner inteligente de datasets para machine learning que detecta automáticamente el tipo de datos, estructura de directorios, formato de archivos y genera información detallada sobre el contenido del dataset.

## 🎯 Objetivo

Reemplazar la función `detect_dataset_type()` actual en `model.py` con un módulo más robusto y extensible que soporte todos los tipos de datasets comunes en ML moderno.

---

## 🔍 Tipos de Datasets Soportados

### 1. **Imágenes (Image)**
| Formato | Extensiones | Descripción |
|---------|-------------|-------------|
| Standard | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp` | Imágenes estándar |
| DICOM | `.dcm`, `.dicom` | Imágenes médicas |

**Estructuras detectadas:**
- **Flat**: Todas las imágenes en un directorio
- **By-class**: Subcarpetas por clase (ej: `cats/`, `dogs/`)
- **Train/Val/Test splits**: `train/`, `val/`, `test/`
- **ImageNet-style**: `train/{class}/`, `val/{class}/`

### 2. **Grafos (Graph)**
| Formato | Extensiones | Descripción |
|---------|-------------|-------------|
| PyTorch Geometric | `.pt` | Data objects de PyG |
| NetworkX | `.gpickle`, `.graphml`, `.gml` | Grafos NetworkX |
| Edge List | `.edgelist`, `.edges` | Lista de aristas |
| Adjacency | `.adj`, `.mtx` | Matrices de adyacencia |
| Custom Pickle | `.pkl` (phases-*, syncro-*) | Datos de sincronización EEG |

**Información extraída:**
- Número de nodos/aristas
- Features de nodos/aristas
- Clases/etiquetas
- Atributos globales del grafo

### 3. **Series Temporales / Señales (TimeSeries)**
| Formato | Extensiones | Descripción |
|---------|-------------|-------------|
| EEG/BioSignal | `.bdf`, `.edf`, `.fif`, `.set` | Señales EEG (MNE-compatible) |
| Audio | `.wav`, `.mp3`, `.flac`, `.ogg` | Archivos de audio |
| Time Series | `.ts`, `.csv` (con timestamp) | Series temporales genéricas |
| NumPy | `.npy`, `.npz` | Arrays NumPy |

**Información extraída:**
- Frecuencia de muestreo
- Número de canales
- Duración total
- Bandas de frecuencia (para EEG)

### 4. **Datos Tabulares (Tabular)**
| Formato | Extensiones | Descripción |
|---------|-------------|-------------|
| CSV/TSV | `.csv`, `.tsv` | Datos delimitados |
| Excel | `.xlsx`, `.xls` | Hojas de cálculo |
| Parquet | `.parquet` | Formato columnar |
| HDF5 | `.h5`, `.hdf5` | Datos jerárquicos |
| Feather | `.feather` | Formato rápido Arrow |

**Información extraída:**
- Número de filas/columnas
- Tipos de columnas
- Valores faltantes
- Columnas categóricas vs numéricas

### 5. **Texto/NLP (Text)**
| Formato | Extensiones | Descripción |
|---------|-------------|-------------|
| Plain Text | `.txt` | Texto plano |
| JSON | `.json`, `.jsonl` | Datos estructurados |
| Tokenized | `.pt`, `.pkl` (con tokens) | Datos pre-tokenizados |

### 6. **Formatos de Frameworks ML (Framework)**
| Formato | Extensiones | Descripción |
|---------|-------------|-------------|
| PyTorch | `.pt`, `.pth` | Modelos/datos PyTorch |
| TensorFlow | `.tfrecord`, `saved_model/` | Datos TF |
| HuggingFace | `.safetensors`, `dataset_dict.json` | Datasets HF |
| Pickle | `.pkl`, `.pickle` | Objetos Python serializados |

---

## 📁 Estructuras de Directorio Detectadas

### Split Types
```
1. SINGLE (sin splits)
   dataset/
   ├── data_001.png
   ├── data_002.png
   └── ...

2. TRAIN_TEST
   dataset/
   ├── train/
   └── test/

3. TRAIN_VAL_TEST
   dataset/
   ├── train/
   ├── val/ (o validation/)
   └── test/

4. BY_CLASS (por clases)
   dataset/
   ├── class_a/
   ├── class_b/
   └── class_c/

5. BY_SUBJECT (por sujetos - común en EEG/médico)
   dataset/
   ├── S01/
   ├── S02/
   └── S03/

6. BY_CONDITION (condiciones experimentales)
   dataset/
   ├── DMT/
   ├── EC/
   └── EO/

7. HIERARCHICAL (combinación)
   dataset/
   ├── train/
   │   ├── class_a/
   │   └── class_b/
   └── test/
       ├── class_a/
       └── class_b/
```

---

## 🏗️ Arquitectura del Módulo

```
dataset_scanner/
├── __init__.py
├── scanner.py          # Clase principal DatasetScanner
├── detectors/
│   ├── __init__.py
│   ├── base.py         # Clase base Detector
│   ├── image.py        # ImageDetector
│   ├── graph.py        # GraphDetector
│   ├── timeseries.py   # TimeSeriesDetector
│   ├── tabular.py      # TabularDetector
│   ├── text.py         # TextDetector
│   └── framework.py    # FrameworkDetector
├── analyzers/
│   ├── __init__.py
│   ├── structure.py    # Análisis de estructura de directorios
│   └── statistics.py   # Estadísticas del dataset
└── models.py           # Dataclasses para resultados
```

---

## 📊 Información Generada (DatasetInfo)

```python
@dataclass
class DatasetInfo:
    # Identificación
    path: Path
    name: str
    
    # Tipo principal
    primary_type: DatasetType  # IMAGE, GRAPH, TIMESERIES, TABULAR, TEXT, MIXED
    
    # Estructura
    structure: StructureInfo
        - split_type: SplitType  # SINGLE, TRAIN_TEST, TRAIN_VAL_TEST, etc.
        - has_classes: bool
        - classes: List[str]
        - has_subjects: bool
        - subjects: List[str]
        - hierarchy_depth: int
    
    # Archivos
    files: FileInfo
        - total_count: int
        - by_extension: Dict[str, int]
        - by_type: Dict[str, int]
        - sample_files: List[Path]
        - total_size_bytes: int
        - total_size_human: str
    
    # Splits (si existen)
    splits: Optional[SplitInfo]
        - train_count: int
        - val_count: int
        - test_count: int
        - train_ratio: float
        - val_ratio: float
        - test_ratio: float
    
    # Estadísticas específicas por tipo
    type_specific: Dict[str, Any]
        # Para imágenes:
        - image_sizes: List[Tuple[int, int]]
        - channels: int (1, 3, 4)
        - color_mode: str (grayscale, RGB, RGBA)
        
        # Para grafos:
        - num_nodes_range: Tuple[int, int]
        - num_edges_range: Tuple[int, int]
        - node_features_dim: int
        - edge_features_dim: int
        - is_directed: bool
        
        # Para series temporales/EEG:
        - sampling_rate: float
        - num_channels: int
        - duration_range: Tuple[float, float]
        - bands: List[str]
        
        # Para tabular:
        - num_rows: int
        - num_columns: int
        - column_types: Dict[str, str]
        - missing_values: Dict[str, int]
    
    # Metadatos
    detected_labels: List[str]
    has_annotations: bool
    annotation_format: Optional[str]
    
    # Compatibilidad
    compatible_frameworks: List[str]  # ["pytorch", "tensorflow", "sklearn"]
    suggested_loaders: List[str]      # ["ImageFolder", "DataLoader", etc.]
    
    # Warnings y sugerencias
    warnings: List[str]
    suggestions: List[str]
```

---

## 🔧 API Principal

```python
from app.utils.dataset_scanner import DatasetScanner, DatasetInfo

# Uso básico
scanner = DatasetScanner()
info: DatasetInfo = scanner.scan("/path/to/dataset")

# Con opciones
info = scanner.scan(
    path="/path/to/dataset",
    deep_scan=True,           # Analizar contenido de archivos
    sample_size=100,          # Número de archivos a muestrear
    detect_labels=True,       # Detectar etiquetas/clases
    analyze_statistics=True,  # Calcular estadísticas
)

# Resultado como dict (para UI)
info_dict = info.to_dict()

# Resumen legible
print(info.summary())

# Verificar compatibilidad
if info.is_compatible_with("pytorch"):
    loader_code = info.generate_loader_code("pytorch")
```

---

## 🧪 Plan de Testing

### Unit Tests

```python
# test_dataset_scanner.py

class TestImageDetector:
    def test_detect_png_folder(self):
        """Detectar carpeta con imágenes PNG."""
    
    def test_detect_imagenet_structure(self):
        """Detectar estructura ImageNet-style."""
    
    def test_detect_train_val_test_splits(self):
        """Detectar splits train/val/test."""
    
    def test_extract_image_dimensions(self):
        """Extraer dimensiones de imágenes."""

class TestGraphDetector:
    def test_detect_pyg_files(self):
        """Detectar archivos PyTorch Geometric."""
    
    def test_detect_phases_pkl(self):
        """Detectar archivos phases-*.pkl del proyecto."""
    
    def test_extract_graph_stats(self):
        """Extraer estadísticas de grafos."""

class TestTimeSeriesDetector:
    def test_detect_eeg_bdf(self):
        """Detectar archivos EEG en formato BDF."""
    
    def test_detect_numpy_timeseries(self):
        """Detectar series temporales en NumPy."""
    
    def test_extract_sampling_rate(self):
        """Extraer frecuencia de muestreo."""

class TestStructureAnalyzer:
    def test_detect_flat_structure(self):
        """Detectar estructura plana."""
    
    def test_detect_by_class_structure(self):
        """Detectar estructura por clases."""
    
    def test_detect_hierarchical_splits(self):
        """Detectar splits jerárquicos."""

class TestIntegration:
    def test_scan_real_phases_dataset(self):
        """Test con dataset real de phases del proyecto."""
    
    def test_scan_real_eeg_dataset(self):
        """Test con dataset real de EEG."""
```

### Test Fixtures (Mock Datasets)

```python
# conftest.py

@pytest.fixture
def mock_image_dataset(tmp_path):
    """Crear dataset de imágenes mock."""
    for split in ['train', 'val', 'test']:
        for cls in ['cat', 'dog']:
            (tmp_path / split / cls).mkdir(parents=True)
            for i in range(5):
                img = Image.new('RGB', (64, 64), color='red')
                img.save(tmp_path / split / cls / f'{i}.png')
    return tmp_path

@pytest.fixture
def mock_graph_dataset(tmp_path):
    """Crear dataset de grafos mock."""
    for i in range(10):
        data = Data(
            x=torch.randn(10, 5),
            edge_index=torch.randint(0, 10, (2, 20)),
            y=torch.tensor([i % 3])
        )
        torch.save(data, tmp_path / f'graph_{i}.pt')
    return tmp_path

@pytest.fixture  
def mock_timeseries_dataset(tmp_path):
    """Crear dataset de series temporales mock."""
    for i in range(5):
        data = np.random.randn(24, 1000)  # 24 channels, 1000 samples
        np.save(tmp_path / f'signal_{i}.npy', data)
    return tmp_path
```

---

## 📝 Implementación Paso a Paso

### Fase 1: Estructura Base (1-2 horas)
1. Crear estructura de directorios
2. Implementar `models.py` con dataclasses
3. Implementar clase base `Detector`
4. Implementar `DatasetScanner` principal

### Fase 2: Detectores de Tipo (2-3 horas)
1. `ImageDetector` - imágenes
2. `GraphDetector` - grafos (incluir phases-*.pkl)
3. `TimeSeriesDetector` - series temporales y EEG
4. `TabularDetector` - datos tabulares
5. `TextDetector` - texto/NLP
6. `FrameworkDetector` - formatos de frameworks

### Fase 3: Analizadores (1-2 horas)
1. `StructureAnalyzer` - análisis de directorios
2. `StatisticsAnalyzer` - estadísticas del dataset

### Fase 4: Integración con model.py (30 min)
1. Reemplazar `detect_dataset_type()` 
2. Actualizar UI para mostrar nueva información
3. Adaptar campos del formulario según tipo detectado

### Fase 5: Testing (1-2 horas)
1. Unit tests para cada detector
2. Integration tests con datasets reales
3. Test de rendimiento (datasets grandes)

---

## 🎨 Integración con UI (model.py)

### Antes (actual)
```python
def scan_dataset():
    info = detect_dataset_type(path)
    # Muestra información básica
```

### Después (nuevo)
```python
from app.utils.dataset_scanner import DatasetScanner

def scan_dataset():
    scanner = DatasetScanner()
    info = scanner.scan(path, deep_scan=True, sample_size=50)
    
    # Actualizar UI con información rica
    with dataset_info_container:
        ui.label(f"📁 {info.name}").classes('text-lg font-bold')
        ui.label(f"Type: {info.primary_type.value}")
        ui.label(f"Structure: {info.structure.split_type.value}")
        ui.label(f"Files: {info.files.total_count} ({info.files.total_size_human})")
        
        if info.splits:
            ui.label(f"Splits: train={info.splits.train_count}, val={info.splits.val_count}, test={info.splits.test_count}")
        
        if info.structure.has_classes:
            ui.label(f"Classes: {', '.join(info.structure.classes[:10])}")
        
        # Warnings
        for warning in info.warnings:
            ui.label(f"⚠️ {warning}").style('color: orange;')
        
        # Suggestions
        for suggestion in info.suggestions:
            ui.label(f"💡 {suggestion}").style('color: cyan;')
```

---

## ✅ Checklist de Implementación

- [ ] Crear `app/utils/dataset_scanner/__init__.py`
- [ ] Crear `app/utils/dataset_scanner/models.py`
- [ ] Crear `app/utils/dataset_scanner/scanner.py`
- [ ] Crear `app/utils/dataset_scanner/detectors/base.py`
- [ ] Crear `app/utils/dataset_scanner/detectors/image.py`
- [ ] Crear `app/utils/dataset_scanner/detectors/graph.py`
- [ ] Crear `app/utils/dataset_scanner/detectors/timeseries.py`
- [ ] Crear `app/utils/dataset_scanner/detectors/tabular.py`
- [ ] Crear `app/utils/dataset_scanner/detectors/text.py`
- [ ] Crear `app/utils/dataset_scanner/analyzers/structure.py`
- [ ] Crear `app/utils/dataset_scanner/analyzers/statistics.py`
- [ ] Crear tests en `tests/test_dataset_scanner.py`
- [ ] Integrar con `model.py`
- [ ] Actualizar UI para mostrar nueva información

---

## 🚀 Beneficios

1. **Detección automática** - No necesita configuración manual
2. **Soporte universal** - Funciona con cualquier tipo de dataset común
3. **Información rica** - Estadísticas detalladas y útiles
4. **Sugerencias inteligentes** - Recomienda loaders y configuraciones
5. **Extensible** - Fácil agregar nuevos tipos de datos
6. **Integración simple** - Una sola llamada a `scanner.scan()`

---

## 📚 Referencias

- [PyTorch Geometric Data](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html)
- [MNE-Python (EEG)](https://mne.tools/stable/index.html)
- [HuggingFace Datasets](https://huggingface.co/docs/datasets/)
- [torchvision.datasets](https://pytorch.org/vision/stable/datasets.html)









