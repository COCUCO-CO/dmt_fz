# Guía de Instalación

## 🚀 Instalación Rápida

### Opción 1: Script Automatizado (Recomendado)

```bash
# Dar permisos de ejecución al script
chmod +x setup_environment.sh

# Ejecutar instalación completa
./setup_environment.sh
```

Esto hace:
1. Crea ambiente conda con Python 3.10
2. Instala **todas** las dependencias con pip desde `requirements.txt`
3. Verifica que todo esté instalado correctamente

---

### Opción 2: Manual (Paso a Paso)

#### 1. Crear el ambiente conda (solo Python + pip)
```bash
conda env create -f environment.yml
```

#### 2. Activar el ambiente
```bash
conda activate dmt_fz
```

#### 3. Instalar todas las dependencias con pip
```bash
pip install -r requirements.txt
```

---

## 📦 ¿Qué se Instala?

### Dependencias Principales (Instaladas Automáticamente)

- **Científicas**: numpy, scipy, pandas
- **Visualización**: matplotlib, seaborn
- **Neurociencia**: mne, mne-bids
- **Machine Learning**: scikit-learn, scikit-learn-extra
- **Optimización**: optuna
- **Estadística**: statsmodels, pingouin
- **Utilidades**: tqdm, h5py, pymatreader, networkx, scikit-image
- **Jupyter**: jupyter, ipykernel

### Dependencias Opcionales (NO Instaladas)

**TensorFlow** - Solo si usas EEGNet
```bash
pip install tensorflow
```

**PyTorch** - Solo para torch-kmeans.py (experimental)
```bash
# Con GPU CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# O CPU only
pip install torch torchvision torchaudio
```

**MDTraj** - Solo para análisis de dinámica molecular
```bash
pip install mdtraj
```

**nice-eeg** - Solo si usas psd.py (puede dar problemas)
```bash
pip install nice-eeg
```

---

## ✅ Verificar Instalación

Después de instalar, verifica que todo funcione:

```bash
# Activar ambiente
conda activate dmt_fz

# Verificar librerías principales
python -c "import mne; print(f'MNE: {mne.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import optuna; print(f'Optuna: {optuna.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

---

## 🔧 Solución de Problemas

### Error: "conda: command not found"
```bash
# Instala Miniconda primero:
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

### Error al instalar pymatreader
```bash
# Instalar alternativa
pip install mat73  # O usar scipy.io.loadmat
```

### Error al instalar nice-eeg
```bash
# Es opcional - puedes omitirlo si no usas psd.py
# O intentar desde fuente:
pip install git+https://github.com/nice-tools/nice.git
```

### Ambiente muy lento al instalar
```bash
# Actualiza pip primero
pip install --upgrade pip setuptools wheel

# Luego instala requirements
pip install -r requirements.txt
```

---

## 🗑️ Desinstalar

Para eliminar el ambiente completamente:

```bash
# Desactivar si está activo
conda deactivate

# Eliminar ambiente
conda env remove -n dmt_fz
```

---

## 💡 Comandos Útiles

```bash
# Listar ambientes conda
conda env list

# Activar ambiente
conda activate dmt_fz

# Desactivar ambiente
conda deactivate

# Ver paquetes instalados
pip list

# Actualizar un paquete
pip install --upgrade nombre_paquete

# Exportar ambiente actual
pip freeze > requirements_actual.txt
```

---

## 🎯 Próximos Pasos

Después de instalar:

1. **Actualizar rutas en scripts** (ver `QUICKSTART.md`)
2. **Verificar datos EEG** disponibles
3. **Ejecutar pipeline** empezando por `fwd.py`

Ver documentación completa en:
- `README.md` - Visión general del proyecto
- `QUICKSTART.md` - Guía de inicio rápido
- `WORKFLOW.md` - Pipeline completo paso a paso
- `TECHNICAL_DETAILS.md` - Detalles técnicos y matemáticos








