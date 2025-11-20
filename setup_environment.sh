#!/bin/bash
# Script de instalación del ambiente conda para dmt_fz

echo "================================================"
echo "   Configurando ambiente conda para dmt_fz"
echo "================================================"
echo ""

# Verificar si conda está instalado
if ! command -v conda &> /dev/null
then
    echo "❌ ERROR: conda no está instalado"
    echo "Por favor instala Anaconda o Miniconda primero:"
    echo "https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✓ Conda encontrado: $(conda --version)"
echo ""

# Crear ambiente desde environment.yml (solo Python + pip)
echo "📦 Creando ambiente conda con Python 3.10..."
conda env create -f environment.yml

# Verificar si la creación fue exitosa
if [ $? -eq 0 ]; then
    echo "✓ Ambiente creado exitosamente"
else
    echo "❌ Error al crear el ambiente"
    exit 1
fi

# Activar el ambiente
eval "$(conda shell.bash hook)"
conda activate dmt_fz

echo ""
echo "================================================"
echo "   Instalando dependencias con pip"
echo "================================================"
echo ""

# Instalar todas las dependencias desde requirements.txt
echo "📥 Instalando paquetes desde requirements.txt..."
pip install -r requirements.txt

# Verificar si la instalación fue exitosa
if [ $? -eq 0 ]; then
    echo "✓ Dependencias instaladas exitosamente"
else
    echo "❌ Error al instalar dependencias"
    exit 1
fi

echo ""
echo "================================================"
echo "   Verificando instalación"
echo "================================================"
echo ""

# Verificar instalaciones clave
python -c "import mne; print(f'✓ MNE: {mne.__version__}')"
python -c "import sklearn; print(f'✓ scikit-learn: {sklearn.__version__}')"
python -c "import tensorflow as tf; print(f'✓ TensorFlow: {tf.__version__}')"
python -c "import optuna; print(f'✓ Optuna: {optuna.__version__}')"
python -c "import numpy; print(f'✓ NumPy: {numpy.__version__}')"
python -c "import pandas; print(f'✓ Pandas: {pandas.__version__}')"

echo ""
echo "================================================"
echo "   ✅ INSTALACIÓN COMPLETA"
echo "================================================"
echo ""
echo "Para activar el ambiente, ejecuta:"
echo "    conda activate dmt_fz"
echo ""
echo "Para desactivar el ambiente:"
echo "    conda deactivate"
echo ""
echo "NOTAS:"
echo "------"
echo "• TensorFlow NO instalado por defecto (solo si usas EEGNet)"
echo "  Para instalarlo: pip install tensorflow"
echo ""
echo "• PyTorch NO instalado (solo para torch-kmeans.py - opcional)"
echo "  Para instalarlo: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
echo ""

