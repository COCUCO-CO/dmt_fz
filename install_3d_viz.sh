#!/bin/bash
# Script de instalación para visualización 3D

echo "========================================="
echo "Instalando dependencias para viz 3D"
echo "========================================="
echo ""

# Verificar Python
python3 --version || { echo "Error: Python 3 no encontrado"; exit 1; }

echo "Selecciona tu opción:"
echo "1) PyVista (videos 3D de alta calidad)"
echo "2) Plotly (visualización web interactiva)"
echo "3) Ambas (recomendado)"
read -p "Opción [1-3]: " choice

case $choice in
    1)
        echo "Instalando PyVista + MNE + OpenCV..."
        pip install pyvista mne opencv-python
        ;;
    2)
        echo "Instalando Plotly..."
        pip install plotly kaleido
        ;;
    3)
        echo "Instalando ambas librerías..."
        pip install pyvista mne opencv-python plotly kaleido
        ;;
    *)
        echo "Opción inválida"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "✓ Instalación completada!"
echo "========================================="
echo ""
echo "Próximos pasos:"
echo ""
echo "1. Para videos 3D (PyVista):"
echo "   python visualize_brain_3d.py --mode snapshot --epoch 0"
echo ""
echo "2. Para visualización web (Plotly):"
echo "   python visualize_brain_3d_plotly.py --mode static --epoch 0"
echo ""
echo "3. Ver guía completa:"
echo "   cat 3D_VISUALIZATION_GUIDE.md"
echo ""

