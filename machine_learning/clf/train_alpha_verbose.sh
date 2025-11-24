#!/bin/bash
# Verbose training script for Alpha band with detailed progress monitoring

set -e

echo "================================================================================"
echo "                   ENTRENAMIENTO ALPHA BAND - MODO VERBOSE"
echo "================================================================================"
echo ""
echo "📍 Directorio: $(pwd)"
echo "⏰ Inicio: $(date)"
echo ""

# ============================================================================
# PASO 1: VERIFICACIONES
# ============================================================================
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 1/6: Verificaciones previas"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Check data
echo "🔍 Verificando datos..."
DATA_DIR="/media/storage_hdd/dmt_fz/fwd-inv-stc"

for cond in DMT EC EO; do
    count=$(ls ${DATA_DIR}/${cond}/phases-*.pkl 2>/dev/null | wc -l)
    echo "  ✓ ${cond}: ${count} archivos"
done

# Check Python packages
echo ""
echo "🔍 Verificando paquetes Python..."
python3 << EOF
import sys
try:
    import torch
    print(f"  ✓ PyTorch: {torch.__version__}")
    print(f"  ✓ CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    
    import torch_geometric
    print(f"  ✓ PyTorch Geometric: {torch_geometric.__version__}")
    
    import tensorboard
    print(f"  ✓ TensorBoard: Instalado")
    
except ImportError as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Error en verificación de paquetes"
    echo "Ejecuta: pip install -r requirements.txt"
    exit 1
fi

# ============================================================================
# PASO 2: CREAR DIRECTORIOS
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 2/6: Preparando directorios"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

mkdir -p output_alpha/logs
mkdir -p output_alpha/analysis
mkdir -p output_alpha/visualizations
mkdir -p checkpoints_alpha
mkdir -p runs_alpha

echo "  ✓ output_alpha/"
echo "  ✓ output_alpha/logs/"
echo "  ✓ output_alpha/analysis/"
echo "  ✓ output_alpha/visualizations/"
echo "  ✓ checkpoints_alpha/"
echo "  ✓ runs_alpha/"

# ============================================================================
# PASO 3: INFORMACIÓN PRE-ENTRENAMIENTO
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 3/6: Configuración del entrenamiento"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

echo "📋 Parámetros:"
echo "  • Banda: Alpha (8-13 Hz)"
echo "  • Condiciones: DMT, EC, EO"
echo "  • Arquitectura: GAT (Graph Attention Network)"
echo "  • Épocas máximas: 300"
echo "  • Early stopping: patience=50"
echo "  • Batch size: 32"
echo "  • Learning rate: 0.001"
echo ""
echo "💾 Outputs:"
echo "  • Dataset cache: data/cache/dataset_DMT_EC_EO_Alpha_eeg.pkl"
echo "  • Modelo: checkpoints_alpha/best_model.pt"
echo "  • Resultados: output_alpha/test_results.json"
echo "  • TensorBoard: runs_alpha/"
echo ""
echo "⏱️  Tiempo estimado: 30-60 minutos"
echo ""

# ============================================================================
# PASO 4: ENTRENAMIENTO
# ============================================================================
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 4/6: Iniciando entrenamiento"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 Entrenando modelo Alpha..."
echo ""
echo "💡 TIP: Abre otra terminal y ejecuta para ver progreso en tiempo real:"
echo "   tensorboard --logdir=runs_alpha --port=6006"
echo "   Luego abre: http://localhost:6006"
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"

START_TIME=$(date +%s)

# Run training
python train_per_band.py --bands Alpha 2>&1 | tee output_alpha/logs/training_output.log

TRAIN_STATUS=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "❌ Error durante el entrenamiento"
    echo "📄 Ver logs en: output_alpha/logs/training_output.log"
    exit 1
fi

echo "✓ Entrenamiento completado en ${MINUTES}m ${SECONDS}s"

# ============================================================================
# PASO 5: RESULTADOS
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 5/6: Resultados del entrenamiento"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "output_alpha/test_results.json" ]; then
    echo "📊 Métricas de Test:"
    echo ""
    
    # Parse JSON and display nicely
    python3 << 'EOF'
import json
with open('output_alpha/test_results.json', 'r') as f:
    results = json.load(f)

print(f"  🎯 Accuracy:   {results.get('test_accuracy', 0)*100:6.2f}%")
print(f"  📈 F1 Score:   {results.get('test_f1', 0):6.4f}")
print(f"  🎪 Precision:  {results.get('test_precision', 0):6.4f}")
print(f"  🎭 Recall:     {results.get('test_recall', 0):6.4f}")
print(f"")
print(f"  ⭐ Mejor val accuracy: {results.get('best_val_acc', 0)*100:6.2f}% (época {results.get('best_epoch', 0)})")
EOF

else
    echo "⚠️  Archivo de resultados no encontrado"
fi

echo ""
echo "📁 Archivos generados:"
echo ""

# Check and list outputs
if [ -f "checkpoints_alpha/best_model.pt" ]; then
    SIZE=$(du -h checkpoints_alpha/best_model.pt | cut -f1)
    echo "  ✓ checkpoints_alpha/best_model.pt (${SIZE})"
fi

if [ -f "output_alpha/test_results.json" ]; then
    echo "  ✓ output_alpha/test_results.json"
fi

if [ -f "output_alpha/confusion_matrix.png" ]; then
    echo "  ✓ output_alpha/confusion_matrix.png"
fi

if [ -f "output_alpha/training_curves.png" ]; then
    echo "  ✓ output_alpha/training_curves.png"
fi

if [ -d "runs_alpha" ]; then
    TB_SIZE=$(du -sh runs_alpha | cut -f1)
    echo "  ✓ runs_alpha/ (${TB_SIZE} - TensorBoard logs)"
fi

LOG_COUNT=$(ls output_alpha/logs/*.log 2>/dev/null | wc -l)
if [ $LOG_COUNT -gt 0 ]; then
    echo "  ✓ output_alpha/logs/ (${LOG_COUNT} archivos)"
fi

# ============================================================================
# PASO 6: PRÓXIMOS PASOS
# ============================================================================
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "PASO 6/6: Próximos pasos"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

echo "🔍 Ver resultados detallados:"
echo ""
echo "  1. Métricas finales:"
echo "     cat output_alpha/test_results.json"
echo ""
echo "  2. Matriz de confusión:"
echo "     eog output_alpha/confusion_matrix.png"
echo "     # o: xdg-open output_alpha/confusion_matrix.png"
echo ""
echo "  3. Curvas de entrenamiento:"
echo "     eog output_alpha/training_curves.png"
echo ""
echo "  4. TensorBoard (logs completos):"
echo "     tensorboard --logdir=runs_alpha --port=6006"
echo "     # Abrir: http://localhost:6006"
echo ""
echo "  5. Logs de entrenamiento:"
echo "     cat output_alpha/logs/training_output.log"
echo ""

# Check accuracy and give recommendations
if [ -f "output_alpha/test_results.json" ]; then
    ACC=$(python3 -c "import json; f=open('output_alpha/test_results.json'); d=json.load(f); print(d.get('test_accuracy', 0)*100)")
    ACC_INT=${ACC%.*}
    
    echo "📈 Interpretación del resultado:"
    echo ""
    
    if [ "$ACC_INT" -ge 80 ]; then
        echo "  🎉 ¡Excelente! Accuracy ≥ 80%"
        echo "     La banda Alpha es muy informativa para clasificar DMT."
        echo ""
        echo "  ➡️  Próximo paso: Entrenar las demás bandas"
        echo "     python train_per_band.py --bands Theta Beta Delta Gamma"
    elif [ "$ACC_INT" -ge 70 ]; then
        echo "  ✓ Bien! Accuracy ≥ 70%"
        echo "    Resultado aceptable. La banda Alpha contiene información útil."
        echo ""
        echo "  ➡️  Próximo paso: Entrenar las demás bandas para comparar"
        echo "     python train_per_band.py --bands Theta Beta Delta Gamma"
    elif [ "$ACC_INT" -ge 60 ]; then
        echo "  ⚠️  Accuracy entre 60-70%"
        echo "     Resultado moderado. Considera:"
        echo "     - Ajustar hiperparámetros"
        echo "     - Verificar calidad de datos"
        echo "     - Probar con otras bandas"
    else
        echo "  ⚠️  Accuracy < 60%"
        echo "     Resultado bajo (baseline random = 33%)."
        echo "     Revisa:"
        echo "     - Logs de entrenamiento"
        echo "     - Calidad de datos"
        echo "     - Configuración del modelo"
    fi
fi

echo ""
echo "🔄 Entrenar todas las bandas:"
echo "   python train_per_band.py"
echo ""
echo "📊 Comparar bandas (después de entrenar todas):"
echo "   cat band_comparison_results.json"
echo ""
echo "🎯 Crear ensemble (después de entrenar todas):"
echo "   python ensemble_bands.py --method average"
echo ""

echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ PROCESO COMPLETADO"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "⏰ Finalizado: $(date)"
echo ""

