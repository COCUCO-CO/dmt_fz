#!/bin/bash
# Quick training script for Alpha band only (fastest way to validate pipeline)

echo "================================================================================"
echo "                   Quick Training: Alpha Band Only"
echo "================================================================================"
echo ""
echo "This script trains a GAT model on Alpha band (8-13 Hz) only."
echo "Alpha is the most studied band for psychedelic effects."
echo ""
echo "Expected time: 30-60 minutes on GPU"
echo "Expected accuracy: 75-85%"
echo ""
echo "================================================================================"

cd "$(dirname "$0")"

# Check Python
if ! command -v python &> /dev/null; then
    echo "Error: Python not found"
    exit 1
fi

echo ""
echo "[1/4] Training Alpha band model..."
python train_per_band.py --bands Alpha

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Training failed. Check error messages above."
    exit 1
fi

echo ""
echo "[2/4] Checking results..."
if [ -f "output_alpha/test_results.json" ]; then
    echo ""
    echo "✓ Results found:"
    cat output_alpha/test_results.json
else
    echo "⚠ Results file not found"
fi

echo ""
echo "[3/4] Results location:"
echo "  - Model:      checkpoints_alpha/best_model.pt"
echo "  - Metrics:    output_alpha/test_results.json"
echo "  - Plots:      output_alpha/*.png"
echo "  - TBoard:     runs_alpha/"

echo ""
echo "[4/4] Next steps:"
echo ""
echo "  View TensorBoard:"
echo "    tensorboard --logdir=runs_alpha"
echo ""
echo "  View confusion matrix:"
echo "    eog output_alpha/confusion_matrix.png"
echo ""
echo "  Train all bands:"
echo "    python train_per_band.py"
echo ""
echo "  Create ensemble:"
echo "    python train_per_band.py  # Train all first"
echo "    python ensemble_bands.py  # Then ensemble"
echo ""
echo "================================================================================"
echo "✓ Alpha band training complete!"
echo "================================================================================"

