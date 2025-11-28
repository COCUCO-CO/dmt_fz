#!/bin/bash
# Run experience predictor training

cd "$(dirname "$0")"

echo "========================================"
echo "Experience Predictor Training"
echo "========================================"

# Check if virtual environment exists
if [ -d "../clf/venv" ]; then
    source ../clf/venv/bin/activate
    echo "Using clf virtual environment"
fi

# Run training
python train.py --config config/config.yaml

echo "========================================"
echo "Training complete!"
echo "Results saved in: output/"
echo "========================================"





