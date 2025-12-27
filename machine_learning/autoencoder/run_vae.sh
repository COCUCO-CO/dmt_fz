#!/bin/bash
# Run VAE training and activation extraction

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  BrainStateVAE Training Pipeline${NC}"
echo -e "${GREEN}======================================${NC}"

# Default values
CONFIG="config/config.yaml"
MAX_EPOCHS=""
EXTRACT_ONLY=false
HYPERPARAM_SEARCH=false
N_EXPERIMENTS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --max-epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --extract-only)
            EXTRACT_ONLY=true
            shift
            ;;
        --hyperparam)
            HYPERPARAM_SEARCH=true
            shift
            ;;
        --n-experiments)
            N_EXPERIMENTS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --config PATH       Path to config file (default: config/config.yaml)"
            echo "  --max-epochs N      Override max epochs"
            echo "  --extract-only      Only extract activations from existing model"
            echo "  --hyperparam        Run hyperparameter search instead of training"
            echo "  --n-experiments N   Number of hyperparam experiments (default: 10)"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

if [ "$HYPERPARAM_SEARCH" = true ]; then
    echo -e "${YELLOW}Running hyperparameter search...${NC}"
    
    CMD="python hyperparam_search.py --config $CONFIG --n_experiments $N_EXPERIMENTS"
    if [ -n "$MAX_EPOCHS" ]; then
        CMD="$CMD --max_epochs $MAX_EPOCHS"
    fi
    
    echo -e "Command: ${CMD}"
    eval $CMD
    
    echo -e "${GREEN}Hyperparameter search complete!${NC}"
    echo -e "View results: tensorboard --logdir=hyperparam_search/"
    exit 0
fi

if [ "$EXTRACT_ONLY" = true ]; then
    echo -e "${YELLOW}Extracting activations only...${NC}"
    
    if [ ! -f "checkpoints/best_model.pt" ]; then
        echo -e "${RED}Error: No checkpoint found at checkpoints/best_model.pt${NC}"
        exit 1
    fi
    
    python extract_activations.py \
        --checkpoint checkpoints/best_model.pt \
        --config "$CONFIG" \
        --per-subject \
        --save-attention
    
    echo -e "${GREEN}Extraction complete!${NC}"
    exit 0
fi

# Full training pipeline
echo -e "${YELLOW}Step 1: Training VAE...${NC}"

CMD="python train.py --config $CONFIG"
echo -e "Command: ${CMD}"
eval $CMD

echo -e "${GREEN}Training complete!${NC}"

echo -e "${YELLOW}Step 2: Extracting activations...${NC}"

python extract_activations.py \
    --checkpoint checkpoints/best_model.pt \
    --config "$CONFIG" \
    --per-subject \
    --save-attention

echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}  Pipeline Complete!${NC}"
echo -e "${GREEN}======================================${NC}"
echo ""
echo -e "Results:"
echo -e "  - Model: checkpoints/best_model.pt"
echo -e "  - Outputs: output/"
echo -e "  - Activations: activations/"
echo ""
echo -e "Next steps:"
echo -e "  1. View TensorBoard: tensorboard --logdir=runs/"
echo -e "  2. Run clustering: python ../pipeline/clustering.py --input activations/vae_activations_mu.pkl"












