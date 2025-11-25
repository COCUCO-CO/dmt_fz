#!/bin/bash

################################################################################
# End-to-End Pipeline for Graph-based EEG Classification
# 
# This script runs the complete workflow:
# 1. Graph dataset construction
# 2. Statistical analysis
# 3. Model training
# 4. Evaluation and visualization
#
# Usage: ./run_pipeline.sh [--force-rebuild] [--skip-analysis] [--config PATH]
################################################################################

set -e  # Exit on error

# Default arguments
FORCE_REBUILD=false
SKIP_ANALYSIS=false
CONFIG="config/config.yaml"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        --skip-analysis)
            SKIP_ANALYSIS=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --force-rebuild    Force rebuild dataset even if cache exists"
            echo "  --skip-analysis    Skip statistical analysis step"
            echo "  --config PATH      Path to config file (default: config/config.yaml)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
echo ""
echo "================================================================================"
echo "          Graph Attention Networks for EEG Brain State Classification"
echo "================================================================================"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG" ]; then
    log_error "Config file not found: $CONFIG"
    exit 1
fi

log_info "Using configuration: $CONFIG"
echo ""

# Check Python environment
if ! command -v python &> /dev/null; then
    log_error "Python not found. Please activate your conda environment."
    exit 1
fi

log_info "Python version: $(python --version)"

# Check required packages
log_info "Checking required packages..."
python -c "import torch; import torch_geometric; import yaml; import sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    log_error "Required packages not found. Please install:"
    echo "  pip install torch torch-geometric pyyaml scikit-learn matplotlib seaborn"
    exit 1
fi
log_success "All required packages found"
echo ""

# Create necessary directories
log_info "Creating output directories..."
mkdir -p checkpoints
mkdir -p output/logs
mkdir -p output/visualizations
mkdir -p data/cache
mkdir -p runs
log_success "Directories created"
echo ""

################################################################################
# STEP 1: Graph Analysis
################################################################################

if [ "$SKIP_ANALYSIS" = false ]; then
    echo "================================================================================"
    echo "STEP 1: Statistical Analysis of Graphs"
    echo "================================================================================"
    echo ""
    
    log_info "Running graph analysis (DMT vs EC vs EO)..."
    log_info "This will take a few minutes..."
    
    python analysis/analyze_graphs.py --config "$CONFIG"
    
    if [ $? -eq 0 ]; then
        log_success "Graph analysis completed"
        log_info "Results saved to: output/analysis/"
    else
        log_error "Graph analysis failed"
        exit 1
    fi
    echo ""
else
    log_warning "Skipping graph analysis (--skip-analysis flag set)"
    echo ""
fi

################################################################################
# STEP 2: Model Training
################################################################################

echo "================================================================================"
echo "STEP 2: Training Graph Attention Network"
echo "================================================================================"
echo ""

log_info "Starting model training..."
log_info "Monitor progress with: tensorboard --logdir=runs"
echo ""

# Build training command
TRAIN_CMD="python train.py --config $CONFIG"
if [ "$FORCE_REBUILD" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --force-rebuild"
    log_info "Force rebuilding dataset..."
fi

# Run training
$TRAIN_CMD

if [ $? -eq 0 ]; then
    log_success "Training completed successfully"
else
    log_error "Training failed"
    exit 1
fi
echo ""

################################################################################
# STEP 3: Summary
################################################################################

echo "================================================================================"
echo "Pipeline Completed!"
echo "================================================================================"
echo ""

# Check for output files
log_info "Checking output files..."

if [ -f "checkpoints/best_model.pt" ]; then
    log_success "✓ Model checkpoint saved"
    MODEL_SIZE=$(du -h checkpoints/best_model.pt | cut -f1)
    echo "    Size: $MODEL_SIZE"
fi

if [ -f "output/test_results.json" ]; then
    log_success "✓ Test results saved"
    echo "    Content:"
    cat output/test_results.json | python -m json.tool | head -20
fi

if [ -f "output/training_curves.png" ]; then
    log_success "✓ Training curves generated"
fi

if [ -f "output/confusion_matrix.png" ]; then
    log_success "✓ Confusion matrix generated"
fi

if [ -d "output/analysis" ]; then
    log_success "✓ Statistical analysis completed"
    NUM_FILES=$(ls output/analysis/*.csv 2>/dev/null | wc -l)
    echo "    CSV files: $NUM_FILES"
fi

echo ""
echo "================================================================================"
echo "Next Steps:"
echo "================================================================================"
echo ""
echo "1. View TensorBoard logs:"
echo "   $ tensorboard --logdir=runs"
echo "   Then open: http://localhost:6006"
echo ""
echo "2. Check test results:"
echo "   $ cat output/test_results.json"
echo ""
echo "3. View visualizations:"
echo "   $ ls output/*.png"
echo "   $ ls output/analysis/*.png"
echo ""
echo "4. Inspect training log:"
echo "   $ tail -100 output/logs/train_*.log"
echo ""
echo "================================================================================"
echo "Happy analyzing! 🧠📊"
echo "================================================================================"
echo ""


