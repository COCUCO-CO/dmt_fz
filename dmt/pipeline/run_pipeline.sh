#!/usr/bin/env bash
################################################################################
# Pipeline Completo de Análisis de Sincronización EEG
# Ejecuta todos los pasos del pipeline con workers configurables
# 
# Los scripts en esta carpeta apuntan al directorio padre para acceder a datos
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKERS=12
CONDITIONS="DMT EC EO"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored messages
print_step() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} ${GREEN}>>> $1${NC}" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
}

print_warning() {
    echo -e "${YELLOW}[WARNING] $1${NC}" | tee -a "$LOG_FILE"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Function to run a command with timing
run_timed() {
    local step_name="$1"
    shift
    local start_time=$(date +%s)
    
    print_step "INICIANDO: $step_name"
    
    if "$@" 2>&1 | tee -a "$LOG_FILE"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local minutes=$((duration / 60))
        local seconds=$((duration % 60))
        print_step "✓ COMPLETADO: $step_name (${minutes}m ${seconds}s)"
        echo ""
    else
        print_error "FALLO: $step_name"
        exit 1
    fi
}

# Check if conda environment is activated
check_environment() {
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        print_warning "No se detectó ambiente conda activo"
        print_info "Intentando activar dmt_fz..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate dmt_fz || {
            print_error "No se pudo activar el ambiente dmt_fz"
            exit 1
        }
    fi
    print_info "Ambiente activo: $CONDA_DEFAULT_ENV"
}

# Main pipeline
main() {
    print_step "INICIANDO PIPELINE COMPLETO"
    print_info "Workers: $WORKERS"
    print_info "Condiciones: $CONDITIONS"
    print_info "Log file: $LOG_FILE"
    print_info "Working directory: $SCRIPT_DIR"
    echo ""
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    # Check environment
    check_environment
    
    # Record start time
    pipeline_start=$(date +%s)
    
    # =========================================================================
    # PASO 1: Procesar EEG - Genera phases-*.pkl y syncro-*.pkl
    # =========================================================================
    run_timed "PASO 1: Procesamiento EEG (fwd.py)" \
        python fwd.py --jobs 0 --workers "$WORKERS" --conditions $CONDITIONS
    
    # =========================================================================
    # PASO 2a: Consolidar archivos de fases - Genera subject_phases_*.pkl
    # =========================================================================
    run_timed "PASO 2a: Consolidación de fases (save_load_pickle.py)" \
        python save_load_pickle.py --conditions $CONDITIONS
    
    # =========================================================================
    # PASO 2b: Filtrar por redes - Genera order_all-*.pkl
    # =========================================================================
    run_timed "PASO 2b: Filtrado por redes (multi2pool2.py)" \
        python multi2pool2.py
    
    # =========================================================================
    # PASO 2c: Calcular sincronización temporal - Genera syncro-*.pkl
    # =========================================================================
    run_timed "PASO 2c: Análisis temporal (calculate_syncro.py)" \
        python calculate_syncro.py --workers "$WORKERS" --conditions $CONDITIONS
    
    # =========================================================================
    # PASO 2d: Calcular order parameter - Genera order-*.pkl
    # =========================================================================
    run_timed "PASO 2d: Order parameter (generate_order.py)" \
        python generate_order.py --workers "$WORKERS" --conditions $CONDITIONS
    
    # =========================================================================
    # PASO 3: Generar gráficos - Genera archivos agregados y plots
    # =========================================================================
    run_timed "PASO 3: Generación de gráficos (plot_order.py)" \
        python plot_order.py --workers "$WORKERS"
    
    # =========================================================================
    # PASO 4: Correlaciones con cuestionarios - Genera heatmaps e histogramas
    # =========================================================================
    run_timed "PASO 4: Correlaciones (pearson.py)" \
        python pearson.py
    
    # =========================================================================
    # PASO 5 (OPCIONAL): Clustering - Identificación de estados
    # =========================================================================
    read -p "¿Ejecutar clustering (2-4 horas)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_timed "PASO 5: Clustering (clustering.py)" \
            python clustering.py
    else
        print_info "Saltando clustering"
    fi
    
    # =========================================================================
    # Summary
    # =========================================================================
    pipeline_end=$(date +%s)
    total_duration=$((pipeline_end - pipeline_start))
    total_minutes=$((total_duration / 60))
    total_seconds=$((total_duration % 60))
    
    echo ""
    echo "=========================================================================="
    print_step "PIPELINE COMPLETADO EXITOSAMENTE"
    echo "=========================================================================="
    print_info "Tiempo total: ${total_minutes}m ${total_seconds}s"
    print_info "Log guardado en: $LOG_FILE"
    echo ""
    print_info "Archivos generados en:"
    print_info "  - $(python -c 'import sys; sys.path.insert(0, \"$SCRIPT_DIR\"); from paths import RESULTS_DIR; print(RESULTS_DIR)')"
    echo ""
    print_info "Próximos pasos opcionales:"
    print_info "  1. Clustering: python clustering.py"
    print_info "  2. Videos 3D: python ../generate_video.py --subject S01 --condition DMT --band Alpha --view 3d --rotate"
    echo "=========================================================================="
}

# Trap errors
trap 'print_error "Pipeline interrumpido en línea $LINENO"; exit 1' ERR

# Run main function
main "$@"

