#!/usr/bin/env bash
################################################################################
# Pipeline RÁPIDO para Testing (3 sujetos)
# Ejecuta el pipeline completo con solo 3 sujetos para verificar funcionamiento
################################################################################

set -e  # Exit on error

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
WORKERS=3
MAX_SUBJECTS=3
CONDITIONS="DMT"  # Solo DMT para test rápido
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${SCRIPT_DIR}/quick_test_$(date +%Y%m%d_%H%M%S).log"

# Print functions
print_step() {
    echo -e "${BLUE}[$(date +%H:%M:%S)]${NC} ${GREEN}>>> $1${NC}" | tee -a "$LOG_FILE"
}

print_error() {
    echo -e "${RED}[ERROR] $1${NC}" | tee -a "$LOG_FILE"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1" | tee -a "$LOG_FILE"
}

# Run with timing
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

# Main
main() {
    print_step "INICIANDO PIPELINE DE TEST RÁPIDO"
    print_info "Sujetos: $MAX_SUBJECTS"
    print_info "Workers: $WORKERS"
    print_info "Condiciones: $CONDITIONS"
    print_info "Log file: $LOG_FILE"
    echo ""
    
    cd "$SCRIPT_DIR"
    
    # Check conda environment
    if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
        print_info "Activando ambiente dmt_fz..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate dmt_fz || {
            print_error "No se pudo activar dmt_fz"
            exit 1
        }
    fi
    print_info "Ambiente: $CONDA_DEFAULT_ENV"
    echo ""
    
    pipeline_start=$(date +%s)
    
    # PASO 1: Procesar EEG (3 sujetos, ~30-40 min)
    run_timed "PASO 1: Procesamiento EEG (3 sujetos)" \
        python fwd.py --max-subjects "$MAX_SUBJECTS" --jobs 0 --workers "$WORKERS" --conditions $CONDITIONS
    
    # PASO 2a: Consolidar fases
    run_timed "PASO 2a: Consolidación de fases (save_load_pickle.py)" \
        python save_load_pickle.py --conditions $CONDITIONS
    
    # PASO 2b: Filtrar por redes
    run_timed "PASO 2b: Filtrado por redes" \
        python multi2pool2.py
    
    # PASO 2c: Análisis temporal
    run_timed "PASO 2c: Análisis temporal (calculate_syncro.py)" \
        python calculate_syncro.py --workers "$WORKERS" --conditions $CONDITIONS
    
    # PASO 2d: Order parameter
    run_timed "PASO 2d: Order parameter" \
        python generate_order.py --workers "$WORKERS" --conditions $CONDITIONS
    
    # PASO 3: Gráficos
    run_timed "PASO 3: Gráficos" \
        python plot_order.py --max-subjects "$MAX_SUBJECTS" --workers "$WORKERS"
    
    # PASO 4: Correlaciones
    run_timed "PASO 4: Correlaciones" \
        python pearson.py
    
    # Summary
    pipeline_end=$(date +%s)
    total_duration=$((pipeline_end - pipeline_start))
    total_minutes=$((total_duration / 60))
    total_seconds=$((total_duration % 60))
    
    echo ""
    echo "=========================================================================="
    print_step "TEST COMPLETADO EXITOSAMENTE"
    echo "=========================================================================="
    print_info "Tiempo total: ${total_minutes}m ${total_seconds}s"
    print_info "Log: $LOG_FILE"
    echo ""
    print_info "Para ejecutar el pipeline completo (29 sujetos):"
    print_info "  ./run_full_pipeline.sh"
    echo ""
    print_info "Para generar un video de prueba:"
    print_info "  python generate_video.py --subject S01 --condition DMT --band Alpha --view graph --epoch-duration 0.5"
    echo "=========================================================================="
}

# Trap errors
trap 'print_error "Pipeline interrumpido"; exit 1' ERR

# Run
main "$@"

