#!/bin/bash
# Script todo-en-uno para generar frames y video de forma óptima

set -e  # Exit on error

# Colores
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default values
EPOCHS=200
WORKERS=20
FPS=30
GPU=false
TEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        --gpu)
            GPU=true
            shift
            ;;
        --test)
            TEST=true
            EPOCHS=10
            shift
            ;;
        --help)
            echo "Uso: $0 [opciones]"
            echo ""
            echo "Opciones:"
            echo "  --epochs N     Número de épocas a procesar (default: 200)"
            echo "  --workers N    Workers paralelos (default: 20)"
            echo "  --fps N        FPS del video (default: 30)"
            echo "  --gpu          Usar aceleración GPU (Nvidia)"
            echo "  --test         Modo test (solo 10 épocas)"
            echo "  --help         Mostrar esta ayuda"
            echo ""
            echo "Ejemplos:"
            echo "  $0                        # Procesamiento completo"
            echo "  $0 --test                 # Test rápido (10 épocas)"
            echo "  $0 --gpu --epochs 200     # Completo con GPU"
            echo "  $0 --workers 10 --fps 60  # 10 workers, 60 FPS"
            exit 0
            ;;
        *)
            echo "Opción desconocida: $1"
            echo "Usa --help para ver opciones disponibles"
            exit 1
            ;;
    esac
done

echo ""
echo "========================================================================"
echo -e "${BLUE}  GENERACIÓN OPTIMIZADA DE VISUALIZACIONES${NC}"
echo "========================================================================"
echo ""

# Mostrar configuración
echo -e "${YELLOW}📊 Configuración actual (desde plot.py):${NC}"
python3 -c "from plot import subject, cond, band; print(f'   Sujeto:    {subject}'); print(f'   Condición: {cond}'); print(f'   Banda:     {band}')"
echo ""

echo -e "${YELLOW}⚙️  Parámetros de procesamiento:${NC}"
echo "   Épocas:   $EPOCHS"
echo "   Workers:  $WORKERS"
echo "   FPS:      $FPS"
echo "   GPU:      $([ "$GPU" = true ] && echo '✓ NVENC' || echo '✗ CPU')"
echo "   Modo:     $([ "$TEST" = true ] && echo 'TEST' || echo 'COMPLETO')"
echo ""
echo "========================================================================"
echo ""

# Estimar tiempos
if [ "$TEST" = true ]; then
    ESTIMATED_FRAMES="30-60 seg"
    ESTIMATED_VIDEO="5-10 seg"
else
    ESTIMATED_FRAMES="2-3 min"
    if [ "$GPU" = true ]; then
        ESTIMATED_VIDEO="5-10 seg"
    else
        ESTIMATED_VIDEO="10-20 seg"
    fi
fi

echo -e "${BLUE}⏱️  Tiempo estimado:${NC}"
echo "   Frames: $ESTIMATED_FRAMES"
echo "   Video:  $ESTIMATED_VIDEO"
echo ""

read -p "¿Continuar? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ ! -z $REPLY ]]; then
    echo "Cancelado."
    exit 0
fi

echo ""
echo "========================================================================"
echo -e "${GREEN}PASO 1/2: Generando frames con paralelización${NC}"
echo "========================================================================"
echo ""

START_TOTAL=$(date +%s)
START_FRAMES=$(date +%s)

python3 generate_frames.py --end $EPOCHS --workers $WORKERS

END_FRAMES=$(date +%s)
ELAPSED_FRAMES=$((END_FRAMES - START_FRAMES))

echo ""
echo -e "${GREEN}✓ Frames generados en ${ELAPSED_FRAMES}s ($(echo "scale=1; $ELAPSED_FRAMES/60" | bc) min)${NC}"
echo ""

echo "========================================================================"
echo -e "${GREEN}PASO 2/2: Creando video con ffmpeg${NC}"
echo "========================================================================"
echo ""

START_VIDEO=$(date +%s)

if [ "$GPU" = true ]; then
    python3 create_video_from_frames.py --fps $FPS --gpu
else
    python3 create_video_from_frames.py --fps $FPS
fi

END_VIDEO=$(date +%s)
ELAPSED_VIDEO=$((END_VIDEO - START_VIDEO))

END_TOTAL=$(date +%s)
ELAPSED_TOTAL=$((END_TOTAL - START_TOTAL))

echo ""
echo "========================================================================"
echo -e "${GREEN}✓✓✓ PROCESO COMPLETADO ✓✓✓${NC}"
echo "========================================================================"
echo ""
echo "⏱️  Tiempos:"
echo "   Frames: ${ELAPSED_FRAMES}s ($(echo "scale=1; $ELAPSED_FRAMES/60" | bc) min)"
echo "   Video:  ${ELAPSED_VIDEO}s"
echo "   TOTAL:  ${ELAPSED_TOTAL}s ($(echo "scale=1; $ELAPSED_TOTAL/60" | bc) min)"
echo ""
echo "💾 Output: ../visualizations/plot/"
echo ""

# Mostrar archivos generados
VIDEO_FILE=$(python3 -c "from plot import subject, cond, band; print(f'{subject}_{cond}_{band}_video.mp4')")
VIDEO_PATH="../visualizations/plot/$VIDEO_FILE"

if [ -f "$VIDEO_PATH" ]; then
    VIDEO_SIZE=$(du -h "$VIDEO_PATH" | cut -f1)
    echo "📹 Video generado: $VIDEO_FILE ($VIDEO_SIZE)"
fi

echo ""
echo "========================================================================"
echo ""
echo -e "${BLUE}💡 TIPS:${NC}"
echo "   • Para cambiar sujeto/condición/banda:"
echo "     ./config_viz.sh --set"
echo ""
echo "   • Para procesar más rápido:"
echo "     ./generate_all.sh --gpu --workers 20"
echo ""
echo "   • Para test rápido:"
echo "     ./generate_all.sh --test"
echo ""
echo "========================================================================"
echo ""

