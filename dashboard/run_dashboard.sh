#!/bin/bash
# =============================================================================
# DMT-EEG Dashboard Launcher
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "🧠 DMT-EEG Analysis Dashboard"
echo "=================================================="
echo ""

# Check if conda environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "dmt_fz" ]]; then
    echo "⚠️  Activating conda environment: dmt_fz"
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate dmt_fz
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    pip install streamlit plotly pandas
fi

# Check port availability
PORT=8501
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is in use. Trying port 8502..."
    PORT=8502
fi

echo ""
echo "🚀 Starting dashboard on port $PORT..."
echo "   Open: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "=================================================="
echo ""

streamlit run app.py --server.port $PORT --server.headless true





