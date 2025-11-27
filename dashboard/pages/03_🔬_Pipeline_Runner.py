#!/usr/bin/env python3
"""
Pipeline Runner Page
====================

Execute pipeline scripts with real-time output.
"""

import streamlit as st
import yaml
import subprocess
import threading
import queue
import time
from pathlib import Path
from datetime import datetime

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title="Pipeline Runner | DMT-EEG",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Pipeline Runner")
st.markdown("Ejecuta los scripts del pipeline de procesamiento EEG")

# Paths
PROJECT_ROOT = Path(config['paths']['project_root'])
PIPELINE_PATH = Path(config['paths']['pipeline'])
VIZ_SCRIPTS_PATH = PROJECT_ROOT / "viz_scripts"
CONDITIONS = config['data']['conditions']

# Pipeline scripts configuration
PIPELINE_SCRIPTS = [
    {
        "name": "fwd.py",
        "display_name": "1️⃣ Forward/Inverse Solution",
        "description": "Procesa archivos EEG: forward solution, inverse, Hilbert, sincronización, Kuramoto",
        "path": PIPELINE_PATH / "fwd.py",
        "estimated_time": "3-4 horas (29 sujetos)",
        "args": ["--jobs", "0", "--workers", "7"],
        "has_conditions": True,
        "has_max_subjects": True
    },
    {
        "name": "multi2pool2.py",
        "display_name": "2️⃣ Filtrado por Redes",
        "description": "Filtra datos por redes cerebrales (FPN, DMN, DAN, etc.)",
        "path": PIPELINE_PATH / "multi2pool2.py",
        "estimated_time": "2-5 minutos",
        "args": [],
        "has_conditions": False,
        "has_max_subjects": False
    },
    {
        "name": "generate_order.py",
        "display_name": "3️⃣ Order Parameter",
        "description": "Calcula el parámetro de orden de Kuramoto por red",
        "path": PIPELINE_PATH / "generate_order.py",
        "estimated_time": "30 seg - 1 min",
        "args": ["--workers", "20"],
        "has_conditions": True,
        "has_max_subjects": False
    },
    {
        "name": "build_order_data.py",
        "display_name": "4️⃣ Datos Agregados",
        "description": "Genera archivos .pkl con datos agregados de Kuramoto",
        "path": PIPELINE_PATH / "build_order_data.py",
        "estimated_time": "2-5 minutos",
        "args": ["--build-all", "--workers", "20"],
        "has_conditions": False,
        "has_max_subjects": False
    },
    {
        "name": "plot_order.py",
        "display_name": "5️⃣ Visualización",
        "description": "Genera histogramas y estadísticas de Kuramoto",
        "path": VIZ_SCRIPTS_PATH / "plot_order.py",
        "estimated_time": "5-10 minutos",
        "args": ["--workers", "20"],
        "has_conditions": False,
        "has_max_subjects": True
    },
    {
        "name": "pearson.py",
        "display_name": "6️⃣ Correlaciones",
        "description": "Correlaciones con cuestionarios subjetivos",
        "path": PIPELINE_PATH / "pearson.py",
        "estimated_time": "3-5 minutos",
        "args": [],
        "has_conditions": False,
        "has_max_subjects": False
    },
    {
        "name": "clustering.py",
        "display_name": "7️⃣ Clustering",
        "description": "Identificación de estados cerebrales con K-Medoids",
        "path": PIPELINE_PATH / "clustering.py",
        "estimated_time": "2-4 horas",
        "args": [],
        "has_conditions": False,
        "has_max_subjects": False
    }
]


def run_script_with_output(cmd: list, output_queue: queue.Queue):
    """Run a script and put output lines in a queue."""
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(PIPELINE_PATH)
        )
        
        for line in iter(process.stdout.readline, ''):
            output_queue.put(line)
        
        process.wait()
        output_queue.put(f"\n{'='*50}\n")
        output_queue.put(f"Proceso terminado con código: {process.returncode}\n")
        output_queue.put(None)  # Signal completion
        
    except Exception as e:
        output_queue.put(f"Error: {str(e)}\n")
        output_queue.put(None)


# Initialize session state
if 'running_process' not in st.session_state:
    st.session_state.running_process = None
if 'output_queue' not in st.session_state:
    st.session_state.output_queue = None
if 'output_lines' not in st.session_state:
    st.session_state.output_lines = []

# Sidebar - Script selection
with st.sidebar:
    st.markdown("### 📋 Scripts Disponibles")
    
    for script in PIPELINE_SCRIPTS:
        exists = script['path'].exists()
        status = "✅" if exists else "❌"
        st.markdown(f"{status} **{script['name']}**")
        if not exists:
            st.caption(f"No encontrado: {script['path']}")

# Main content
st.markdown("### ⚙️ Configuración")

# Script selector
selected_script = st.selectbox(
    "Selecciona el script a ejecutar",
    options=range(len(PIPELINE_SCRIPTS)),
    format_func=lambda x: PIPELINE_SCRIPTS[x]['display_name']
)

script_info = PIPELINE_SCRIPTS[selected_script]

# Script info card
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f"""
    <div style='background: #f0f2f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
        <h4>{script_info['display_name']}</h4>
        <p>{script_info['description']}</p>
        <small>⏱️ Tiempo estimado: {script_info['estimated_time']}</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if script_info['path'].exists():
        st.success("✅ Script encontrado")
    else:
        st.error("❌ Script no encontrado")

# Configuration options
st.markdown("### 🎛️ Opciones")

col1, col2, col3 = st.columns(3)

with col1:
    if script_info.get('has_conditions'):
        selected_conditions = st.multiselect(
            "Condiciones",
            CONDITIONS,
            default=CONDITIONS,
            help="Selecciona las condiciones a procesar"
        )
    else:
        selected_conditions = None

with col2:
    if script_info.get('has_max_subjects'):
        max_subjects = st.number_input(
            "Máx. sujetos (0=todos)",
            min_value=0,
            max_value=50,
            value=0,
            help="Limitar número de sujetos (para testing)"
        )
    else:
        max_subjects = 0

with col3:
    workers = st.slider(
        "Workers",
        min_value=1,
        max_value=24,
        value=7,
        help="Número de procesos paralelos"
    )

# Build command
cmd = ["python", str(script_info['path'])]

# Add default args
for arg in script_info['args']:
    if arg == "7" and 'workers' in str(script_info['args']):
        cmd.append(str(workers))
    elif arg == "20" and 'workers' in str(script_info['args']):
        cmd.append(str(workers))
    else:
        cmd.append(arg)

# Add conditions
if selected_conditions:
    cmd.extend(["--conditions"] + selected_conditions)

# Add max subjects
if max_subjects > 0:
    cmd.extend(["--max-subjects", str(max_subjects)])

# Show command
st.markdown("### 📝 Comando a ejecutar")
st.code(" ".join(cmd), language="bash")

# Run button
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    run_button = st.button(
        "▶️ Ejecutar",
        use_container_width=True,
        type="primary",
        disabled=not script_info['path'].exists()
    )

with col2:
    if st.button("🗑️ Limpiar Output", use_container_width=True):
        st.session_state.output_lines = []
        st.rerun()

# Output area
st.markdown("### 📺 Output")

output_container = st.container()

if run_button:
    st.session_state.output_lines = []
    st.session_state.output_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Iniciando: {script_info['name']}\n")
    st.session_state.output_lines.append(f"[{datetime.now().strftime('%H:%M:%S')}] Comando: {' '.join(cmd)}\n")
    st.session_state.output_lines.append("=" * 60 + "\n")
    
    # Create queue for output
    output_queue = queue.Queue()
    
    # Start process in thread
    thread = threading.Thread(target=run_script_with_output, args=(cmd, output_queue))
    thread.start()
    
    # Display output in real-time
    output_placeholder = output_container.empty()
    
    while True:
        try:
            line = output_queue.get(timeout=0.1)
            if line is None:
                break
            st.session_state.output_lines.append(line)
            
            # Update display
            output_text = "".join(st.session_state.output_lines[-100:])  # Last 100 lines
            output_placeholder.code(output_text, language="bash")
            
        except queue.Empty:
            continue
    
    thread.join()
    st.success("✅ Proceso completado")
    st.balloons()

# Show existing output
if st.session_state.output_lines:
    output_text = "".join(st.session_state.output_lines[-100:])
    with output_container:
        st.code(output_text, language="bash")
else:
    with output_container:
        st.info("👆 Presiona 'Ejecutar' para iniciar el script seleccionado")

# Footer with tips
st.markdown("---")
st.markdown("""
### 💡 Tips

- **Testing rápido**: Usa `Máx. sujetos = 3` para probar con pocos datos
- **Memoria**: Reduce `Workers` si hay errores de memoria
- **fwd.py**: Es el más largo (~3-4h), ejecuta primero
- **Orden recomendado**: fwd.py → multi2pool2.py → generate_order.py → build_order_data.py → plot_order.py
""")
