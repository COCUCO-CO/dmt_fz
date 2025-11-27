#!/usr/bin/env python3
"""
ML Training Page
================

Train GAT models and run hyperparameter search.
"""

import streamlit as st
import yaml
import subprocess
import os
from pathlib import Path
from datetime import datetime
import json

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title="ML Training | DMT-EEG",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Model Training")
st.markdown("Entrena modelos GAT para clasificación de estados cerebrales")

# Paths
ML_CLF_PATH = Path(config['paths']['ml_clf'])
CONDITIONS = config['data']['conditions']
BANDS = config['data']['bands']


def get_recent_experiments():
    """Get recent hyperparameter search experiments."""
    search_dirs = sorted(ML_CLF_PATH.glob("hyperparam_search/search_*"), reverse=True)
    experiments = []
    
    for search_dir in search_dirs[:10]:
        # Try to load summary
        summary_file = search_dir / "all_experiments.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r') as f:
                    data = json.load(f)
                experiments.append({
                    "name": search_dir.name,
                    "path": search_dir,
                    "n_experiments": len(data),
                    "data": data
                })
            except:
                experiments.append({
                    "name": search_dir.name,
                    "path": search_dir,
                    "n_experiments": len(list(search_dir.glob("exp*"))),
                    "data": None
                })
        else:
            experiments.append({
                "name": search_dir.name,
                "path": search_dir,
                "n_experiments": len(list(search_dir.glob("exp*"))),
                "data": None
            })
    
    return experiments


def get_checkpoints():
    """Get available model checkpoints."""
    checkpoints = []
    for pt_file in ML_CLF_PATH.rglob("*.pt"):
        checkpoints.append({
            "name": pt_file.name,
            "path": pt_file,
            "size_mb": pt_file.stat().st_size / (1024 * 1024),
            "modified": datetime.fromtimestamp(pt_file.stat().st_mtime)
        })
    return sorted(checkpoints, key=lambda x: x['modified'], reverse=True)


# Tabs
tab1, tab2, tab3 = st.tabs([
    "🧪 Quick Train",
    "🔍 Hyperparameter Search",
    "📊 Experiments"
])

with tab1:
    st.markdown("### 🧪 Entrenamiento Rápido")
    st.markdown("Entrena un modelo con la configuración actual")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚙️ Configuración")
        
        selected_bands = st.multiselect(
            "Bandas de Frecuencia",
            BANDS,
            default=["Alpha"],
            help="Bandas a incluir en el entrenamiento"
        )
        
        selected_conditions = st.multiselect(
            "Condiciones",
            CONDITIONS,
            default=["DMT", "EC"],
            help="Condiciones a clasificar"
        )
        
        epochs = st.slider(
            "Epochs",
            min_value=10,
            max_value=1000,
            value=config['ml']['default_epochs'],
            step=10
        )
        
        batch_size = st.selectbox(
            "Batch Size",
            [16, 32, 64, 128],
            index=1
        )
    
    with col2:
        st.markdown("#### 📋 Comando")
        
        cmd = [
            "python", "train.py",
            "--config", "config/config.yaml"
        ]
        
        st.code(" ".join(cmd), language="bash")
        
        st.markdown("#### 📁 Output")
        st.info(f"""
        - Checkpoints: `{ML_CLF_PATH}/checkpoints/`
        - TensorBoard: `{ML_CLF_PATH}/runs/`
        - Resultados: `{ML_CLF_PATH}/output/`
        """)
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("▶️ Iniciar Entrenamiento", type="primary", use_container_width=True):
            st.info("Iniciando entrenamiento...")
            st.warning("⚠️ Para ver el progreso en tiempo real, ejecuta desde terminal")
            st.code(f"cd {ML_CLF_PATH} && python train.py --config config/config.yaml", language="bash")
    
    with col2:
        if st.button("📊 Abrir TensorBoard", use_container_width=True):
            tensorboard_cmd = f"tensorboard --logdir={ML_CLF_PATH}/runs --port=6006"
            st.code(tensorboard_cmd, language="bash")
            st.markdown("[Abrir TensorBoard](http://localhost:6006)")

with tab2:
    st.markdown("### 🔍 Búsqueda de Hiperparámetros")
    st.markdown("Random search para encontrar la mejor configuración")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### ⚙️ Configuración")
        
        n_experiments = st.slider(
            "Número de Experimentos",
            min_value=5,
            max_value=100,
            value=config['ml']['hyperparam_experiments'],
            step=5
        )
        
        hp_bands = st.multiselect(
            "Bandas",
            BANDS,
            default=["Alpha"],
            key="hp_bands"
        )
        
        hp_conditions = st.multiselect(
            "Condiciones",
            CONDITIONS,
            default=["DMT", "EC"],
            key="hp_conditions"
        )
        
        hp_epochs = st.slider(
            "Max Epochs por Experimento",
            min_value=50,
            max_value=500,
            value=300,
            step=50
        )
    
    with col2:
        st.markdown("#### 📋 Comando")
        
        hp_cmd = [
            "python", "hyperparam_search.py",
            "--n_experiments", str(n_experiments),
            "--bands"] + hp_bands + [
            "--conditions"] + hp_conditions + [
            "--max_epochs", str(hp_epochs)
        ]
        
        st.code(" ".join(hp_cmd), language="bash")
        
        st.markdown("#### ⏱️ Tiempo Estimado")
        
        # Rough estimate: 20-30 min per experiment
        min_time = n_experiments * 15 / 60
        max_time = n_experiments * 40 / 60
        st.info(f"Aproximadamente **{min_time:.1f} - {max_time:.1f} horas**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🚀 Iniciar Búsqueda", type="primary", use_container_width=True):
            st.warning("⚠️ La búsqueda puede tomar varias horas")
            st.code(f"cd {ML_CLF_PATH} && {' '.join(hp_cmd)}", language="bash")
    
    with col2:
        if st.button("📊 Ver TensorBoard", use_container_width=True):
            st.code(f"tensorboard --logdir={ML_CLF_PATH}/hyperparam_search --port=6007", language="bash")

with tab3:
    st.markdown("### 📊 Experimentos Anteriores")
    
    experiments = get_recent_experiments()
    
    if experiments:
        for exp in experiments:
            with st.expander(f"📁 {exp['name']} ({exp['n_experiments']} experimentos)"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**Path:** `{exp['path']}`")
                    
                    if exp['data']:
                        # Show best experiment
                        best = None
                        best_acc = 0
                        for e in exp['data']:
                            if 'results' in e and e['results']:
                                acc = e['results'].get('best_val_accuracy', 0)
                                if acc > best_acc:
                                    best_acc = acc
                                    best = e
                        
                        if best:
                            st.success(f"🏆 Mejor accuracy: **{best_acc*100:.2f}%**")
                            st.json(best.get('sampled_params', {}))
                
                with col2:
                    if st.button("📊 TensorBoard", key=f"tb_{exp['name']}"):
                        st.code(f"tensorboard --logdir={exp['path']}/tensorboard")
    else:
        st.info("No hay experimentos anteriores. Ejecuta una búsqueda de hiperparámetros para comenzar.")
    
    st.markdown("---")
    
    st.markdown("### 💾 Checkpoints Disponibles")
    
    checkpoints = get_checkpoints()
    
    if checkpoints:
        for ckpt in checkpoints[:10]:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{ckpt['name']}**")
                st.caption(f"{ckpt['path']}")
            with col2:
                st.markdown(f"{ckpt['size_mb']:.1f} MB")
            with col3:
                st.markdown(ckpt['modified'].strftime("%Y-%m-%d %H:%M"))
    else:
        st.info("No hay checkpoints disponibles.")

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Estado")
    
    # Check if training is running
    st.info("💡 Para monitorear entrenamientos en curso, usa TensorBoard")
    
    st.markdown("---")
    
    st.markdown("### 🔗 Links Rápidos")
    st.markdown("""
    - [TensorBoard runs](http://localhost:6006)
    - [TensorBoard hyperparam](http://localhost:6007)
    """)
    
    st.markdown("---")
    
    st.markdown("### 📚 Documentación")
    st.markdown(f"[Ver docs]({ML_CLF_PATH}/docs/)")
