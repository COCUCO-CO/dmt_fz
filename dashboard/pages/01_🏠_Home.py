#!/usr/bin/env python3
"""
Home Page
=========

Main dashboard overview with quick stats and navigation.
"""

import streamlit as st
import yaml
from pathlib import Path
from datetime import datetime
import json

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title="Home | DMT-EEG",
    page_icon="🏠",
    layout="wide"
)

# Paths
paths = config['paths']


def get_quick_stats():
    """Get quick statistics for the dashboard."""
    stats = {}
    
    # Subjects
    eeg_path = Path(paths['eeg_clean'])
    if (eeg_path / "DMT").exists():
        stats['subjects'] = len(list((eeg_path / "DMT").glob("*.set")))
    else:
        stats['subjects'] = 0
    
    # Phases files
    fwd_path = Path(paths['fwd_inv_stc'])
    if fwd_path.exists():
        stats['phases_files'] = len(list(fwd_path.rglob("phases-*.pkl")))
    else:
        stats['phases_files'] = 0
    
    # ML experiments
    ml_path = Path(paths['ml_clf'])
    if ml_path.exists():
        stats['experiments'] = len(list(ml_path.glob("hyperparam_search/search_*")))
        stats['checkpoints'] = len(list(ml_path.rglob("*.pt")))
    else:
        stats['experiments'] = 0
        stats['checkpoints'] = 0
    
    # Best accuracy (try to find from experiments)
    stats['best_accuracy'] = None
    for search_dir in ml_path.glob("hyperparam_search/search_*"):
        json_file = search_dir / "all_experiments.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for exp in data:
                    if 'results' in exp and exp['results']:
                        acc = exp['results'].get('best_val_accuracy', 0)
                        if stats['best_accuracy'] is None or acc > stats['best_accuracy']:
                            stats['best_accuracy'] = acc
            except:
                pass
    
    return stats


# Title
st.title("🧠 DMT-EEG Analysis Dashboard")
st.markdown("**Pipeline integrado para análisis de sincronización EEG y clasificación de estados cerebrales**")

# Quick stats
stats = get_quick_stats()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📁 Sujetos", stats['subjects'])

with col2:
    st.metric("📊 Archivos Procesados", stats['phases_files'])

with col3:
    st.metric("🧪 Experimentos ML", stats['experiments'])

with col4:
    if stats['best_accuracy']:
        st.metric("🎯 Mejor Accuracy", f"{stats['best_accuracy']*100:.1f}%")
    else:
        st.metric("🎯 Mejor Accuracy", "N/A")

st.markdown("---")

# Two columns layout
left_col, right_col = st.columns([1.5, 1])

with left_col:
    st.markdown("### 📋 Estado del Pipeline")
    
    fwd_path = Path(paths['fwd_inv_stc'])
    
    # Check each step
    steps = [
        ("fwd.py", "Forward/Inverse + Métricas", "phases-*.pkl", "DMT"),
        ("multi2pool2.py", "Filtrado por redes", "order_all-*.pkl", "DMT"),
        ("generate_order.py", "Order Parameter", "order-*.pkl", "DMT"),
        ("build_order_data.py", "Datos agregados", "r_kuramoto_*.pkl", ""),
        ("plot_order.py", "Visualización", "", "plot_order_results/"),
        ("pearson.py", "Correlaciones", "", ""),  # Check pearson_results
        ("clustering.py", "Estados cerebrales", "clusters_*.pkl", ""),
    ]
    
    for script, desc, pattern, subdir in steps:
        if pattern:
            search_path = fwd_path / subdir if subdir else fwd_path
            files = list(search_path.glob(pattern)) if search_path.exists() else []
            n_files = len(files)
            
            if n_files > 0:
                status = "✅"
                status_text = f"{n_files} archivos"
            else:
                status = "⏳"
                status_text = "Pendiente"
        else:
            status = "⏳"
            status_text = "Pendiente"
        
        col_icon, col_name, col_status = st.columns([0.5, 3, 1.5])
        with col_icon:
            st.markdown(f"### {status}")
        with col_name:
            st.markdown(f"**{script}**")
            st.caption(desc)
        with col_status:
            if status == "✅":
                st.success(status_text)
            else:
                st.info(status_text)

with right_col:
    st.markdown("### 🔬 Últimos Experimentos")
    
    ml_path = Path(paths['ml_clf'])
    search_dirs = sorted(ml_path.glob("hyperparam_search/search_*"), reverse=True)[:5]
    
    if search_dirs:
        for exp_dir in search_dirs:
            # Try to get best accuracy
            json_file = exp_dir / "all_experiments.json"
            best_acc = None
            n_exps = 0
            
            if json_file.exists():
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                    n_exps = len(data)
                    for exp in data:
                        if 'results' in exp and exp['results']:
                            acc = exp['results'].get('best_val_accuracy', 0)
                            if best_acc is None or acc > best_acc:
                                best_acc = acc
                except:
                    pass
            
            with st.container():
                exp_date = exp_dir.name.replace("search_", "")
                if best_acc:
                    st.success(f"📁 {exp_date} | {n_exps} exp | 🎯 {best_acc*100:.1f}%")
                else:
                    st.info(f"📁 {exp_date} | {n_exps} experimentos")
    else:
        st.info("No hay experimentos. Ve a 'ML Training' para comenzar.")

st.markdown("---")

# Quick actions
st.markdown("### ⚡ Acciones Rápidas")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.page_link("pages/02_📊_Data_Explorer.py", label="📊 Explorar Datos", use_container_width=True)

with col2:
    st.page_link("pages/03_🔬_Pipeline_Runner.py", label="🔬 Pipeline", use_container_width=True)

with col3:
    st.page_link("pages/04_🤖_ML_Training.py", label="🤖 Entrenar", use_container_width=True)

with col4:
    st.page_link("pages/05_📈_Experiments.py", label="📈 Experimentos", use_container_width=True)

with col5:
    st.page_link("pages/06_🎨_Visualizations.py", label="🎨 Visualizar", use_container_width=True)

with col6:
    st.page_link("pages/07_🎬_Animations.py", label="🎬 Animaciones", use_container_width=True)

st.markdown("---")

# Info cards
st.markdown("### 📚 Documentación")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### 🔬 Pipeline EEG
    
    Procesamiento de señales EEG:
    - Forward/Inverse solution
    - Transformada de Hilbert
    - Sincronización de fases
    - Parámetro de Kuramoto
    
    [Ver documentación](docs/)
    """)

with col2:
    st.markdown("""
    #### 🤖 Machine Learning
    
    Clasificación con Graph Attention Networks:
    - GAT con múltiples capas
    - Attention weights interpretables
    - Hyperparameter search
    
    [Ver docs ML](machine_learning/clf/docs/)
    """)

with col3:
    st.markdown("""
    #### 📊 Análisis
    
    Herramientas de análisis:
    - Correlaciones (Pearson)
    - Clustering de estados
    - Visualización temporal
    
    [Ver resultados](visualizations/)
    """)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 0.8rem;'>
    DMT-EEG Analysis Dashboard v1.0 | 
    Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)
