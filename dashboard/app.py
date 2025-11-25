#!/usr/bin/env python3
"""
DMT-EEG Analysis Dashboard
==========================

Main entry point for the Streamlit dashboard.

Run with:
    streamlit run app.py --server.port 8501
"""

import streamlit as st
import yaml
from pathlib import Path

# Configuration
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config():
    """Load dashboard configuration."""
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def main():
    """Main dashboard application."""
    config = load_config()
    
    # Page configuration
    st.set_page_config(
        page_title=config['app']['title'],
        page_icon=config['app']['icon'],
        layout=config['app']['layout'],
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 1rem 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Status badges */
    .status-running {
        background-color: #ffc107;
        color: black;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    
    .status-completed {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    
    .status-pending {
        background-color: #6c757d;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
    }
    
    /* Section headers */
    .section-header {
        border-left: 4px solid #667eea;
        padding-left: 1rem;
        margin: 1.5rem 0 1rem 0;
    }
    
    /* Cards */
    .info-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Terminal output */
    .terminal {
        background-color: #1e1e1e;
        color: #00ff00;
        font-family: 'Courier New', monospace;
        padding: 1rem;
        border-radius: 8px;
        max-height: 400px;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://raw.githubusercontent.com/streamlit/streamlit/develop/docs/logo.svg", width=50)
        st.title("🧠 DMT-EEG")
        st.markdown("---")
        
        # Navigation info
        st.markdown("""
        ### 📍 Navegación
        Usa el menú lateral para navegar entre las diferentes secciones del dashboard.
        """)
        
        st.markdown("---")
        
        # Quick status
        st.markdown("### ⚡ Estado Rápido")
        
        # Check if paths exist
        paths = config['paths']
        
        if Path(paths['fwd_inv_stc']).exists():
            phases_files = list(Path(paths['fwd_inv_stc']).rglob("phases-*.pkl"))
            st.success(f"✅ {len(phases_files)} archivos phases-*.pkl")
        else:
            st.warning("⚠️ Directorio fwd-inv-stc no encontrado")
        
        if Path(paths['ml_clf']).exists():
            checkpoints = list(Path(paths['ml_clf']).rglob("*.pt"))
            st.success(f"✅ {len(checkpoints)} checkpoints de modelos")
        else:
            st.warning("⚠️ Directorio ML no encontrado")
        
        st.markdown("---")
        st.markdown("v1.0.0 | [Docs](docs/)")
    
    # Main content - Home page
    st.title("🧠 DMT-EEG Analysis Dashboard")
    st.markdown("**Pipeline integrado para análisis de sincronización EEG y clasificación de estados cerebrales**")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Count subjects
        eeg_path = Path(paths['eeg_clean'])
        if eeg_path.exists():
            subjects = len(list((eeg_path / "DMT").glob("*.set"))) if (eeg_path / "DMT").exists() else 0
        else:
            subjects = 0
        st.metric("📁 Sujetos", subjects)
    
    with col2:
        # Count experiments
        ml_path = Path(paths['ml_clf'])
        experiments = len(list(ml_path.rglob("hyperparam_search/search_*"))) if ml_path.exists() else 0
        st.metric("🧪 Experimentos", experiments)
    
    with col3:
        # Best accuracy (placeholder)
        st.metric("🎯 Mejor Accuracy", "78.5%", "+2.3%")
    
    with col4:
        # Phases files
        fwd_path = Path(paths['fwd_inv_stc'])
        phases_count = len(list(fwd_path.rglob("phases-*.pkl"))) if fwd_path.exists() else 0
        st.metric("📊 Datos Procesados", f"{phases_count} archivos")
    
    st.markdown("---")
    
    # Two columns layout
    left_col, right_col = st.columns([1.5, 1])
    
    with left_col:
        st.markdown("### 📊 Estado del Pipeline")
        
        pipeline_status = [
            ("fwd.py", "Forward/Inverse + Métricas", "completed", "✅"),
            ("multi2pool2.py", "Filtrado por redes", "completed", "✅"),
            ("generate_order.py", "Order Parameter", "completed", "✅"),
            ("plot_order.py", "Visualización", "completed", "✅"),
            ("pearson.py", "Correlaciones", "pending", "⏳"),
            ("clustering.py", "Estados cerebrales", "pending", "⏳"),
        ]
        
        for script, desc, status, icon in pipeline_status:
            col_icon, col_name, col_status = st.columns([0.5, 3, 1])
            with col_icon:
                st.markdown(f"### {icon}")
            with col_name:
                st.markdown(f"**{script}**")
                st.caption(desc)
            with col_status:
                if status == "completed":
                    st.success("Completado")
                elif status == "running":
                    st.warning("Ejecutando...")
                else:
                    st.info("Pendiente")
    
    with right_col:
        st.markdown("### 🔬 Últimos Experimentos")
        
        # Check for recent experiments
        ml_path = Path(paths['ml_clf'])
        hyperparam_dirs = sorted(ml_path.glob("hyperparam_search/search_*"), reverse=True)[:5] if ml_path.exists() else []
        
        if hyperparam_dirs:
            for exp_dir in hyperparam_dirs:
                exp_name = exp_dir.name
                with st.container():
                    st.markdown(f"""
                    <div class="info-card">
                        <strong>📁 {exp_name}</strong><br>
                        <small>Hyperparameter search</small>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No hay experimentos recientes. Usa la página 'ML Training' para empezar.")
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("### ⚡ Acciones Rápidas")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Explorar Datos", use_container_width=True):
            st.switch_page("pages/02_📊_Data_Explorer.py")
    
    with col2:
        if st.button("🔬 Ejecutar Pipeline", use_container_width=True):
            st.switch_page("pages/03_🔬_Pipeline_Runner.py")
    
    with col3:
        if st.button("🤖 Entrenar Modelo", use_container_width=True):
            st.switch_page("pages/04_🤖_ML_Training.py")
    
    with col4:
        if st.button("🎨 Ver Visualizaciones", use_container_width=True):
            st.switch_page("pages/06_🎨_Visualizations.py")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        DMT-EEG Analysis Dashboard | Pipeline de Sincronización de Fases | 
        <a href='https://github.com/your-repo'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
