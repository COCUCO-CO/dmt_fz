#!/usr/bin/env python3
"""
Experiments Page
================

View and compare ML experiments.
"""

import streamlit as st
import yaml
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title="Experiments | DMT-EEG",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Experiments Tracker")
st.markdown("Visualiza y compara experimentos de ML")

# Paths
ML_CLF_PATH = Path(config['paths']['ml_clf'])


def load_experiments():
    """Load all experiment data."""
    experiments = []
    
    # Load from hyperparam_search directories
    for search_dir in ML_CLF_PATH.glob("hyperparam_search/search_*"):
        json_file = search_dir / "all_experiments.json"
        if json_file.exists():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                for exp in data:
                    exp['search_name'] = search_dir.name
                    exp['search_path'] = str(search_dir)
                    experiments.append(exp)
            except Exception as e:
                st.warning(f"Error loading {json_file}: {e}")
    
    return experiments


def experiments_to_dataframe(experiments):
    """Convert experiments to DataFrame for analysis."""
    rows = []
    
    for exp in experiments:
        row = {
            'search': exp.get('search_name', 'unknown'),
            'name': exp.get('name', 'unknown'),
            'status': exp.get('status', 'unknown'),
        }
        
        # Extract sampled parameters
        params = exp.get('sampled_params', {})
        for key, value in params.items():
            row[f'param_{key}'] = value
        
        # Extract results
        results = exp.get('results', {})
        if results:
            row['best_val_accuracy'] = results.get('best_val_accuracy', 0)
            row['best_val_loss'] = results.get('best_val_loss', 0)
            row['best_epoch'] = results.get('best_epoch', 0)
            row['training_time'] = results.get('training_time', 0)
        
        rows.append(row)
    
    return pd.DataFrame(rows)


# Load data
experiments = load_experiments()

if experiments:
    df = experiments_to_dataframe(experiments)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Experimentos", len(experiments))
    
    with col2:
        completed = len([e for e in experiments if e.get('status') == 'completed'])
        st.metric("Completados", completed)
    
    with col3:
        if 'best_val_accuracy' in df.columns:
            best_acc = df['best_val_accuracy'].max()
            st.metric("Mejor Accuracy", f"{best_acc*100:.2f}%")
    
    with col4:
        n_searches = df['search'].nunique()
        st.metric("Búsquedas", n_searches)
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "🔍 Comparación",
        "📋 Tabla Completa"
    ])
    
    with tab1:
        st.markdown("### 📊 Distribución de Accuracy")
        
        if 'best_val_accuracy' in df.columns:
            fig = px.histogram(
                df,
                x='best_val_accuracy',
                nbins=20,
                title="Distribución de Accuracy de Validación",
                labels={'best_val_accuracy': 'Validation Accuracy'}
            )
            fig.add_vline(x=df['best_val_accuracy'].mean(), line_dash="dash", 
                         annotation_text=f"Mean: {df['best_val_accuracy'].mean():.3f}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Top experiments
        st.markdown("### 🏆 Top 10 Experimentos")
        
        if 'best_val_accuracy' in df.columns:
            top_10 = df.nlargest(10, 'best_val_accuracy')[
                ['search', 'name', 'best_val_accuracy', 'best_epoch']
            ].reset_index(drop=True)
            top_10.index = top_10.index + 1
            st.dataframe(top_10, use_container_width=True)
    
    with tab2:
        st.markdown("### 🔍 Comparación de Hiperparámetros")
        
        # Get parameter columns
        param_cols = [col for col in df.columns if col.startswith('param_')]
        
        if param_cols and 'best_val_accuracy' in df.columns:
            selected_param = st.selectbox(
                "Selecciona parámetro para comparar",
                param_cols,
                format_func=lambda x: x.replace('param_', '')
            )
            
            if selected_param:
                fig = px.box(
                    df,
                    x=selected_param,
                    y='best_val_accuracy',
                    title=f"Accuracy vs {selected_param.replace('param_', '')}",
                    points="all"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot
        st.markdown("### 📈 Scatter Plot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_param = st.selectbox("Eje X", param_cols, key="scatter_x")
        with col2:
            y_param = st.selectbox("Eje Y", ['best_val_accuracy', 'best_epoch'] + param_cols, key="scatter_y")
        
        if x_param and y_param:
            fig = px.scatter(
                df,
                x=x_param,
                y=y_param,
                color='search',
                hover_data=['name'],
                title=f"{x_param} vs {y_param}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 📋 Todos los Experimentos")
        
        # Column filter
        available_cols = df.columns.tolist()
        default_cols = ['search', 'name', 'status', 'best_val_accuracy', 'best_epoch']
        selected_cols = st.multiselect(
            "Columnas a mostrar",
            available_cols,
            default=[c for c in default_cols if c in available_cols]
        )
        
        if selected_cols:
            st.dataframe(df[selected_cols], use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            "📥 Descargar CSV",
            csv,
            "experiments.csv",
            "text/csv"
        )

else:
    st.info("No hay experimentos registrados. Ejecuta una búsqueda de hiperparámetros en la página 'ML Training'.")
    
    st.markdown("""
    ### 🚀 Cómo empezar
    
    1. Ve a la página **ML Training**
    2. Configura y ejecuta una búsqueda de hiperparámetros
    3. Los resultados aparecerán aquí automáticamente
    """)

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Estadísticas")
    
    if experiments:
        st.markdown(f"""
        - **Total experimentos:** {len(experiments)}
        - **Búsquedas:** {df['search'].nunique() if 'search' in df.columns else 0}
        """)
    
    st.markdown("---")
    
    st.markdown("### 🔄 Actualizar")
    if st.button("🔄 Recargar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
