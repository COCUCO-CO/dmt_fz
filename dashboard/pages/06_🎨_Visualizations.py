#!/usr/bin/env python3
"""
Visualizations Page
===================

View attention heatmaps, MST graphs, and other visualizations.
"""

import streamlit as st
import yaml
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title="Visualizations | DMT-EEG",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 Visualizaciones")
st.markdown("Explora visualizaciones de atención, MST y análisis de grafos")

# Paths
ML_CLF_PATH = Path(config['paths']['ml_clf'])
VIZ_PATH = Path(config['paths']['visualizations'])
OUTPUT_PATH = ML_CLF_PATH / "output"


def get_attention_images():
    """Get available attention visualization images."""
    images = []
    
    # Check output/attention folder
    attention_dir = OUTPUT_PATH / "attention"
    if attention_dir.exists():
        for epoch_dir in sorted(attention_dir.glob("epoch_*")):
            for img in epoch_dir.glob("*.png"):
                images.append({
                    "name": img.name,
                    "path": img,
                    "epoch": epoch_dir.name,
                    "type": "attention"
                })
    
    # Check output/analysis folder
    analysis_dir = OUTPUT_PATH / "analysis"
    if analysis_dir.exists():
        for img in analysis_dir.glob("*.png"):
            images.append({
                "name": img.name,
                "path": img,
                "epoch": "analysis",
                "type": "analysis"
            })
    
    return images


def get_visualization_images():
    """Get images from visualizations folder."""
    images = []
    
    if VIZ_PATH.exists():
        for img in VIZ_PATH.glob("*.png"):
            images.append({
                "name": img.name,
                "path": img,
                "type": "visualization"
            })
    
    return images[:50]  # Limit to 50


# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔥 Attention Heatmaps",
    "🌳 MST Graphs",
    "📊 Analysis",
    "🖼️ Gallery"
])

with tab1:
    st.markdown("### 🔥 Attention Heatmaps")
    st.markdown("Visualiza las matrices de atención por capa y condición")
    
    attention_images = [img for img in get_attention_images() if img['type'] == 'attention']
    
    if attention_images:
        # Group by epoch
        epochs = sorted(set(img['epoch'] for img in attention_images))
        
        selected_epoch = st.selectbox("Época", epochs)
        
        epoch_images = [img for img in attention_images if img['epoch'] == selected_epoch]
        
        # Display images in grid
        cols = st.columns(3)
        for i, img in enumerate(epoch_images):
            with cols[i % 3]:
                try:
                    image = Image.open(img['path'])
                    st.image(image, caption=img['name'], use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading {img['name']}: {e}")
    else:
        st.info("No hay imágenes de atención disponibles. Entrena un modelo primero.")
        
        # Show placeholder
        st.markdown("""
        #### Cómo generar visualizaciones de atención:
        
        1. Entrena un modelo en la página **ML Training**
        2. Las visualizaciones se generan automáticamente durante el entrenamiento
        3. También puedes generarlas manualmente:
        
        ```python
        from utils.attention_logger import log_attention_to_tensorboard
        log_attention_to_tensorboard(writer, model, data_loader, device, epoch)
        ```
        """)

with tab2:
    st.markdown("### 🌳 MST Graphs")
    st.markdown("Minimum Spanning Trees de las matrices de atención")
    
    # Look for MST images
    mst_images = []
    
    analysis_dir = OUTPUT_PATH / "analysis"
    if analysis_dir.exists():
        mst_images = list(analysis_dir.glob("mst_*.png"))
    
    if mst_images:
        cols = st.columns(2)
        for i, img_path in enumerate(mst_images[:6]):
            with cols[i % 2]:
                try:
                    image = Image.open(img_path)
                    st.image(image, caption=img_path.name, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("No hay gráficos MST disponibles.")
        
        st.markdown("""
        #### Cómo generar MST graphs:
        
        Los MST se generan automáticamente durante el entrenamiento.
        También puedes usar el script de animaciones:
        
        ```bash
        python generate_attention_animation.py --checkpoint best_model.pt --subject sub-01 --mst
        ```
        """)

with tab3:
    st.markdown("### 📊 Análisis")
    st.markdown("Gráficos de análisis y estadísticas")
    
    analysis_images = [img for img in get_attention_images() if img['type'] == 'analysis']
    
    if analysis_images:
        # Filter by type
        image_types = sorted(set(img['name'].split('_')[0] for img in analysis_images))
        
        if image_types:
            selected_type = st.selectbox("Tipo de análisis", image_types)
            
            filtered = [img for img in analysis_images if img['name'].startswith(selected_type)]
            
            cols = st.columns(2)
            for i, img in enumerate(filtered[:6]):
                with cols[i % 2]:
                    try:
                        image = Image.open(img['path'])
                        st.image(image, caption=img['name'], use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
    else:
        st.info("No hay análisis disponibles.")

with tab4:
    st.markdown("### 🖼️ Galería de Visualizaciones")
    st.markdown("Todas las visualizaciones generadas por el pipeline")
    
    viz_images = get_visualization_images()
    
    if viz_images:
        # Search/filter
        search = st.text_input("🔍 Buscar", placeholder="Escribe para filtrar...")
        
        if search:
            viz_images = [img for img in viz_images if search.lower() in img['name'].lower()]
        
        st.markdown(f"Mostrando {len(viz_images)} imágenes")
        
        cols = st.columns(3)
        for i, img in enumerate(viz_images):
            with cols[i % 3]:
                try:
                    image = Image.open(img['path'])
                    st.image(image, caption=img['name'], use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("No hay visualizaciones en la galería.")

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Directorios")
    
    st.markdown(f"""
    - **Output ML:** `{OUTPUT_PATH}`
    - **Visualizations:** `{VIZ_PATH}`
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔄 Actualizar")
    if st.button("🔄 Recargar imágenes", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📥 Exportar")
    st.info("Haz clic derecho en cualquier imagen para guardarla")
