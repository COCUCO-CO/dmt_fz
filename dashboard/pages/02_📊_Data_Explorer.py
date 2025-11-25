#!/usr/bin/env python3
"""
Data Explorer Page
==================

Explore EEG data by subject, condition, and frequency band.
"""

import streamlit as st
import yaml
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List, Optional

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title="Data Explorer | DMT-EEG",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Explorer")
st.markdown("Explora los datos EEG por sujeto, condición y banda de frecuencia")

# Paths
FWD_PATH = Path(config['paths']['fwd_inv_stc'])
CONDITIONS = config['data']['conditions']
BANDS = config['data']['bands']


@st.cache_data
def get_available_subjects() -> Dict[str, List[str]]:
    """Get available subjects for each condition."""
    subjects = {}
    for cond in CONDITIONS:
        cond_path = FWD_PATH / cond
        if cond_path.exists():
            files = list(cond_path.glob("phases-*.pkl"))
            subjects[cond] = sorted([f.stem.replace("phases-", "") for f in files])
        else:
            subjects[cond] = []
    return subjects


@st.cache_data
def load_phases_data(subject: str, condition: str) -> Optional[Dict]:
    """Load phases data for a subject."""
    file_path = FWD_PATH / condition / f"phases-{subject}.pkl"
    if file_path.exists():
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None


def plot_kuramoto_series(data: Dict, band: str, use_stc: bool = False) -> go.Figure:
    """Plot Kuramoto order parameter over epochs."""
    suffix = "_stc" if use_stc else "_eeg"
    key = f"kuramoto{suffix}"
    
    if key not in data or band not in data[key]:
        return None
    
    kuramoto_data = data[key][band]
    
    # Each element is an array of r values over time for one epoch
    fig = go.Figure()
    
    for i, epoch_data in enumerate(kuramoto_data):
        if hasattr(epoch_data, '__len__'):
            fig.add_trace(go.Scatter(
                y=epoch_data,
                mode='lines',
                name=f'Epoch {i+1}',
                opacity=0.5,
                line=dict(width=1)
            ))
    
    fig.update_layout(
        title=f"Kuramoto Order Parameter - {band} Band",
        xaxis_title="Time (samples)",
        yaxis_title="r (coherence)",
        yaxis=dict(range=[0, 1]),
        showlegend=False,
        height=400
    )
    
    return fig


def plot_sync_matrix(data: Dict, band: str, epoch_idx: int, use_stc: bool = False) -> go.Figure:
    """Plot synchronization matrix for a specific epoch."""
    suffix = "_stc" if use_stc else "_eeg"
    key = f"syncros{suffix}"
    
    if key not in data or band not in data[key]:
        return None
    
    sync_matrices = data[key][band]
    
    if epoch_idx >= len(sync_matrices):
        return None
    
    matrix = sync_matrices[epoch_idx]
    
    fig = px.imshow(
        matrix,
        color_continuous_scale='RdYlBu_r',
        aspect='equal',
        title=f"Synchronization Matrix - {band} Band - Epoch {epoch_idx + 1}"
    )
    
    fig.update_layout(
        xaxis_title="Channel",
        yaxis_title="Channel",
        height=500
    )
    
    return fig


def plot_phase_distribution(data: Dict, band: str, epoch_idx: int, use_stc: bool = False) -> go.Figure:
    """Plot phase distribution histogram."""
    suffix = "_stc" if use_stc else "_eeg"
    key = f"phases{suffix}"
    
    if key not in data or band not in data[key]:
        return None
    
    phases = data[key][band]
    
    if epoch_idx >= len(phases):
        return None
    
    phase_data = phases[epoch_idx].flatten()
    
    fig = px.histogram(
        x=phase_data,
        nbins=50,
        title=f"Phase Distribution - {band} Band - Epoch {epoch_idx + 1}"
    )
    
    fig.update_layout(
        xaxis_title="Phase (radians)",
        yaxis_title="Count",
        height=300
    )
    
    return fig


# Main content
subjects_dict = get_available_subjects()

# Sidebar filters
with st.sidebar:
    st.markdown("### 🎛️ Filtros")
    
    # Condition selector
    condition = st.selectbox(
        "Condición",
        CONDITIONS,
        help="Selecciona la condición experimental"
    )
    
    # Subject selector
    available_subjects = subjects_dict.get(condition, [])
    if available_subjects:
        subject = st.selectbox(
            "Sujeto",
            available_subjects,
            help="Selecciona el sujeto"
        )
    else:
        st.warning(f"No hay sujetos disponibles para {condition}")
        subject = None
    
    # Band selector
    band = st.selectbox(
        "Banda de Frecuencia",
        BANDS,
        index=2,  # Default to Alpha
        help="Selecciona la banda de frecuencia"
    )
    
    # Source space toggle
    use_stc = st.toggle(
        "Usar Source Space (STC)",
        value=False,
        help="Toggle entre EEG (24 canales) y Source Space (100 parcelas)"
    )
    
    st.markdown("---")
    
    # Data info
    if subject:
        data = load_phases_data(subject, condition)
        if data:
            suffix = "_stc" if use_stc else "_eeg"
            kuramoto_key = f"kuramoto{suffix}"
            if kuramoto_key in data and band in data[kuramoto_key]:
                n_epochs = len(data[kuramoto_key][band])
                st.info(f"📊 {n_epochs} epochs disponibles")

# Main content
if subject:
    data = load_phases_data(subject, condition)
    
    if data:
        st.markdown(f"### 📁 Sujeto: `{subject}` | Condición: `{condition}` | Banda: `{band}`")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Kuramoto",
            "🌐 Sincronización",
            "📊 Fases",
            "📋 Estadísticas"
        ])
        
        with tab1:
            st.markdown("#### Parámetro de Orden de Kuramoto")
            st.caption("Coherencia global de fases a lo largo del tiempo para cada época")
            
            fig = plot_kuramoto_series(data, band, use_stc)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos de Kuramoto disponibles para esta configuración")
        
        with tab2:
            st.markdown("#### Matriz de Sincronización")
            
            suffix = "_stc" if use_stc else "_eeg"
            key = f"syncros{suffix}"
            
            if key in data and band in data[key]:
                n_epochs = len(data[key][band])
                epoch_idx = st.slider("Época", 0, n_epochs - 1, 0)
                
                fig = plot_sync_matrix(data, band, epoch_idx, use_stc)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos de sincronización disponibles")
        
        with tab3:
            st.markdown("#### Distribución de Fases")
            
            suffix = "_stc" if use_stc else "_eeg"
            key = f"phases{suffix}"
            
            if key in data and band in data[key]:
                n_epochs = len(data[key][band])
                epoch_idx = st.slider("Época ", 0, n_epochs - 1, 0, key="phase_epoch")
                
                fig = plot_phase_distribution(data, band, epoch_idx, use_stc)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay datos de fases disponibles")
        
        with tab4:
            st.markdown("#### Estadísticas del Sujeto")
            
            suffix = "_stc" if use_stc else "_eeg"
            kuramoto_key = f"kuramoto{suffix}"
            
            if kuramoto_key in data and band in data[kuramoto_key]:
                kuramoto_data = data[kuramoto_key][band]
                
                # Calculate statistics
                all_r_values = []
                for epoch in kuramoto_data:
                    if hasattr(epoch, '__len__'):
                        all_r_values.extend(epoch)
                
                if all_r_values:
                    all_r = np.array(all_r_values)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Epochs", len(kuramoto_data))
                    with col2:
                        st.metric("Mean r", f"{np.mean(all_r):.4f}")
                    with col3:
                        st.metric("Std r", f"{np.std(all_r):.4f}")
                    with col4:
                        st.metric("Max r", f"{np.max(all_r):.4f}")
                    
                    # Distribution of mean r per epoch
                    mean_per_epoch = [np.mean(ep) for ep in kuramoto_data if hasattr(ep, '__len__')]
                    
                    fig = px.histogram(
                        x=mean_per_epoch,
                        nbins=20,
                        title="Distribución de Coherencia Media por Época"
                    )
                    fig.update_layout(
                        xaxis_title="Mean r",
                        yaxis_title="Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No hay estadísticas disponibles")
    else:
        st.error(f"No se pudo cargar el archivo para {subject}")
else:
    st.info("👆 Selecciona un sujeto en el menú lateral para comenzar a explorar los datos")
    
    # Show summary
    st.markdown("### 📋 Resumen de Datos Disponibles")
    
    summary_data = []
    for cond in CONDITIONS:
        n_subjects = len(subjects_dict.get(cond, []))
        summary_data.append({
            "Condición": cond,
            "Sujetos": n_subjects,
            "Estado": "✅ Disponible" if n_subjects > 0 else "❌ No disponible"
        })
    
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
