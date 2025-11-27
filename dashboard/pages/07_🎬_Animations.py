#!/usr/bin/env python3
"""
Animations Page
===============

Generate attention animations for specific subjects.
"""

import streamlit as st
import yaml
import subprocess
from pathlib import Path
from datetime import datetime
import os

# Load config
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

st.set_page_config(
    page_title="Animations | DMT-EEG",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Attention Animations")
st.markdown("Genera animaciones de la evolución temporal de la atención")

# Paths
ML_CLF_PATH = Path(config['paths']['ml_clf'])
CONDITIONS = config['data']['conditions']


def get_checkpoints():
    """Get available model checkpoints."""
    checkpoints = []
    for pt_file in ML_CLF_PATH.rglob("*.pt"):
        if "checkpoint" in str(pt_file) or "best" in pt_file.name.lower():
            checkpoints.append(pt_file)
    return sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)


def get_available_videos():
    """Get already generated videos."""
    video_dir = ML_CLF_PATH / "output" / "animations"
    if video_dir.exists():
        return list(video_dir.glob("*.mp4"))
    return []


# Tabs
tab1, tab2 = st.tabs([
    "🎬 Generar Animación",
    "📹 Videos Existentes"
])

with tab1:
    st.markdown("### 🎬 Generar Nueva Animación")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📁 Modelo")
        
        checkpoints = get_checkpoints()
        
        if checkpoints:
            checkpoint_options = {str(cp): cp.name for cp in checkpoints}
            selected_checkpoint = st.selectbox(
                "Checkpoint",
                options=list(checkpoint_options.keys()),
                format_func=lambda x: checkpoint_options[x]
            )
        else:
            st.warning("No hay checkpoints disponibles")
            selected_checkpoint = None
        
        st.markdown("#### 👤 Sujeto")
        
        subject_id = st.text_input(
            "Subject ID",
            value="",
            placeholder="ej: S01-DMT",
            help="ID del sujeto (ejecuta con --list-subjects para ver disponibles)"
        )
        
        if st.button("📋 Ver sujetos disponibles"):
            if selected_checkpoint:
                st.code(f"""
cd {ML_CLF_PATH}
python generate_attention_animation.py \\
    --checkpoint {selected_checkpoint} \\
    --list-subjects
                """, language="bash")
    
    with col2:
        st.markdown("#### ⚙️ Configuración")
        
        animation_type = st.radio(
            "Tipo de animación",
            ["Atención completa", "MST (Minimum Spanning Tree)"],
            help="MST muestra solo las conexiones más importantes"
        )
        
        layer = st.slider(
            "Capa GAT",
            min_value=0,
            max_value=3,
            value=0,
            help="Índice de la capa (0-based)"
        )
        
        fps = st.slider(
            "FPS",
            min_value=1,
            max_value=30,
            value=5,
            help="Frames por segundo"
        )
        
        transitions = st.slider(
            "Frames de transición",
            min_value=0,
            max_value=10,
            value=3,
            help="Suavizado entre epochs"
        )
        
        conditions_filter = st.multiselect(
            "Condiciones",
            CONDITIONS,
            default=CONDITIONS[:2] if len(CONDITIONS) >= 2 else CONDITIONS
        )
    
    st.markdown("---")
    
    # Build command
    if selected_checkpoint and subject_id:
        cmd_parts = [
            "python", "generate_attention_animation.py",
            "--checkpoint", selected_checkpoint,
            "--subject", subject_id,
            "--layer", str(layer),
            "--fps", str(fps),
            "--transitions", str(transitions)
        ]
        
        if animation_type == "MST (Minimum Spanning Tree)":
            cmd_parts.append("--mst")
        
        if conditions_filter:
            cmd_parts.extend(["--conditions"] + conditions_filter)
        
        st.markdown("#### 📝 Comando")
        st.code(f"cd {ML_CLF_PATH} && " + " ".join(cmd_parts), language="bash")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🎬 Generar Animación", type="primary", use_container_width=True):
                st.info("Generando animación... Esto puede tomar unos minutos.")
                st.warning("⚠️ Para ver el progreso, ejecuta el comando en terminal")
        
        with col2:
            if st.button("📋 Copiar comando", use_container_width=True):
                st.code(" ".join(cmd_parts))
    else:
        st.warning("Selecciona un checkpoint y especifica un sujeto")

with tab2:
    st.markdown("### 📹 Videos Generados")
    
    videos = get_available_videos()
    
    if videos:
        for video_path in videos:
            with st.expander(f"🎬 {video_path.name}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Try to display video
                    try:
                        video_file = open(video_path, 'rb')
                        video_bytes = video_file.read()
                        st.video(video_bytes)
                    except Exception as e:
                        st.error(f"Error cargando video: {e}")
                        st.info(f"Path: {video_path}")
                
                with col2:
                    st.markdown(f"""
                    **Detalles:**
                    - Tamaño: {video_path.stat().st_size / (1024*1024):.1f} MB
                    - Creado: {datetime.fromtimestamp(video_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M')}
                    """)
                    
                    # Download button
                    with open(video_path, 'rb') as f:
                        st.download_button(
                            "📥 Descargar",
                            f,
                            file_name=video_path.name,
                            mime="video/mp4"
                        )
    else:
        st.info("No hay videos generados. Usa la pestaña 'Generar Animación' para crear uno.")
        
        st.markdown(f"""
        Los videos se guardan en:
        ```
        {ML_CLF_PATH}/output/animations/
        ```
        """)

# Sidebar
with st.sidebar:
    st.markdown("### 📋 Info")
    
    st.markdown("""
    **Tipos de animación:**
    
    - **Atención completa**: Muestra todas las conexiones con pesos de atención
    - **MST**: Muestra solo el árbol de expansión mínima (conexiones más fuertes)
    
    **Consejos:**
    - FPS bajo (3-5) para análisis detallado
    - FPS alto (15-30) para presentaciones
    - Más transiciones = animación más suave
    """)
    
    st.markdown("---")
    
    st.markdown("### 🔄 Actualizar")
    if st.button("🔄 Recargar", use_container_width=True):
        st.rerun()




