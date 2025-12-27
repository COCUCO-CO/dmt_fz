"""Model analysis page."""
from pathlib import Path
import asyncio
import numpy as np
from nicegui import ui
import plotly.graph_objects as go

from config import (
    THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM,
    EEG_CLEAN_DIR
)
from app.state import AS
from app.visualization.styles.css import STYLE
from app.visualization.components.global_header import render_global_header
from app.core.kuramoto_proxy import compute_kuramoto_comparison
# EEG sync is now manual - user selects EEG file directly
from eeg_loader import load_eeg_file, get_channel_data
from app.core.signal import process_data as signal_process_data

AUTOENCODER_DIR = Path(__file__).parent.parent.parent.parent / "machine_learning" / "autoencoder"
AUTOENCODER_CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "autoencoder"

# Electrode positions (10-20 system) for brain topology visualization
# Coordinates are normalized to fit in [-1, 1] range with head facing up
ELECTRODE_POSITIONS_10_20 = {
    # Frontal pole
    'Fp1': (-0.31, 0.95), 'Fpz': (0.0, 0.98), 'Fp2': (0.31, 0.95),
    # Frontal
    'F7': (-0.81, 0.59), 'F3': (-0.39, 0.67), 'Fz': (0.0, 0.72), 'F4': (0.39, 0.67), 'F8': (0.81, 0.59),
    # Frontal-Central
    'FC5': (-0.63, 0.39), 'FC1': (-0.22, 0.42), 'FCz': (0.0, 0.45), 'FC2': (0.22, 0.42), 'FC6': (0.63, 0.39),
    # Temporal & Central
    'T7': (-0.99, 0.0), 'T3': (-0.99, 0.0),  # T7/T3 same position
    'C3': (-0.49, 0.0), 'Cz': (0.0, 0.0), 'C4': (0.49, 0.0),
    'T8': (0.99, 0.0), 'T4': (0.99, 0.0),  # T8/T4 same position
    # Central-Parietal
    'CP5': (-0.63, -0.39), 'CP1': (-0.22, -0.42), 'CPz': (0.0, -0.45), 'CP2': (0.22, -0.42), 'CP6': (0.63, -0.39),
    # Parietal & Temporal
    'T5': (-0.81, -0.59), 'P7': (-0.81, -0.59),  # T5/P7 same
    'P3': (-0.39, -0.67), 'Pz': (0.0, -0.72), 'P4': (0.39, -0.67),
    'T6': (0.81, -0.59), 'P8': (0.81, -0.59),  # T6/P8 same
    # Occipital
    'O1': (-0.31, -0.95), 'Oz': (0.0, -0.98), 'O2': (0.31, -0.95),
    # Mastoid/Ear references
    'M1': (-1.05, 0.0), 'A1': (-1.05, 0.0),
    'M2': (1.05, 0.0), 'A2': (1.05, 0.0),
}

# Standard 24-channel EEG montage order (common clinical setup)
STANDARD_24_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'Oz', 'O2',
    'FC1', 'FC2', 'CP1', 'CP2'
]

# Alternative 24-channel montage (10-20 extended)
STANDARD_24_CHANNELS_ALT = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'CP5', 'CP1', 'CP2', 'CP6',
    'P3', 'Pz', 'P4',
    'O1', 'O2'
]


def generate_eeg_positions(n_nodes: int, channel_names: list = None) -> dict:
    """
    Generate EEG electrode positions for visualization.
    
    Args:
        n_nodes: Number of nodes/channels
        channel_names: Optional list of channel names to use for positioning
        
    Returns:
        Dict mapping node index to (x, y) position
    """
    positions = {}
    
    # If channel names provided, try to match to 10-20 positions
    if channel_names:
        for i, name in enumerate(channel_names):
            # Clean channel name (remove spaces, handle case)
            clean_name = name.strip()
            # Try exact match first
            if clean_name in ELECTRODE_POSITIONS_10_20:
                positions[i] = ELECTRODE_POSITIONS_10_20[clean_name]
            # Try uppercase
            elif clean_name.upper() in ELECTRODE_POSITIONS_10_20:
                positions[i] = ELECTRODE_POSITIONS_10_20[clean_name.upper()]
            # Try title case
            elif clean_name.title() in ELECTRODE_POSITIONS_10_20:
                positions[i] = ELECTRODE_POSITIONS_10_20[clean_name.title()]
        
        # If we matched all, return
        if len(positions) == n_nodes:
            return positions
    
    # Use standard montage for common node counts
    if n_nodes == 24:
        montage = STANDARD_24_CHANNELS_ALT
        for i, ch in enumerate(montage[:n_nodes]):
            if ch in ELECTRODE_POSITIONS_10_20:
                positions[i] = ELECTRODE_POSITIONS_10_20[ch]
    elif n_nodes == 19:
        # Standard 10-20 montage
        montage_19 = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 
                      'T3', 'C3', 'Cz', 'C4', 'T4', 
                      'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
        for i, ch in enumerate(montage_19):
            if ch in ELECTRODE_POSITIONS_10_20:
                positions[i] = ELECTRODE_POSITIONS_10_20[ch]
    elif n_nodes == 32:
        # Extended 10-20
        montage_32 = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                      'FC5', 'FC1', 'FCz', 'FC2', 'FC6',
                      'T7', 'C3', 'Cz', 'C4', 'T8',
                      'CP5', 'CP1', 'CPz', 'CP2', 'CP6',
                      'P7', 'P3', 'Pz', 'P4', 'P8',
                      'O1', 'Oz', 'O2']
        for i, ch in enumerate(montage_32[:n_nodes]):
            if ch in ELECTRODE_POSITIONS_10_20:
                positions[i] = ELECTRODE_POSITIONS_10_20[ch]
    
    # Fill any missing positions with interpolated grid
    if len(positions) < n_nodes:
        # Create a grid layout for remaining nodes
        remaining = [i for i in range(n_nodes) if i not in positions]
        n_remaining = len(remaining)
        
        if n_remaining == n_nodes:
            # No positions matched, create full EEG-like grid
            # Arrange in rows like EEG cap
            rows = [
                (0.9, 3),   # Frontal pole: 3 electrodes
                (0.6, 5),   # Frontal: 5 electrodes
                (0.3, 5),   # FC: 5 electrodes
                (0.0, 5),   # Central: 5 electrodes
                (-0.3, 5),  # CP: 5 electrodes
                (-0.6, 5),  # Parietal: 5 electrodes
                (-0.9, 3),  # Occipital: 3 electrodes
            ]
            
            idx = 0
            for y, n_in_row in rows:
                if idx >= n_nodes:
                    break
                n_this_row = min(n_in_row, n_nodes - idx)
                for j in range(n_this_row):
                    if n_this_row == 1:
                        x = 0
                    else:
                        x = -0.8 + (1.6 * j / (n_this_row - 1))
                    positions[idx] = (x, y)
                    idx += 1
        else:
            # Fill remaining with circular positions in unused space
            for j, node_idx in enumerate(remaining):
                angle = 2 * np.pi * j / n_remaining - np.pi / 2
                x = 0.5 * np.cos(angle)
                y = 0.5 * np.sin(angle)
                positions[node_idx] = (x, y)
    
    return positions


def generate_circular_positions(n_nodes: int) -> dict:
    """Generate evenly spaced circular positions for n nodes (fallback)."""
    positions = {}
    for i in range(n_nodes):
        angle = 2 * np.pi * i / n_nodes - np.pi / 2  # Start from top
        x = 0.8 * np.cos(angle)
        y = 0.8 * np.sin(angle)
        positions[i] = (x, y)
    return positions


def build_sync_matrix_from_edges(edge_index, edge_attr, num_nodes, normalize=False):
    """Build synchronization matrix from edge representation.
    
    Args:
        edge_index: [2, num_edges] source and target indices
        edge_attr: [num_edges, ...] edge weights
        num_nodes: Number of nodes
        normalize: If True, normalize weights to [0, 1] range (useful for reconstruction)
    
    Returns:
        [num_nodes, num_nodes] symmetric sync matrix
    """
    if isinstance(edge_index, np.ndarray):
        src, dst = edge_index[0], edge_index[1]
    else:
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    
    if isinstance(edge_attr, np.ndarray):
        weights = edge_attr.copy()
    else:
        weights = edge_attr.cpu().numpy().copy()
    
    # Get PLV values (first column if multi-dimensional)
    if weights.ndim > 1:
        weights = weights[:, 0]
    
    # Normalize if requested (handles reconstructed values that may be out of range)
    if normalize:
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights = np.ones_like(weights) * 0.5
        # Apply sigmoid if values were very large (likely raw decoder output)
        # This keeps values already in [0,1] mostly unchanged
    
    # Clip to valid range
    weights = np.clip(weights, 0, 1)
    
    sync_matrix = np.zeros((num_nodes, num_nodes))
    for i, (s, d) in enumerate(zip(src, dst)):
        if i < len(weights) and s < num_nodes and d < num_nodes:
            sync_matrix[int(s), int(d)] = weights[i]
            sync_matrix[int(d), int(s)] = weights[i]  # Symmetric
    
    return sync_matrix


def make_sync_network_fig(sync_matrix, num_nodes, title="Sync Network", 
                          primary_color=THEME_PRIMARY, edge_colorscale='Viridis',
                          threshold=0.3, channel_names=None):
    """
    Create a brain network visualization showing synchronization between nodes.
    
    Args:
        sync_matrix: [num_nodes, num_nodes] synchronization matrix (PLV values)
        num_nodes: Number of nodes in the graph
        title: Plot title
        primary_color: Color for nodes and head outline
        edge_colorscale: Colorscale for edges
        threshold: Minimum PLV to show edge (reduces clutter)
        channel_names: Optional list of EEG channel names for positioning
    
    Returns:
        Plotly figure with network visualization
    """
    fig = go.Figure()
    
    # Head outline
    theta = np.linspace(0, 2 * np.pi, 100)
    head_color = f'rgba({int(primary_color[1:3], 16)}, {int(primary_color[3:5], 16)}, {int(primary_color[5:7], 16)}, 0.4)'
    
    fig.add_trace(go.Scatter(
        x=np.cos(theta), y=np.sin(theta), mode='lines',
        line=dict(color=head_color, width=1.5), showlegend=False, hoverinfo='skip'
    ))
    # Nose (pointing up)
    fig.add_trace(go.Scatter(
        x=[-0.08, 0, 0.08], y=[0.98, 1.12, 0.98], mode='lines',
        line=dict(color=head_color, width=1.5), showlegend=False, hoverinfo='skip'
    ))
    # Ears
    fig.add_trace(go.Scatter(
        x=[-1.01, -1.08, -1.01], y=[0.12, 0, -0.12], mode='lines',
        line=dict(color=head_color, width=1), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[1.01, 1.08, 1.01], y=[0.12, 0, -0.12], mode='lines',
        line=dict(color=head_color, width=1), showlegend=False, hoverinfo='skip'
    ))
    
    # Generate node positions (use EEG layout)
    positions = generate_eeg_positions(num_nodes, channel_names)
    
    # Draw edges (connections) - only above threshold
    edge_x, edge_y, edge_colors, edge_texts = [], [], [], []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Upper triangle only
            plv = sync_matrix[i, j]
            if plv > threshold:
                x0, y0 = positions[i]
                x1, y1 = positions[j]
                # Add edge as line segment with None separator
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
    
    # Draw all edges at once (more efficient)
    if edge_x:
        # Create edge traces with varying opacity based on PLV
        # Group edges by PLV strength for better visualization
        for plv_min, plv_max, opacity, width in [
            (threshold, 0.5, 0.15, 0.5),
            (0.5, 0.7, 0.35, 1.0),
            (0.7, 0.85, 0.6, 1.5),
            (0.85, 1.0, 0.9, 2.0)
        ]:
            ex, ey = [], []
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    plv = sync_matrix[i, j]
                    if plv_min <= plv < plv_max:
                        x0, y0 = positions[i]
                        x1, y1 = positions[j]
                        ex.extend([x0, x1, None])
                        ey.extend([y0, y1, None])
            
            if ex:
                # Map PLV range to color
                avg_plv = (plv_min + plv_max) / 2
                if edge_colorscale == 'Viridis':
                    # Blue -> Green -> Yellow
                    if avg_plv < 0.5:
                        color = f'rgba(68, 1, 84, {opacity})'
                    elif avg_plv < 0.7:
                        color = f'rgba(59, 82, 139, {opacity})'
                    elif avg_plv < 0.85:
                        color = f'rgba(33, 145, 140, {opacity})'
                    else:
                        color = f'rgba(94, 201, 98, {opacity})'
                else:  # Pink/magenta scale for reconstructed
                    if avg_plv < 0.5:
                        color = f'rgba(131, 24, 67, {opacity})'
                    elif avg_plv < 0.7:
                        color = f'rgba(190, 24, 93, {opacity})'
                    elif avg_plv < 0.85:
                        color = f'rgba(244, 114, 182, {opacity})'
                    else:
                        color = f'rgba(253, 164, 175, {opacity})'
                
                fig.add_trace(go.Scatter(
                    x=ex, y=ey, mode='lines',
                    line=dict(color=color, width=width),
                    hoverinfo='skip', showlegend=False
                ))
    
    # Draw nodes
    node_x = [positions[i][0] for i in range(num_nodes)]
    node_y = [positions[i][1] for i in range(num_nodes)]
    
    # Node color based on total connectivity (degree)
    node_degrees = sync_matrix.sum(axis=1)
    if node_degrees.max() > 0:
        node_colors = node_degrees / node_degrees.max()
    else:
        node_colors = np.ones(num_nodes) * 0.5
    
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers',
        marker=dict(
            size=10,
            color=node_colors,
            colorscale='Viridis' if edge_colorscale == 'Viridis' else 'RdPu',
            cmin=0, cmax=1,
            line=dict(width=1, color=primary_color),
            showscale=False
        ),
        text=[f'Node {i}<br>Degree: {node_degrees[i]:.2f}' for i in range(num_nodes)],
        hoverinfo='text',
        showlegend=False
    ))
    
    # Layout
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(8,8,8,1)',
        plot_bgcolor='rgba(12,12,12,1)',
        margin=dict(l=5, r=5, t=25, b=5),
        height=200,
        title=dict(text=title, font=dict(size=10, color=primary_color), x=0.5, y=0.98),
        xaxis=dict(
            range=[-1.2, 1.2], showgrid=False, zeroline=False, 
            showticklabels=False, fixedrange=True
        ),
        yaxis=dict(
            range=[-0.95, 1.15], showgrid=False, zeroline=False, 
            showticklabels=False, fixedrange=True, scaleanchor='x'
        ),
        showlegend=False,
        hovermode='closest'
    )
    
    return fig


def analysis_log(msg: str, msg_type: str = 'info'):
    """Add message to analysis log."""
    if AS.log_container:
        colors = {'info': THEME_TEXT, 'success': THEME_PRIMARY, 'warning': THEME_WARN, 'error': THEME_ERROR}
        with AS.log_container:
            ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.7rem;')


def load_trained_model(model_path: Path):
    """Load a trained VAE model from checkpoint. Supports both graph and image models."""
    import torch
    import pickle
    import sys
    
    try:
        # For external models, add the project root to sys.path
        # This handles models that were saved with project-specific module references
        # Search upward from model path to find project root (containing 'src' folder)
        current = model_path.parent
        for _ in range(5):  # Search up to 5 levels
            if (current / 'src').exists():
                if str(current) not in sys.path:
                    sys.path.insert(0, str(current))
                    analysis_log(f"Added {current} to Python path", 'info')
                break
            parent = current.parent
            if parent == current:  # Reached root
                break
            current = parent
        
        # Try to load as pickle first (for image models with full_model)
        is_pickle = False
        checkpoint = None
        
        if model_path.suffix == '.pkl':
            try:
                with open(model_path, 'rb') as f:
                    checkpoint = pickle.load(f)
                is_pickle = True
            except Exception:
                pass
        
        if checkpoint is None:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Detect model type: image model has 'full_model' key, graph model has 'model_state_dict'
        if 'full_model' in checkpoint:
            # Image-based VAE (convolutional)
            return load_image_model(checkpoint, model_path)
        else:
            # Graph-based VAE
            return load_graph_model(checkpoint, model_path)
            
    except Exception as e:
        raise Exception(f"Failed to load model: {e}")


def load_image_model(checkpoint, model_path: Path):
    """Load an image-based convolutional VAE model."""
    import torch
    
    # The model is stored directly in the checkpoint
    model = checkpoint['full_model']
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Extract model info
    model_config = checkpoint.get('model_config', {})
    model_params = {
        'latent_dim': checkpoint.get('latent_dim', model_config.get('latent_dim', 256)),
        'input_size': checkpoint.get('input_size', model_config.get('input_size', 512)),
        'num_classes': checkpoint.get('num_classes', model_config.get('num_classes', 17)),
        'model_type': checkpoint.get('model_type', 'image_vae'),
    }
    
    AS.model = model
    AS.model_config = model_config
    AS.model_path = str(model_path)
    AS.device = device
    AS.model_params = model_params
    AS.model_type = 'image'
    
    # Register hooks for activation extraction (image model)
    register_image_activation_hooks(model)
    
    return {
        'epoch': checkpoint.get('epoch', '?'),
        'val_loss': checkpoint.get('best_val_loss', '?'),
        'config': model_config,
        'model_params': model_params,
        'model_type': 'image'
    }


def load_graph_model(checkpoint, model_path: Path):
    """Load a graph-based VAE model."""
    import torch
    import pickle
    import sys
    
    config = checkpoint.get('config', {})
    
    # Get model parameters from checkpoint
    model_params = checkpoint.get('model_params', {})
    
    # If no model_params in checkpoint, try to infer from dataset or config
    if not model_params:
        # Try to load a sample from dataset cache to get dimensions
        dataset_cache = AUTOENCODER_CACHE_DIR / 'dataset_cache'
        try:
            # Try both .pt and .pkl files
            for pattern in ['*.pt', '*.pkl']:
                for cache_file in dataset_cache.glob(pattern):
                    try:
                        if cache_file.suffix == '.pkl':
                            with open(cache_file, 'rb') as f:
                                data = pickle.load(f)
                        else:
                            data = torch.load(cache_file, weights_only=False)
                        
                        # Handle dict with train/val/test splits
                        if isinstance(data, dict) and 'train' in data:
                            data = data['train']
                        
                        sample = data[0] if isinstance(data, list) and len(data) > 0 else data
                        if hasattr(sample, 'x'):
                            model_params = {
                                'num_node_features': sample.x.shape[1],
                                'num_edge_features': sample.edge_attr.shape[1] if hasattr(sample, 'edge_attr') and sample.edge_attr is not None else 1,
                                'num_graph_features': sample.graph_attr.shape[0] if hasattr(sample, 'graph_attr') and sample.graph_attr is not None else 3,
                                'num_nodes': sample.num_nodes
                            }
                            break
                    except Exception:
                        continue
                if model_params:
                    break
        except Exception:
            pass
    
    # Use values from model_params or defaults
    num_node_features = model_params.get('num_node_features', 68)
    num_edge_features = model_params.get('num_edge_features', 1)
    num_graph_features = model_params.get('num_graph_features', 3)
    num_nodes = model_params.get('num_nodes', 68)
    
    # Import model creation function
    if str(AUTOENCODER_DIR) not in sys.path:
        sys.path.insert(0, str(AUTOENCODER_DIR))
    from models import create_vae_from_config
    
    model = create_vae_from_config(
        config,
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Check for GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    AS.model = model
    AS.model_config = config
    AS.model_path = str(model_path)
    AS.device = device
    AS.model_params = model_params
    AS.model_type = 'graph'
    
    # Register hooks for activation extraction
    register_activation_hooks(model)
    
    return {
        'epoch': checkpoint.get('epoch', '?'),
        'val_loss': checkpoint.get('val_loss', '?'),
        'config': config,
        'model_params': model_params,
        'model_type': 'graph'
    }


def register_image_activation_hooks(model):
    """Register forward hooks to capture activations for image models."""
    AS.activations = {}
    AS.attention_weights = {}
    AS.layer_info = {'encoder': [], 'decoder': []}  # Store layer info for UI
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                AS.activations[name] = output[0].detach().cpu()
            else:
                AS.activations[name] = output.detach().cpu()
        return hook
    
    # Register hooks on stem
    if hasattr(model, 'stem'):
        model.stem.register_forward_hook(get_activation('stem'))
    
    # Register hooks on encoder layers
    if hasattr(model, 'encoder_layers'):
        for i, layer in enumerate(model.encoder_layers):
            layer.register_forward_hook(get_activation(f'encoder.layer_{i}'))
            # Get output channels info
            out_ch = None
            for m in layer.modules():
                if hasattr(m, 'out_channels'):
                    out_ch = m.out_channels
                    break
            AS.layer_info['encoder'].append({'name': f'Layer {i}', 'channels': out_ch or 'N/A'})
    elif hasattr(model, 'encoder'):
        # Fallback for other architectures
        for name, child in model.encoder.named_children():
            if 'stage' in name or 'layer' in name:
                child.register_forward_hook(get_activation(f'encoder.{name}'))
                AS.layer_info['encoder'].append({'name': name, 'channels': 'N/A'})
    
    # Register hooks on decoder layers  
    if hasattr(model, 'decoder_layers'):
        for i, layer in enumerate(model.decoder_layers):
            layer.register_forward_hook(get_activation(f'decoder.layer_{i}'))
            out_ch = None
            for m in layer.modules():
                if hasattr(m, 'out_channels'):
                    out_ch = m.out_channels
                    break
            AS.layer_info['decoder'].append({'name': f'Layer {i}', 'channels': out_ch or 'N/A'})
    
    # Register hook on latent
    if hasattr(model, 'fc_mu'):
        model.fc_mu.register_forward_hook(get_activation('latent_mu'))
    if hasattr(model, 'fc_logvar'):
        model.fc_logvar.register_forward_hook(get_activation('latent_logvar'))
    
    # Register hook on output
    if hasattr(model, 'output'):
        model.output.register_forward_hook(get_activation('output'))


def register_activation_hooks(model):
    """Register forward hooks to capture activations."""
    AS.activations = {}
    AS.attention_weights = {}
    
    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                AS.activations[name] = output[0].detach().cpu()
            else:
                AS.activations[name] = output.detach().cpu()
        return hook
    
    def get_attention(name):
        def hook(module, input, output):
            # GAT layers store attention in return_attention_weights
            if hasattr(module, 'return_attention_weights') and module.return_attention_weights:
                if isinstance(output, tuple) and len(output) > 1:
                    AS.attention_weights[name] = output[1].detach().cpu()
        return hook
    
    # Register hooks on encoder layers
    if hasattr(model, 'encoder'):
        conv_layers = getattr(model.encoder, 'conv_layers', None)
        if conv_layers is not None:
            for i, layer in enumerate(conv_layers):
                layer.register_forward_hook(get_activation(f'encoder.conv_{i}'))
    
    # Register hooks on decoder layers
    if hasattr(model, 'decoder'):
        conv_layers = getattr(model.decoder, 'conv_layers', None)
        if conv_layers is not None:
            for i, layer in enumerate(conv_layers):
                layer.register_forward_hook(get_activation(f'decoder.conv_{i}'))
        elif hasattr(model.decoder, 'layers'):
            for i, layer in enumerate(model.decoder.layers):
                layer.register_forward_hook(get_activation(f'decoder.conv_{i}'))
    
    # Register hook on latent
    if hasattr(model, 'fc_mu'):
        model.fc_mu.register_forward_hook(get_activation('latent_mu'))
    if hasattr(model, 'fc_logvar'):
        model.fc_logvar.register_forward_hook(get_activation('latent_logvar'))


def process_sample(sample, store_latent=True):
    """Process a single sample through the model and extract activations."""
    import torch
    
    if AS.model is None:
        analysis_log("No model loaded", 'warning')
        return None
    
    try:
        AS.model.eval()
        with torch.no_grad():
            if AS.model_type == 'image':
                return process_image_sample(sample, store_latent)
            else:
                return process_graph_sample(sample, store_latent)
    except Exception as e:
        analysis_log(f"Error in process_sample: {e}", 'error')
        import traceback
        analysis_log(traceback.format_exc(), 'error')
        return None


def process_image_sample(sample, store_latent=True):
    """Process an image sample through the model."""
    
    # sample is a tuple (image_tensor, label) or just image_tensor
    if isinstance(sample, tuple):
        image, label = sample
        AS.current_label = label.item() if hasattr(label, 'item') else label
    else:
        image = sample
        AS.current_label = None
    
    # Ensure batch dimension
    if image.dim() == 3:
        image = image.unsqueeze(0)
    
    image = image.to(AS.device)
    
    # Forward pass
    output = AS.model(image)
    
    AS.current_sample = image
    AS.current_recon = output
    AS.current_z = output.get('mu', None) if isinstance(output, dict) else None
    
    if store_latent and AS.current_z is not None:
        z_np = AS.current_z.detach().cpu().numpy().flatten()
        AS.latent_codes.append(z_np)
        label = AS.current_label if AS.current_label is not None else 0
        AS.latent_labels.append(label)
        # Keep only last 500 for PCA
        if len(AS.latent_codes) > 500:
            AS.latent_codes = AS.latent_codes[-500:]
            AS.latent_labels = AS.latent_labels[-500:]
    
    return output


def process_graph_sample(sample, store_latent=True):
    """Process a graph sample through the model."""
    import torch
    from torch_geometric.data import Batch
    
    # Create a batch from single sample (required by PyG)
    if not isinstance(sample, Batch):
        batch = Batch.from_data_list([sample])
    else:
        batch = sample
    
    batch = batch.to(AS.device)
    
    # Forward pass with activation storage ENABLED to capture GAT attention
    output = AS.model(batch, store_activations=True)
    
    AS.current_sample = batch
    AS.current_recon = output
    AS.current_z = output.get('mu', None) if isinstance(output, dict) else None
    
    # Compute Kuramoto proxy from reconstructed edges
    AS.current_kuramoto_comparison = None
    if isinstance(output, dict) and 'edge_attr_recon' in output:
        try:
            comparison = compute_kuramoto_comparison(
                original_graph_attr=batch.graph_attr[0] if batch.graph_attr.dim() == 2 else batch.graph_attr,
                reconstructed_edge_attr=output['edge_attr_recon'],
                edge_index=batch.edge_index,
                num_nodes=batch.num_nodes if hasattr(batch, 'num_nodes') else batch.x.shape[0],
                method='mean_plv'
            )
            AS.current_kuramoto_comparison = comparison
        except Exception as e:
            analysis_log(f"Kuramoto proxy error: {e}", 'warning')
    
    # Extract REAL attention weights from encoder
    if hasattr(AS.model, 'encoder') and hasattr(AS.model.encoder, 'get_all_attention_weights'):
        AS.attention_weights = AS.model.encoder.get_all_attention_weights()
    
    # Extract attention weights from decoder (if using GAT decoder)
    if hasattr(AS.model, 'get_decoder_attention_weights'):
        decoder_att = AS.model.get_decoder_attention_weights()
        for key, val in decoder_att.items():
            AS.attention_weights[key] = val
    
    # Extract layer activations from encoder
    if hasattr(AS.model, 'encoder') and hasattr(AS.model.encoder, 'get_layer_activations'):
        layer_acts = AS.model.encoder.get_layer_activations()
        for key, val in layer_acts.items():
            AS.activations[f'encoder.{key}'] = val
    
    # Extract decoder activations/attention
    if hasattr(AS.model, 'decoder'):
        decoder = AS.model.decoder
        decoder_att = decoder.get_attention_weights() if hasattr(decoder, 'get_attention_weights') else {}
        
        if decoder_att:
            # GAT decoder - use attention matrices as "activations"
            for key, att_data in decoder_att.items():
                if 'attention' in att_data:
                    # Reshape attention to 2D matrix for visualization
                    attn = att_data['attention'].cpu()
                    edge_index = att_data['edge_index'].cpu()
                    
                    # Build attention matrix
                    num_nodes = batch.x.shape[0]
                    if attn.ndim > 1:
                        attn_flat = attn.mean(dim=1)  # Average across heads
                    else:
                        attn_flat = attn
                    
                    attn_matrix = torch.zeros(num_nodes, num_nodes)
                    for e_idx in range(min(edge_index.shape[1], len(attn_flat))):
                        src, dst = int(edge_index[0, e_idx]), int(edge_index[1, e_idx])
                        if src < num_nodes and dst < num_nodes:
                            attn_matrix[src, dst] = attn_flat[e_idx]
                    
                    AS.activations[f'decoder.{key}'] = attn_matrix
        elif hasattr(decoder, 'node_decoder'):
            # MLP decoder - use reconstructed features as visualization
            if 'x_recon' in output:
                AS.activations['decoder.output'] = output['x_recon'].detach().cpu()
    
    if store_latent and AS.current_z is not None:
        z_np = AS.current_z.detach().cpu().numpy().flatten()
        AS.latent_codes.append(z_np)
        # Get label from sample
        label = batch.y[0].item() if hasattr(batch, 'y') and batch.y is not None else 0
        AS.latent_labels.append(label)
        # Keep only last 500 for PCA
        if len(AS.latent_codes) > 500:
            AS.latent_codes = AS.latent_codes[-500:]
            AS.latent_labels = AS.latent_labels[-500:]
    
    return output


def compute_latent_pca():
    """Compute PCA on accumulated latent codes."""
    if len(AS.latent_codes) < 10:
        return None, None
    
    from sklearn.decomposition import PCA
    import numpy as np
    
    X = np.array(AS.latent_codes)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    return X_pca, np.array(AS.latent_labels)


def load_hdf5_image_dataset(h5_path, split, dataset_info_container, progress_slider):
    """Load an image dataset from HDF5 file using lazy loading."""
    import h5py
    import torch
    import numpy as np
    
    try:
        # First, just read metadata without loading all data
        with h5py.File(h5_path, 'r') as f:
            total_samples = len(f['labels'])
            resolution = f.attrs.get('resolution', 512)
            labels_all = f['labels'][:]
            
            # Get group names if available
            if 'group_names' in f:
                AS.class_names = [n.decode() if isinstance(n, bytes) else n for n in f['group_names'][:]]
            else:
                AS.class_names = [str(i) for i in range(len(np.unique(labels_all)))]
        
        # Check for splits file
        indices = None
        splits_path = h5_path.parent / 'splits.npz'
        if splits_path.exists():
            splits = np.load(splits_path)
            if split in splits:
                indices = splits[split]
                analysis_log(f"Using '{split}' split: {len(indices)} samples", 'info')
        
        # Create lazy-loading wrapper for HDF5
        class HDF5DatasetWrapper:
            """Lazy-loading wrapper for HDF5 image dataset."""
            def __init__(self, h5_path, indices=None):
                self.h5_path = str(h5_path)
                self.indices = indices
                self._file = None
                self._labels = labels_all[indices] if indices is not None else labels_all
                
            def _open(self):
                if self._file is None:
                    self._file = h5py.File(self.h5_path, 'r')
                return self._file
            
            def __len__(self):
                return len(self.indices) if self.indices is not None else len(self._labels)
            
            def __getitem__(self, idx):
                f = self._open()
                real_idx = self.indices[idx] if self.indices is not None else idx
                # Load single image on demand
                pattern = f['patterns'][real_idx]  # [H, W, 3]
                label = self._labels[idx]
                # Convert to tensor [3, H, W]
                tensor = torch.from_numpy(pattern.astype(np.float32)).permute(2, 0, 1)
                return tensor, int(label)
            
            def __del__(self):
                if self._file is not None:
                    try:
                        self._file.close()
                    except:
                        pass
        
        AS.dataset = HDF5DatasetWrapper(h5_path, indices)
        AS.dataset_type = 'image'
        AS.total_samples = len(AS.dataset)
        AS.current_idx = 0
        AS.dataset_path = str(h5_path)
        
        # Clear latent history
        AS.latent_codes = []
        AS.latent_labels = []
        
        dataset_info_container.clear()
        with dataset_info_container:
            ui.label("✓ Image dataset loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
            ui.label(f"  Split: {split}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Samples: {AS.total_samples}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Size: {resolution}×{resolution} RGB").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Classes: {len(AS.class_names)}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
        
        analysis_log(f"HDF5 dataset loaded: {AS.total_samples} images (lazy loading)", 'success')
        
        if progress_slider:
            progress_slider.set_value(0)
            progress_slider._props['max'] = max(1, AS.total_samples - 1)
        
        return True
    except Exception as e:
        analysis_log(f"Error loading HDF5: {e}", 'error')
        import traceback
        analysis_log(traceback.format_exc(), 'error')
        return False


def load_image_folder_dataset(images_dir, split, base_path, dataset_info_container, progress_slider):
    """Load an image dataset from folder structure (class_name/image.png)."""
    import torch
    import numpy as np
    from PIL import Image
    
    try:
        # Find all class directories
        class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])
        AS.class_names = [d.name for d in class_dirs]
        
        # Collect all images
        samples = []
        for class_idx, class_dir in enumerate(class_dirs):
            for img_path in sorted(class_dir.glob('*.png')) + sorted(class_dir.glob('*.jpg')):
                samples.append((str(img_path), class_idx))
        
        # Check for splits file
        splits_path = base_path / 'splits.npz'
        if splits_path.exists():
            splits = np.load(splits_path)
            if split in splits:
                indices = splits[split]
                samples = [samples[i] for i in indices if i < len(samples)]
                analysis_log(f"Using '{split}' split: {len(samples)} samples", 'info')
        
        # Create lazy-loading dataset
        class ImageDatasetWrapper:
            def __init__(self, samples, class_names):
                self.samples = samples
                self.class_names = class_names
            
            def __len__(self):
                return len(self.samples)
            
            def __getitem__(self, idx):
                img_path, label = self.samples[idx]
                img = Image.open(img_path).convert('RGB')
                img_np = np.array(img, dtype=np.float32) / 255.0
                # [H, W, 3] -> [3, H, W]
                tensor = torch.from_numpy(img_np).permute(2, 0, 1)
                return tensor, label
        
        AS.dataset = ImageDatasetWrapper(samples, AS.class_names)
        AS.dataset_type = 'image'
        AS.total_samples = len(AS.dataset)
        AS.current_idx = 0
        AS.dataset_path = str(images_dir)
        
        # Clear latent history
        AS.latent_codes = []
        AS.latent_labels = []
        
        # Get sample info
        sample_img, _ = AS.dataset[0]
        resolution = sample_img.shape[-1]
        
        dataset_info_container.clear()
        with dataset_info_container:
            ui.label("✓ Image folder loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
            ui.label(f"  Split: {split}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Samples: {AS.total_samples}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Size: {resolution}×{resolution} RGB").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            ui.label(f"  Classes: {len(AS.class_names)}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
        
        analysis_log(f"Image folder loaded: {AS.total_samples} images, {len(AS.class_names)} classes", 'success')
        
        if progress_slider:
            progress_slider.set_value(0)
            progress_slider._props['max'] = max(1, AS.total_samples - 1)
        
        return True
    except Exception as e:
        analysis_log(f"Error loading image folder: {e}", 'error')
        import traceback
        analysis_log(traceback.format_exc(), 'error')
        return False


def compute_dataset_ranges():
    """
    Analyze dataset and model to compute fixed axis ranges.
    Should be called after loading both model and dataset.
    """
    import torch
    from torch_geometric.data import Batch
    import numpy as np
    
    if not AS.dataset or not AS.model:
        return
    
    analysis_log("Computing axis ranges from dataset sample...", 'info')
    
    # Compute Kuramoto statistics from entire dataset
    all_kuramoto = []
    for sample in AS.dataset:
        if hasattr(sample, 'graph_attr') and sample.graph_attr is not None and len(sample.graph_attr) > 0:
            k_val = sample.graph_attr[0].item() if hasattr(sample.graph_attr[0], 'item') else float(sample.graph_attr[0])
            all_kuramoto.append(k_val)
    
    if all_kuramoto:
        AS.kuramoto_avg = float(np.mean(all_kuramoto))
        AS.kuramoto_std = float(np.std(all_kuramoto))
        analysis_log(f"Kuramoto: avg={AS.kuramoto_avg:.3f}, std={AS.kuramoto_std:.3f} (metastability)", 'info')
    
    # Clear kuramoto history for fresh start
    AS.kuramoto_history = []
    AS.kuramoto_proxy_history = []
    
    # Sample a subset of the dataset for analysis
    n_samples = min(100, len(AS.dataset))
    indices = np.linspace(0, len(AS.dataset)-1, n_samples, dtype=int)
    
    all_features = []
    all_latents = []
    all_activations = {f'encoder.conv_{i}': [] for i in range(3)}
    
    AS.model.eval()
    with torch.no_grad():
        for idx in indices:
            try:
                sample = AS.dataset[idx]
                batch = Batch.from_data_list([sample]).to(AS.device)
                
                # Get node features
                all_features.append(batch.x.cpu().numpy())
                
                # Forward pass with activations
                output = AS.model(batch, store_activations=True)
                
                # Get latent
                if isinstance(output, dict) and 'mu' in output:
                    all_latents.append(output['mu'].cpu().numpy())
                
                # Get activations
                for name, act in AS.activations.items():
                    if name in all_activations:
                        act_np = act.cpu().numpy() if hasattr(act, 'cpu') else act.numpy()
                        all_activations[name].append(act_np.flatten())
            except Exception:
                continue
    
    # Compute ranges for node features
    if all_features:
        all_feat = np.concatenate(all_features, axis=0)
        feat_min, feat_max = np.percentile(all_feat, [2, 98])  # Use percentiles to ignore outliers
        margin = (feat_max - feat_min) * 0.1
        AS.axis_ranges['node_features'] = {
            'min': float(feat_min - margin),
            'max': float(feat_max + margin)
        }
        analysis_log(f"Node features range: [{feat_min:.2f}, {feat_max:.2f}]", 'info')
    
    # Compute ranges for latent space (will be updated as PCA accumulates)
    if all_latents:
        all_lat = np.concatenate(all_latents, axis=0)
        from sklearn.decomposition import PCA
        if all_lat.shape[0] >= 10:
            pca = PCA(n_components=2)
            lat_pca = pca.fit_transform(all_lat)
            x_min, x_max = np.percentile(lat_pca[:, 0], [2, 98])
            y_min, y_max = np.percentile(lat_pca[:, 1], [2, 98])
            margin_x = (x_max - x_min) * 0.15
            margin_y = (y_max - y_min) * 0.15
            AS.axis_ranges['latent'] = {
                'x_min': float(x_min - margin_x),
                'x_max': float(x_max + margin_x),
                'y_min': float(y_min - margin_y),
                'y_max': float(y_max + margin_y)
            }
            analysis_log(f"Latent PCA range: x[{x_min:.1f}, {x_max:.1f}], y[{y_min:.1f}, {y_max:.1f}]", 'info')
    
    # Compute ranges for activations
    for name, acts in all_activations.items():
        if acts:
            all_act = np.concatenate(acts)
            act_min, act_max = np.percentile(all_act, [2, 98])
            margin = (act_max - act_min) * 0.1
            if 'activations_per_layer' not in AS.axis_ranges:
                AS.axis_ranges['activations_per_layer'] = {}
            AS.axis_ranges['activations_per_layer'][name] = {
                'min': float(act_min - margin),
                'max': float(act_max + margin)
            }
    
    # Global activation range
    if all_activations:
        all_acts = []
        for acts in all_activations.values():
            if acts:
                all_acts.extend(acts)
        if all_acts:
            all_act = np.concatenate(all_acts)
            act_min, act_max = np.percentile(all_act, [2, 98])
            margin = (act_max - act_min) * 0.1
            AS.axis_ranges['activations'] = {
                'min': float(act_min - margin),
                'max': float(act_max + margin)
            }
            analysis_log(f"Activations range: [{act_min:.1f}, {act_max:.1f}]", 'info')
    
    AS.ranges_computed = True
    analysis_log("✓ Axis ranges computed", 'success')


@ui.page('/analysis')
def analysis_page():
    """Model Analysis Page - Visualize trained model internals."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    # Global header with system monitor (CPU/RAM/GPU)
    render_global_header('analysis')
    
    with ui.row().classes('w-full p-4 gap-4').style('height: calc(100vh - 50px); overflow: hidden;'):
        
        # LEFT PANEL: Controls
        with ui.column().classes('gap-3').style('width: 280px; flex-shrink: 0; overflow-y: auto; max-height: 100%;'):
            
            # MODEL LOADER
            with ui.card().classes('dark-card p-3 w-full').style(f'border: 1px solid {THEME_WARN};'):
                ui.label('// LOAD MODEL').classes('terminal-header')
                
                model_path_input = ui.input(
                    value=str(AUTOENCODER_CACHE_DIR / 'checkpoints' / 'best_model.pt'),
                    placeholder='Path to model checkpoint'
                ).props('dense dark').classes('w-full mt-2')
                
                model_info_container = ui.column().classes('w-full mt-2 gap-1')
                
                async def load_model():
                    try:
                        path = Path(model_path_input.value.strip())
                        if not path.exists():
                            ui.notify(f'Model not found: {path}', type='negative')
                            return
                        
                        # Show loading indicator
                        model_info_container.clear()
                        with model_info_container:
                            ui.label("⏳ Loading model...").style(f'color:{THEME_WARN}; font-size: 0.75rem;')
                            ui.label("  This may take a moment").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Load model in background thread to avoid blocking UI
                        loop = asyncio.get_event_loop()
                        info = await loop.run_in_executor(None, load_trained_model, path)
                        
                        model_info_container.clear()
                        with model_info_container:
                            ui.label("✓ Model loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                            ui.label(f"  Type: {info.get('model_type', 'graph')}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                            ui.label(f"  Epoch: {info['epoch']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            ui.label(f"  Val Loss: {info['val_loss']:.4f}" if isinstance(info['val_loss'], float) else f"  Val Loss: {info['val_loss']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            ui.label(f"  Device: {AS.device}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                        analysis_log(f"Model loaded: {path.name}", 'success')
                        
                        # Compute axis ranges if dataset is loaded
                        if AS.dataset is not None:
                            compute_dataset_ranges()
                    except Exception as e:
                        model_info_container.clear()
                        with model_info_container:
                            ui.label("✗ Error loading model").style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                        ui.notify(f'Error: {e}', type='negative')
                        analysis_log(f"Error loading model: {e}", 'error')
                        import traceback
                        analysis_log(traceback.format_exc(), 'error')
                
                ui.button('Load Model', on_click=load_model, icon='upload').props('dense').classes('mt-2').style(f'background:{THEME_WARN}; color:black;')
            
            # DATASET LOADER
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// DATASET').classes('terminal-header')
                
                with ui.row().classes('items-center gap-2 mt-2'):
                    ui.label('Split:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    split_select = ui.select(['train', 'val', 'test'], value='test').props('dense dark').classes('flex-1')
                
                dataset_path_input = ui.input(
                    value=str(AUTOENCODER_CACHE_DIR / 'dataset_cache'),
                    placeholder='Path to dataset cache'
                ).props('dense dark').classes('w-full mt-2')
                
                dataset_info_container = ui.column().classes('w-full mt-2 gap-1')
                
                def load_dataset():
                    import torch
                    import pickle
                    try:
                        cache_path = Path(dataset_path_input.value.strip())
                        split = split_select.value
                        
                        # Detect dataset type
                        dataset_loaded = False
                        
                        # Check for HDF5 image dataset
                        h5_files = list(cache_path.glob('*.h5')) if cache_path.is_dir() else []
                        if cache_path.suffix == '.h5' or h5_files:
                            h5_path = cache_path if cache_path.suffix == '.h5' else h5_files[0]
                            dataset_loaded = load_hdf5_image_dataset(h5_path, split, dataset_info_container, progress_slider)
                        
                        # Check for image folder dataset
                        if not dataset_loaded and cache_path.is_dir():
                            images_dir = cache_path / 'images' if (cache_path / 'images').exists() else cache_path
                            subdirs = [d for d in images_dir.iterdir() if d.is_dir()]
                            has_images = any(list(d.glob('*.png'))[:1] or list(d.glob('*.jpg'))[:1] for d in subdirs[:3])
                            if has_images:
                                dataset_loaded = load_image_folder_dataset(images_dir, split, cache_path, dataset_info_container, progress_slider)
                        
                        # Fallback to graph dataset loading
                        if not dataset_loaded:
                            cache_file = None
                            
                            # Check if path is a file directly
                            if cache_path.is_file():
                                cache_file = cache_path
                            else:
                                # It's a directory, search for dataset files
                                possible_files = [
                                    cache_path / f'{split}_dataset.pt',
                                    cache_path / 'processed_dataset.pt',
                                    cache_path / f'{split}_dataset.pkl',
                                    cache_path / 'processed_dataset.pkl',
                                ]
                                for f in possible_files:
                                    if f.exists():
                                        cache_file = f
                                        break
                                
                                # If still not found, search for any dataset file
                                if not cache_file:
                                    for pattern in ['dataset*.pkl', 'dataset*.pt', '*.pkl', '*.pt']:
                                        files = list(cache_path.glob(pattern))
                                        if files:
                                            cache_file = files[0]
                                            break
                            
                            if cache_file and cache_file.exists():
                                # Load based on file extension
                                if cache_file.suffix == '.pkl':
                                    with open(cache_file, 'rb') as f:
                                        data = pickle.load(f)
                                else:
                                    data = torch.load(cache_file, weights_only=False)
                                
                                # Handle dict with train/val/test splits
                                if isinstance(data, dict) and split in data:
                                    graphs = data[split]
                                    analysis_log(f"Using '{split}' split from dataset", 'info')
                                elif isinstance(data, dict) and 'train' in data:
                                    # Default to train if requested split not found
                                    available = list(data.keys())
                                    graphs = data.get(split, data['train'])
                                    analysis_log(f"Available splits: {available}, using '{split}'", 'info')
                                elif isinstance(data, list):
                                    graphs = data
                                elif hasattr(data, '__len__'):
                                    graphs = list(data)
                                else:
                                    graphs = [data]
                                
                                AS.dataset = graphs
                                AS.dataset_type = 'graph'
                                AS.total_samples = len(AS.dataset)
                                AS.current_idx = 0
                                AS.dataset_path = str(cache_file)
                                
                                # Clear latent history for fresh PCA
                                AS.latent_codes = []
                                AS.latent_labels = []
                                
                                dataset_info_container.clear()
                                with dataset_info_container:
                                    ui.label("✓ Graph dataset loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                    ui.label(f"  Split: {split}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    ui.label(f"  Samples: {AS.total_samples}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    if AS.dataset and hasattr(AS.dataset[0], 'x'):
                                        sample = AS.dataset[0]
                                        ui.label(f"  Nodes: {sample.x.shape[0]}, Features: {sample.x.shape[1]}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                
                                analysis_log(f"Dataset loaded: {AS.total_samples} samples from {cache_file.name}", 'success')
                                if progress_slider:
                                    progress_slider.set_value(0)
                                    progress_slider._props['max'] = max(1, AS.total_samples - 1)
                                
                                # Compute axis ranges if model is loaded
                                if AS.model is not None:
                                    compute_dataset_ranges()
                                
                                # EEG sync is now manual - user loads EEG file via the EEG SYNC panel
                                analysis_log("Use EEG SYNC panel to load EEG file for visualization", 'info')
                                
                                dataset_loaded = True
                        
                        if not dataset_loaded:
                            ui.notify(f'Dataset not found at {cache_path}', type='warning')
                            analysis_log(f"Dataset not found: {cache_path}", 'warning')
                    except Exception as e:
                        import traceback
                        ui.notify(f'Error: {e}', type='negative')
                        analysis_log(f"Error loading dataset: {e}", 'error')
                        analysis_log(traceback.format_exc(), 'error')
                
                ui.button('Load Dataset', on_click=load_dataset, icon='dataset').props('dense').classes('mt-2')
            
            # PLAYBACK CONTROLS
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// PLAYBACK').classes('terminal-header')
                
                with ui.row().classes('items-center gap-2 mt-2'):
                    ui.label('Speed:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    speed_select = ui.select(
                        ['0.5x', '1x', '2x', '5x', '10x'],
                        value='1x'
                    ).props('dense dark').classes('w-20')
                
                with ui.row().classes('items-center gap-2 mt-1'):
                    ui.label('Epoch:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    epoch_dur_input = ui.number(value=AS.epoch_duration, min=0.5, max=10, step=0.5).props('dense dark').classes('w-16')
                    ui.label('sec').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                    
                    def on_epoch_dur_change(e):
                        if e.value and e.value > 0:
                            AS.epoch_duration = float(e.value)
                    epoch_dur_input.on('update:model-value', on_epoch_dur_change)
                
                progress_slider = ui.slider(min=0, max=100, value=0).props('label-always').classes('w-full mt-2')
                AS.progress_slider = progress_slider
                
                sample_label = ui.label('Sample: 0 / 0').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;').classes('mt-1')
                
                with ui.row().classes('gap-2 mt-2 justify-center'):
                    def prev_sample():
                        if AS.dataset and AS.current_idx > 0:
                            AS.current_idx -= 1
                            progress_slider.set_value(AS.current_idx)
                            process_current_sample()
                    
                    def next_sample():
                        if AS.dataset and AS.current_idx < AS.total_samples - 1:
                            AS.current_idx += 1
                            progress_slider.set_value(AS.current_idx)
                            process_current_sample()
                    
                    async def toggle_play():
                        AS.playing = not AS.playing
                        if AS.playing:
                            play_btn.props('icon=pause color=negative')
                            analysis_log("Playback started", 'info')
                            # Start playback loop
                            while AS.playing and AS.dataset and AS.current_idx < AS.total_samples - 1:
                                AS.current_idx += 1
                                progress_slider.set_value(AS.current_idx)
                                process_current_sample()
                                # Speed control
                                speed_map = {'0.5x': 2.0, '1x': 1.0, '2x': 0.5, '5x': 0.2, '10x': 0.1}
                                delay = speed_map.get(speed_select.value, 1.0)
                                await asyncio.sleep(delay)
                            AS.playing = False
                            play_btn.props('icon=play_arrow color=primary')
                            analysis_log("Playback stopped", 'info')
                        else:
                            play_btn.props('icon=play_arrow color=primary')
                    
                    def stop_play():
                        AS.playing = False
                        AS.current_idx = 0
                        progress_slider.set_value(0)
                        play_btn.props('icon=play_arrow color=primary')
                        process_current_sample()
                    
                    ui.button(icon='skip_previous', on_click=prev_sample).props('round dense size=sm')
                    play_btn = ui.button(icon='play_arrow', on_click=toggle_play).props('round dense size=sm color=primary')
                    ui.button(icon='skip_next', on_click=next_sample).props('round dense size=sm')
                    ui.button(icon='stop', on_click=stop_play).props('round dense size=sm color=negative')
                
                def on_slider_change(e):
                    if AS.dataset:
                        AS.current_idx = int(e.args)
                        process_current_sample()
                
                progress_slider.on('update:model-value', on_slider_change)
            
            # SAMPLE INFO
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// CURRENT SAMPLE').classes('terminal-header')
                AS.sample_info_container = ui.column().classes('w-full mt-2 gap-1')
                with AS.sample_info_container:
                    ui.label('No sample loaded').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
            
            # LOG
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('// LOG').classes('terminal-header')
                with ui.scroll_area().classes('w-full').style('height: 100px; background: #050505; border-radius: 4px;'):
                    AS.log_container = ui.column().classes('w-full p-2 gap-0')
                    with AS.log_container:
                        ui.label('Ready. Load a model and dataset.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.7rem;')
        
        # RIGHT PANEL: Visualizations
        with ui.column().classes('flex-1 gap-3').style('min-height: 0; overflow-y: auto;'):
            
            # EEG VISUALIZATION PANEL (synchronized with dataset playback)
            with ui.card().classes('dark-card p-3 w-full'):
                with ui.row().classes('items-center gap-2 mb-2'):
                    ui.label('▌EEG SYNC').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
                    eeg_sync_toggle = ui.switch('Sync', value=False).props('dense size=xs')
                    eeg_sync_info = ui.label('Load EEG to sync').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.7rem; margin-left: auto;')
                    
                    def on_sync_toggle(e):
                        AS.eeg_sync_enabled = e.args
                    eeg_sync_toggle.on('update:model-value', on_sync_toggle)
                
                # EEG File selector
                with ui.row().classes('items-center gap-2 w-full'):
                    eeg_path_input = ui.input(
                        value=str(EEG_CLEAN_DIR / 'DMT' / 'S01-DMT_ICA_pruned.set'),
                        placeholder='Path to EEG file'
                    ).props('dense dark').classes('flex-1').style('font-size: 0.7rem;')
                    
                    def load_sync_eeg():
                        """Load EEG file for sync visualization."""
                        try:
                            path = Path(eeg_path_input.value.strip())
                            if not path.exists():
                                ui.notify(f'File not found: {path}', type='negative')
                                return
                            
                            AS.eeg_data = load_eeg_file(path)
                            AS.current_eeg_file = str(path)
                            AS.eeg_view_start = 0.0
                            
                            # Get EEG channel names (native strings)
                            eeg_chs = [str(ch) for ch, t in AS.eeg_data.channel_types.items() if t == 'eeg']
                            AS.eeg_channels = eeg_chs[:10]  # Limit to 10 channels
                            
                            # Enable sync
                            AS.eeg_sync_enabled = True
                            eeg_sync_toggle.value = True
                            
                            # Update info
                            total_epochs = int(AS.eeg_data.duration_sec // AS.epoch_duration)
                            eeg_sync_info.set_text(f'{path.stem} | {total_epochs} epochs @ {AS.epoch_duration}s')
                            
                            # Update plot
                            AS.eeg_plot.figure = make_analysis_eeg_fig()
                            AS.eeg_plot.update()
                            
                            ui.notify(f'EEG loaded: {path.name}', type='positive')
                        except Exception as e:
                            ui.notify(f'Error: {e}', type='negative')
                            analysis_log(f"Error loading EEG: {e}", 'error')
                    
                    ui.button(icon='folder_open', on_click=load_sync_eeg).props('dense flat size=sm')
                
                # EEG Plot
                def make_analysis_eeg_fig():
                    """Create EEG figure for analysis sync visualization."""
                    fig = go.Figure()
                    
                    if AS.eeg_data is None or not AS.eeg_channels:
                        fig.add_annotation(
                            text="Select EEG file and click folder icon to load",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(color=THEME_TEXT_DIM, size=11)
                        )
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=150,
                            margin=dict(l=10, r=10, t=10, b=10),
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False)
                        )
                        return fig
                    
                    try:
                        # Get data for current epoch
                        channels = AS.eeg_channels[:10]
                        data, times, chs = get_channel_data(
                            AS.eeg_data, channels,
                            AS.eeg_view_start, AS.epoch_duration
                        )
                        # Process data (filter and convert to µV)
                        data = signal_process_data(data, AS.eeg_data.sfreq) * 1e6
                        n = len(chs)
                        
                        # Spacing for stacked display
                        spacing_factor = 0.35 if n <= 8 else 0.25
                        signal_color = 'rgba(0, 255, 136, 0.7)'
                        
                        y_ticks, y_labels = [], []
                        for i in range(n):
                            offset = (n - 1 - i)
                            y = data[i].copy()
                            y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
                            std_val = np.std(y)
                            if std_val < 1e-10:
                                y_norm = np.zeros_like(y) + offset
                            else:
                                y_norm = (y - np.mean(y)) / std_val * spacing_factor + offset
                            
                            y_ticks.append(offset)
                            # Ensure channel name is native string
                            y_labels.append(str(chs[i]))
                            
                            fig.add_trace(go.Scatter(
                                x=times, y=y_norm, name=str(chs[i]),
                                line=dict(color=signal_color, width=1),
                                hovertemplate=f'{str(chs[i])}: %{{customdata:.1f}} µV<extra></extra>',
                                customdata=y
                            ))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=150,
                            margin=dict(l=50, r=10, t=5, b=30),
                            xaxis=dict(
                                title=dict(text='TIME [s]', font=dict(size=9, color=THEME_PRIMARY)),
                                gridcolor='rgba(0,255,136,0.08)',
                                tickfont=dict(size=8, color=THEME_TEXT_DIM),
                                fixedrange=True
                            ),
                            yaxis=dict(
                                tickmode='array', tickvals=y_ticks, ticktext=y_labels,
                                range=[-0.5, n - 0.5],
                                gridcolor='rgba(0,255,136,0.03)',
                                tickfont=dict(size=8, color=THEME_PRIMARY),
                                fixedrange=True
                            ),
                            showlegend=False,
                            font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                        )
                    except Exception as e:
                        fig.add_annotation(
                            text=f"Error: {e}", x=0.5, y=0.5, showarrow=False,
                            font=dict(color=THEME_ERROR, size=10)
                        )
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=150,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                    
                    return fig
                
                AS.eeg_plot = ui.plotly(make_analysis_eeg_fig()).classes('w-full').style('height: 150px;')
            
            # ROW 1: KURAMOTO ORDER PARAMETER
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('▌KURAMOTO ORDER PARAMETER (r)').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;').classes('mb-1')
                
                def make_kuramoto_fig():
                    """Create Kuramoto order parameter visualization - Original vs Proxy."""
                    fig = go.Figure()
                    
                    n_total = AS.total_samples if AS.total_samples > 0 else 100
                    
                    # Metastability band (±1 std around mean) - subtle gray
                    if AS.kuramoto_avg > 0 and AS.kuramoto_std > 0:
                        fig.add_trace(go.Scatter(
                            x=list(range(n_total)) + list(range(n_total-1, -1, -1)),
                            y=[AS.kuramoto_avg + AS.kuramoto_std] * n_total + [AS.kuramoto_avg - AS.kuramoto_std] * n_total,
                            fill='toself',
                            fillcolor='rgba(128, 128, 128, 0.15)',
                            line=dict(width=0),
                            name='±1σ',
                            showlegend=True,
                            hoverinfo='skip'
                        ))
                    
                    # Mean line (thin gray dashed)
                    fig.add_trace(go.Scatter(
                        x=[0, n_total-1],
                        y=[AS.kuramoto_avg, AS.kuramoto_avg],
                        mode='lines',
                        line=dict(color='rgba(180, 180, 180, 0.5)', width=1, dash='dot'),
                        name=f'μ={AS.kuramoto_avg:.2f}',
                        hoverinfo='skip'
                    ))
                    
                    # Original Kuramoto (from dataset) - blue line
                    if AS.kuramoto_history:
                        indices, values = zip(*AS.kuramoto_history)
                        fig.add_trace(go.Scatter(
                            x=indices,
                            y=values,
                            mode='lines+markers',
                            line=dict(color='rgba(100, 180, 255, 0.9)', width=1.5),
                            marker=dict(size=3, color='rgba(100, 180, 255, 0.9)'),
                            name='Original',
                            hovertemplate='idx:%{x}<br>r=%{y:.3f}<extra></extra>'
                        ))
                    
                    # Reconstructed Kuramoto proxy (from VAE) - pink/magenta line
                    if AS.kuramoto_proxy_history:
                        proxy_indices, proxy_values = zip(*AS.kuramoto_proxy_history)
                        fig.add_trace(go.Scatter(
                            x=proxy_indices,
                            y=proxy_values,
                            mode='lines+markers',
                            line=dict(color='rgba(244, 114, 182, 0.9)', width=1.5),
                            marker=dict(size=3, color='rgba(244, 114, 182, 0.9)'),
                            name='Proxy (VAE)',
                            hovertemplate='idx:%{x}<br>proxy=%{y:.3f}<extra></extra>'
                        ))
                    
                    # Current position marker
                    if AS.kuramoto_history:
                        indices, values = zip(*AS.kuramoto_history)
                        if len(indices) > 0:
                            current_idx = indices[-1]
                            fig.add_trace(go.Scatter(
                                x=[current_idx, current_idx],
                                y=[0, 1],
                                mode='lines',
                                line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dash'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(12,12,12,1)',
                        height=145,
                        margin=dict(l=45, r=15, t=5, b=35),
                        xaxis=dict(
                            title=dict(text='Sample', font=dict(size=9)),
                            range=[0, n_total], 
                            gridcolor='rgba(80,80,80,0.3)',
                            tickfont=dict(size=8)
                        ),
                        yaxis=dict(
                            title=dict(text='r', font=dict(size=9)),
                            range=[0, 1], 
                            gridcolor='rgba(80,80,80,0.3)',
                            tickfont=dict(size=8),
                            dtick=0.2
                        ),
                        legend=dict(
                            orientation='h', 
                            y=1.02, 
                            x=0,
                            font=dict(size=8),
                            bgcolor='rgba(0,0,0,0)'
                        ),
                        font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT_DIM),
                        hovermode='x unified'
                    )
                    return fig
                
                kuramoto_plot = ui.plotly(make_kuramoto_fig()).classes('w-full').style('height: 145px;')
                AS.activation_plots['kuramoto'] = kuramoto_plot
            
            # ROW 1.5: SYNC NETWORK VISUALIZATION (Original vs Reconstructed)
            with ui.row().classes('gap-3 w-full'):
                # ORIGINAL SYNC NETWORK
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-1'):
                        ui.label('▌ORIGINAL SYNC').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
                        sync_threshold_slider = ui.slider(min=0.1, max=0.8, step=0.1, value=0.3).props('label-always').classes('w-24')
                        ui.label('threshold').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                    
                    def make_original_sync_fig():
                        """Create original synchronization network visualization."""
                        # Default empty figure
                        if AS.current_sample is None or not hasattr(AS.current_sample, 'edge_attr'):
                            fig = go.Figure()
                            fig.add_annotation(
                                text="Load sample to visualize",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(color=THEME_TEXT_DIM, size=10)
                            )
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(8,8,8,1)',
                                plot_bgcolor='rgba(12,12,12,1)',
                                height=200,
                                margin=dict(l=5, r=5, t=10, b=5),
                                xaxis=dict(showticklabels=False, showgrid=False),
                                yaxis=dict(showticklabels=False, showgrid=False)
                            )
                            return fig
                        
                        # Build sync matrix from original edges
                        batch = AS.current_sample
                        num_nodes = batch.x.shape[0]
                        sync_matrix = build_sync_matrix_from_edges(
                            batch.edge_index, batch.edge_attr, num_nodes
                        )
                        
                        threshold = sync_threshold_slider.value if sync_threshold_slider else 0.3
                        return make_sync_network_fig(
                            sync_matrix, num_nodes,
                            title="Original PLV Network",
                            primary_color=THEME_PRIMARY,
                            edge_colorscale='Viridis',
                            threshold=threshold
                        )
                    
                    original_sync_plot = ui.plotly(make_original_sync_fig()).classes('w-full').style('height: 200px;')
                    AS.activation_plots['original_sync'] = original_sync_plot
                
                # RECONSTRUCTED SYNC NETWORK
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-1'):
                        ui.label('▌RECONSTRUCTED SYNC').style('color:#f472b6; font-family: JetBrains Mono; font-size: 0.75rem;')
                        ui.label('(VAE decoder)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                    
                    def make_recon_sync_fig():
                        """Create reconstructed synchronization network visualization.
                        
                        Uses decoder attention weights if available (GAT decoder),
                        otherwise falls back to edge_attr_recon.
                        """
                        # Default empty figure
                        if AS.current_sample is None:
                            fig = go.Figure()
                            fig.add_annotation(
                                text="No reconstruction available",
                                x=0.5, y=0.5, showarrow=False,
                                font=dict(color=THEME_TEXT_DIM, size=10)
                            )
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(8,8,8,1)',
                                plot_bgcolor='rgba(12,12,12,1)',
                                height=200,
                                margin=dict(l=5, r=5, t=10, b=5),
                                xaxis=dict(showticklabels=False, showgrid=False),
                                yaxis=dict(showticklabels=False, showgrid=False)
                            )
                            return fig
                        
                        batch = AS.current_sample
                        num_nodes = batch.x.shape[0]
                        sync_matrix = None
                        title_suffix = ""
                        
                        # Priority 1: Use decoder attention weights if available (GAT decoder)
                        decoder_att_keys = [k for k in AS.attention_weights.keys() if k.startswith('decoder_layer_')]
                        if decoder_att_keys:
                            try:
                                # Use last decoder layer attention as sync proxy
                                last_layer_key = sorted(decoder_att_keys)[-1]
                                att_data = AS.attention_weights[last_layer_key]
                                
                                # Safe conversion to numpy (handle both CPU and CUDA tensors)
                                ei = att_data['edge_index']
                                at = att_data['attention']
                                edge_index = ei.detach().cpu().numpy() if hasattr(ei, 'detach') else (ei.cpu().numpy() if hasattr(ei, 'cpu') else np.array(ei))
                                attention = at.detach().cpu().numpy() if hasattr(at, 'detach') else (at.cpu().numpy() if hasattr(at, 'cpu') else np.array(at))
                                
                                # Average across heads if multi-head
                                if attention.ndim > 1:
                                    attention = attention.mean(axis=1)
                                
                                # Build sync matrix from attention
                                sync_matrix = np.zeros((num_nodes, num_nodes))
                                for e_idx in range(edge_index.shape[1]):
                                    src, dst = edge_index[0, e_idx], edge_index[1, e_idx]
                                    if src < num_nodes and dst < num_nodes:
                                        sync_matrix[int(src), int(dst)] = attention[e_idx]
                                        sync_matrix[int(dst), int(src)] = attention[e_idx]
                                
                                title_suffix = " (GAT attention)"
                            except Exception:
                                sync_matrix = None  # Fall back to other methods
                        
                        # Priority 2: Use edge_attr_recon if available and has variance
                        if sync_matrix is None and AS.current_recon is not None and 'edge_attr_recon' in AS.current_recon:
                            recon_attr = AS.current_recon['edge_attr_recon']
                            if hasattr(recon_attr, 'cpu'):
                                recon_np = recon_attr.cpu().numpy()
                            else:
                                recon_np = recon_attr
                            
                            recon_std = np.std(recon_np)
                            
                            if recon_std >= 0.001:
                                # Has variance - use it
                                sync_matrix = build_sync_matrix_from_edges(
                                    batch.edge_index, AS.current_recon['edge_attr_recon'], num_nodes,
                                    normalize=True
                                )
                                title_suffix = " (edge recon)"
                        
                        # Handle visualization
                        if sync_matrix is None:
                            # No data at all - show empty with message
                            fig = go.Figure()
                            theta_head = np.linspace(0, 2 * np.pi, 100)
                            fig.add_trace(go.Scatter(
                                x=np.cos(theta_head), y=np.sin(theta_head), mode='lines',
                                line=dict(color='rgba(244,114,182,0.4)', width=1.5), 
                                showlegend=False, hoverinfo='skip'
                            ))
                            positions = generate_circular_positions(num_nodes)
                            node_x = [positions[i][0] for i in range(num_nodes)]
                            node_y = [positions[i][1] for i in range(num_nodes)]
                            fig.add_trace(go.Scatter(
                                x=node_x, y=node_y, mode='markers',
                                marker=dict(size=8, color='#f472b6', opacity=0.5),
                                showlegend=False, hoverinfo='skip'
                            ))
                            fig.add_annotation(
                                text="No reconstruction data", x=0.5, y=0, xref='paper', yref='paper',
                                showarrow=False, font=dict(color=THEME_TEXT_DIM, size=10)
                            )
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(8,8,8,1)',
                                plot_bgcolor='rgba(12,12,12,1)',
                                height=200,
                                margin=dict(l=5, r=5, t=25, b=25),
                                title=dict(text="Reconstructed Sync", font=dict(size=10, color='#f472b6'), x=0.5),
                                xaxis=dict(range=[-1.2, 1.2], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True),
                                yaxis=dict(range=[-0.95, 1.15], showgrid=False, zeroline=False, showticklabels=False, fixedrange=True, scaleanchor='x')
                            )
                            return fig
                        
                        # Check if has variance
                        has_variance = np.std(sync_matrix) > 0.001
                        threshold = sync_threshold_slider.value if sync_threshold_slider else 0.3
                        
                        # If uniform, use threshold=0 to show all connections equally
                        if not has_variance:
                            # Set all values to 0.5 to show uniform connections
                            sync_matrix = np.where(sync_matrix > 0, 0.5, 0)
                            title_suffix = " ⚠uniform"
                            threshold = 0.1  # Lower threshold to show all
                        
                        return make_sync_network_fig(
                            sync_matrix, num_nodes,
                            title=f"Recon PLV{title_suffix}",
                            primary_color='#f472b6',
                            edge_colorscale='RdPu',
                            threshold=threshold
                        )
                    
                    recon_sync_plot = ui.plotly(make_recon_sync_fig()).classes('w-full').style('height: 200px;')
                    AS.activation_plots['recon_sync'] = recon_sync_plot
                
                # Update both plots when threshold changes
                def on_threshold_change(e):
                    try:
                        original_sync_plot.figure = make_original_sync_fig()
                        original_sync_plot.update()
                        recon_sync_plot.figure = make_recon_sync_fig()
                        recon_sync_plot.update()
                    except Exception:
                        pass
                
                sync_threshold_slider.on('update:model-value', on_threshold_change)
            
            # ROW 2: ENCODER + DECODER ACTIVATIONS
            with ui.row().classes('gap-3 w-full'):
                
                # ENCODER ACTIVATIONS
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('▌ENCODER').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
                        encoder_layer_select = ui.select([], value=None).props('dense dark').classes('w-24').style('font-size: 0.7rem;')
                        encoder_channel_select = ui.select([], value=None).props('dense dark').classes('w-20').style('font-size: 0.7rem;')
                    
                    def update_encoder_selectors():
                        """Update encoder layer/channel selectors based on available activations."""
                        layers = [k for k in AS.activations.keys() if k.startswith('encoder.') or k == 'stem']
                        if layers:
                            encoder_layer_select.options = layers
                            if encoder_layer_select.value not in layers:
                                encoder_layer_select.value = layers[0]
                            # Update channel selector
                            if encoder_layer_select.value and encoder_layer_select.value in AS.activations:
                                act = AS.activations[encoder_layer_select.value]
                                if len(act.shape) >= 2:
                                    n_channels = act.shape[1] if len(act.shape) == 4 else act.shape[0]
                                    ch_options = [f'Ch {i}' for i in range(min(n_channels, 64))]
                                    encoder_channel_select.options = ch_options
                                    if not encoder_channel_select.value or encoder_channel_select.value not in ch_options:
                                        encoder_channel_select.value = ch_options[0] if ch_options else None
                    
                    def make_encoder_fig():
                        """Create encoder activation visualization."""
                        fig = go.Figure()
                        
                        layer_name = encoder_layer_select.value
                        channel_str = encoder_channel_select.value
                        
                        if layer_name and layer_name in AS.activations and channel_str:
                            act = AS.activations[layer_name]
                            channel_idx = int(channel_str.split(' ')[1]) if channel_str else 0
                            
                            # Handle different activation shapes - ensure CPU before numpy
                            if hasattr(act, 'cpu'):
                                act = act.cpu()
                            if len(act.shape) == 4:  # [B, C, H, W]
                                act_2d = act[0, channel_idx].numpy()
                            elif len(act.shape) == 3:  # [C, H, W]
                                act_2d = act[channel_idx].numpy()
                            elif len(act.shape) == 2:  # [H, W] or [B, features]
                                act_2d = act[0].numpy() if act.shape[0] == 1 else act.numpy()
                            else:
                                act_2d = act.numpy().flatten().reshape(-1, 1)
                            
                            fig.add_trace(go.Heatmap(
                                z=act_2d,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(len=0.8, thickness=10)
                            ))
                        else:
                            fig.add_annotation(text="No activations", x=0.5, y=0.5, showarrow=False,
                                             font=dict(color=THEME_TEXT_DIM))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=180,
                            margin=dict(l=10, r=40, t=5, b=10),
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
                            font=dict(family='JetBrains Mono', size=8, color=THEME_TEXT)
                        )
                        return fig
                    
                    encoder_plot = ui.plotly(make_encoder_fig()).classes('w-full').style('height: 180px;')
                    AS.activation_plots['encoder'] = encoder_plot
                    
                    def on_encoder_layer_change(e):
                        update_encoder_selectors()
                        encoder_plot.figure = make_encoder_fig()
                        encoder_plot.update()
                    
                    def on_encoder_channel_change(e):
                        encoder_plot.figure = make_encoder_fig()
                        encoder_plot.update()
                    
                    encoder_layer_select.on('update:model-value', on_encoder_layer_change)
                    encoder_channel_select.on('update:model-value', on_encoder_channel_change)
                
                # DECODER ACTIVATIONS
                with ui.card().classes('dark-card p-3 flex-1'):
                    with ui.row().classes('items-center gap-2 mb-2'):
                        ui.label('▌DECODER').style('color:#f472b6; font-family: JetBrains Mono; font-size: 0.75rem;')
                        decoder_layer_select = ui.select([], value=None).props('dense dark').classes('w-24').style('font-size: 0.7rem;')
                        decoder_channel_select = ui.select([], value=None).props('dense dark').classes('w-20').style('font-size: 0.7rem;')
                    
                    def update_decoder_selectors():
                        """Update decoder layer/channel selectors based on available activations."""
                        layers = [k for k in AS.activations.keys() if k.startswith('decoder.') or k == 'output']
                        if layers:
                            decoder_layer_select.options = layers
                            if decoder_layer_select.value not in layers:
                                decoder_layer_select.value = layers[0]
                            # Update channel selector
                            if decoder_layer_select.value and decoder_layer_select.value in AS.activations:
                                act = AS.activations[decoder_layer_select.value]
                                if len(act.shape) >= 2:
                                    n_channels = act.shape[1] if len(act.shape) == 4 else act.shape[0]
                                    ch_options = [f'Ch {i}' for i in range(min(n_channels, 64))]
                                    decoder_channel_select.options = ch_options
                                    if not decoder_channel_select.value or decoder_channel_select.value not in ch_options:
                                        decoder_channel_select.value = ch_options[0] if ch_options else None
                    
                    def make_decoder_fig():
                        """Create decoder activation visualization."""
                        fig = go.Figure()
                        
                        layer_name = decoder_layer_select.value
                        channel_str = decoder_channel_select.value
                        
                        if layer_name and layer_name in AS.activations and channel_str:
                            act = AS.activations[layer_name]
                            channel_idx = int(channel_str.split(' ')[1]) if channel_str else 0
                            
                            # Handle different activation shapes - ensure CPU before numpy
                            if hasattr(act, 'cpu'):
                                act = act.cpu()
                            if len(act.shape) == 4:  # [B, C, H, W]
                                act_2d = act[0, channel_idx].numpy()
                            elif len(act.shape) == 3:  # [C, H, W]
                                act_2d = act[channel_idx].numpy()
                            elif len(act.shape) == 2:
                                act_2d = act[0].numpy() if act.shape[0] == 1 else act.numpy()
                            else:
                                act_2d = act.numpy().flatten().reshape(-1, 1)
                            
                            fig.add_trace(go.Heatmap(
                                z=act_2d,
                                colorscale='Magma',
                                showscale=True,
                                colorbar=dict(len=0.8, thickness=10)
                            ))
                        else:
                            fig.add_annotation(text="No activations", x=0.5, y=0.5, showarrow=False,
                                             font=dict(color=THEME_TEXT_DIM))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=180,
                            margin=dict(l=10, r=40, t=5, b=10),
                            xaxis=dict(showticklabels=False, showgrid=False),
                            yaxis=dict(showticklabels=False, showgrid=False, scaleanchor='x'),
                            font=dict(family='JetBrains Mono', size=8, color=THEME_TEXT)
                        )
                        return fig
                    
                    decoder_plot = ui.plotly(make_decoder_fig()).classes('w-full').style('height: 180px;')
                    AS.activation_plots['decoder'] = decoder_plot
                    
                    def on_decoder_layer_change(e):
                        update_decoder_selectors()
                        decoder_plot.figure = make_decoder_fig()
                        decoder_plot.update()
                    
                    def on_decoder_channel_change(e):
                        decoder_plot.figure = make_decoder_fig()
                        decoder_plot.update()
                    
                    decoder_layer_select.on('update:model-value', on_decoder_layer_change)
                    decoder_channel_select.on('update:model-value', on_decoder_channel_change)
            
            # ROW 3: LATENT SPACE (full width, larger)
            with ui.card().classes('dark-card p-3 w-full'):
                with ui.row().classes('items-center gap-4 mb-1'):
                    ui.label('▌LATENT SPACE (PCA)').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                    ui.label('').bind_text_from(AS, 'latent_codes', lambda x: f'n={len(x)}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                
                def make_latent_fig():
                    """Create latent space PCA visualization - scientific style."""
                    fig = go.Figure()
                    
                    X_pca, labels = compute_latent_pca()
                    if X_pca is not None and len(X_pca) > 0:
                        # Scientific color palette (colorblind-friendly)
                        n_classes = len(np.unique(labels))
                        # Muted scientific colors
                        colors = ['#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', 
                                 '#AA3377', '#BBBBBB', '#44AA99', '#999933', '#882255']
                        
                        for label in np.unique(labels):
                            mask = labels == label
                            label_name = AS.class_names[int(label)] if int(label) < len(AS.class_names) else f'C{int(label)}'
                            fig.add_trace(go.Scatter(
                                x=X_pca[mask, 0], y=X_pca[mask, 1],
                                mode='markers',
                                marker=dict(size=5, color=colors[int(label) % len(colors)], opacity=0.6),
                                name=label_name,
                                hovertemplate=f'{label_name}<br>PC1=%{{x:.2f}}<br>PC2=%{{y:.2f}}<extra></extra>'
                            ))
                        
                        # Current point - simple ring marker
                        if len(X_pca) > 0:
                            fig.add_trace(go.Scatter(
                                x=[X_pca[-1, 0]], y=[X_pca[-1, 1]],
                                mode='markers',
                                marker=dict(size=10, color='rgba(255,255,255,0)', 
                                           line=dict(width=2, color='white')),
                                name='current',
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                    else:
                        fig.add_annotation(text="Process samples to visualize latent space", 
                                         x=0.5, y=0.5, showarrow=False,
                                         font=dict(color=THEME_TEXT_DIM, size=10))
                    
                    lat_range = AS.axis_ranges['latent']
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(12,12,12,1)',
                        height=280,
                        margin=dict(l=45, r=15, t=5, b=40),
                        xaxis=dict(
                            title=dict(text='PC1', font=dict(size=9)),
                            gridcolor='rgba(80,80,80,0.3)', 
                            range=[lat_range['x_min'], lat_range['x_max']],
                            tickfont=dict(size=8)
                        ),
                        yaxis=dict(
                            title=dict(text='PC2', font=dict(size=9)),
                            gridcolor='rgba(80,80,80,0.3)',
                            range=[lat_range['y_min'], lat_range['y_max']],
                            tickfont=dict(size=8)
                        ),
                        legend=dict(
                            orientation='h', y=-0.12, x=0.5, xanchor='center', 
                            font=dict(size=8), bgcolor='rgba(0,0,0,0)'
                        ),
                        font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT_DIM)
                    )
                    return fig
                
                latent_plot = ui.plotly(make_latent_fig()).classes('w-full').style('height: 280px;')
                AS.latent_plot = latent_plot
            
            # ROW 3: ATTENTION WEIGHTS (per layer)
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('▌ATTENTION WEIGHTS (GAT)').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;').classes('mb-1')
                
                def make_attention_fig():
                    """Create attention heatmap visualization using REAL GAT attention weights."""
                    from plotly.subplots import make_subplots
                    
                    n_nodes = AS.model_params.get('num_nodes', 24) if AS.model_params else 24
                    
                    # Check how many layers have attention
                    n_layers = len(AS.attention_weights) if AS.attention_weights else 0
                    
                    if n_layers == 0:
                        # Fallback: no attention data yet
                        fig = go.Figure()
                        fig.add_annotation(text="No attention data", x=0.5, y=0.5, showarrow=False)
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=150,
                            margin=dict(l=10, r=10, t=10, b=10)
                        )
                        return fig
                    
                    # Create subplot for each layer
                    fig = make_subplots(rows=1, cols=n_layers, 
                                       subplot_titles=[f'Layer {i+1}' for i in range(n_layers)])
                    
                    for i, (layer_name, att_data) in enumerate(AS.attention_weights.items()):
                        if 'attention' in att_data and 'edge_index' in att_data:
                            edge_index = att_data['edge_index'].detach().cpu().numpy()
                            alpha = att_data['attention'].detach().cpu().numpy()
                            
                            # Build attention matrix from sparse edge data
                            # alpha shape: (num_edges, num_heads) - average across heads
                            if len(alpha.shape) > 1:
                                alpha_avg = alpha.mean(axis=1)
                            else:
                                alpha_avg = alpha
                            
                            # Create dense attention matrix
                            actual_nodes = min(n_nodes, int(edge_index.max()) + 1) if edge_index.size > 0 else n_nodes
                            attn_matrix = np.zeros((actual_nodes, actual_nodes))
                            
                            for e_idx in range(edge_index.shape[1]):
                                src, tgt = edge_index[0, e_idx], edge_index[1, e_idx]
                                if src < actual_nodes and tgt < actual_nodes:
                                    attn_matrix[src, tgt] = alpha_avg[e_idx] if e_idx < len(alpha_avg) else 0
                            
                            fig.add_trace(go.Heatmap(
                                z=attn_matrix,
                                colorscale='Viridis',
                                showscale=(i == n_layers - 1),  # Only show colorbar on last
                                zmin=0,
                                zmax=1,
                                colorbar=dict(title='α', len=0.8) if i == n_layers - 1 else None
                            ), row=1, col=i+1)
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(8,8,8,1)',
                        height=150,
                        margin=dict(l=30, r=50, t=25, b=25),
                        font=dict(family='JetBrains Mono', size=8, color=THEME_TEXT)
                    )
                    
                    # Update axes for all subplots
                    for i in range(n_layers):
                        fig.update_xaxes(title_text='Target' if i == 0 else '', row=1, col=i+1)
                        fig.update_yaxes(title_text='Source' if i == 0 else '', row=1, col=i+1)
                    
                    return fig
                
                attention_plot = ui.plotly(make_attention_fig()).classes('w-full').style('height: 180px;')
                AS.attention_plots['main'] = attention_plot
            
            # BOTTOM ROW: Reconstruction
            with ui.row().classes('gap-3 w-full'):
                
                # ORIGINAL
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌ORIGINAL').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                    
                    def make_original_fig():
                        fig = go.Figure()
                        
                        if AS.model_type == 'image' and AS.current_sample is not None:
                            # Image model: show RGB image
                            img = AS.current_sample.cpu().numpy()
                            if img.ndim == 4:
                                img = img[0]  # Remove batch dim
                            # [C, H, W] -> [H, W, C]
                            if img.shape[0] == 3:
                                img = np.transpose(img, (1, 2, 0))
                            # Clip to [0, 1] for display
                            img = np.clip(img, 0, 1)
                            # Resize for faster display if too large
                            display_size = min(256, img.shape[0])
                            if img.shape[0] > display_size:
                                from scipy.ndimage import zoom
                                scale = display_size / img.shape[0]
                                img = zoom(img, (scale, scale, 1), order=1)
                            fig.add_trace(go.Image(z=(img * 255).astype(np.uint8)))
                            fig.update_layout(height=200)
                        elif AS.current_sample is not None and hasattr(AS.current_sample, 'x'):
                            # Graph model: show heatmap
                            feat_range = AS.axis_ranges['node_features']
                            x = AS.current_sample.x.cpu().numpy()
                            fig.add_trace(go.Heatmap(
                                z=x[:24, :].T, 
                                colorscale='Viridis', 
                                showscale=False,
                                zmin=feat_range['min'],
                                zmax=feat_range['max']
                            ))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=200 if AS.model_type == 'image' else 150,
                            margin=dict(l=10, r=10, t=10, b=10) if AS.model_type == 'image' else dict(l=30, r=10, t=10, b=30),
                            xaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Nodes'),
                            yaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Features'),
                            font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                        )
                        return fig
                    
                    original_plot = ui.plotly(make_original_fig()).classes('w-full').style('height: 200px;')
                    AS.activation_plots['original'] = original_plot
                
                # RECONSTRUCTED
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌RECONSTRUCTED').style('color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                    
                    def make_recon_fig():
                        fig = go.Figure()
                        
                        if AS.model_type == 'image' and AS.current_recon is not None:
                            # Image model: show RGB reconstruction
                            recon_key = 'reconstruction' if 'reconstruction' in AS.current_recon else 'x_recon'
                            if recon_key in AS.current_recon:
                                img = AS.current_recon[recon_key].cpu().numpy()
                                if img.ndim == 4:
                                    img = img[0]  # Remove batch dim
                                # [C, H, W] -> [H, W, C]
                                if img.shape[0] == 3:
                                    img = np.transpose(img, (1, 2, 0))
                                img = np.clip(img, 0, 1)
                                # Resize for display
                                display_size = min(256, img.shape[0])
                                if img.shape[0] > display_size:
                                    from scipy.ndimage import zoom
                                    scale = display_size / img.shape[0]
                                    img = zoom(img, (scale, scale, 1), order=1)
                                fig.add_trace(go.Image(z=(img * 255).astype(np.uint8)))
                                fig.update_layout(height=200)
                        elif AS.current_recon is not None and 'x_recon' in AS.current_recon:
                            # Graph model: show heatmap
                            feat_range = AS.axis_ranges['node_features']
                            x_recon = AS.current_recon['x_recon'].cpu().numpy()
                            fig.add_trace(go.Heatmap(
                                z=x_recon[:24, :].T, 
                                colorscale='Viridis', 
                                showscale=False,
                                zmin=feat_range['min'],
                                zmax=feat_range['max']
                            ))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=200 if AS.model_type == 'image' else 150,
                            margin=dict(l=10, r=10, t=10, b=10) if AS.model_type == 'image' else dict(l=30, r=10, t=10, b=30),
                            xaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Nodes'),
                            yaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Features'),
                            font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                        )
                        return fig
                    
                    recon_plot = ui.plotly(make_recon_fig()).classes('w-full').style('height: 200px;')
                    AS.recon_plot = recon_plot
                
                # DIFFERENCE
                with ui.card().classes('dark-card p-3 flex-1'):
                    ui.label('▌DIFFERENCE').style(f'color:{THEME_ERROR}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                    
                    def make_diff_fig():
                        fig = go.Figure()
                        
                        if AS.model_type == 'image' and AS.current_sample is not None and AS.current_recon is not None:
                            # Image model: show difference image
                            recon_key = 'reconstruction' if 'reconstruction' in AS.current_recon else 'x_recon'
                            if recon_key in AS.current_recon:
                                orig = AS.current_sample.cpu().numpy()
                                recon = AS.current_recon[recon_key].cpu().numpy()
                                if orig.ndim == 4:
                                    orig = orig[0]
                                if recon.ndim == 4:
                                    recon = recon[0]
                                # Compute absolute difference
                                diff = np.abs(orig - recon)
                                # [C, H, W] -> [H, W, C]
                                if diff.shape[0] == 3:
                                    diff = np.transpose(diff, (1, 2, 0))
                                # Amplify for visibility and convert to grayscale-ish
                                diff_gray = np.mean(diff, axis=2)
                                diff_display = np.stack([diff_gray, diff_gray * 0.3, diff_gray * 0.3], axis=2)
                                diff_display = np.clip(diff_display * 3, 0, 1)  # Amplify
                                # Resize
                                display_size = min(256, diff_display.shape[0])
                                if diff_display.shape[0] > display_size:
                                    from scipy.ndimage import zoom
                                    scale = display_size / diff_display.shape[0]
                                    diff_display = zoom(diff_display, (scale, scale, 1), order=1)
                                fig.add_trace(go.Image(z=(diff_display * 255).astype(np.uint8)))
                                fig.update_layout(height=200)
                        elif AS.current_sample is not None and AS.current_recon is not None:
                            # Graph model
                            if hasattr(AS.current_sample, 'x') and 'x_recon' in AS.current_recon:
                                feat_range = AS.axis_ranges['node_features']
                                diff_max = (feat_range['max'] - feat_range['min']) * 0.5
                                x = AS.current_sample.x.cpu().numpy()
                                x_recon = AS.current_recon['x_recon'].cpu().numpy()
                                diff = np.abs(x - x_recon)
                                fig.add_trace(go.Heatmap(
                                    z=diff[:24, :].T, 
                                    colorscale='Reds', 
                                    showscale=True,
                                    zmin=0,
                                    zmax=diff_max,
                                    colorbar=dict(title='|Δ|', len=0.8)
                                ))
                        
                        fig.update_layout(
                            template='plotly_dark',
                            paper_bgcolor='rgba(8,8,8,1)',
                            plot_bgcolor='rgba(8,8,8,1)',
                            height=200 if AS.model_type == 'image' else 150,
                            margin=dict(l=10, r=10, t=10, b=10) if AS.model_type == 'image' else dict(l=30, r=50, t=10, b=30),
                            xaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Nodes'),
                            yaxis=dict(showticklabels=False, showgrid=False) if AS.model_type == 'image' else dict(title='Features'),
                            font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                        )
                        return fig
                    
                    diff_plot = ui.plotly(make_diff_fig()).classes('w-full').style('height: 200px;')
                    AS.activation_plots['diff'] = diff_plot
        
        # Process sample and update all plots
        def process_current_sample():
            if not AS.dataset:
                analysis_log("No dataset loaded", 'warning')
                return
            if not AS.model:
                analysis_log("No model loaded", 'warning')
                return
            
            try:
                sample = AS.dataset[AS.current_idx]
                result = process_sample(sample)
                
                if result is None:
                    analysis_log(f"Failed to process sample {AS.current_idx}", 'error')
                    return
                
                # Update sample info based on model type
                if AS.sample_info_container:
                    AS.sample_info_container.clear()
                    with AS.sample_info_container:
                        ui.label(f'Index: {AS.current_idx}').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                        
                        if AS.model_type == 'image':
                            # Image model info
                            if AS.current_label is not None:
                                lbl = AS.current_label
                                class_name = AS.class_names[lbl] if lbl < len(AS.class_names) else str(lbl)
                                ui.label(f'Class: {class_name} ({lbl})').style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                            if AS.current_sample is not None:
                                shape = AS.current_sample.shape
                                if len(shape) == 4:
                                    ui.label(f'Size: {shape[2]}×{shape[3]} RGB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                elif len(shape) == 3:
                                    ui.label(f'Size: {shape[1]}×{shape[2]} RGB').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            # Show classification prediction if available
                            if result is not None and 'class_logits' in result:
                                import torch
                                pred = torch.argmax(result['class_logits'], dim=-1).item()
                                pred_name = AS.class_names[pred] if pred < len(AS.class_names) else str(pred)
                                ui.label(f'Predicted: {pred_name}').style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                        else:
                            # Graph model info
                            if hasattr(sample, 'y') and sample.y is not None:
                                lbl = sample.y.item() if hasattr(sample.y, "item") else sample.y
                                ui.label(f'Label: {lbl}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            if hasattr(sample, 'x'):
                                ui.label(f'Nodes: {sample.x.shape[0]}, Feat: {sample.x.shape[1]}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            if hasattr(sample, 'graph_attr') and sample.graph_attr is not None:
                                k_mean = sample.graph_attr[0].item() if hasattr(sample.graph_attr[0], 'item') else sample.graph_attr[0]
                                ui.label(f'Kuramoto: {k_mean:.3f}').style(f'color:{THEME_WARN}; font-size: 0.7rem;')
                                # Track Kuramoto original
                                AS.kuramoto_history.append((AS.current_idx, k_mean))
                            
                            # Track Kuramoto proxy from reconstruction
                            if AS.current_kuramoto_comparison is not None:
                                k_proxy = AS.current_kuramoto_comparison['kuramoto_proxy']
                                k_error = AS.current_kuramoto_comparison['absolute_error']
                                AS.kuramoto_proxy_history.append((AS.current_idx, k_proxy))
                                ui.label(f'K. Proxy: {k_proxy:.3f} (Δ={k_error:.3f})').style('color:#f472b6; font-size: 0.7rem;')
                
                # EEG Synchronization - simple: current_idx * epoch_duration = view_start
                if AS.eeg_sync_enabled and AS.eeg_data is not None:
                    try:
                        # Calculate view position from sample index
                        AS.eeg_view_start = AS.current_idx * AS.epoch_duration
                        # Clamp to EEG duration
                        max_start = max(0, AS.eeg_data.duration_sec - AS.epoch_duration)
                        AS.eeg_view_start = min(AS.eeg_view_start, max_start)
                        
                        # Update info label
                        total_epochs = int(AS.eeg_data.duration_sec // AS.epoch_duration)
                        eeg_sync_info.set_text(f'Epoch {AS.current_idx}/{total_epochs} | {AS.eeg_view_start:.1f}s')
                    except Exception as e:
                        analysis_log(f"EEG sync error: {e}", 'warning')
                
                # Update sample label
                sample_label.set_text(f'Sample: {AS.current_idx + 1} / {AS.total_samples}')
                
                # Update all plots
                
                # Update EEG sync plot if enabled
                if AS.eeg_sync_enabled and AS.eeg_plot is not None and AS.eeg_data is not None:
                    try:
                        AS.eeg_plot.figure = make_analysis_eeg_fig()
                        AS.eeg_plot.update()
                    except Exception as e:
                        analysis_log(f"EEG plot error: {e}", 'warning')
                
                try:
                    kuramoto_plot.figure = make_kuramoto_fig()
                    kuramoto_plot.update()
                except Exception as e:
                    analysis_log(f"Kuramoto plot error: {e}", 'warning')
                
                # Update sync network visualizations (Original vs Reconstructed)
                try:
                    original_sync_plot.figure = make_original_sync_fig()
                    original_sync_plot.update()
                except Exception as e:
                    analysis_log(f"Original sync plot error: {e}", 'warning')
                
                try:
                    recon_sync_plot.figure = make_recon_sync_fig()
                    recon_sync_plot.update()
                except Exception as e:
                    analysis_log(f"Recon sync plot error: {e}", 'warning')
                
                try:
                    # Update encoder selectors and plot
                    update_encoder_selectors()
                    encoder_plot.figure = make_encoder_fig()
                    encoder_plot.update()
                except Exception as e:
                    analysis_log(f"Encoder plot error: {e}", 'warning')
                
                try:
                    # Update decoder selectors and plot
                    update_decoder_selectors()
                    decoder_plot.figure = make_decoder_fig()
                    decoder_plot.update()
                except Exception as e:
                    analysis_log(f"Decoder plot error: {e}", 'warning')
                
                try:
                    latent_plot.figure = make_latent_fig()
                    latent_plot.update()
                except Exception as e:
                    analysis_log(f"Latent plot error: {e}", 'warning')
                
                try:
                    attention_plot.figure = make_attention_fig()
                    attention_plot.update()
                except Exception as e:
                    analysis_log(f"Attention plot error: {e}", 'warning')
                
                try:
                    original_plot.figure = make_original_fig()
                    original_plot.update()
                except Exception as e:
                    analysis_log(f"Original plot error: {e}", 'warning')
                
                try:
                    recon_plot.figure = make_recon_fig()
                    recon_plot.update()
                except Exception as e:
                    analysis_log(f"Recon plot error: {e}", 'warning')
                
                try:
                    diff_plot.figure = make_diff_fig()
                    diff_plot.update()
                except Exception as e:
                    analysis_log(f"Diff plot error: {e}", 'warning')
                
            except Exception as e:
                import traceback
                analysis_log(f"Error processing sample: {e}", 'error')
                analysis_log(traceback.format_exc(), 'error')


