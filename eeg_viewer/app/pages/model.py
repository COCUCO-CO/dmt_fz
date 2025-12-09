"""Model training page."""
from pathlib import Path
import asyncio
import subprocess
import json
import sys
import os
import re
from nicegui import ui

from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM
)
from app.state import MS
from app.visualization.components.debug_console import render_debug_toggle, render_debug_console, render_training_badge
from app.visualization.styles.css import STYLE

AUTOENCODER_DIR = Path(__file__).parent.parent.parent.parent / "machine_learning" / "autoencoder"
AUTOENCODER_CACHE_DIR = Path(__file__).parent.parent / "cache" / "autoencoder"

def detect_dataset_type(path: Path) -> dict:
    """
    Detect dataset type and structure from a given path.
    
    Returns dict with:
        - type: 'graph', 'image', 'unknown'
        - structure: 'single_file', 'split_files', 'hierarchical'
        - conditions: list of detected conditions (e.g., ['DMT', 'EC', 'EO'])
        - file_count: number of data files
        - sample_file: path to a sample file
        - data_sources: what data is available (phases, syncro, etc.)
        - has_stc: whether STC (source localized) data is available
        - has_eeg: whether EEG data is available
        - num_nodes: number of nodes (channels/parcels)
    """
    info = {
        'type': 'unknown',
        'structure': 'unknown',
        'conditions': [],
        'file_count': 0,
        'sample_file': None,
        'bands': [],
        'data_sources': [],
        'has_stc': False,
        'has_eeg': False,
        'num_nodes_eeg': 0,
        'num_nodes_stc': 0,
        'num_epochs_sample': 0,
        'error': None
    }
    
    if not path.exists():
        info['error'] = f"Path does not exist: {path}"
        return info
    
    # Check for phases-*.pkl files (graph data from pipeline)
    phases_files = list(path.rglob("phases-*.pkl"))
    syncro_files = list(path.rglob("syncro-*.pkl"))
    order_files = list(path.rglob("order-*.pkl"))
    
    if phases_files:
        info['type'] = 'graph'
        info['file_count'] = len(phases_files)
        info['sample_file'] = str(phases_files[0])
        info['data_sources'].append('phases (syncro + phases + amplitudes + kuramoto)')
        
        # Detect conditions from folder structure
        conditions = set()
        for f in phases_files:
            parent = f.parent.name
            if parent in ['DMT', 'EC', 'EO']:
                conditions.add(parent)
        info['conditions'] = sorted(list(conditions)) if conditions else ['Unknown']
        
        # Check structure
        subdirs = [d for d in path.iterdir() if d.is_dir()]
        info['structure'] = 'hierarchical' if subdirs else 'flat'
        
        # Analyze sample file for detailed info
        try:
            import pickle
            with open(info['sample_file'], 'rb') as f:
                data = pickle.load(f)
            
            # Check what data is available
            if 'phases_stc' in data:
                info['has_stc'] = True
                info['bands'] = list(data['phases_stc'].keys())
                # Get number of parcels from first band, first epoch
                first_band = info['bands'][0]
                if data['phases_stc'][first_band]:
                    info['num_nodes_stc'] = data['phases_stc'][first_band][0].shape[0]
                    info['num_epochs_sample'] = len(data['phases_stc'][first_band])
            
            if 'phases_eeg' in data:
                info['has_eeg'] = True
                if not info['bands']:
                    info['bands'] = list(data['phases_eeg'].keys())
                first_band = info['bands'][0]
                if data['phases_eeg'][first_band]:
                    info['num_nodes_eeg'] = data['phases_eeg'][first_band][0].shape[0]
                    if info['num_epochs_sample'] == 0:
                        info['num_epochs_sample'] = len(data['phases_eeg'][first_band])
            
            # Check what else is in the file
            available_keys = list(data.keys())
            if 'syncros_stc' in data or 'syncros_eeg' in data:
                if 'syncro' not in str(info['data_sources']):
                    pass  # Already included in phases
            if 'kuramoto_stc' in data or 'kuramoto_eeg' in data:
                pass  # Already included in phases
                
        except Exception as e:
            info['error'] = f"Could not analyze sample file: {e}"
        
        return info
    
    # Fallback: check for standalone syncro files
    if syncro_files:
        info['type'] = 'graph'
        info['file_count'] = len(syncro_files)
        info['sample_file'] = str(syncro_files[0])
        info['data_sources'].append('syncro (only synchronization matrices)')
        
        conditions = set()
        for f in syncro_files:
            parent = f.parent.name
            if parent in ['DMT', 'EC', 'EO']:
                conditions.add(parent)
        info['conditions'] = sorted(list(conditions)) if conditions else ['Unknown']
        
        return info
    
    # Check for order files
    if order_files:
        info['type'] = 'order'
        info['file_count'] = len(order_files)
        info['sample_file'] = str(order_files[0])
        info['data_sources'].append('order (Kuramoto order parameter)')
        info['error'] = "Order files contain pre-computed Kuramoto values, not suitable for VAE training. Use phases-*.pkl files instead."
        return info
    
    # Check for image files
    image_files = list(path.rglob("*.png")) + list(path.rglob("*.jpg")) + list(path.rglob("*.jpeg"))
    if image_files:
        info['type'] = 'image'
        info['file_count'] = len(image_files)
        info['structure'] = 'flat' if not any(d.is_dir() for d in path.iterdir()) else 'hierarchical'
        info['sample_file'] = str(image_files[0])
        return info
    
    # Check for numpy arrays
    npy_files = list(path.rglob("*.npy")) + list(path.rglob("*.npz"))
    if npy_files:
        info['type'] = 'array'
        info['file_count'] = len(npy_files)
        info['sample_file'] = str(npy_files[0])
        return info
    
    info['error'] = "No recognized data files found (phases-*.pkl, images, or numpy arrays)"
    return info


def model_log(msg: str, msg_type: str = 'info'):
    """Add message to model training log and global debug console."""
    from app.visualization.components.debug_console import debug_log
    
    # Store in history for persistence
    MS.log_history.append((msg, msg_type))
    # Keep only last 500 messages
    if len(MS.log_history) > 500:
        MS.log_history = MS.log_history[-500:]
    
    # Also send to global debug console
    debug_log(msg, msg_type, 'model')
    
    if MS.log_container:
        colors = {
            'info': THEME_TEXT,
            'success': THEME_PRIMARY,
            'warning': THEME_WARN,
            'error': THEME_ERROR
        }
        with MS.log_container:
            ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')


def update_status_indicator(status: str):
    """Update the training status. The global render_training_badge() handles UI."""
    MS.status = status
    # Update training flag for global badge
    MS.training = (status == 'training')


def create_default_config(dataset_path: str, dataset_info: dict) -> dict:
    """Create default VAE configuration based on detected dataset."""
    config = {
        'paths': {
            'phases_dir': dataset_path,
            'output_dir': str(AUTOENCODER_CACHE_DIR / 'output'),
            'dataset_cache': str(AUTOENCODER_CACHE_DIR / 'dataset_cache'),
            'checkpoints': str(AUTOENCODER_CACHE_DIR / 'checkpoints'),
            'tensorboard': str(AUTOENCODER_CACHE_DIR / 'runs'),
        },
        'data': {
            'conditions': dataset_info.get('conditions', ['DMT', 'EC', 'EO']),
            'bands': dataset_info.get('bands', ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']),
            'use_stc': dataset_info.get('has_stc', False),
            'graph': {
                'fully_connected': True,
                'edge_threshold': 0.3,
                'edge_threshold_percentile': None,
                'max_edges_per_node': None,
                'self_loops': False,
                'directed': False,
            },
            'node_features': {
                'use_phase_stats': True,
                'use_amplitude_stats': True,
                'use_temporal_complexity': True,
                'use_network_label': False,
            },
            'graph_features': {
                'use_kuramoto': True,
                'use_global_sync': True,
                'use_topology': True,
            },
            'split': {
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15,
                'stratify': True,
                'random_state': 42,
                'group_by_subject': True,
            }
        },
        'model': {
            'name': 'BrainStateVAE',
            'encoder': {
                'conv_type': 'gatv2',
                'hidden_dim': 64,
                'num_gat_layers': 3,
                'num_attention_heads': 4,
                'cheby_k': 3,
                'dropout': 0.2,
                'attention_dropout': 0.1,
                'use_edge_attr': True,
                'concat_heads': True,
                'negative_slope': 0.2,
                'use_skip_connections': True,
            },
            'latent': {
                'dim': 64,
            },
            'decoder': {
                'hidden_dims': [256, 128],
                'reconstruct_edges': True,
                'activation': 'leaky_relu',
                'dropout': 0.1,
            },
            'pooling': {
                'method': 'mean',
            }
        },
        'loss': {
            'reconstruction': {
                'node_weight': 0.3,
                'edge_weight': 1.0,
                'type': 'mse',
            },
            'kl': {
                'weight': 0.01,
                'annealing': {
                    'enabled': True,
                    'start': 0.0,
                    'end': 0.05,
                    'epochs': 50,
                    'type': 'linear',
                }
            },
            'free_bits': 0.1,
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 256,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'optimizer': 'adamw',
            'scheduler': {
                'type': 'cosine',
                'patience': 10,
                'factor': 0.5,
                'min_lr': 1e-6,
            },
            'early_stopping': {
                'patience': 20,
                'min_delta': 0.0001,
                'monitor': 'val_loss',
            },
            'gradient_clipping': {
                'enabled': True,
                'max_norm': 0.5,
            }
        },
        'logging': {
            'level': 'INFO',
            'tensorboard': True,
            'save_frequency': 10,
            'log_frequency': 1,
            'tensorboard_extras': {
                'latent_space': True,
                'reconstructions': True,
                'attention_weights': False,
                'kl_per_dim': True,
                'gradient_norms': False,
            }
        },
        'visualization': {
            'plot_training_curves': True,
            'plot_latent_space': True,
            'plot_reconstructions': True,
            'num_examples_to_visualize': 10,
            'latent_method': 'tsne',
        },
        'seed': 42,
        'deterministic': True,
        'device': 'cuda',  # Will fallback to CPU if not available
        'num_workers': 4,  # Use workers with GPU
        'dataset_workers': 4,
        'pin_memory': True,  # Enabled for GPU
    }
    return config


@ui.page('/model')
def model_page():
    """Model training page."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    
    # Header with navigation
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem; letter-spacing: 2px;')
        ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono; letter-spacing: 1px;')
        ui.label('// MODEL').classes('text-xs ml-2').style(f'color: #f472b6; font-family: JetBrains Mono;')
        
        render_debug_toggle()
        render_training_badge()  # Global training indicator - shows on all pages
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('CLEANER', on_click=lambda: ui.navigate.to('/cleaner')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('PIPELINE', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('MODEL', on_click=lambda: ui.navigate.to('/model')).props('flat dense').style(f'color:#f472b6;')
            ui.button('ANALYSIS', on_click=lambda: ui.navigate.to('/analysis')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
    
    with ui.row().classes('w-full p-4 gap-4').style('height: calc(100vh - 50px); align-items: stretch;'):
        
        # LEFT PANEL: Configuration
        with ui.column().classes('gap-4').style('width: 400px; overflow-y: auto;'):
            
            # DATASET CONFIGURATION
            with ui.card().classes('dark-card p-4 w-full').style(f'border: 1px solid #f472b6;'):
                ui.label('// DATASET').classes('terminal-header')
                
                with ui.row().classes('items-center gap-2 mt-2 w-full'):
                    ui.label('Path:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem; min-width: 50px;')
                    dataset_path_input = ui.input(
                        value='/media/storage_hdd/dmt_fz/fwd-inv-stc'
                    ).props('dense dark').classes('flex-1')
                
                dataset_info_container = ui.column().classes('w-full mt-2 gap-1')
                
                def scan_dataset():
                    """Scan and detect dataset."""
                    path = Path(dataset_path_input.value.strip())
                    MS.dataset_path = str(path)
                    
                    dataset_info_container.clear()
                    with dataset_info_container:
                        ui.label('Scanning...').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    
                    info = detect_dataset_type(path)
                    MS.dataset_info = info
                    MS.dataset_type = info['type']
                    
                    dataset_info_container.clear()
                    with dataset_info_container:
                        if info.get('error'):
                            ui.label(f"❌ {info['error']}").style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                        else:
                            ui.label(f"✓ Type: {info['type'].upper()}").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                            ui.label(f"  Files: {info['file_count']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            
                            if info['conditions']:
                                ui.label(f"  Conditions: {', '.join(info['conditions'])}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            if info['bands']:
                                ui.label(f"  Bands: {', '.join(info['bands'])}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            
                            # Show data sources
                            if info['data_sources']:
                                ui.label(f"  Data: {info['data_sources'][0]}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                            
                            # Show EEG/STC availability
                            if info['has_eeg'] or info['has_stc']:
                                sources = []
                                if info['has_eeg']:
                                    sources.append(f"EEG ({info['num_nodes_eeg']} ch)")
                                if info['has_stc']:
                                    sources.append(f"STC ({info['num_nodes_stc']} parcels)")
                                ui.label(f"  Sources: {' | '.join(sources)}").style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                            
                            if info['num_epochs_sample'] > 0:
                                ui.label(f"  Epochs/subject: ~{info['num_epochs_sample']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                            
                            # Update source selector options
                            if info['has_eeg'] and info['has_stc']:
                                data_source_select.options = ['EEG (channels)', 'STC (parcels)']
                                data_source_select.value = 'STC (parcels)' if info['has_stc'] else 'EEG (channels)'
                            elif info['has_eeg']:
                                data_source_select.options = ['EEG (channels)']
                                data_source_select.value = 'EEG (channels)'
                            elif info['has_stc']:
                                data_source_select.options = ['STC (parcels)']
                                data_source_select.value = 'STC (parcels)'
                            
                            # Create default config
                            MS.config = create_default_config(str(path), info)
                            model_log(f"Dataset detected: {info['type']} ({info['file_count']} files)", 'success')
                
                ui.button('Scan Dataset', on_click=scan_dataset, icon='search').props('dense').classes('mt-2').style(f'background:#f472b6; color:black;')
                
                # Data source selector (EEG vs STC)
                ui.separator().classes('my-2')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Source:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 50px;')
                    data_source_select = ui.select(
                        ['EEG (channels)', 'STC (parcels)'],
                        value='EEG (channels)'
                    ).props('dense dark').classes('flex-1')
                
                ui.label('EEG: 24 electrodes | STC: ~200 brain parcels').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                
                # Band selection
                ui.separator().classes('my-2')
                ui.label('Bands to use:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                with ui.row().classes('gap-2 flex-wrap'):
                    band_checks = {}
                    for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                        band_checks[band] = ui.checkbox(band, value=(band == 'Alpha')).props('dense')
                
                ui.label('Tip: Start with 1-2 bands for faster training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                
                # Subsample option
                with ui.row().classes('items-center gap-2 mt-2'):
                    ui.label('Subsample:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 70px;')
                    subsample_slider = ui.slider(min=0.1, max=1.0, step=0.1, value=0.3).props('label-always').classes('flex-1')
                
                ui.label('Use 0.1-0.3 for quick tests, 1.0 for full training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
            
            # MODEL CONFIGURATION
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// MODEL CONFIG').classes('terminal-header')
                
                with ui.column().classes('gap-1 mt-2'):
                    # Model type selector
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Type:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem; min-width: 70px;')
                        model_type_select = ui.select(
                            ['VAE (Graph)', 'VAE (Image)', 'AE (Graph)'],
                            value='VAE (Graph)'
                        ).props('dense dark').classes('flex-1')
                    
                    ui.separator().classes('my-1')
                    
                    # Architecture params
                    ui.label('Architecture').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Latent:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        latent_dim = ui.number(value=64, min=8, max=512, step=8).props('dense').classes('w-16')
                        ui.label('Hidden:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        hidden_dim = ui.number(value=64, min=16, max=256, step=16).props('dense').classes('w-16')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('GAT layers:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        gat_layers = ui.number(value=3, min=1, max=6).props('dense').classes('w-16')
                        ui.label('Heads:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        attention_heads = ui.number(value=4, min=1, max=8).props('dense').classes('w-16')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Dropout:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        dropout = ui.number(value=0.2, min=0.0, max=0.5, step=0.05).props('dense').classes('w-16')
                        ui.label('Attn drop:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        attn_dropout = ui.number(value=0.1, min=0.0, max=0.3, step=0.05).props('dense').classes('w-16')
                    
                    ui.separator().classes('my-1')
                    
                    # Training params
                    ui.label('Training').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Epochs:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        num_epochs = ui.number(value=100, min=10, max=500, step=10).props('dense').classes('w-16')
                        ui.label('Batch:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        batch_size = ui.number(value=256, min=16, max=1024, step=16).props('dense').classes('w-16')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Learn rate:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        learning_rate = ui.select(
                            ['1e-2', '5e-3', '1e-3', '5e-4', '1e-4'],
                            value='1e-3'
                        ).props('dense dark').classes('w-20')
                        ui.label('Decay:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        weight_decay = ui.select(
                            ['0', '1e-5', '1e-4', '1e-3'],
                            value='1e-5'
                        ).props('dense dark').classes('w-20')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Optimizer:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        optimizer_select = ui.select(
                            ['adamw', 'adam', 'sgd'],
                            value='adamw'
                        ).props('dense dark').classes('w-20')
                        ui.label('Scheduler:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        scheduler_select = ui.select(
                            ['cosine', 'reduce_on_plateau', 'step', 'none'],
                            value='cosine'
                        ).props('dense dark').classes('w-24')
                    
                    ui.separator().classes('my-1')
                    
                    # Early stopping & regularization
                    ui.label('Regularization').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Patience:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        patience = ui.number(value=25, min=5, max=100, step=5).props('dense').classes('w-16')
                        ui.label('Grad clip:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        grad_clip = ui.number(value=0.5, min=0.0, max=5.0, step=0.1).props('dense').classes('w-16')
                    
                    ui.separator().classes('my-1')
                    
                    # Loss params
                    ui.label('Loss').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('KL weight:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        kl_weight = ui.number(value=0.01, min=0.0, max=1.0, step=0.01).props('dense').classes('w-16')
                        ui.label('β anneal:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        beta_annealing = ui.switch(value=True).props('dense')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Node wt:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        node_weight = ui.number(value=0.3, min=0.0, max=1.0, step=0.1).props('dense').classes('w-16')
                        ui.label('Edge wt:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        edge_weight = ui.number(value=1.0, min=0.0, max=2.0, step=0.1).props('dense').classes('w-16')
            
            # TRAINING CONTROLS
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// TRAINING').classes('terminal-header')
                
                training_status = ui.label('Ready').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;').classes('mt-2')
                
                with ui.row().classes('gap-2 mt-3'):
                    async def start_training():
                        if MS.training:
                            ui.notify('Training already in progress', type='warning')
                            return
                        
                        if not MS.dataset_path or not MS.dataset_info.get('type'):
                            ui.notify('Please scan a dataset first', type='warning')
                            return
                        
                        if MS.dataset_info.get('type') == 'order':
                            ui.notify('Order files not suitable for VAE. Use phases-*.pkl', type='error')
                            return
                        
                        # Update config with UI values - Architecture
                        MS.config['model']['latent']['dim'] = int(latent_dim.value)
                        MS.config['model']['encoder']['hidden_dim'] = int(hidden_dim.value)
                        MS.config['model']['encoder']['num_gat_layers'] = int(gat_layers.value)
                        MS.config['model']['encoder']['num_attention_heads'] = int(attention_heads.value)
                        MS.config['model']['encoder']['dropout'] = float(dropout.value)
                        MS.config['model']['encoder']['attention_dropout'] = float(attn_dropout.value)
                        
                        # Training params
                        MS.config['training']['num_epochs'] = int(num_epochs.value)
                        MS.config['training']['batch_size'] = int(batch_size.value)
                        MS.config['training']['learning_rate'] = float(learning_rate.value)
                        MS.config['training']['weight_decay'] = float(weight_decay.value) if weight_decay.value != '0' else 0.0
                        MS.config['training']['optimizer'] = optimizer_select.value
                        MS.config['training']['scheduler']['type'] = scheduler_select.value if scheduler_select.value != 'none' else None
                        MS.config['training']['early_stopping']['patience'] = int(patience.value)
                        MS.config['training']['gradient_clipping']['max_norm'] = float(grad_clip.value)
                        MS.config['training']['gradient_clipping']['enabled'] = grad_clip.value > 0
                        
                        # Loss params
                        MS.config['loss']['kl']['weight'] = float(kl_weight.value)
                        MS.config['loss']['kl']['annealing']['enabled'] = beta_annealing.value
                        MS.config['loss']['reconstruction']['node_weight'] = float(node_weight.value)
                        MS.config['loss']['reconstruction']['edge_weight'] = float(edge_weight.value)
                        
                        # Set data source (EEG vs STC)
                        use_stc = 'STC' in data_source_select.value
                        MS.config['data']['use_stc'] = use_stc
                        model_log(f"Using {'STC (parcels)' if use_stc else 'EEG (channels)'} data", 'info')
                        
                        # Set selected bands
                        selected_bands = [band for band, cb in band_checks.items() if cb.value]
                        if not selected_bands:
                            ui.notify('Select at least one band', type='warning')
                            return
                        MS.config['data']['bands'] = selected_bands
                        model_log(f"Bands: {', '.join(selected_bands)}", 'info')
                        
                        # Estimate dataset size
                        n_files = MS.dataset_info.get('file_count', 0)
                        n_epochs = MS.dataset_info.get('num_epochs_sample', 50)
                        n_bands = len(selected_bands)
                        estimated_graphs = n_files * n_epochs * n_bands
                        subsample = subsample_slider.value
                        final_estimate = int(estimated_graphs * subsample)
                        model_log(f"Estimated graphs: ~{final_estimate} (subsample={subsample:.0%})", 'info')
                        
                        # Check GPU availability
                        try:
                            import torch
                            if torch.cuda.is_available():
                                gpu_name = torch.cuda.get_device_name(0)
                                model_log(f"GPU: {gpu_name}", 'success')
                            else:
                                model_log("⚠ No GPU available - training on CPU (slower)", 'warning')
                        except:
                            model_log("⚠ Could not detect GPU", 'warning')
                        
                        # Create cache directories
                        AUTOENCODER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'output').mkdir(exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'checkpoints').mkdir(exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'runs').mkdir(exist_ok=True)
                        (AUTOENCODER_CACHE_DIR / 'dataset_cache').mkdir(exist_ok=True)
                        
                        # Save config to temp file
                        import yaml
                        config_path = AUTOENCODER_CACHE_DIR / 'train_config.yaml'
                        with open(config_path, 'w') as f:
                            yaml.dump(MS.config, f, default_flow_style=False)
                        
                        MS.training = True
                        MS.history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [], 'epoch': []}
                        MS.log_history = []  # Clear log history
                        
                        # Clear previous logs and reset plot
                        if MS.log_container:
                            MS.log_container.clear()
                        update_loss_plot()  # Reset the plot with empty data
                        
                        # Update status indicator
                        update_status_indicator('training')
                        
                        training_status.text = 'Training...'
                        training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                        model_log(f"Starting training with config: {config_path}", 'info')
                        
                        # Run training in subprocess
                        import subprocess
                        cmd = [
                            sys.executable,
                            '-u',  # Unbuffered output - critical for real-time logs
                            str(AUTOENCODER_DIR / 'train.py'),
                            '--config', str(config_path),
                            '--subsample', str(subsample_slider.value)
                        ]
                        
                        env = os.environ.copy()
                        env['PYTHONPATH'] = str(AUTOENCODER_DIR.parent)
                        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
                        
                        try:
                            process = await asyncio.create_subprocess_exec(
                                *cmd,
                                stdout=asyncio.subprocess.PIPE,
                                stderr=asyncio.subprocess.STDOUT,
                                env=env,
                                cwd=str(AUTOENCODER_DIR)
                            )
                            MS.current_process = process
                            
                            # Read output line by line
                            epoch_pattern = re.compile(r'Epoch (\d+).*Loss: ([\d.]+).*Recon: ([\d.]+).*KL: ([\d.]+)')
                            val_pattern = re.compile(r'Epoch \d+.*\[val\].*Loss: ([\d.]+)')
                            
                            while True:
                                line = await process.stdout.readline()
                                if not line:
                                    break
                                line = line.decode('utf-8', errors='replace').strip()
                                if line:
                                    model_log(line, 'info')
                                    
                                    # Parse training metrics
                                    epoch_match = epoch_pattern.search(line)
                                    if epoch_match and '[train]' in line:
                                        epoch = int(epoch_match.group(1))
                                        loss = float(epoch_match.group(2))
                                        recon = float(epoch_match.group(3))
                                        kl = float(epoch_match.group(4))
                                        
                                        MS.history['epoch'].append(epoch)
                                        MS.history['train_loss'].append(loss)
                                        MS.history['recon_loss'].append(recon)
                                        MS.history['kl_loss'].append(kl)
                                        
                                        # Update loss plot
                                        update_loss_plot()
                                    
                                    val_match = val_pattern.search(line)
                                    if val_match:
                                        val_loss = float(val_match.group(1))
                                        MS.history['val_loss'].append(val_loss)
                                        update_loss_plot()
                            
                            await process.wait()
                            
                            if process.returncode == 0:
                                model_log("Training completed successfully!", 'success')
                                training_status.text = 'Completed'
                                training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                update_status_indicator('completed')
                            else:
                                model_log(f"Training failed with code {process.returncode}", 'error')
                                training_status.text = 'Failed'
                                training_status.style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                                update_status_indicator('error')
                        
                        except Exception as e:
                            model_log(f"Error: {e}", 'error')
                            training_status.text = 'Error'
                            training_status.style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                            update_status_indicator('error')
                        
                        finally:
                            MS.training = False
                            MS.current_process = None
                    
                    async def stop_training():
                        if MS.current_process:
                            MS.current_process.terminate()
                            model_log("Training stopped by user", 'warning')
                            training_status.text = 'Stopped'
                            training_status.style(f'color:{THEME_WARN}; font-size: 0.75rem;')
                            MS.training = False
                            update_status_indicator('idle')
                    
                    ui.button('Train', on_click=start_training, icon='play_arrow').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                    ui.button('Stop', on_click=stop_training, icon='stop').props('dense color=negative')
        
        # RIGHT PANEL: Visualization & Logs
        with ui.column().classes('flex-1').style('min-height: 0; display: flex; flex-direction: column;'):
            
            with ui.card().classes('dark-card p-2 w-full flex-1').style('display: flex; flex-direction: column; min-height: 0;'):
                with ui.tabs().classes('w-full').style(f'background: {THEME_BG};') as model_tabs:
                    tab_metrics = ui.tab('METRICS', icon='show_chart').style(f'color:#f472b6;')
                    tab_recon = ui.tab('RECON', icon='compare').style(f'color:{THEME_SECONDARY};')
                    tab_console = ui.tab('CONSOLE', icon='terminal').style(f'color:{THEME_PRIMARY};')
                
                with ui.tab_panels(model_tabs, value=tab_console).classes('w-full').style('flex: 1; min-height: 0; overflow: hidden;'):
                    
                    # METRICS TAB
                    with ui.tab_panel(tab_metrics).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        ui.label('▌TRAINING METRICS').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                        
                        loss_plot_container = ui.column().classes('w-full flex-1')
                        
                        def make_loss_figure():
                            """Create separate plots for different metrics."""
                            from plotly.subplots import make_subplots
                            
                            epochs = MS.history.get('epoch', [])
                            train_loss = MS.history.get('train_loss', [])
                            val_loss = MS.history.get('val_loss', [])
                            recon_loss = MS.history.get('recon_loss', [])
                            kl_loss = MS.history.get('kl_loss', [])
                            
                            # Create 1x3 subplot grid
                            fig = make_subplots(
                                rows=1, cols=3,
                                subplot_titles=('Total Loss', 'Recon Loss', 'KL Loss'),
                                horizontal_spacing=0.08
                            )
                            
                            if epochs:
                                # Plot 1: Train + Val Loss
                                fig.add_trace(go.Scatter(
                                    x=epochs, y=train_loss,
                                    mode='lines', name='Train',
                                    line=dict(color=THEME_PRIMARY, width=2)
                                ), row=1, col=1)
                                
                                if val_loss:
                                    fig.add_trace(go.Scatter(
                                        x=epochs[:len(val_loss)], y=val_loss,
                                        mode='lines', name='Val',
                                        line=dict(color='#f472b6', width=2)
                                    ), row=1, col=1)
                                
                                # Plot 2: Recon Loss
                                fig.add_trace(go.Scatter(
                                    x=epochs, y=recon_loss,
                                    mode='lines', name='Recon',
                                    line=dict(color=THEME_SECONDARY, width=2),
                                    showlegend=False
                                ), row=1, col=2)
                                
                                # Plot 3: KL Loss
                                fig.add_trace(go.Scatter(
                                    x=epochs, y=kl_loss,
                                    mode='lines', name='KL',
                                    line=dict(color=THEME_WARN, width=2),
                                    showlegend=False
                                ), row=1, col=3)
                            
                            fig.update_layout(
                                template='plotly_dark',
                                paper_bgcolor='rgba(8,8,8,1)',
                                plot_bgcolor='rgba(8,8,8,1)',
                                margin=dict(l=40, r=20, t=40, b=40),
                                height=280,
                                legend=dict(
                                    orientation='h',
                                    yanchor='bottom',
                                    y=1.08,
                                    xanchor='left',
                                    x=0
                                ),
                                font=dict(family='JetBrains Mono', size=10, color=THEME_TEXT)
                            )
                            
                            # Update axes
                            fig.update_xaxes(gridcolor='rgba(0,255,136,0.1)', showgrid=True)
                            fig.update_yaxes(gridcolor='rgba(0,255,136,0.1)', showgrid=True)
                            
                            return fig
                        
                        def update_loss_plot():
                            """Update the loss plot with current history."""
                            loss_plot_container.clear()
                            with loss_plot_container:
                                fig = make_loss_figure()
                                MS.loss_plot = ui.plotly(fig).classes('w-full').style('height: 280px;')
                        
                        # Initial empty plot
                        update_loss_plot()
                        
                        # Stats summary
                        with ui.row().classes('w-full gap-4 mt-4'):
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Best Val Loss').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                best_val_label = ui.label('--').style(f'color:#f472b6; font-size: 1.2rem; font-weight: bold;')
                            
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Current Epoch').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                current_epoch_label = ui.label('0').style(f'color:{THEME_PRIMARY}; font-size: 1.2rem; font-weight: bold;')
                            
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Train Loss').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                train_loss_label = ui.label('--').style(f'color:{THEME_SECONDARY}; font-size: 1.2rem; font-weight: bold;')
                        
                        def update_stats():
                            """Update stats labels."""
                            if MS.history['epoch']:
                                current_epoch_label.text = str(MS.history['epoch'][-1])
                            if MS.history['train_loss']:
                                train_loss_label.text = f"{MS.history['train_loss'][-1]:.4f}"
                            if MS.history['val_loss']:
                                best_val_label.text = f"{min(MS.history['val_loss']):.4f}"
                        
                        ui.timer(2.0, update_stats)
                    
                    # RECONSTRUCTION TAB - Show original vs reconstructed with epoch slider
                    with ui.tab_panel(tab_recon).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        ui.label('▌RECONSTRUCTION QUALITY').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                        
                        recon_container = ui.column().classes('w-full flex-1')
                        
                        # Epoch selector
                        with ui.row().classes('items-center gap-3 mb-3'):
                            ui.label('Epoch:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                            epoch_slider = ui.slider(min=1, max=100, step=10, value=10).props('label-always').classes('flex-1')
                            refresh_btn = ui.button('Refresh', icon='refresh').props('flat dense size=sm')
                        
                        def load_reconstruction(epoch_val):
                            """Load and display reconstruction for given epoch."""
                            recon_dir = AUTOENCODER_CACHE_DIR / 'output' / 'reconstructions'
                            recon_file = recon_dir / f'recon_epoch_{int(epoch_val):03d}.npz'
                            
                            recon_container.clear()
                            with recon_container:
                                if not recon_file.exists():
                                    ui.label(f'No reconstruction for epoch {int(epoch_val)}').style(f'color:{THEME_TEXT_DIM};')
                                    ui.label('Reconstructions are saved every 10 epochs during training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                    
                                    # List available epochs
                                    if recon_dir.exists():
                                        available = sorted([f.stem.split('_')[-1] for f in recon_dir.glob('*.npz')])
                                        if available:
                                            ui.label(f'Available: {", ".join(available)}').style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                                    return
                                
                                try:
                                    data = np.load(recon_file)
                                    original = data['original']
                                    reconstructed = data['reconstructed']
                                    
                                    # Create side-by-side heatmaps
                                    from plotly.subplots import make_subplots
                                    
                                    # Use first 100 samples, reshape to 10x10 grid
                                    n_features = min(original.shape[1], 10) if len(original.shape) > 1 else 10
                                    n_samples = min(100, original.shape[0])
                                    
                                    # Take mean across features for visualization
                                    if len(original.shape) > 1:
                                        orig_grid = original[:n_samples, :n_features]
                                        recon_grid = reconstructed[:n_samples, :n_features]
                                    else:
                                        orig_grid = original[:n_samples].reshape(-1, 1)
                                        recon_grid = reconstructed[:n_samples].reshape(-1, 1)
                                    
                                    diff_grid = np.abs(orig_grid - recon_grid)
                                    
                                    fig = make_subplots(
                                        rows=1, cols=3,
                                        subplot_titles=(f'Original (Epoch {int(epoch_val)})', 'Reconstructed', 'Difference'),
                                        horizontal_spacing=0.05
                                    )
                                    
                                    # Original heatmap
                                    fig.add_trace(go.Heatmap(
                                        z=orig_grid, colorscale='Viridis', showscale=False,
                                        name='Original'
                                    ), row=1, col=1)
                                    
                                    # Reconstructed heatmap  
                                    fig.add_trace(go.Heatmap(
                                        z=recon_grid, colorscale='Viridis', showscale=False,
                                        name='Reconstructed'
                                    ), row=1, col=2)
                                    
                                    # Difference heatmap
                                    fig.add_trace(go.Heatmap(
                                        z=diff_grid, colorscale='Reds', showscale=True,
                                        colorbar=dict(title='|Δ|', x=1.02, len=0.9),
                                        name='Difference'
                                    ), row=1, col=3)
                                    
                                    fig.update_layout(
                                        template='plotly_dark',
                                        paper_bgcolor='rgba(8,8,8,1)',
                                        plot_bgcolor='rgba(8,8,8,1)',
                                        height=500,
                                        margin=dict(l=40, r=60, t=50, b=40),
                                        font=dict(family='JetBrains Mono', size=10, color=THEME_TEXT)
                                    )
                                    
                                    ui.plotly(fig).classes('w-full').style('height: 500px;')
                                    
                                    # Stats
                                    mse = np.mean(diff_grid ** 2)
                                    mae = np.mean(diff_grid)
                                    with ui.row().classes('gap-4 mt-2'):
                                        ui.label(f'MSE: {mse:.6f}').style(f'color:{THEME_PRIMARY}; font-size: 0.8rem;')
                                        ui.label(f'MAE: {mae:.6f}').style(f'color:{THEME_SECONDARY}; font-size: 0.8rem;')
                                        ui.label(f'Samples: {n_samples} × {n_features} features').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                                    
                                except Exception as e:
                                    ui.label(f'Error loading: {e}').style(f'color:{THEME_ERROR};')
                        
                        # Bind events
                        epoch_slider.on('update:model-value', lambda e: load_reconstruction(e.args))
                        refresh_btn.on('click', lambda: load_reconstruction(epoch_slider.value))
                        
                        # Initial load
                        load_reconstruction(10)
                    
                    # CONSOLE TAB
                    with ui.tab_panel(tab_console).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        with ui.row().classes('items-center gap-3 mb-2'):
                            ui.label('// TRAINING_LOG').classes('terminal-header')
                            
                            def clear_log():
                                if MS.log_container:
                                    MS.log_container.clear()
                                MS.log_history = []
                            ui.button('CLEAR', on_click=clear_log, icon='delete').props('flat dense size=sm').classes('ml-auto')
                        
                        with ui.scroll_area().classes('w-full flex-1').style('background: #050505; border-radius: 4px; min-height: 200px;'):
                            MS.log_container = ui.column().classes('w-full p-3 gap-0')
                            with MS.log_container:
                                # Restore previous logs if any
                                if MS.log_history:
                                    colors = {'info': THEME_TEXT, 'success': THEME_PRIMARY, 'warning': THEME_WARN, 'error': THEME_ERROR}
                                    for msg, msg_type in MS.log_history[-100:]:  # Show last 100
                                        ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                else:
                                    ui.label('Ready. Select a dataset and click Train.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
    
    render_debug_console()


# ============================================================================
# ANALYSIS PAGE - Model Analysis & Visualization
# ============================================================================

