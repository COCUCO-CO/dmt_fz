"""Model training page."""
from pathlib import Path
import asyncio
import subprocess
import json
import sys
import os
import re
import yaml
import numpy as np
from nicegui import ui
import plotly.graph_objects as go

from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM
)
from app.state import MS
from app.visualization.styles.css import STYLE
from app.visualization.components.running_indicator import render_running_indicator
from app.visualization.components.global_header import render_global_header
from app.core.dataset_scanner import DatasetScanner
from app.core.dataset_scanner.models import DatasetType

AUTOENCODER_DIR = Path(__file__).parent.parent.parent.parent / "machine_learning" / "autoencoder"
AUTOENCODER_CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "autoencoder"

# Initialize global scanner instance
_dataset_scanner = DatasetScanner()


def detect_dataset_type(path: Path) -> dict:
    """
    Detect dataset type and structure using the intelligent DatasetScanner.
    
    Uses the new DatasetScanner module which supports:
    - Images (PNG, JPG, TIFF, WEBP, DICOM)
    - Graphs (PyG .pt, .gpickle, .graphml, phases-*.pkl)
    - Time Series (EEG: .bdf, .edf, .fif, .set; Audio: .wav; Arrays: .npy, .npz)
    - Tabular (.csv, .xlsx, .parquet, .h5)
    - Text (.txt, .json, .jsonl)
    
    Returns dict with detailed dataset information for compatibility with existing code.
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
        'error': None,
        # New fields from DatasetScanner
        'scanner_info': None,
        'type_specific': {},
        'classes': [],
        'subjects': [],
        'warnings': [],
        'suggestions': [],
    }
    
    if not path.exists():
        info['error'] = f"Path does not exist: {path}"
        return info
    
    # Use the new intelligent scanner
    try:
        scanner_info = _dataset_scanner.scan(path, deep_scan=True, sample_size=10)
        info['scanner_info'] = scanner_info
        
        # Convert DatasetType to string
        type_map = {
            DatasetType.IMAGE: 'image',
            DatasetType.GRAPH: 'graph',
            DatasetType.TIMESERIES: 'timeseries',
            DatasetType.TABULAR: 'tabular',
            DatasetType.TEXT: 'text',
            DatasetType.MIXED: 'mixed',
            DatasetType.UNKNOWN: 'unknown',
        }
        info['type'] = type_map.get(scanner_info.primary_type, 'unknown')
        
        # File information
        info['file_count'] = scanner_info.files.total_count
        if scanner_info.files.sample_files:
            info['sample_file'] = str(scanner_info.files.sample_files[0])
        
        # Structure information
        info['structure'] = scanner_info.structure.split_type.value
        
        # Classes/conditions
        if scanner_info.structure.has_classes:
            info['conditions'] = scanner_info.structure.classes
            info['classes'] = scanner_info.structure.classes
        
        # Subjects
        if scanner_info.structure.has_subjects:
            info['subjects'] = scanner_info.structure.subjects
        
        # Type-specific information
        info['type_specific'] = scanner_info.type_specific
        
        # Graph-specific (phases format)
        if scanner_info.primary_type == DatasetType.GRAPH:
            ts = scanner_info.type_specific
            if ts.get('format') == 'phases':
                info['bands'] = ts.get('bands', [])
                info['has_eeg'] = ts.get('has_eeg', False)
                info['has_stc'] = ts.get('has_stc', False)
                info['num_nodes_eeg'] = ts.get('n_channels_eeg', 0)
                info['num_nodes_stc'] = ts.get('n_parcels', 0)
                info['num_epochs_sample'] = ts.get('n_epochs_per_file', 0)
                info['data_sources'].append('phases (syncro + phases + amplitudes + kuramoto)')
            elif ts.get('format') == 'pyg':
                info['data_sources'].append('PyTorch Geometric graphs')
        
        # TimeSeries-specific
        elif scanner_info.primary_type == DatasetType.TIMESERIES:
            ts = scanner_info.type_specific
            if ts.get('is_eeg_format'):
                info['data_sources'].append('EEG raw data')
                info['has_eeg'] = True
                info['num_nodes_eeg'] = ts.get('num_channels', 0)
            elif ts.get('is_audio_format'):
                info['data_sources'].append('Audio data')
            else:
                info['data_sources'].append('NumPy arrays')
        
        # Image-specific
        elif scanner_info.primary_type == DatasetType.IMAGE:
            ts = scanner_info.type_specific
            if ts.get('sizes'):
                info['data_sources'].append(f"Images {ts.get('sizes', [])[0] if ts.get('sizes') else ''}")
        
        # Tabular-specific
        elif scanner_info.primary_type == DatasetType.TABULAR:
            ts = scanner_info.type_specific
            info['data_sources'].append(f"Tabular ({ts.get('num_rows', 0)} rows × {ts.get('num_columns', 0)} cols)")
        
        # Text-specific
        elif scanner_info.primary_type == DatasetType.TEXT:
            ts = scanner_info.type_specific
            info['data_sources'].append(f"Text ({ts.get('document_count', 0)} documents)")
        
        # Warnings and suggestions
        info['warnings'] = scanner_info.warnings
        info['suggestions'] = scanner_info.suggestions
        
        # Check for errors/warnings that should be shown
        if scanner_info.warnings:
            info['error'] = '; '.join(scanner_info.warnings[:2])  # Show first 2 warnings
        
        if not scanner_info.is_valid:
            info['error'] = "No recognized data files found"
            
    except Exception as e:
        info['error'] = f"Scanner error: {e}"
    
    return info


def model_log(msg: str, msg_type: str = 'info'):
    """Add message to model training log - persists even when tab switches."""
    # Store in history for persistence
    MS.log_history.append((msg, msg_type))
    # Keep only last 500 messages
    if len(MS.log_history) > 500:
        MS.log_history = MS.log_history[-500:]
    
    # Try to update UI if container exists and client is connected
    if MS.log_container:
        try:
            colors = {
                'info': THEME_TEXT,
                'success': THEME_PRIMARY,
                'warning': THEME_WARN,
                'error': THEME_ERROR
            }
            with MS.log_container:
                # Add line break before major sections
                if any(x in msg for x in ['Starting', 'Training completed', '====', 'Epoch 001 ']):
                    ui.label('').style('height: 8px;')
                ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')
            
            # Auto-scroll to bottom
            if hasattr(MS, 'log_scroll') and MS.log_scroll:
                MS.log_scroll.scroll_to(percent=1.0)
        except RuntimeError:
            # Client disconnected (tab switched), log is still stored in history
            pass


def update_status_indicator(status: str):
    """Update the training status (used by global indicator)."""
    MS.status = status


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
    
    # Global header with system monitor (CPU/RAM/GPU)
    render_global_header('model')
    
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
                    """Scan and detect dataset using the intelligent DatasetScanner."""
                    path = Path(dataset_path_input.value.strip())
                    MS.dataset_path = str(path)
                    
                    dataset_info_container.clear()
                    with dataset_info_container:
                        ui.label('🔍 Scanning...').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                    
                    info = detect_dataset_type(path)
                    MS.dataset_info = info
                    MS.dataset_type = info['type']
                    
                    dataset_info_container.clear()
                    with dataset_info_container:
                        # Show errors first but don't stop if we have valid data
                        has_valid_data = info['file_count'] > 0 and info['type'] != 'unknown'
                        
                        if not has_valid_data and info.get('error'):
                            ui.label(f"❌ {info['error']}").style(f'color:{THEME_ERROR}; font-size: 0.7rem;')
                            return
                        
                        # Main type indicator with icon
                        type_icons = {
                            'image': '🖼️',
                            'graph': '🔗',
                            'timeseries': '📈',
                            'tabular': '📊',
                            'text': '📝',
                            'mixed': '📦',
                        }
                        icon = type_icons.get(info['type'], '❓')
                        ui.label(f"{icon} Type: {info['type'].upper()}").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem; font-weight: bold;')
                        
                        # File count and size
                        scanner_info = info.get('scanner_info')
                        size_str = ''
                        if scanner_info:
                            size_str = f" ({scanner_info.files.total_size_human})"
                        ui.label(f"  📁 Files: {info['file_count']}{size_str}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Structure
                        if info['structure'] != 'unknown':
                            struct_label = info['structure'].replace('_', ' ').title()
                            ui.label(f"  📂 Structure: {struct_label}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Classes/Conditions
                        if info.get('conditions') or info.get('classes'):
                            classes = info.get('conditions') or info.get('classes', [])
                            if classes:
                                class_str = ', '.join(classes[:5])
                                if len(classes) > 5:
                                    class_str += f" (+{len(classes)-5})"
                                ui.label(f"  🏷️ Classes: {class_str}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Subjects
                        if info.get('subjects'):
                            ui.label(f"  👥 Subjects: {len(info['subjects'])}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Bands (for graph/phases data)
                        if info.get('bands'):
                            ui.label(f"  🎵 Bands: {', '.join(info['bands'])}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                        
                        # Data sources
                        if info.get('data_sources'):
                            ui.label(f"  💾 Data: {info['data_sources'][0]}").style(f'color:{THEME_SECONDARY}; font-size: 0.7rem;')
                        
                        # EEG/STC availability (graph data)
                        if info.get('has_eeg') or info.get('has_stc'):
                            sources = []
                            if info['has_eeg']:
                                sources.append(f"EEG ({info['num_nodes_eeg']} ch)")
                            if info['has_stc']:
                                sources.append(f"STC ({info['num_nodes_stc']} parcels)")
                            ui.label(f"  🧠 Sources: {' | '.join(sources)}").style(f'color:{THEME_PRIMARY}; font-size: 0.7rem;')
                        
                        # Epochs
                        if info.get('num_epochs_sample', 0) > 0:
                            ui.label(f"  📊 Epochs/subject: ~{info['num_epochs_sample']}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Type-specific info
                        ts = info.get('type_specific', {})
                        if info['type'] == 'image' and ts.get('sizes'):
                            sizes = ts.get('sizes', [])
                            if sizes:
                                ui.label(f"  📐 Size: {sizes[0]}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        elif info['type'] == 'tabular':
                            rows = ts.get('num_rows', ts.get('total_rows', 0))
                            cols = ts.get('num_columns', 0)
                            if rows or cols:
                                ui.label(f"  📋 Shape: {rows} × {cols}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        elif info['type'] == 'text':
                            docs = ts.get('document_count', 0)
                            if docs:
                                ui.label(f"  📄 Documents: {docs}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        elif info['type'] == 'timeseries':
                            channels = ts.get('num_channels', 0)
                            sr = ts.get('sampling_rate')
                            if channels:
                                sr_str = f" @ {sr}Hz" if sr else ""
                                ui.label(f"  📡 Channels: {channels}{sr_str}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Warnings (if any but still valid)
                        if info.get('warnings') and has_valid_data:
                            ui.label(f"  ⚠️ {info['warnings'][0]}").style(f'color:{THEME_WARN}; font-size: 0.65rem;')
                        
                        # Suggestions
                        if info.get('suggestions') and has_valid_data:
                            ui.label(f"  💡 {info['suggestions'][0]}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; font-style: italic;')
                        
                        # Compatible frameworks
                        if scanner_info and scanner_info.compatible_frameworks:
                            frameworks = ', '.join(scanner_info.compatible_frameworks[:4])
                            ui.label(f"  🔧 Compatible: {frameworks}").style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        
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
                        model_log(f"Dataset scanned: {info['type'].upper()} ({info['file_count']} files)", 'success')
                        if info.get('data_sources'):
                            model_log(f"  Format: {info['data_sources'][0]}", 'info')
                
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
                with ui.row().classes('items-center gap-2 mt-2 w-full'):
                    ui.label('Subsample:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 70px;')
                    subsample_slider = ui.slider(min=0.1, max=1.0, step=0.1, value=0.3).classes('flex-1')
                    subsample_label = ui.label('0.3').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem; min-width: 30px; text-align: right;')
                
                # Use value from event args directly for immediate sync
                subsample_slider.on_value_change(lambda e: subsample_label.set_text(f'{e.value:.1f}'))
                
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
                        MS.running_task_name = 'train.py'  # For global indicator
                        MS.history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [], 'epoch': []}
                        MS.log_history = []  # Clear log history
                        
                        # Clear previous logs and reset plot
                        try:
                            if MS.log_container:
                                MS.log_container.clear()
                            update_loss_plot()  # Reset the plot with empty data
                            
                            # Update status indicator
                            update_status_indicator('training')
                            
                            training_status.text = 'Training...'
                            training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                        except RuntimeError:
                            pass
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
                                        
                                        # Update loss plot (may fail if tab switched)
                                        try:
                                            update_loss_plot()
                                        except RuntimeError:
                                            pass
                                    
                                    val_match = val_pattern.search(line)
                                    if val_match:
                                        val_loss = float(val_match.group(1))
                                        MS.history['val_loss'].append(val_loss)
                                        try:
                                            update_loss_plot()
                                        except RuntimeError:
                                            pass
                            
                            await process.wait()
                            
                            if process.returncode == 0:
                                model_log("Training completed successfully!", 'success')
                                try:
                                    training_status.text = 'Completed'
                                    training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                                    update_status_indicator('completed')
                                except RuntimeError:
                                    pass
                            else:
                                model_log(f"Training failed with code {process.returncode}", 'error')
                                try:
                                    training_status.text = 'Failed'
                                    training_status.style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                                    update_status_indicator('error')
                                except RuntimeError:
                                    pass
                        
                        except Exception as e:
                            model_log(f"Error: {e}", 'error')
                            try:
                                training_status.text = 'Error'
                                training_status.style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                                update_status_indicator('error')
                            except RuntimeError:
                                pass
                        
                        finally:
                            MS.training = False
                            MS.running_task_name = ""
                            MS.current_process = None
                    
                    async def stop_training():
                        if MS.current_process:
                            MS.current_process.terminate()
                            model_log("Training stopped by user", 'warning')
                            try:
                                training_status.text = 'Stopped'
                                training_status.style(f'color:{THEME_WARN}; font-size: 0.75rem;')
                                update_status_indicator('idle')
                            except RuntimeError:
                                pass
                            MS.training = False
                            MS.running_task_name = ""
                    
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
                            epoch_slider = ui.slider(min=1, max=100, step=10, value=10).props('label label-always :label-value="value"').classes('flex-1')
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
                    with ui.tab_panel(tab_console).classes('p-2').style('height: 100%; display: flex; flex-direction: column; overflow: hidden;'):
                        with ui.row().classes('items-center gap-3 mb-2 shrink-0'):
                            ui.label('// TRAINING_LOG').classes('terminal-header')
                            
                            def clear_log():
                                if MS.log_container:
                                    MS.log_container.clear()
                                MS.log_history = []
                            ui.button('CLEAR', on_click=clear_log, icon='delete').props('flat dense size=sm').classes('ml-auto')
                        
                        MS.log_scroll = ui.scroll_area().classes('w-full').style('background: #050505; border-radius: 4px; flex: 1; min-height: 0;')
                        with MS.log_scroll:
                            MS.log_container = ui.column().classes('w-full p-3 gap-0')
                            with MS.log_container:
                                # Restore previous logs if any
                                if MS.log_history:
                                    colors = {'info': THEME_TEXT, 'success': THEME_PRIMARY, 'warning': THEME_WARN, 'error': THEME_ERROR}
                                    for msg, msg_type in MS.log_history[-100:]:  # Show last 100
                                        ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                else:
                                    ui.label('Ready. Select a dataset and click Train.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')


# ============================================================================
# ANALYSIS PAGE - Model Analysis & Visualization
# ============================================================================

