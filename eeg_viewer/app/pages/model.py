"""Model training page."""
from pathlib import Path
import asyncio
import signal
import sys
import os
import re
import numpy as np
from nicegui import ui, background_tasks
import plotly.graph_objects as go

from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM
)
from app.state import MS
from app.visualization.styles.css import STYLE
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
                info['num_timepoints'] = ts.get('n_timepoints', 0)
                info['eeg_feature_types'] = ts.get('eeg_feature_types', [])
                info['stc_feature_types'] = ts.get('stc_feature_types', [])
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
    """Add message to model training log - persists even when tab switches.
    
    NOTE: This function only stores logs in history. The UI is updated
    by a timer that polls this history, which handles page switches gracefully.
    """
    # Store in history for persistence
    MS.log_history.append((msg, msg_type))
    # Keep only last 500 messages
    if len(MS.log_history) > 500:
        MS.log_history = MS.log_history[-500:]
    
    # Track last displayed count for polling
    if not hasattr(MS, '_last_log_count'):
        MS._last_log_count = 0


def update_status_indicator(status: str):
    """Update the training status (used by global indicator)."""
    MS.status = status


async def _run_training_background(cmd: list, env: dict, cwd: str):
    """Background task to run training subprocess.
    
    This runs independently of the client connection, so it continues
    even when the user navigates away from the model page.
    """
    epoch_pattern = re.compile(r'Epoch (\d+).*Loss: ([\d.]+).*Recon: ([\d.]+).*KL: ([\d.]+)')
    val_pattern = re.compile(r'Epoch \d+.*\[val\].*Loss: ([\d.]+)')
    
    process = None
    try:
        # Start subprocess with new process group for clean termination
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            cwd=cwd,
            start_new_session=True  # Create new process group
        )
        MS.current_process = process
        
        # Read output line by line with timeout to prevent blocking
        buffer = ""
        while True:
            try:
                # Read chunks with small timeout
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(1024),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
                    continue
                
                if not chunk:
                    # Process any remaining buffer
                    if buffer.strip():
                        model_log(buffer.strip(), 'info')
                    break
                
                # Add to buffer and process complete lines
                buffer += chunk.decode('utf-8', errors='replace')
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    line = line.strip()
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
                        
                        val_match = val_pattern.search(line)
                        if val_match:
                            val_loss = float(val_match.group(1))
                            MS.history['val_loss'].append(val_loss)
                
                # Also process carriage returns (for tqdm)
                while '\r' in buffer and '\n' not in buffer:
                    line, buffer = buffer.split('\r', 1)
                    line = line.strip()
                    if line:
                        model_log(line, 'info')
                        
            except asyncio.CancelledError:
                raise  # Re-raise to handle in outer except
        
        await process.wait()
        
        if process.returncode == 0:
            model_log("Training completed successfully!", 'success')
            update_status_indicator('completed')
        elif process.returncode == -signal.SIGTERM or process.returncode == -signal.SIGKILL:
            # Process was terminated by us
            pass
        else:
            model_log(f"Training failed with code {process.returncode}", 'error')
            update_status_indicator('error')
    
    except asyncio.CancelledError:
        model_log("Training task cancelled", 'warning')
        update_status_indicator('idle')
        # Kill entire process group to stop workers too
        if process and process.returncode is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
    
    except Exception as e:
        model_log(f"Error: {e}", 'error')
        update_status_indicator('error')
        # Try to kill process on error
        if process and process.returncode is None:
            try:
                process.kill()
            except (ProcessLookupError, OSError):
                pass
    
    finally:
        MS.training = False
        MS.running_task_name = ""
        MS.current_process = None


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
            # Use detected conditions from dataset scan (no hardcoded fallback)
            'conditions': dataset_info.get('conditions') or dataset_info.get('classes') or [],
            'bands': dataset_info.get('bands') or ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
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
                'gat_layers': 0,  # 0 = MLP decoder, >0 = GAT decoder
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
            with ui.card().classes('dark-card p-4 w-full').style('border: 1px solid #f472b6;'):
                ui.label('// DATASET').classes('terminal-header')
                
                # Restore dataset path from state if available
                default_path = MS.dataset_path if MS.dataset_path else '/media/storage_hdd/dmt_fz/fwd-inv-stc'
                with ui.row().classes('items-center gap-2 mt-2 w-full'):
                    ui.label('Path:').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem; min-width: 50px;')
                    dataset_path_input = ui.input(
                        value=default_path
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
                        
                        # Update data preview
                        update_data_preview(info)
                        
                        # Update source info label with real data
                        eeg_info = f"EEG: {info.get('num_nodes_eeg', 0)} ch" if info.get('has_eeg') else ""
                        stc_info = f"STC: {info.get('num_nodes_stc', 0)} parcels" if info.get('has_stc') else ""
                        if eeg_info and stc_info:
                            source_info_label.text = f"{eeg_info} | {stc_info}"
                        elif eeg_info:
                            source_info_label.text = eeg_info
                        elif stc_info:
                            source_info_label.text = stc_info
                        else:
                            source_info_label.text = "No EEG/STC data found"
                
                ui.button('Scan Dataset', on_click=scan_dataset, icon='search').props('dense').classes('mt-2').style('background:#f472b6; color:black;')
                
                # Data Preview Panel (appears after scan)
                data_preview_container = ui.column().classes('w-full mt-2')
                
                def update_data_preview(info):
                    """Update data preview panel with scanned dataset info."""
                    data_preview_container.clear()
                    
                    if not info or info.get('type') == 'unknown':
                        return
                    
                    with data_preview_container:
                        ui.separator().classes('my-2')
                        ui.label('📊 Data Preview').style(f'color:{THEME_SECONDARY}; font-size: 0.7rem; font-weight: bold;')
                        
                        # Show sample info based on type
                        if info.get('type') == 'graph':
                            # Graph/phases data preview
                            with ui.column().classes('gap-1 mt-1'):
                                # Show both EEG and STC if available
                                has_eeg = info.get('has_eeg', False)
                                has_stc = info.get('has_stc', False)
                                nodes_eeg = info.get('num_nodes_eeg', 0)
                                nodes_stc = info.get('num_nodes_stc', 0)
                                
                                if has_eeg:
                                    ui.label(f'📡 EEG: {nodes_eeg} channels').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem;')
                                if has_stc:
                                    ui.label(f'🧠 STC: {nodes_stc} parcels').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                                
                                # Bands info (extracted from actual data)
                                bands = info.get('bands', [])
                                if bands:
                                    ui.label(f'🎵 Bands: {", ".join(bands)} ({len(bands)} total)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                
                                # Timepoints per epoch (raw data shape)
                                timepoints = info.get('num_timepoints', 0)
                                if timepoints:
                                    ui.label(f'⏱️ Timepoints/epoch: {timepoints}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                
                                # Node features count (feature_types × bands)
                                eeg_ft = info.get('eeg_feature_types', [])
                                stc_ft = info.get('stc_feature_types', [])
                                n_bands = len(info.get('bands', []))
                                if eeg_ft and n_bands:
                                    eeg_node_features = len(eeg_ft) * n_bands
                                    ui.label(f'🔢 EEG node features: {eeg_node_features} ({len(eeg_ft)} types × {n_bands} bands)').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem; font-weight: bold;')
                                if stc_ft and n_bands:
                                    stc_node_features = len(stc_ft) * n_bands
                                    ui.label(f'🔢 STC node features: {stc_node_features} ({len(stc_ft)} types × {n_bands} bands)').style(f'color:{THEME_PRIMARY}; font-size: 0.65rem; font-weight: bold;')
                                
                                # Epochs
                                epochs = info.get('num_epochs_sample', 0)
                                if epochs:
                                    ui.label(f'📊 Epochs/subject: {epochs}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                
                                # Edge estimate
                                nodes = nodes_eeg if has_eeg else nodes_stc
                                if nodes > 0:
                                    ui.label(f'🔗 Edges: ~{nodes*(nodes-1)//2} (fully conn.)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                
                                # Total graphs estimate
                                file_count = info.get('file_count', 0)
                                if epochs and bands and file_count:
                                    total = file_count * epochs * len(bands)
                                    ui.label(f'📦 Total graphs: ~{total:,}').style(f'color:{THEME_WARN}; font-size: 0.6rem;')
                        
                        elif info.get('type') == 'image':
                            ts = info.get('type_specific', {})
                            sizes = ts.get('sizes', [])
                            channels = ts.get('channels', 3)
                            color_mode = ts.get('color_mode', 'RGB')
                            
                            with ui.column().classes('gap-1 mt-1'):
                                if sizes:
                                    ui.label(f'📐 Size: {sizes[0][0]}×{sizes[0][1]}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                ui.label(f'🎨 {color_mode} ({channels} ch)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                
                                classes = info.get('classes', [])
                                if classes:
                                    ui.label(f'🏷️ Classes: {len(classes)}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        
                        elif info.get('type') == 'timeseries':
                            ts = info.get('type_specific', {})
                            channels = ts.get('num_channels', 0)
                            sr = ts.get('sampling_rate')
                            
                            with ui.column().classes('gap-1 mt-1'):
                                if channels:
                                    ui.label(f'📡 Channels: {channels}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                                if sr:
                                    ui.label(f'⏱️ Sample rate: {sr} Hz').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                
                # Data source selector (EEG vs STC)
                ui.separator().classes('my-2')
                with ui.row().classes('items-center gap-2'):
                    ui.label('Source:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 50px;')
                    data_source_select = ui.select(
                        ['EEG (channels)', 'STC (parcels)'],
                        value=MS.ui_params.get('data_source', 'EEG (channels)')
                    ).props('dense dark').classes('flex-1')
                    data_source_select.on_value_change(lambda e: MS.ui_params.update({'data_source': e.value}))
                
                source_info_label = ui.label('Scan dataset to see channel info').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                
                # Band selection (restore from state)
                ui.separator().classes('my-2')
                ui.label('Bands to use:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem;')
                with ui.row().classes('gap-2 flex-wrap'):
                    band_checks = {}
                    for band in ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']:
                        band_val = MS.ui_params.get('bands', {}).get(band, band == 'Alpha')
                        band_checks[band] = ui.checkbox(band, value=band_val).props('dense')
                        # Save on change
                        band_checks[band].on_value_change(
                            lambda e, b=band: MS.ui_params['bands'].update({b: e.value})
                        )
                
                ui.label('Tip: Start with 1-2 bands for faster training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                
                # Subsample option (restore from state)
                subsample_val = MS.ui_params.get('subsample', 0.3)
                with ui.row().classes('items-center gap-2 mt-2 w-full'):
                    ui.label('Subsample:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.75rem; min-width: 70px;')
                    subsample_slider = ui.slider(min=0.1, max=1.0, step=0.1, value=subsample_val).classes('flex-1')
                    subsample_label = ui.label(f'{subsample_val:.1f}').style(f'color:{THEME_PRIMARY}; font-size: 0.75rem; min-width: 30px; text-align: right;')
                
                def on_subsample_change(e):
                    subsample_label.set_text(f'{e.value:.1f}')
                    MS.ui_params['subsample'] = e.value
                subsample_slider.on_value_change(on_subsample_change)
                
                ui.label('Use 0.1-0.3 for quick tests, 1.0 for full training').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
            
            # MODEL CONFIGURATION
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// MODEL CONFIG').classes('terminal-header')
                
                # Helper to save params on change
                def save_param(key):
                    return lambda e: MS.ui_params.update({key: e.value})
                
                with ui.column().classes('gap-1 mt-2'):
                    # Model type selector
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Type:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem; min-width: 70px;')
                        model_type_select = ui.select(
                            ['VAE (Graph)', 'VAE (Image)', 'AE (Graph)'],
                            value='VAE (Graph)'
                        ).props('dense dark').classes('flex-1')
                    
                    ui.separator().classes('my-1')
                    
                    # Architecture params (restored from state)
                    ui.label('Architecture').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Latent:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        latent_dim = ui.number(value=MS.ui_params.get('latent_dim', 64), min=8, max=512, step=8).props('dense').classes('w-16')
                        latent_dim.on_value_change(save_param('latent_dim'))
                        ui.label('Hidden:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        hidden_dim = ui.number(value=MS.ui_params.get('hidden_dim', 64), min=16, max=256, step=16).props('dense').classes('w-16')
                        hidden_dim.on_value_change(save_param('hidden_dim'))
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('GAT layers:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        gat_layers = ui.number(value=MS.ui_params.get('gat_layers', 3), min=1, max=6).props('dense').classes('w-16')
                        gat_layers.on_value_change(save_param('gat_layers'))
                        ui.label('Heads:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        attention_heads = ui.number(value=MS.ui_params.get('attention_heads', 4), min=1, max=8).props('dense').classes('w-16')
                        attention_heads.on_value_change(save_param('attention_heads'))
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Dropout:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        dropout = ui.number(value=MS.ui_params.get('dropout', 0.2), min=0.0, max=0.5, step=0.05).props('dense').classes('w-16')
                        dropout.on_value_change(save_param('dropout'))
                        ui.label('Attn drop:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        attn_dropout = ui.number(value=MS.ui_params.get('attn_dropout', 0.1), min=0.0, max=0.3, step=0.05).props('dense').classes('w-16')
                        attn_dropout.on_value_change(save_param('attn_dropout'))
                    
                    ui.separator().classes('my-1')
                    
                    # Decoder params
                    ui.label('Decoder').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Dec GAT:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        decoder_gat_layers = ui.number(value=MS.ui_params.get('decoder_gat_layers', 0), min=0, max=4).props('dense').classes('w-16')
                        decoder_gat_layers.on_value_change(save_param('decoder_gat_layers'))
                        ui.label('(0=MLP)').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Dec dims:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        decoder_hidden_dims = ui.input(value=MS.ui_params.get('decoder_hidden_dims', '256,128')).props('dense').classes('w-24')
                        decoder_hidden_dims.on_value_change(save_param('decoder_hidden_dims'))
                    
                    ui.separator().classes('my-1')
                    
                    # Training params
                    ui.label('Training').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Epochs:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        num_epochs = ui.number(value=MS.ui_params.get('num_epochs', 100), min=10, max=500, step=10).props('dense').classes('w-16')
                        num_epochs.on_value_change(save_param('num_epochs'))
                        ui.label('Batch:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        batch_size = ui.number(value=MS.ui_params.get('batch_size', 256), min=16, max=1024, step=16).props('dense').classes('w-16')
                        batch_size.on_value_change(save_param('batch_size'))
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Learn rate:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        learning_rate = ui.select(
                            ['1e-2', '5e-3', '1e-3', '5e-4', '1e-4'],
                            value=MS.ui_params.get('learning_rate', '1e-3')
                        ).props('dense dark').classes('w-20')
                        learning_rate.on_value_change(save_param('learning_rate'))
                        ui.label('Decay:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        weight_decay = ui.select(
                            ['0', '1e-5', '1e-4', '1e-3'],
                            value=MS.ui_params.get('weight_decay', '1e-5')
                        ).props('dense dark').classes('w-20')
                        weight_decay.on_value_change(save_param('weight_decay'))
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Optimizer:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        optimizer_select = ui.select(
                            ['adamw', 'adam', 'sgd'],
                            value=MS.ui_params.get('optimizer', 'adamw')
                        ).props('dense dark').classes('w-20')
                        optimizer_select.on_value_change(save_param('optimizer'))
                        ui.label('Scheduler:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        scheduler_select = ui.select(
                            ['cosine', 'reduce_on_plateau', 'step', 'none'],
                            value=MS.ui_params.get('scheduler', 'cosine')
                        ).props('dense dark').classes('w-24')
                        scheduler_select.on_value_change(save_param('scheduler'))
                    
                    ui.separator().classes('my-1')
                    
                    # Early stopping & regularization
                    ui.label('Regularization').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Patience:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        patience = ui.number(value=MS.ui_params.get('patience', 25), min=5, max=100, step=5).props('dense').classes('w-16')
                        patience.on_value_change(save_param('patience'))
                        ui.label('Grad clip:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        grad_clip = ui.number(value=MS.ui_params.get('grad_clip', 0.5), min=0.0, max=5.0, step=0.1).props('dense').classes('w-16')
                        grad_clip.on_value_change(save_param('grad_clip'))
                    
                    ui.separator().classes('my-1')
                    
                    # Loss params
                    ui.label('Loss').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('KL weight:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        kl_weight = ui.number(value=MS.ui_params.get('kl_weight', 0.01), min=0.0, max=1.0, step=0.01).props('dense').classes('w-16')
                        kl_weight.on_value_change(save_param('kl_weight'))
                        ui.label('β anneal:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        beta_annealing = ui.switch(value=MS.ui_params.get('beta_annealing', True)).props('dense')
                        beta_annealing.on_value_change(save_param('beta_annealing'))
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Node wt:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        node_weight = ui.number(value=MS.ui_params.get('node_weight', 0.3), min=0.0, max=1.0, step=0.1).props('dense').classes('w-16')
                        node_weight.on_value_change(save_param('node_weight'))
                        ui.label('Edge wt:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        edge_weight = ui.number(value=MS.ui_params.get('edge_weight', 1.0), min=0.0, max=2.0, step=0.1).props('dense').classes('w-16')
                        edge_weight.on_value_change(save_param('edge_weight'))
                    
                    ui.separator().classes('my-1')
                    
                    # Workers (parallelization)
                    ui.label('Parallelization').style(f'color:{THEME_SECONDARY}; font-size: 0.65rem;')
                    
                    with ui.row().classes('items-center gap-2'):
                        ui.label('Workers:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem; min-width: 70px;')
                        num_workers = ui.number(value=MS.ui_params.get('num_workers', 4), min=0, max=32, step=1).props('dense').classes('w-16')
                        num_workers.on_value_change(save_param('num_workers'))
                        ui.label('Dataset:').style(f'color:{THEME_TEXT_DIM}; font-size: 0.65rem;')
                        dataset_workers = ui.number(value=MS.ui_params.get('dataset_workers', 8), min=1, max=32, step=1).props('dense').classes('w-16')
                        dataset_workers.on_value_change(save_param('dataset_workers'))
                    
                    ui.label('Workers=0 uses main thread. Dataset workers for building graphs.').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
            
            # TRAINING CONTROLS
            with ui.card().classes('dark-card p-4 w-full'):
                ui.label('// TRAINING').classes('terminal-header')
                
                # Initialize status based on current state
                initial_status = 'Training...' if MS.training else MS.status.title() if MS.status != 'idle' else 'Ready'
                initial_color = THEME_PRIMARY if MS.training else (THEME_PRIMARY if MS.status == 'completed' else THEME_TEXT_DIM)
                training_status = ui.label(initial_status).style(f'color:{initial_color}; font-size: 0.75rem;').classes('mt-2')
                
                # Timer to sync training status
                def poll_training_status():
                    try:
                        if MS.training:
                            training_status.text = 'Training...'
                            training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                        elif MS.status == 'completed':
                            training_status.text = 'Completed'
                            training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                        elif MS.status == 'error':
                            training_status.text = 'Error'
                            training_status.style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
                    except Exception:
                        pass
                
                ui.timer(1.0, poll_training_status)
                
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
                        
                        # Decoder config
                        MS.config['model']['decoder']['gat_layers'] = int(decoder_gat_layers.value)
                        try:
                            dims_str = decoder_hidden_dims.value.strip()
                            if dims_str:
                                MS.config['model']['decoder']['hidden_dims'] = [int(x.strip()) for x in dims_str.split(',') if x.strip()]
                        except ValueError:
                            MS.config['model']['decoder']['hidden_dims'] = [256, 128]  # Default
                        
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
                        
                        # Validate conditions were detected
                        conditions = MS.config['data'].get('conditions', [])
                        if not conditions:
                            ui.notify('No conditions detected in dataset. Check directory structure.', type='error')
                            return
                        model_log(f"Conditions: {', '.join(conditions)}", 'info')
                        
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
                        MS._last_log_count = 0  # Reset log polling counter
                        MS._last_epoch_count = 0  # Reset epoch polling counter
                        
                        # Update status indicator
                        update_status_indicator('training')
                        
                        try:
                            training_status.text = 'Training...'
                            training_status.style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
                        except Exception:
                            pass
                        model_log(f"Starting training with config: {config_path}", 'info')
                        
                        # Get worker settings
                        n_workers = int(num_workers.value) if num_workers.value else 4
                        n_dataset_workers = int(dataset_workers.value) if dataset_workers.value else 8
                        model_log(f"Workers: DataLoader={n_workers}, Dataset={n_dataset_workers}", 'info')
                        
                        # Prepare command and environment
                        cmd = [
                            sys.executable,
                            '-u',  # Unbuffered output - critical for real-time logs
                            str(AUTOENCODER_DIR / 'train.py'),
                            '--config', str(config_path),
                            '--subsample', str(subsample_slider.value),
                            '--workers', str(n_workers),
                            '--dataset-workers', str(n_dataset_workers),
                        ]
                        
                        env = os.environ.copy()
                        env['PYTHONPATH'] = str(AUTOENCODER_DIR.parent)
                        env['PYTHONUNBUFFERED'] = '1'  # Force unbuffered output
                        
                        # Run training as background task (continues even when navigating away)
                        background_tasks.create(
                            _run_training_background(cmd, env, str(AUTOENCODER_DIR))
                        )
                    
                    async def stop_training():
                        if MS.current_process:
                            try:
                                # Kill entire process group (terminates all child workers)
                                try:
                                    pgid = os.getpgid(MS.current_process.pid)
                                    os.killpg(pgid, signal.SIGTERM)
                                except (ProcessLookupError, OSError):
                                    # Fallback to just terminating main process
                                    try:
                                        MS.current_process.terminate()
                                    except ProcessLookupError:
                                        pass
                                
                                # Wait a bit for cleanup
                                try:
                                    await asyncio.wait_for(MS.current_process.wait(), timeout=2.0)
                                except asyncio.TimeoutError:
                                    # Force kill if it doesn't stop
                                    try:
                                        os.killpg(os.getpgid(MS.current_process.pid), signal.SIGKILL)
                                    except (ProcessLookupError, OSError):
                                        pass
                            except Exception as e:
                                model_log(f"Error stopping process: {e}", 'warning')
                            
                            model_log("Training stopped by user", 'warning')
                            try:
                                training_status.text = 'Stopped'
                                training_status.style(f'color:{THEME_WARN}; font-size: 0.75rem;')
                                update_status_indicator('idle')
                            except RuntimeError:
                                pass
                            MS.training = False
                            MS.running_task_name = ""
                            MS.current_process = None
                    
                    ui.button('Train', on_click=start_training, icon='play_arrow').props('dense').style(f'background:{THEME_PRIMARY}; color:black;')
                    ui.button('Stop', on_click=stop_training, icon='stop').props('dense color=negative')
        
        # RIGHT PANEL: Visualization & Logs
        with ui.column().classes('flex-1').style('min-height: 0; display: flex; flex-direction: column;'):
            
            with ui.card().classes('dark-card p-2 w-full flex-1').style('display: flex; flex-direction: column; min-height: 0;'):
                with ui.tabs().classes('w-full').style(f'background: {THEME_BG};') as model_tabs:
                    tab_arch = ui.tab('ARCH', icon='account_tree').style(f'color:{THEME_WARN};')
                    tab_metrics = ui.tab('METRICS', icon='show_chart').style('color:#f472b6;')
                    tab_recon = ui.tab('RECON', icon='compare').style(f'color:{THEME_SECONDARY};')
                    tab_console = ui.tab('CONSOLE', icon='terminal').style(f'color:{THEME_PRIMARY};')
                
                with ui.tab_panels(model_tabs, value=tab_arch).classes('w-full').style('flex: 1; min-height: 0; overflow: hidden;'):
                    
                    # ARCHITECTURE TAB
                    with ui.tab_panel(tab_arch).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        ui.label('▌MODEL ARCHITECTURE').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                        
                        arch_container = ui.column().classes('w-full flex-1')
                        
                        def update_architecture_viz():
                            """Update architecture visualization based on current config."""
                            arch_container.clear()
                            
                            with arch_container:
                                # Get current config values
                                enc_layers = int(gat_layers.value) if gat_layers.value else 3
                                enc_hidden = int(hidden_dim.value) if hidden_dim.value else 64
                                enc_heads = int(attention_heads.value) if attention_heads.value else 4
                                lat_dim = int(latent_dim.value) if latent_dim.value else 64
                                dec_gat = int(decoder_gat_layers.value) if decoder_gat_layers.value else 0
                                
                                try:
                                    dec_dims = [int(x.strip()) for x in decoder_hidden_dims.value.split(',') if x.strip()]
                                except:
                                    dec_dims = [256, 128]
                                
                                # Get real values from dataset info (no hardcoded fallbacks)
                                selected_bands = [b for b, cb in band_checks.items() if cb.value]
                                n_bands = len(selected_bands) if selected_bands else 0
                                
                                # Get node count based on selected source (EEG vs STC)
                                if MS.dataset_info:
                                    use_stc = 'STC' in data_source_select.value if data_source_select else False
                                    if use_stc:
                                        input_nodes = MS.dataset_info.get('num_nodes_stc', 0)
                                    else:
                                        input_nodes = MS.dataset_info.get('num_nodes_eeg', 0)
                                else:
                                    input_nodes = 0
                                
                                # Features depend on node_features config (calculated at training time)
                                # Show as "configured" since actual count depends on settings
                                input_features = "cfg"
                                
                                # Create visual representation using cards
                                with ui.row().classes('w-full gap-2 items-center justify-center flex-wrap'):
                                    # Input
                                    with ui.card().classes('p-2').style(f'background: {THEME_CARD}; border: 1px solid {THEME_BORDER}; min-width: 80px;'):
                                        ui.label('INPUT').style(f'color:{THEME_PRIMARY}; font-size: 0.6rem; font-weight: bold;')
                                        nodes_text = f'{input_nodes} nodes' if input_nodes > 0 else 'Scan first'
                                        ui.label(nodes_text).style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                        bands_text = f'{n_bands} bands' if n_bands > 0 else 'Select bands'
                                        ui.label(bands_text).style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                    
                                    ui.label('→').style(f'color:{THEME_TEXT_DIM}; font-size: 1.2rem;')
                                    
                                    # Encoder
                                    with ui.card().classes('p-2').style(f'background: {THEME_CARD}; border: 1px solid {THEME_SECONDARY}; min-width: 90px;'):
                                        ui.label('ENCODER').style(f'color:{THEME_SECONDARY}; font-size: 0.6rem; font-weight: bold;')
                                        ui.label(f'GAT × {enc_layers}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                        ui.label(f'{enc_hidden}d × {enc_heads}h').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                    
                                    ui.label('→').style(f'color:{THEME_TEXT_DIM}; font-size: 1.2rem;')
                                    
                                    # Latent
                                    with ui.card().classes('p-2').style(f'background: {THEME_CARD}; border: 1px solid #f472b6; min-width: 70px;'):
                                        ui.label('LATENT').style('color:#f472b6; font-size: 0.6rem; font-weight: bold;')
                                        ui.label(f'{lat_dim} dim').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                        ui.label('μ + σ').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                    
                                    ui.label('→').style(f'color:{THEME_TEXT_DIM}; font-size: 1.2rem;')
                                    
                                    # Decoder
                                    with ui.card().classes('p-2').style(f'background: {THEME_CARD}; border: 1px solid {THEME_WARN}; min-width: 90px;'):
                                        ui.label('DECODER').style(f'color:{THEME_WARN}; font-size: 0.6rem; font-weight: bold;')
                                        if dec_gat > 0:
                                            ui.label(f'GAT × {dec_gat}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                        else:
                                            ui.label('MLP').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                        ui.label(f'{" → ".join(map(str, dec_dims))}').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                    
                                    ui.label('→').style(f'color:{THEME_TEXT_DIM}; font-size: 1.2rem;')
                                    
                                    # Output
                                    with ui.card().classes('p-2').style(f'background: {THEME_CARD}; border: 1px solid {THEME_BORDER}; min-width: 80px;'):
                                        ui.label('OUTPUT').style(f'color:{THEME_PRIMARY}; font-size: 0.6rem; font-weight: bold;')
                                        ui.label(nodes_text).style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                        ui.label('node features').style(f'color:{THEME_TEXT_DIM}; font-size: 0.55rem;')
                                
                                # Summary stats (only show real data)
                                ui.separator().classes('my-3')
                                with ui.row().classes('w-full gap-4 justify-center'):
                                    with ui.column().classes('items-center'):
                                        ui.label(f'{input_nodes if input_nodes > 0 else "?"}').style(f'color:{THEME_SECONDARY}; font-size: 1rem; font-weight: bold;')
                                        ui.label('Input Nodes').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                    
                                    with ui.column().classes('items-center'):
                                        ui.label(f'{lat_dim}').style('color:#f472b6; font-size: 1rem; font-weight: bold;')
                                        ui.label('Latent Dim').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                    
                                    with ui.column().classes('items-center'):
                                        ui.label(f'{n_bands if n_bands > 0 else "?"}').style(f'color:{THEME_WARN}; font-size: 1rem; font-weight: bold;')
                                        ui.label('Bands Selected').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                                
                                # Refresh button
                                ui.button('Refresh', on_click=update_architecture_viz, icon='refresh').props('flat dense size=sm').classes('mt-2')
                        
                        # Initial render
                        update_architecture_viz()
                    
                    # METRICS TAB
                    with ui.tab_panel(tab_metrics).classes('p-2').style('height: 100%; display: flex; flex-direction: column;'):
                        ui.label('▌TRAINING METRICS').style('color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                        
                        loss_plot_container = ui.column().classes('w-full flex-1')
                        
                        # Track last epoch count for polling
                        if not hasattr(MS, '_last_epoch_count'):
                            MS._last_epoch_count = 0
                        
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
                            
                            # Update axes with log scale for better visualization
                            fig.update_xaxes(gridcolor='rgba(0,255,136,0.1)', showgrid=True)
                            fig.update_yaxes(
                                gridcolor='rgba(0,255,136,0.1)', 
                                showgrid=True,
                                type='log',  # Log scale for loss values
                                dtick=1,  # Show 10^0, 10^1, etc.
                            )
                            
                            return fig
                        
                        def update_loss_plot():
                            """Update the loss plot with current history."""
                            try:
                                loss_plot_container.clear()
                                with loss_plot_container:
                                    fig = make_loss_figure()
                                    ui.plotly(fig).classes('w-full').style('height: 280px;')
                            except Exception:
                                pass  # Ignore errors when page is destroyed
                        
                        # Initial plot
                        update_loss_plot()
                        
                        # Stats summary
                        with ui.row().classes('w-full gap-4 mt-4'):
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Best Val Loss').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                best_val_label = ui.label('--').style('color:#f472b6; font-size: 1.2rem; font-weight: bold;')
                            
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Current Epoch').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                current_epoch_label = ui.label('0').style(f'color:{THEME_PRIMARY}; font-size: 1.2rem; font-weight: bold;')
                            
                            with ui.card().classes('dark-card p-3 flex-1'):
                                ui.label('Train Loss').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                                train_loss_label = ui.label('--').style(f'color:{THEME_SECONDARY}; font-size: 1.2rem; font-weight: bold;')
                        
                        def poll_metrics():
                            """Poll for new metrics and update UI."""
                            try:
                                current_epoch_count = len(MS.history.get('epoch', []))
                                
                                # Update stats labels
                                if MS.history['epoch']:
                                    current_epoch_label.text = str(MS.history['epoch'][-1])
                                if MS.history['train_loss']:
                                    train_loss_label.text = f"{MS.history['train_loss'][-1]:.4f}"
                                if MS.history['val_loss']:
                                    best_val_label.text = f"{min(MS.history['val_loss']):.4f}"
                                
                                # Update plot only when new epochs come in
                                if current_epoch_count > MS._last_epoch_count:
                                    MS._last_epoch_count = current_epoch_count
                                    update_loss_plot()
                            except Exception:
                                pass  # Ignore errors when page is destroyed
                        
                        ui.timer(1.0, poll_metrics)  # Poll every second
                    
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
                        # Store reference for clear function
                        console_state = {'log_container': None}
                        
                        with ui.row().classes('items-center gap-3 mb-2 shrink-0'):
                            ui.label('// TRAINING_LOG').classes('terminal-header')
                            ui.element('div').classes('flex-1')  # Spacer
                            
                            def clear_log():
                                if console_state['log_container']:
                                    console_state['log_container'].clear()
                                    MS.log_history = []
                                    MS._last_log_count = 0
                                    with console_state['log_container']:
                                        ui.label('Log cleared.').style(f'color:{THEME_TEXT_DIM}; font-family: JetBrains Mono; font-size: 0.75rem;')
                            
                            ui.button('CLEAR', on_click=clear_log, icon='delete').props('flat dense size=sm')
                        
                        log_scroll_area = ui.scroll_area().classes('w-full').style('background: #050505; border-radius: 4px; flex: 1; min-height: 0;')
                        with log_scroll_area:
                            log_container = ui.column().classes('w-full p-3 gap-0')
                            console_state['log_container'] = log_container  # Store reference
                            
                            # Initialize tracking
                            MS._last_log_count = 0
                            
                            # Restore previous logs if any
                            colors = {'info': THEME_TEXT, 'success': THEME_PRIMARY, 'warning': THEME_WARN, 'error': THEME_ERROR}
                            if MS.log_history:
                                with log_container:
                                    for msg, msg_type in MS.log_history[-100:]:  # Show last 100
                                        ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                MS._last_log_count = len(MS.log_history)
                            else:
                                with log_container:
                                    ui.label('Ready. Select a dataset and click Train.').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
                        
                        # Status indicator for polling
                        with ui.row().classes('items-center gap-2 mt-2'):
                            poll_status = ui.label('').style(f'color:{THEME_TEXT_DIM}; font-size: 0.6rem;')
                        
                        # Polling timer for log updates (runs while on this page)
                        def poll_logs():
                            """Poll for new log messages and update UI."""
                            try:
                                current_count = len(MS.log_history)
                                # Update status indicator
                                status_text = f"📊 Logs: {current_count} | Last: {MS._last_log_count} | Training: {'🟢' if MS.training else '⚫'}"
                                poll_status.text = status_text
                                
                                if current_count > MS._last_log_count:
                                    # Add only new messages
                                    new_messages = MS.log_history[MS._last_log_count:]
                                    with log_container:
                                        for msg, msg_type in new_messages:
                                            if any(x in msg for x in ['Starting', 'Training completed', '====', 'Epoch 001 ']):
                                                ui.label('').style('height: 8px;')
                                            ui.label(msg).style(f'color:{colors.get(msg_type, THEME_TEXT)}; font-family: JetBrains Mono; font-size: 0.75rem;')
                                    MS._last_log_count = current_count
                                    # Auto-scroll to bottom
                                    log_scroll_area.scroll_to(percent=1.0)
                            except Exception as e:
                                poll_status.text = f"❌ Error: {e}"
                        
                        ui.timer(0.5, poll_logs)  # Poll every 500ms


# ============================================================================
# ANALYSIS PAGE - Model Analysis & Visualization
# ============================================================================

