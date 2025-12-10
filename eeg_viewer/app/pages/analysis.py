"""Model analysis page."""
from pathlib import Path
import asyncio
import numpy as np
from nicegui import ui
import plotly.graph_objects as go

from config import (
    THEME_BG, THEME_CARD, THEME_BORDER, THEME_PRIMARY, THEME_SECONDARY,
    THEME_WARN, THEME_ERROR, THEME_TEXT, THEME_TEXT_DIM
)
from app.state import AS
from app.visualization.styles.css import STYLE
from app.visualization.components.running_indicator import render_running_indicator

AUTOENCODER_DIR = Path(__file__).parent.parent.parent.parent / "machine_learning" / "autoencoder"
AUTOENCODER_CACHE_DIR = Path(__file__).parent.parent.parent / "cache" / "autoencoder"

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
    import sys
    
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
    import torch.nn as nn
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
    import torch
    
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
        z_np = AS.current_z.cpu().numpy().flatten()
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
    
    # Extract REAL attention weights from encoder
    if hasattr(AS.model, 'encoder') and hasattr(AS.model.encoder, 'get_all_attention_weights'):
        AS.attention_weights = AS.model.encoder.get_all_attention_weights()
    
    # Extract layer activations
    if hasattr(AS.model, 'encoder') and hasattr(AS.model.encoder, 'get_layer_activations'):
        layer_acts = AS.model.encoder.get_layer_activations()
        for key, val in layer_acts.items():
            AS.activations[f'encoder.{key}'] = val
    
    if store_latent and AS.current_z is not None:
        z_np = AS.current_z.cpu().numpy().flatten()
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
            ui.label(f"✓ Image dataset loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
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
            ui.label(f"✓ Image folder loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
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
                        all_activations[name].append(act.numpy().flatten())
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
    
    # Header
    with ui.header().classes('items-center px-4 py-1').style(f'background: {THEME_BG}; border-bottom: 1px solid {THEME_BORDER};'):
        ui.label('▶').style(f'color:{THEME_PRIMARY}; font-family: JetBrains Mono; font-size: 0.75rem;')
        ui.label('EEG_VIEWER').classes('text-base font-medium ml-2').style(f'color: {THEME_PRIMARY}; font-family: JetBrains Mono;')
        ui.label('// ANALYSIS').classes('text-sm ml-3').style(f'color:{THEME_WARN}; font-family: JetBrains Mono;')
        
        # Global running indicator
        render_running_indicator()
        
        with ui.row().classes('ml-auto gap-2'):
            ui.button('VIEWER', on_click=lambda: ui.navigate.to('/')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('CLEANER', on_click=lambda: ui.navigate.to('/cleaner')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('PIPELINE', on_click=lambda: ui.navigate.to('/pipeline')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('MODEL', on_click=lambda: ui.navigate.to('/model')).props('flat dense').style(f'color:{THEME_TEXT_DIM};')
            ui.button('ANALYSIS', on_click=lambda: ui.navigate.to('/analysis')).props('flat dense').style(f'color:{THEME_WARN};')
    
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
                            ui.label(f"⏳ Loading model...").style(f'color:{THEME_WARN}; font-size: 0.75rem;')
                            ui.label(f"  This may take a moment").style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                        
                        # Load model in background thread to avoid blocking UI
                        loop = asyncio.get_event_loop()
                        info = await loop.run_in_executor(None, load_trained_model, path)
                        
                        model_info_container.clear()
                        with model_info_container:
                            ui.label(f"✓ Model loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
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
                            ui.label(f"✗ Error loading model").style(f'color:{THEME_ERROR}; font-size: 0.75rem;')
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
                    import h5py
                    import numpy as np
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
                                    ui.label(f"✓ Graph dataset loaded").style(f'color:{THEME_PRIMARY}; font-size: 0.75rem;')
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
            
            # ROW 1: KURAMOTO ORDER PARAMETER (progress bar style)
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('▌KURAMOTO ORDER PARAMETER').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                
                def make_kuramoto_fig():
                    """Create Kuramoto order parameter visualization."""
                    fig = go.Figure()
                    
                    n_total = AS.total_samples if AS.total_samples > 0 else 100
                    
                    # Metastability band (std around mean)
                    if AS.kuramoto_avg > 0:
                        fig.add_trace(go.Scatter(
                            x=list(range(n_total)) + list(range(n_total-1, -1, -1)),
                            y=[AS.kuramoto_avg + AS.kuramoto_std] * n_total + [AS.kuramoto_avg - AS.kuramoto_std] * n_total,
                            fill='toself',
                            fillcolor='rgba(100, 150, 200, 0.2)',
                            line=dict(width=0),
                            name='Metastability',
                            showlegend=True
                        ))
                    
                    # Average line (dashed)
                    fig.add_trace(go.Scatter(
                        x=[0, n_total-1],
                        y=[AS.kuramoto_avg, AS.kuramoto_avg],
                        mode='lines',
                        line=dict(color='rgba(150, 200, 255, 0.7)', width=2, dash='dash'),
                        name=f'Avg: {AS.kuramoto_avg:.3f}'
                    ))
                    
                    # Scatter points for processed samples
                    if AS.kuramoto_history:
                        indices, values = zip(*AS.kuramoto_history)
                        # Color by label
                        colors = [['#00ff88', '#f472b6', '#00d4ff'][AS.latent_labels[i] % 3] if i < len(AS.latent_labels) else THEME_PRIMARY for i in range(len(indices))]
                        fig.add_trace(go.Scatter(
                            x=indices,
                            y=values,
                            mode='markers',
                            marker=dict(size=5, color=colors, opacity=0.8),
                            name='Samples'
                        ))
                        
                        # Current point highlighted
                        if len(indices) > 0:
                            fig.add_trace(go.Scatter(
                                x=[indices[-1]],
                                y=[values[-1]],
                                mode='markers',
                                marker=dict(size=12, color=THEME_WARN, symbol='star', line=dict(width=2, color='white')),
                                name='Current'
                            ))
                    
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(8,8,8,1)',
                        height=120,
                        margin=dict(l=40, r=10, t=5, b=30),
                        xaxis=dict(title='Sample Index', range=[0, n_total], gridcolor='rgba(255,204,0,0.1)'),
                        yaxis=dict(title='r', range=[0, 1], gridcolor='rgba(255,204,0,0.1)'),
                        legend=dict(orientation='h', y=1.15, font=dict(size=8)),
                        font=dict(family='JetBrains Mono', size=9, color=THEME_TEXT)
                    )
                    return fig
                
                kuramoto_plot = ui.plotly(make_kuramoto_fig()).classes('w-full').style('height: 120px;')
                AS.activation_plots['kuramoto'] = kuramoto_plot
            
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
                            
                            # Handle different activation shapes
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
                        ui.label('▌DECODER').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.75rem;')
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
                            
                            # Handle different activation shapes
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
                with ui.row().classes('items-center gap-4 mb-2'):
                    ui.label('▌LATENT SPACE (PCA)').style(f'color:{THEME_SECONDARY}; font-family: JetBrains Mono; font-size: 0.8rem;')
                    ui.label('').bind_text_from(AS, 'latent_codes', lambda x: f'{len(x)} samples').style(f'color:{THEME_TEXT_DIM}; font-size: 0.7rem;')
                
                def make_latent_fig():
                    """Create latent space PCA visualization."""
                    fig = go.Figure()
                    
                    X_pca, labels = compute_latent_pca()
                    if X_pca is not None and len(X_pca) > 0:
                        # Use more colors for more classes
                        n_classes = len(np.unique(labels))
                        if n_classes <= 10:
                            colors = ['#00ff88', '#f472b6', '#00d4ff', '#ffcc00', '#a78bfa', 
                                     '#00ffcc', '#ff9f43', '#74b9ff', '#55efc4', '#fd79a8']
                        else:
                            # Use colorscale for many classes
                            import plotly.express as px
                            colors = px.colors.qualitative.Alphabet[:n_classes]
                        
                        for label in np.unique(labels):
                            mask = labels == label
                            label_name = AS.class_names[int(label)] if int(label) < len(AS.class_names) else f'C{int(label)}'
                            fig.add_trace(go.Scatter(
                                x=X_pca[mask, 0], y=X_pca[mask, 1],
                                mode='markers',
                                marker=dict(size=6, color=colors[int(label) % len(colors)], opacity=0.7),
                                name=label_name
                            ))
                        
                        # Current point - bright yellow circle instead of star
                        if len(X_pca) > 0:
                            fig.add_trace(go.Scatter(
                                x=[X_pca[-1, 0]], y=[X_pca[-1, 1]],
                                mode='markers',
                                marker=dict(size=14, color='#ffff00', opacity=1, 
                                           line=dict(width=2, color='#000000')),
                                name='Current',
                                showlegend=False
                            ))
                    else:
                        fig.add_annotation(text="Process samples to visualize latent space", 
                                         x=0.5, y=0.5, showarrow=False,
                                         font=dict(color=THEME_TEXT_DIM, size=12))
                    
                    lat_range = AS.axis_ranges['latent']
                    fig.update_layout(
                        template='plotly_dark',
                        paper_bgcolor='rgba(8,8,8,1)',
                        plot_bgcolor='rgba(8,8,8,1)',
                        height=300,
                        margin=dict(l=40, r=20, t=10, b=40),
                        xaxis=dict(title='PC1', gridcolor='rgba(0,255,136,0.1)', 
                                  range=[lat_range['x_min'], lat_range['x_max']]),
                        yaxis=dict(title='PC2', gridcolor='rgba(0,255,136,0.1)',
                                  range=[lat_range['y_min'], lat_range['y_max']]),
                        legend=dict(orientation='h', y=-0.15, x=0.5, xanchor='center', 
                                   font=dict(size=9), bgcolor='rgba(0,0,0,0.5)'),
                        font=dict(family='JetBrains Mono', size=10, color=THEME_TEXT)
                    )
                    return fig
                
                latent_plot = ui.plotly(make_latent_fig()).classes('w-full').style('height: 300px;')
                AS.latent_plot = latent_plot
            
            # ROW 3: ATTENTION WEIGHTS (per layer)
            with ui.card().classes('dark-card p-3 w-full'):
                ui.label('▌ATTENTION WEIGHTS (GAT layers)').style(f'color:{THEME_WARN}; font-family: JetBrains Mono; font-size: 0.75rem;').classes('mb-1')
                
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
                            edge_index = att_data['edge_index'].numpy()
                            alpha = att_data['attention'].numpy()
                            
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
                    ui.label('▌RECONSTRUCTED').style(f'color:#f472b6; font-family: JetBrains Mono; font-size: 0.8rem;').classes('mb-2')
                    
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
                                # Track Kuramoto
                                AS.kuramoto_history.append((AS.current_idx, k_mean))
                
                # Update sample label
                sample_label.set_text(f'Sample: {AS.current_idx + 1} / {AS.total_samples}')
                
                # Update all plots
                try:
                    kuramoto_plot.figure = make_kuramoto_fig()
                    kuramoto_plot.update()
                except Exception as e:
                    analysis_log(f"Kuramoto plot error: {e}", 'warning')
                
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


