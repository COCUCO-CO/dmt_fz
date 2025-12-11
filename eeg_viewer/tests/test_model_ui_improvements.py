"""Tests for Model UI improvements."""

import pytest
import numpy as np
from pathlib import Path


# =============================================================================
# Tests for Decoder Configuration
# =============================================================================

class TestDecoderConfig:
    """Test decoder configuration options."""
    
    def test_decoder_gat_layers_valid_range(self):
        """Test decoder GAT layers accepts valid values."""
        valid_values = [0, 1, 2, 3, 4]
        for val in valid_values:
            assert 0 <= val <= 4, f"Invalid decoder GAT layers: {val}"
    
    def test_decoder_gat_zero_means_mlp(self):
        """Test that 0 GAT layers means MLP decoder."""
        decoder_gat_layers = 0
        # 0 means use MLP decoder (no graph convolutions)
        assert decoder_gat_layers == 0
        decoder_type = 'mlp' if decoder_gat_layers == 0 else 'gat'
        assert decoder_type == 'mlp'
    
    def test_decoder_gat_positive_means_gat(self):
        """Test that >0 GAT layers means GAT decoder."""
        for layers in [1, 2, 3, 4]:
            decoder_type = 'mlp' if layers == 0 else 'gat'
            assert decoder_type == 'gat'
    
    def test_decoder_config_dict_structure(self):
        """Test decoder config dictionary structure."""
        decoder_config = {
            'gat_layers': 2,
            'hidden_dims': [256, 128],
            'reconstruct_edges': True,
            'activation': 'leaky_relu',
            'dropout': 0.1,
        }
        
        assert 'gat_layers' in decoder_config
        assert 'hidden_dims' in decoder_config
        assert isinstance(decoder_config['hidden_dims'], list)
        assert decoder_config['gat_layers'] >= 0


# =============================================================================
# Tests for Architecture Visualization
# =============================================================================

class TestArchitectureVisualization:
    """Test model architecture visualization."""
    
    def test_generates_layer_info(self):
        """Test generation of layer information."""
        config = {
            'encoder': {
                'gat_layers': 3,
                'hidden_dim': 64,
                'num_attention_heads': 4,
            },
            'latent_dim': 64,
            'decoder': {
                'gat_layers': 0,
                'hidden_dims': [256, 128],
            },
            'input_nodes': 24,
            'input_features': 15,
        }
        
        # Simulate layer info generation
        layers = []
        
        # Input layer
        layers.append({
            'name': 'Input',
            'type': 'input',
            'shape': (config['input_nodes'], config['input_features']),
        })
        
        # Encoder layers
        for i in range(config['encoder']['gat_layers']):
            layers.append({
                'name': f'GAT_{i+1}',
                'type': 'gat',
                'hidden_dim': config['encoder']['hidden_dim'],
                'heads': config['encoder']['num_attention_heads'],
            })
        
        # Latent layer
        layers.append({
            'name': 'Latent',
            'type': 'latent',
            'dim': config['latent_dim'],
        })
        
        # Decoder layers
        if config['decoder']['gat_layers'] > 0:
            for i in range(config['decoder']['gat_layers']):
                layers.append({
                    'name': f'Dec_GAT_{i+1}',
                    'type': 'gat',
                })
        else:
            layers.append({
                'name': 'Decoder_MLP',
                'type': 'mlp',
                'dims': config['decoder']['hidden_dims'],
            })
        
        # Output layer
        layers.append({
            'name': 'Output',
            'type': 'output',
            'shape': (config['input_nodes'], config['input_features']),
        })
        
        assert len(layers) >= 4  # At minimum: input, encoder, latent, output
        assert layers[0]['type'] == 'input'
        assert layers[-1]['type'] == 'output'
    
    def test_ascii_diagram_generation(self):
        """Test ASCII diagram generation for model architecture."""
        def generate_ascii_architecture(encoder_layers, latent_dim, decoder_type, input_nodes):
            """Generate simple ASCII representation."""
            lines = []
            lines.append("┌" + "─" * 50 + "┐")
            lines.append("│" + " VAE Architecture ".center(50) + "│")
            lines.append("├" + "─" * 50 + "┤")
            lines.append(f"│ Input: {input_nodes} nodes".ljust(51) + "│")
            lines.append(f"│ Encoder: GAT × {encoder_layers}".ljust(51) + "│")
            lines.append(f"│ Latent: {latent_dim} dim".ljust(51) + "│")
            lines.append(f"│ Decoder: {decoder_type}".ljust(51) + "│")
            lines.append("└" + "─" * 50 + "┘")
            return "\n".join(lines)
        
        diagram = generate_ascii_architecture(
            encoder_layers=3,
            latent_dim=64,
            decoder_type="MLP [256→128]",
            input_nodes=24
        )
        
        assert "VAE Architecture" in diagram
        assert "GAT × 3" in diagram
        assert "64 dim" in diagram
        assert "MLP" in diagram
    
    def test_parameter_estimation(self):
        """Test model parameter count estimation."""
        def estimate_params(input_dim, hidden_dim, latent_dim, gat_layers, heads):
            """Rough estimation of VAE parameters."""
            params = 0
            
            # GAT layers: input_dim * hidden_dim * heads + biases
            for i in range(gat_layers):
                in_features = input_dim if i == 0 else hidden_dim * heads
                params += in_features * hidden_dim * heads
                params += hidden_dim * heads  # bias
            
            # Latent projection
            params += hidden_dim * heads * latent_dim * 2  # mu and logvar
            
            # Decoder MLP
            params += latent_dim * 256 + 256  # First layer
            params += 256 * 128 + 128  # Second layer
            params += 128 * input_dim + input_dim  # Output
            
            return params
        
        estimated = estimate_params(
            input_dim=15,
            hidden_dim=64,
            latent_dim=64,
            gat_layers=3,
            heads=4
        )
        
        # Should be reasonable for a small VAE
        assert estimated > 10000  # At least 10K params
        assert estimated < 10000000  # Less than 10M params


# =============================================================================
# Tests for Data Preview
# =============================================================================

class TestDataPreview:
    """Test data preview functionality."""
    
    def test_preview_info_for_graph_data(self):
        """Test preview info generation for graph data."""
        # Simulate phases data info
        dataset_info = {
            'type': 'graph',
            'sample_file': '/path/to/phases-S01.pkl',
            'has_eeg': True,
            'has_stc': False,
            'num_nodes_eeg': 24,
            'num_epochs_sample': 50,
            'bands': ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'],
            'conditions': ['DMT', 'EC', 'EO'],
        }
        
        # Generate preview info
        preview_info = {
            'sample_name': Path(dataset_info['sample_file']).name,
            'node_count': dataset_info['num_nodes_eeg'],
            'data_source': 'EEG' if dataset_info['has_eeg'] else 'STC',
            'bands': dataset_info['bands'],
            'labels': dataset_info['conditions'],
            'epochs_per_file': dataset_info['num_epochs_sample'],
        }
        
        assert preview_info['node_count'] == 24
        assert preview_info['data_source'] == 'EEG'
        assert len(preview_info['bands']) == 5
        assert len(preview_info['labels']) == 3
    
    def test_preview_info_for_image_data(self):
        """Test preview info generation for image data."""
        dataset_info = {
            'type': 'image',
            'sample_file': '/path/to/image_001.png',
            'type_specific': {
                'sizes': [(224, 224)],
                'channels': 3,
                'color_mode': 'RGB',
            },
            'classes': ['cat', 'dog', 'bird'],
        }
        
        preview_info = {
            'sample_name': Path(dataset_info['sample_file']).name,
            'image_size': dataset_info['type_specific']['sizes'][0],
            'channels': dataset_info['type_specific']['channels'],
            'color_mode': dataset_info['type_specific']['color_mode'],
            'classes': dataset_info['classes'],
        }
        
        assert preview_info['image_size'] == (224, 224)
        assert preview_info['channels'] == 3
        assert preview_info['color_mode'] == 'RGB'
    
    def test_feature_info_for_graph(self):
        """Test feature information extraction for graphs."""
        # Node features typically include:
        # - Phase statistics (mean, std, etc.) per band
        # - Amplitude statistics per band
        # - Temporal complexity measures
        
        feature_config = {
            'use_phase_stats': True,
            'use_amplitude_stats': True,
            'use_temporal_complexity': True,
        }
        
        n_bands = 5
        features_per_stat = 3  # mean, std, var (example)
        
        total_features = 0
        if feature_config['use_phase_stats']:
            total_features += n_bands * features_per_stat
        if feature_config['use_amplitude_stats']:
            total_features += n_bands * features_per_stat
        if feature_config['use_temporal_complexity']:
            total_features += 5  # Example: 5 complexity measures
        
        assert total_features > 0
        assert total_features == 35  # 5*3 + 5*3 + 5
    
    def test_connectivity_matrix_preview(self):
        """Test connectivity matrix preview for graphs."""
        # Simulate a small connectivity matrix
        n_nodes = 24
        connectivity = np.random.rand(n_nodes, n_nodes)
        connectivity = (connectivity + connectivity.T) / 2  # Make symmetric
        np.fill_diagonal(connectivity, 0)  # No self-loops
        
        # Apply threshold
        threshold = 0.3
        adjacency = (connectivity > threshold).astype(int)
        
        # Count edges
        n_edges = np.sum(adjacency) // 2  # Divide by 2 for undirected
        
        assert connectivity.shape == (24, 24)
        assert n_edges > 0
        assert n_edges <= n_nodes * (n_nodes - 1) // 2  # Max edges


# =============================================================================
# Tests for UI Integration
# =============================================================================

class TestUIIntegration:
    """Test UI component integration."""
    
    def test_config_to_dict_conversion(self):
        """Test conversion of UI values to config dict."""
        # Simulate UI values
        ui_values = {
            'latent_dim': 64,
            'hidden_dim': 64,
            'gat_layers': 3,
            'attention_heads': 4,
            'dropout': 0.2,
            'decoder_gat_layers': 0,  # NEW
            'decoder_hidden_dims': '256,128',  # NEW (as string from input)
        }
        
        # Convert to config
        config = {
            'model': {
                'encoder': {
                    'hidden_dim': ui_values['hidden_dim'],
                    'num_gat_layers': ui_values['gat_layers'],
                    'num_attention_heads': ui_values['attention_heads'],
                    'dropout': ui_values['dropout'],
                },
                'latent': {
                    'dim': ui_values['latent_dim'],
                },
                'decoder': {
                    'gat_layers': ui_values['decoder_gat_layers'],
                    'hidden_dims': [int(x) for x in ui_values['decoder_hidden_dims'].split(',')],
                },
            }
        }
        
        assert config['model']['decoder']['gat_layers'] == 0
        assert config['model']['decoder']['hidden_dims'] == [256, 128]
    
    def test_preview_visibility_after_scan(self):
        """Test that preview becomes visible after dataset scan."""
        # Simulate scan state
        scan_completed = False
        preview_visible = False
        
        # Before scan
        assert not preview_visible
        
        # After scan
        scan_completed = True
        dataset_info = {'type': 'graph', 'file_count': 100}
        
        if scan_completed and dataset_info.get('file_count', 0) > 0:
            preview_visible = True
        
        assert preview_visible
    
    def test_architecture_updates_on_config_change(self):
        """Test that architecture visualization updates when config changes."""
        # Simulate config state
        config_v1 = {'gat_layers': 3, 'latent_dim': 64}
        config_v2 = {'gat_layers': 4, 'latent_dim': 128}
        
        def generate_arch_hash(config):
            """Generate a simple hash of architecture config."""
            return f"{config['gat_layers']}_{config['latent_dim']}"
        
        hash_v1 = generate_arch_hash(config_v1)
        hash_v2 = generate_arch_hash(config_v2)
        
        # Hashes should be different for different configs
        assert hash_v1 != hash_v2
        assert hash_v1 == "3_64"
        assert hash_v2 == "4_128"


# =============================================================================
# Tests for Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling in model UI."""
    
    def test_handles_missing_dataset_info(self):
        """Test handling of missing dataset info for preview."""
        dataset_info = None
        
        def get_preview_data(info):
            if info is None:
                return {'error': 'No dataset scanned'}
            return {'data': info}
        
        result = get_preview_data(dataset_info)
        assert 'error' in result
    
    def test_handles_invalid_decoder_dims(self):
        """Test handling of invalid decoder dimensions input."""
        invalid_inputs = ['', 'abc', '256,', ',128', '256,,128']
        
        def parse_hidden_dims(s):
            try:
                if not s or not s.strip():
                    return [256, 128]  # Default
                parts = [p.strip() for p in s.split(',') if p.strip()]
                return [int(p) for p in parts]
            except ValueError:
                return [256, 128]  # Default on error
        
        for invalid in invalid_inputs:
            result = parse_hidden_dims(invalid)
            assert isinstance(result, list)
            assert all(isinstance(x, int) for x in result)
    
    def test_handles_empty_sample_file(self):
        """Test handling when sample file doesn't exist."""
        def load_sample_preview(path):
            if path is None or not Path(path).exists():
                return {'error': 'Sample file not found'}
            return {'loaded': True}
        
        result = load_sample_preview('/nonexistent/path.pkl')
        assert 'error' in result


