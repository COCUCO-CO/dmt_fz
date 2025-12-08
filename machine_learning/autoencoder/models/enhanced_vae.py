"""
Enhanced Variational Autoencoder with GAT Encoder AND GAT Decoder.

Key improvements:
- GAT-based decoder with attention layers (symmetric architecture)
- Attention weight extraction from both encoder and decoder
- Support for multi-band features
- Temporal window support
- Improved reconstruction with attention-guided decoding

Author: Assistant
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


# =============================================================================
# ATTENTION POOLING
# =============================================================================

class AttentionPooling(nn.Module):
    """Attention-based global pooling layer."""
    
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, 1)
        )
        self.global_attention = GlobalAttention(self.attention_net)
    
    def forward(self, x, batch):
        return self.global_attention(x, batch)


# =============================================================================
# GAT ENCODER
# =============================================================================

class GATEncoder(nn.Module):
    """
    GAT-based encoder with multiple layers and skip connections.
    Extracts hierarchical node representations and attention weights.
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 config: Dict[str, Any]):
        super().__init__()
        
        enc_config = config['model']['encoder']
        
        hidden_dim = enc_config['hidden_dim']
        num_layers = enc_config['num_gat_layers']
        num_heads = enc_config.get('num_attention_heads', 4)
        dropout = enc_config['dropout']
        att_dropout = enc_config.get('attention_dropout', 0.1)
        use_edge_attr = enc_config.get('use_edge_attr', True)
        concat_heads = enc_config.get('concat_heads', True)
        
        self.use_skip = enc_config.get('use_skip_connections', True)
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Edge encoder
        if use_edge_attr and num_edge_features > 0:
            self.edge_encoder = nn.Linear(num_edge_features, num_heads)
        else:
            self.edge_encoder = None
        
        # GAT layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        
        # First layer
        out_dim = hidden_dim * num_heads if concat_heads else hidden_dim
        self.conv_layers.append(
            GATv2Conv(
                num_node_features, hidden_dim,
                heads=num_heads, concat=concat_heads,
                dropout=att_dropout,
                edge_dim=num_heads if use_edge_attr else None
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        if self.use_skip and num_node_features != out_dim:
            self.skip_projs.append(nn.Linear(num_node_features, out_dim))
        else:
            self.skip_projs.append(None)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                GATv2Conv(
                    out_dim, hidden_dim,
                    heads=num_heads, concat=concat_heads,
                    dropout=att_dropout,
                    edge_dim=num_heads if use_edge_attr else None
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
            self.skip_projs.append(None)  # Same dimension
        
        self.output_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        
        # Storage for activations
        self._activations = {}
        self._attention = {}
    
    def forward(self, x, edge_index, edge_attr=None, batch=None, 
                store_activations=False):
        """Forward pass with optional activation storage."""
        
        if store_activations:
            self._activations = {}
            self._attention = {}
        
        # Encode edge features
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        h = x
        
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            h_prev = h
            
            if store_activations:
                h_new, (edge_idx, alpha) = conv(
                    h, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True
                )
                self._attention[f'encoder_layer_{i}'] = {
                    'edge_index': edge_idx.detach().cpu(),
                    'alpha': alpha.detach().cpu()
                }
            else:
                h_new = conv(h, edge_index, edge_attr=edge_attr)
            
            h_new = bn(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            
            # Skip connection
            if self.use_skip and i < len(self.skip_projs):
                if self.skip_projs[i] is not None:
                    h_prev = self.skip_projs[i](h_prev)
                if h_prev.shape[-1] == h_new.shape[-1]:
                    h = h_new + h_prev
                else:
                    h = h_new
            else:
                h = h_new
            
            if store_activations:
                self._activations[f'encoder_layer_{i}'] = h.detach().cpu()
        
        return h
    
    def get_attention_weights(self) -> Dict:
        return self._attention
    
    def get_activations(self) -> Dict:
        return self._activations


# =============================================================================
# GAT DECODER
# =============================================================================

class GATDecoder(nn.Module):
    """
    GAT-based decoder with attention layers.
    
    Decodes latent representation back to node and edge features
    using graph attention to capture structural dependencies.
    """
    
    def __init__(self,
                 latent_dim: int,
                 num_nodes: int,
                 num_node_features: int,
                 num_edge_features: int,
                 config: Dict[str, Any]):
        super().__init__()
        
        dec_config = config['model']['decoder']
        
        hidden_dim = dec_config.get('hidden_dim', 64)
        num_layers = dec_config.get('num_gat_layers', 2)
        num_heads = dec_config.get('num_attention_heads', 4)
        dropout = dec_config.get('dropout', 0.1)
        att_dropout = dec_config.get('attention_dropout', 0.1)
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.reconstruct_edges = dec_config.get('reconstruct_edges', True)
        
        # Project latent to node dimension
        gat_input_dim = hidden_dim * num_heads
        self.latent_to_node = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, gat_input_dim)
        )
        
        # GAT decoder layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = gat_input_dim
            out_dim = hidden_dim * num_heads
            
            self.conv_layers.append(
                GATv2Conv(
                    in_dim, hidden_dim,
                    heads=num_heads, concat=True,
                    dropout=att_dropout
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Output heads
        final_dim = hidden_dim * num_heads
        self.node_output = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_node_features)
        )
        
        if self.reconstruct_edges:
            self.edge_output = nn.Sequential(
                nn.Linear(final_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_edge_features)
            )
        
        # Storage
        self._activations = {}
        self._attention = {}
    
    def forward(self, z, edge_index, batch, store_activations=False):
        """
        Decode latent vectors to node and edge features.
        
        Args:
            z: Graph-level latent [batch_size, latent_dim]
            edge_index: Edge indices
            batch: Batch assignment per node
            store_activations: Store intermediate values
            
        Returns:
            Dict with x_recon and edge_attr_recon
        """
        if store_activations:
            self._activations = {}
            self._attention = {}
        
        # Broadcast z to nodes: z[batch] gives per-node latent
        h = self.latent_to_node(z[batch])
        
        # Pass through GAT decoder layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            if store_activations:
                h_new, (edge_idx, alpha) = conv(
                    h, edge_index,
                    return_attention_weights=True
                )
                self._attention[f'decoder_layer_{i}'] = {
                    'edge_index': edge_idx.detach().cpu(),
                    'alpha': alpha.detach().cpu()
                }
            else:
                h_new = conv(h, edge_index)
            
            h_new = bn(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual
            if h.shape == h_new.shape:
                h = h + h_new
            else:
                h = h_new
            
            if store_activations:
                self._activations[f'decoder_layer_{i}'] = h.detach().cpu()
        
        # Reconstruct node features
        x_recon = self.node_output(h)
        
        result = {'x_recon': x_recon}
        
        # Reconstruct edge features
        if self.reconstruct_edges:
            src, dst = edge_index
            edge_features = torch.cat([h[src], h[dst]], dim=1)
            edge_attr_recon = self.edge_output(edge_features)
            result['edge_attr_recon'] = edge_attr_recon
        
        return result
    
    def get_attention_weights(self) -> Dict:
        return self._attention
    
    def get_activations(self) -> Dict:
        return self._activations


# =============================================================================
# ENHANCED VAE MODEL
# =============================================================================

class EnhancedBrainStateVAE(nn.Module):
    """
    Enhanced VAE with GAT encoder AND GAT decoder.
    
    Features:
    - Symmetric attention architecture
    - Attention extraction from encoder and decoder
    - Multi-band support
    - Temporal window support
    - β-VAE with KL annealing
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 num_graph_features: int,
                 num_nodes: int,
                 config: Dict[str, Any]):
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.num_graph_features = num_graph_features
        self.num_nodes = num_nodes
        self.config = config
        
        latent_dim = config['model']['latent']['dim']
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = GATEncoder(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            config=config
        )
        
        encoder_output_dim = self.encoder.output_dim
        
        # Pooling
        pool_method = config['model']['pooling']['method']
        self.pooling_method = pool_method
        
        if pool_method == 'attention':
            self.pool = AttentionPooling(encoder_output_dim)
            pooled_dim = encoder_output_dim
        elif pool_method == 'mean+max':
            self.pool = None
            pooled_dim = encoder_output_dim * 2
        else:
            self.pool = None
            pooled_dim = encoder_output_dim
        
        # Include graph features
        self.use_graph_features = num_graph_features > 0
        total_pooled = pooled_dim + (num_graph_features if self.use_graph_features else 0)
        
        # Latent projections
        self.fc_mu = nn.Linear(total_pooled, latent_dim)
        self.fc_logvar = nn.Linear(total_pooled, latent_dim)
        
        # Decoder (GAT-based)
        self.decoder = GATDecoder(
            latent_dim=latent_dim,
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            config=config
        )
        
        # Storage
        self._last_mu = None
        self._last_logvar = None
        self._last_z = None
        self._last_pooled = None
        
        self._init_weights()
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"EnhancedBrainStateVAE: {n_params:,} parameters")
        logger.info(f"  Encoder output: {encoder_output_dim}")
        logger.info(f"  Latent dim: {latent_dim}")
        logger.info(f"  Decoder: GAT-based with {config['model']['decoder'].get('num_gat_layers', 2)} layers")
    
    def _init_weights(self):
        """Initialize linear layers."""
        for module in [self.fc_mu, self.fc_logvar]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def encode(self, data: Batch, store_activations: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode to latent distribution parameters."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Encoder forward
        h = self.encoder(x, edge_index, edge_attr, batch, store_activations)
        
        # Pooling
        if self.pooling_method == 'attention':
            h_graph = self.pool(h, batch)
        elif self.pooling_method == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pooling_method == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pooling_method == 'mean+max':
            h_graph = torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch)
            ], dim=1)
        else:
            h_graph = global_mean_pool(h, batch)
        
        # Add graph features
        if self.use_graph_features and hasattr(data, 'graph_attr') and data.graph_attr is not None:
            graph_attr = data.graph_attr
            if graph_attr.dim() == 1:
                graph_attr = graph_attr.view(-1, self.num_graph_features)
            h_graph = torch.cat([h_graph, graph_attr], dim=1)
        
        self._last_pooled = h_graph.detach().cpu()
        
        # Project to latent
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        logvar = torch.clamp(logvar, -20, 20)
        
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu
    
    def decode(self, z: torch.Tensor, data: Batch, 
               store_activations: bool = False) -> Dict[str, torch.Tensor]:
        """Decode latent to reconstructed features."""
        return self.decoder(
            z, data.edge_index, data.batch,
            store_activations=store_activations
        )
    
    def forward(self, data: Batch, 
                store_activations: bool = False) -> Dict[str, torch.Tensor]:
        """Full forward pass."""
        # Encode
        mu, logvar = self.encode(data, store_activations)
        self._last_mu = mu.detach().cpu()
        self._last_logvar = logvar.detach().cpu()
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        self._last_z = z.detach().cpu()
        
        # Decode
        recon = self.decode(z, data, store_activations)
        
        return {
            'x_recon': recon['x_recon'],
            'edge_attr_recon': recon.get('edge_attr_recon'),
            'mu': mu,
            'log_var': logvar,
            'z': z
        }
    
    def loss_function(self, output: Dict[str, torch.Tensor],
                      data: Batch,
                      beta: float = 1.0,
                      free_bits: float = 0.0) -> Dict[str, torch.Tensor]:
        """Compute VAE loss with reconstruction and KL terms."""
        loss_config = self.config['loss']
        
        # Node reconstruction
        x_recon = output['x_recon']
        x_target = data.x
        
        if loss_config['reconstruction']['type'] == 'mse':
            recon_loss_node = F.mse_loss(x_recon, x_target, reduction='mean')
        else:
            recon_loss_node = F.binary_cross_entropy_with_logits(
                x_recon, x_target, reduction='mean'
            )
        
        recon_loss = loss_config['reconstruction']['node_weight'] * recon_loss_node
        
        # Edge reconstruction
        recon_loss_edge = torch.tensor(0.0, device=x_recon.device)
        if output['edge_attr_recon'] is not None and hasattr(data, 'edge_attr'):
            edge_recon = output['edge_attr_recon']
            edge_target = data.edge_attr
            
            if loss_config['reconstruction']['type'] == 'mse':
                recon_loss_edge = F.mse_loss(edge_recon, edge_target, reduction='mean')
            else:
                recon_loss_edge = F.binary_cross_entropy_with_logits(
                    edge_recon, edge_target, reduction='mean'
                )
            
            recon_loss += loss_config['reconstruction']['edge_weight'] * recon_loss_edge
        
        # KL divergence
        mu = output['mu']
        logvar = torch.clamp(output['log_var'], -20, 20)
        
        mu_sq = torch.clamp(mu.pow(2), max=1e6)
        var = torch.clamp(logvar.exp(), max=1e6)
        
        kl_per_dim = 0.5 * (mu_sq + var - 1 - logvar)
        
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        
        kl_loss = kl_per_dim.sum(dim=1).mean()
        
        if torch.isnan(kl_loss):
            kl_loss = torch.tensor(0.0, device=mu.device)
        
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'recon_loss_node': recon_loss_node,
            'recon_loss_edge': recon_loss_edge,
            'kl_loss': kl_loss,
            'kl_per_dim': kl_per_dim.mean(dim=0).detach()
        }
    
    def get_latent(self, data: Batch, use_mean: bool = True) -> torch.Tensor:
        """Get latent representation."""
        with torch.no_grad():
            mu, logvar = self.encode(data)
            if use_mean:
                return mu
            return self.reparameterize(mu, logvar)
    
    def get_all_attention_weights(self) -> Dict:
        """Get attention weights from encoder and decoder."""
        attention = {}
        attention.update(self.encoder.get_attention_weights())
        attention.update(self.decoder.get_attention_weights())
        return attention
    
    def get_all_activations(self, data: Batch) -> Dict:
        """Extract all intermediate activations."""
        with torch.no_grad():
            _ = self.forward(data, store_activations=True)
            
            activations = {}
            
            # Encoder
            activations.update(self.encoder.get_activations())
            
            # Decoder
            for k, v in self.decoder.get_activations().items():
                activations[k] = v
            
            # Attention weights
            for k, v in self.encoder.get_attention_weights().items():
                activations[f'attention.{k}'] = v
            for k, v in self.decoder.get_attention_weights().items():
                activations[f'attention.{k}'] = v
            
            # Graph-level
            activations['pooled'] = self._last_pooled
            activations['mu'] = self._last_mu
            activations['z'] = self._last_z
            
            return activations


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_enhanced_vae(config: Dict[str, Any],
                        num_node_features: int,
                        num_edge_features: int,
                        num_graph_features: int,
                        num_nodes: int) -> EnhancedBrainStateVAE:
    """Create EnhancedBrainStateVAE from config."""
    return EnhancedBrainStateVAE(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes,
        config=config
    )






