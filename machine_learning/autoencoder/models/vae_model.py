"""
Variational Autoencoder (VAE) with Graph Attention Networks for EEG data.

This module implements a VAE architecture using GAT for the encoder,
designed to learn latent representations of brain synchronization graphs.
Supports extraction of intermediate GAT layer activations for clustering analysis.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


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


class GATEncoder(nn.Module):
    """
    GAT-based encoder for the VAE.
    
    Extracts hierarchical representations from graph-structured data
    using multiple GAT layers with skip connections.
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 config: Dict[str, Any]):
        super().__init__()
        
        enc_config = config['model']['encoder']
        
        conv_type = enc_config.get('conv_type', 'gatv2').lower()
        self.conv_type = conv_type
        
        hidden_dim = enc_config['hidden_dim']
        num_layers = enc_config['num_gat_layers']
        dropout = enc_config['dropout']
        use_edge_attr = enc_config['use_edge_attr']
        
        self.use_skip_connections = enc_config.get('use_skip_connections', True)
        
        # Store type-specific parameters as instance attributes
        # Default values for ChebConv case (avoids NameError)
        self.num_heads = enc_config.get('num_attention_heads', 1)
        self.att_dropout = enc_config.get('attention_dropout', 0.0)
        self.concat_heads = enc_config.get('concat_heads', True)
        self.negative_slope = enc_config.get('negative_slope', 0.2)
        self.cheby_k = enc_config.get('cheby_k', 3)
        
        # Edge encoder (for GATv2 only)
        if conv_type == 'gatv2' and use_edge_attr and num_edge_features > 0:
            self.edge_encoder = nn.Linear(num_edge_features, self.num_heads)
        else:
            self.edge_encoder = None
        
        # Build conv layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Calculate output dimension for first layer (needed for skip connections)
        if conv_type == 'gatv2':
            first_layer_output_dim = hidden_dim * self.num_heads if self.concat_heads else hidden_dim
        else:
            first_layer_output_dim = hidden_dim
        
        # First layer
        if conv_type == 'gatv2':
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=num_node_features,
                    out_channels=hidden_dim,
                    heads=self.num_heads,
                    concat=self.concat_heads,
                    dropout=self.att_dropout,
                    negative_slope=self.negative_slope,
                    edge_dim=self.num_heads if use_edge_attr else None
                )
            )
            self.output_dim = first_layer_output_dim
        elif conv_type == 'cheby':
            self.conv_layers.append(
                ChebConv(
                    in_channels=num_node_features,
                    out_channels=hidden_dim,
                    K=self.cheby_k
                )
            )
            self.output_dim = hidden_dim
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}. Use 'gatv2' or 'cheby'")
        
        self.batch_norms.append(nn.BatchNorm1d(self.output_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if conv_type == 'gatv2':
                self.conv_layers.append(
                    GATv2Conv(
                        in_channels=self.output_dim,
                        out_channels=hidden_dim,
                        heads=self.num_heads,
                        concat=self.concat_heads,
                        dropout=self.att_dropout,
                        negative_slope=self.negative_slope,
                        edge_dim=self.num_heads if use_edge_attr else None
                    )
                )
                self.output_dim = hidden_dim * self.num_heads if self.concat_heads else hidden_dim
            elif conv_type == 'cheby':
                self.conv_layers.append(
                    ChebConv(
                        in_channels=self.output_dim,
                        out_channels=hidden_dim,
                        K=self.cheby_k
                    )
                )
                self.output_dim = hidden_dim
            
            self.batch_norms.append(nn.BatchNorm1d(self.output_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection projections
        # Use first_layer_output_dim for the first skip projection
        self.skip_projections = nn.ModuleList()
        if self.use_skip_connections:
            # First layer: project input features to first layer output dimension
            if num_node_features != first_layer_output_dim:
                self.skip_projections.append(nn.Linear(num_node_features, first_layer_output_dim))
            else:
                self.skip_projections.append(None)
            
            # Subsequent layers: dimensions match after first layer (all use output_dim)
            for _ in range(num_layers - 1):
                self.skip_projections.append(None)
        
        # Store for activation extraction
        self.num_layers = num_layers
        self._layer_activations = {}
        self._attention_weights = {}
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                store_activations: bool = False) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_edge_features]
            batch: Batch assignment [num_nodes]
            store_activations: If True, store intermediate activations
            
        Returns:
            Node embeddings [num_nodes, output_dim]
        """
        # Clear stored activations
        if store_activations:
            self._layer_activations = {}
            self._attention_weights = {}
        
        # Encode edge attributes
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        h = x
        
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            h_prev = h
            
            # Forward through conv
            if self.conv_type == 'gatv2':
                if store_activations:
                    h_new, (edge_index_att, alpha) = conv(
                        h, edge_index, edge_attr=edge_attr, 
                        return_attention_weights=True
                    )
                    self._attention_weights[f'layer_{i}'] = {
                        'edge_index': edge_index_att.detach().cpu(),
                        'attention': alpha.detach().cpu()
                    }
                else:
                    h_new = conv(h, edge_index, edge_attr=edge_attr)
            elif self.conv_type == 'cheby':
                h_new = conv(h, edge_index)
            
            h_new = bn(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            
            # Skip connection
            if self.use_skip_connections and i < len(self.skip_projections):
                if self.skip_projections[i] is not None:
                    h_prev = self.skip_projections[i](h_prev)
                
                if h_prev.shape[-1] == h_new.shape[-1]:
                    h = h_new + h_prev
                else:
                    h = h_new
            else:
                h = h_new
            
            # Store activations
            if store_activations:
                self._layer_activations[f'layer_{i}'] = h.detach().cpu()
        
        return h
    
    def get_attention_weights(self, layer_idx: int = -1) -> Optional[Dict]:
        """Get stored attention weights for a layer."""
        if layer_idx == -1:
            layer_idx = self.num_layers - 1
        key = f'layer_{layer_idx}'
        return self._attention_weights.get(key, None)
    
    def get_all_attention_weights(self) -> Dict:
        """Get all stored attention weights."""
        return self._attention_weights
    
    def get_layer_activations(self) -> Dict:
        """Get all stored layer activations."""
        return self._layer_activations


class GraphDecoder(nn.Module):
    """
    Decoder for reconstructing node features and optionally edge weights.
    """
    
    def __init__(self,
                 latent_dim: int,
                 num_nodes: int,
                 num_node_features: int,
                 num_edge_features: int,
                 config: Dict[str, Any]):
        super().__init__()
        
        dec_config = config['model']['decoder']
        
        hidden_dims = dec_config['hidden_dims']
        dropout = dec_config['dropout']
        activation = dec_config['activation']
        self.reconstruct_edges = dec_config.get('reconstruct_edges', True)
        
        # Node feature decoder
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2))
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer for node features
        layers.append(nn.Linear(prev_dim, num_node_features))
        
        self.node_decoder = nn.Sequential(*layers)
        
        # Edge decoder (optional)
        if self.reconstruct_edges:
            # Use inner product for edge reconstruction
            # or a small MLP
            self.edge_decoder = nn.Sequential(
                nn.Linear(latent_dim * 2, 64),
                nn.ReLU(),
                nn.Linear(64, num_edge_features)
            )
        else:
            self.edge_decoder = None
        
        self.latent_dim = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
    
    def forward(self, z: torch.Tensor, 
                edge_index: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                num_nodes_per_graph: Optional[List[int]] = None) -> Dict[str, torch.Tensor]:
        """
        Decode latent representation to node features and edge weights.
        
        Args:
            z: Latent node representations [num_nodes, latent_dim]
            edge_index: Edge indices for edge reconstruction [2, num_edges]
            batch: Batch assignment
            num_nodes_per_graph: Number of nodes per graph in batch
            
        Returns:
            Dict with 'x_recon' and optionally 'edge_attr_recon'
        """
        # Decode node features
        x_recon = self.node_decoder(z)
        
        result = {'x_recon': x_recon}
        
        # Decode edge features
        if self.reconstruct_edges and edge_index is not None and self.edge_decoder is not None:
            # Get node pairs for each edge
            src, dst = edge_index[0], edge_index[1]
            z_src = z[src]  # [num_edges, latent_dim]
            z_dst = z[dst]  # [num_edges, latent_dim]
            
            # Concatenate source and destination embeddings
            z_edge = torch.cat([z_src, z_dst], dim=1)  # [num_edges, 2*latent_dim]
            
            edge_attr_recon = self.edge_decoder(z_edge)
            result['edge_attr_recon'] = edge_attr_recon
        
        return result


class BrainStateVAE(nn.Module):
    """
    Variational Autoencoder with GAT encoder for brain state analysis.
    
    Architecture:
        1. GAT Encoder: Graph -> Node embeddings
        2. Pooling: Node embeddings -> Graph embedding  
        3. Latent: Graph embedding -> (mu, log_var) -> z
        4. Broadcast: z -> Node latents
        5. Decoder: Node latents -> Reconstructed features
    
    Supports:
        - Extraction of intermediate GAT layer activations
        - Attention weight logging
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
        pool_config = config['model']['pooling']
        self.pooling_method = pool_config['method']
        
        if self.pooling_method == "attention":
            self.pool = AttentionPooling(encoder_output_dim)
            pooled_dim = encoder_output_dim
        elif self.pooling_method == "mean+max":
            self.pool = None
            pooled_dim = encoder_output_dim * 2
        else:
            self.pool = None
            pooled_dim = encoder_output_dim
        
        # Include graph-level features
        self.use_graph_features = num_graph_features > 0
        total_pooled_dim = pooled_dim + (num_graph_features if self.use_graph_features else 0)
        
        # Latent space projections
        self.fc_mu = nn.Linear(total_pooled_dim, latent_dim)
        self.fc_log_var = nn.Linear(total_pooled_dim, latent_dim)
        
        # Project latent back to node-level
        self.latent_to_node = nn.Sequential(
            nn.Linear(latent_dim, encoder_output_dim),
            nn.ReLU(),
            nn.Linear(encoder_output_dim, latent_dim)
        )
        
        # Decoder
        self.decoder = GraphDecoder(
            latent_dim=latent_dim,
            num_nodes=num_nodes,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            config=config
        )
        
        # Store last activations
        self._last_mu = None
        self._last_log_var = None
        self._last_z = None
        self._last_pooled = None
        
        self.reset_parameters()
        
        logger.info(f"BrainStateVAE initialized with {self.count_parameters()} parameters")
        logger.info(f"  Encoder output dim: {encoder_output_dim}")
        logger.info(f"  Latent dim: {latent_dim}")
        logger.info(f"  Pooling method: {self.pooling_method}")
    
    def reset_parameters(self):
        """Initialize model parameters.
        
        Only resets the VAE-specific linear layers (fc_mu, fc_log_var, latent_to_node, decoder).
        Does NOT reset GATv2Conv/ChebConv internal parameters to preserve their initialization.
        """
        # Reset only VAE-specific modules, not encoder conv layers
        modules_to_reset = [
            self.fc_mu, 
            self.fc_log_var, 
            self.latent_to_node, 
            self.decoder
        ]
        
        for module in modules_to_reset:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.zeros_(submodule.bias)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def encode(self, data: Batch, 
               store_activations: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input graph to latent distribution parameters.
        
        Args:
            data: PyTorch Geometric batch
            store_activations: Store intermediate activations
            
        Returns:
            Tuple of (mu, log_var)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Handle graph attributes
        graph_attr = None
        if self.use_graph_features and hasattr(data, 'graph_attr') and data.graph_attr is not None:
            graph_attr = data.graph_attr
            if graph_attr.dim() == 1:
                num_features = self.num_graph_features
                if num_features > 0 and graph_attr.numel() % num_features == 0:
                    graph_attr = graph_attr.view(-1, num_features)
        
        # Encode through GAT
        h = self.encoder(x, edge_index, edge_attr, batch, store_activations)
        
        # Global pooling
        if self.pooling_method == "attention":
            h_graph = self.pool(h, batch)
        elif self.pooling_method == "mean":
            h_graph = global_mean_pool(h, batch)
        elif self.pooling_method == "max":
            h_graph = global_max_pool(h, batch)
        elif self.pooling_method == "add":
            h_graph = global_add_pool(h, batch)
        elif self.pooling_method == "mean+max":
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_graph = torch.cat([h_mean, h_max], dim=1)
        else:
            h_graph = global_mean_pool(h, batch)
        
        # Concatenate graph features
        if graph_attr is not None:
            h_graph = torch.cat([h_graph, graph_attr], dim=1)
        
        # Store pooled representation
        self._last_pooled = h_graph.detach().cpu()
        
        # Project to latent distribution
        mu = self.fc_mu(h_graph)
        log_var = self.fc_log_var(h_graph)
        
        return mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon
        
        Includes clamping of log_var for numerical stability.
        """
        # Clamp log_var to prevent extreme values (exp overflow/underflow)
        log_var = torch.clamp(log_var, min=-20.0, max=20.0)
        
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu  # Use mean for inference
    
    def decode(self, z: torch.Tensor, 
               data: Batch) -> Dict[str, torch.Tensor]:
        """
        Decode latent representation to reconstructed graph.
        
        Args:
            z: Graph-level latent [batch_size, latent_dim]
            data: Original data batch (for structure info)
            
        Returns:
            Dict with reconstructed features
        """
        batch = data.batch
        edge_index = data.edge_index
        
        # Broadcast graph-level z to all nodes
        # z is [batch_size, latent_dim], we need [num_nodes, latent_dim]
        z_nodes = z[batch]  # Index expansion
        
        # Project to decoder input
        z_nodes = self.latent_to_node(z_nodes)
        
        # Decode
        recon = self.decoder(z_nodes, edge_index, batch)
        
        return recon
    
    def forward(self, data: Batch, 
                store_activations: bool = False) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> reparameterize -> decode.
        
        Args:
            data: PyTorch Geometric batch
            store_activations: Store intermediate activations
            
        Returns:
            Dict with:
                - 'x_recon': Reconstructed node features
                - 'edge_attr_recon': Reconstructed edge features (if enabled)
                - 'mu': Latent mean
                - 'log_var': Latent log variance
                - 'z': Sampled latent
        """
        # Encode
        mu, log_var = self.encode(data, store_activations)
        
        # Store for logging
        self._last_mu = mu.detach().cpu()
        self._last_log_var = log_var.detach().cpu()
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        self._last_z = z.detach().cpu()
        
        # Decode
        recon = self.decode(z, data)
        
        return {
            'x_recon': recon['x_recon'],
            'edge_attr_recon': recon.get('edge_attr_recon', None),
            'mu': mu,
            'log_var': log_var,
            'z': z
        }
    
    def loss_function(self, 
                      output: Dict[str, torch.Tensor],
                      data: Batch,
                      beta: float = 1.0,
                      free_bits: float = 0.0) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss: Reconstruction + β * KL divergence.
        
        Args:
            output: Forward pass output
            data: Original data batch
            beta: Weight for KL term (β-VAE)
            free_bits: Minimum KL per dimension
            
        Returns:
            Dict with loss components
        """
        loss_config = self.config['loss']
        
        # Reconstruction loss for node features
        x_recon = output['x_recon']
        x_target = data.x
        
        if loss_config['reconstruction']['type'] == 'mse':
            recon_loss_node = F.mse_loss(x_recon, x_target, reduction='mean')
        else:  # bce
            recon_loss_node = F.binary_cross_entropy_with_logits(
                x_recon, x_target, reduction='mean'
            )
        
        recon_loss = loss_config['reconstruction']['node_weight'] * recon_loss_node
        
        # Edge reconstruction loss (optional)
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
        else:
            recon_loss_edge = torch.tensor(0.0, device=x_recon.device)
        
        # KL divergence
        mu = output['mu']
        log_var = output['log_var']
        
        # Clamp log_var for numerical stability
        log_var = torch.clamp(log_var, min=-20.0, max=20.0)
        
        # KL per dimension: 0.5 * (mu^2 + var - 1 - log_var)
        # Clamp mu^2 to prevent extreme values
        mu_sq = torch.clamp(mu.pow(2), max=1e6)
        var = torch.clamp(log_var.exp(), max=1e6)
        
        kl_per_dim = 0.5 * (mu_sq + var - 1 - log_var)
        
        # Apply free bits (minimum KL per dimension)
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        
        # Sum over dimensions, mean over batch
        kl_loss = kl_per_dim.sum(dim=1).mean()
        
        # Safety check for NaN
        if torch.isnan(kl_loss):
            kl_loss = torch.tensor(0.0, device=mu.device)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'recon_loss_node': recon_loss_node,
            'recon_loss_edge': recon_loss_edge,
            'kl_loss': kl_loss,
            'kl_per_dim': kl_per_dim.mean(dim=0).detach()  # For logging
        }
    
    def get_latent(self, data: Batch, 
                   use_mean: bool = True) -> torch.Tensor:
        """
        Get latent representation for a batch.
        
        Args:
            data: PyTorch Geometric batch
            use_mean: If True, return mu. If False, sample.
            
        Returns:
            Latent representations [batch_size, latent_dim]
        """
        with torch.no_grad():
            mu, log_var = self.encode(data, store_activations=False)
            if use_mean:
                return mu
            else:
                return self.reparameterize(mu, log_var)
    
    def get_all_activations(self, data: Batch) -> Dict[str, torch.Tensor]:
        """
        Extract all intermediate activations for clustering analysis.
        
        Args:
            data: PyTorch Geometric batch
            
        Returns:
            Dict with activations from each layer:
                - 'encoder.layer_0', 'encoder.layer_1', ...
                - 'pooled': Graph-level pooled representation
                - 'mu': Latent mean
                - 'z': Sampled latent
                - 'attention.layer_0', 'attention.layer_1', ...
        """
        with torch.no_grad():
            # Forward with activation storage
            _ = self.forward(data, store_activations=True)
            
            activations = {}
            
            # Layer activations
            layer_acts = self.encoder.get_layer_activations()
            for key, value in layer_acts.items():
                activations[f'encoder.{key}'] = value
            
            # Attention weights
            att_weights = self.encoder.get_all_attention_weights()
            for key, value in att_weights.items():
                activations[f'attention.{key}'] = value
            
            # Graph-level representations
            activations['pooled'] = self._last_pooled
            activations['mu'] = self._last_mu
            activations['z'] = self._last_z
            
            return activations
    
    def extract_gat_activations_per_node(self, data: Batch, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract GAT layer activations per node (for clustering at node level).
        
        Args:
            data: PyTorch Geometric batch
            layer_idx: Which layer (-1 = last)
            
        Returns:
            Node activations [num_nodes, hidden_dim]
        """
        with torch.no_grad():
            _ = self.forward(data, store_activations=True)
            
            layer_acts = self.encoder.get_layer_activations()
            
            if layer_idx == -1:
                layer_idx = self.encoder.num_layers - 1
            
            key = f'layer_{layer_idx}'
            return layer_acts.get(key, None)


def create_vae_from_config(config: Dict[str, Any],
                           num_node_features: int,
                           num_edge_features: int,
                           num_graph_features: int,
                           num_nodes: int) -> BrainStateVAE:
    """
    Factory function to create VAE from configuration.
    
    Args:
        config: Configuration dictionary
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        num_graph_features: Number of graph-level features
        num_nodes: Number of nodes per graph
        
    Returns:
        Initialized BrainStateVAE model
    """
    model = BrainStateVAE(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_nodes=num_nodes,
        config=config
    )
    
    return model

