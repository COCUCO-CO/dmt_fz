"""
Graph Attention Network model for brain state classification.

This module implements a GAT-based architecture specifically designed for
EEG synchronization graphs with edge attributes and rich node features.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention, Set2Set
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


class BrainStateGAT(nn.Module):
    """
    Graph Neural Network for EEG brain state classification.
    
    Supports multiple convolution types:
    - GATv2Conv: Graph Attention Networks v2 (with edge attributes)
    - ChebConv: Chebyshev spectral graph convolutions
    
    Architecture:
        1. Multiple graph conv layers (GATv2 or Cheby)
        2. Skip connections for better gradient flow
        3. Attention-based or statistical pooling
        4. MLP classifier with batch normalization
    
    Args:
        num_node_features: Dimension of input node features
        num_edge_features: Dimension of edge features
        num_graph_features: Dimension of graph-level features
        num_classes: Number of output classes
        config: Model configuration dictionary
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 num_graph_features: int,
                 num_classes: int,
                 config: Dict[str, Any]):
        super(BrainStateGAT, self).__init__()
        
        self.num_classes = num_classes
        self.num_graph_features = num_graph_features
        arch_config = config['model']['architecture']
        pool_config = config['model']['pooling']
        mlp_config = config['model']['mlp']
        
        # Get convolution type
        conv_type = arch_config.get('conv_type', 'gatv2').lower()
        self.conv_type = conv_type
        
        hidden_dim = arch_config['hidden_dim']
        num_layers = arch_config['num_gat_layers']
        dropout = arch_config['dropout']
        use_edge_attr = arch_config['use_edge_attr']
        
        # Type-specific parameters
        if conv_type == 'gatv2':
            num_heads = arch_config['num_attention_heads']
            att_dropout = arch_config['attention_dropout']
            concat_heads = arch_config['concat_heads']
            negative_slope = arch_config['negative_slope']
        elif conv_type == 'cheby':
            cheby_k = arch_config.get('cheby_k', 3)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}. Use 'gatv2' or 'cheby'")
        
        # ====================================================================
        # INPUT PROJECTION
        # ====================================================================
        
        # Project edge features if using edge attributes (only for GATv2)
        if conv_type == 'gatv2' and use_edge_attr and num_edge_features > 0:
            self.edge_encoder = nn.Linear(num_edge_features, num_heads)
        else:
            self.edge_encoder = None
        
        # ====================================================================
        # GRAPH CONVOLUTION LAYERS
        # ====================================================================
        
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if conv_type == 'gatv2':
            self.conv_layers.append(
                GATv2Conv(
                    in_channels=num_node_features,
                    out_channels=hidden_dim,
                    heads=num_heads,
                    concat=concat_heads,
                    dropout=att_dropout,
                    negative_slope=negative_slope,
                    edge_dim=num_heads if use_edge_attr else None
                )
            )
            current_dim = hidden_dim * num_heads if concat_heads else hidden_dim
        elif conv_type == 'cheby':
            self.conv_layers.append(
                ChebConv(
                    in_channels=num_node_features,
                    out_channels=hidden_dim,
                    K=cheby_k
                )
            )
            current_dim = hidden_dim
        
        self.batch_norms.append(nn.BatchNorm1d(current_dim))
        logger.info(f"First layer output dim: {current_dim}")
        
        # Hidden layers
        for layer_idx in range(num_layers - 1):
            if conv_type == 'gatv2':
                self.conv_layers.append(
                    GATv2Conv(
                        in_channels=current_dim,
                        out_channels=hidden_dim,
                        heads=num_heads,
                        concat=concat_heads,
                        dropout=att_dropout,
                        negative_slope=negative_slope,
                        edge_dim=num_heads if use_edge_attr else None
                    )
                )
                # Update current_dim for next layer
                current_dim = hidden_dim * num_heads if concat_heads else hidden_dim
                logger.info(f"Hidden layer {layer_idx+1} output dim: {current_dim}")
            elif conv_type == 'cheby':
                self.conv_layers.append(
                    ChebConv(
                        in_channels=current_dim,
                        out_channels=hidden_dim,
                        K=cheby_k
                    )
                )
                current_dim = hidden_dim
                logger.info(f"Hidden layer {layer_idx+1} output dim: {current_dim}")
            self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # ====================================================================
        # POOLING
        # ====================================================================
        
        pooling_method = pool_config['method']
        logger.info(f"Pooling method: {pooling_method}, current_dim before pooling: {current_dim}")
        
        if pooling_method == "attention":
            self.pool = AttentionPooling(current_dim)
            pooled_dim = current_dim
        elif pooling_method == "set2set":
            self.pool = Set2Set(current_dim, processing_steps=3)
            pooled_dim = current_dim * 2
        elif pooling_method in ["mean", "max", "add"]:
            self.pool = None
            pooled_dim = current_dim  # Single pooling
        else:  # mean+max combined
            self.pool = None
            pooled_dim = current_dim * 2  # mean + max concatenated
        
        self.pooling_method = pooling_method
        
        # ====================================================================
        # MLP CLASSIFIER
        # ====================================================================
        
        # Combine pooled features with graph-level features
        mlp_input_dim = pooled_dim + num_graph_features
        
        logger.info(f"Pooled dim: {pooled_dim}, Graph features: {num_graph_features}, MLP input dim: {mlp_input_dim}")
        
        mlp_layers = []
        prev_dim = mlp_input_dim
        
        for hidden in mlp_config['hidden_dims']:
            mlp_layers.append(nn.Linear(prev_dim, hidden))
            
            if mlp_config['use_batch_norm']:
                mlp_layers.append(nn.BatchNorm1d(hidden))
            
            if mlp_config['activation'] == 'relu':
                mlp_layers.append(nn.ReLU())
            elif mlp_config['activation'] == 'elu':
                mlp_layers.append(nn.ELU())
            elif mlp_config['activation'] == 'leaky_relu':
                mlp_layers.append(nn.LeakyReLU(negative_slope))
            
            mlp_layers.append(nn.Dropout(mlp_config['dropout']))
            prev_dim = hidden
        
        # Output layer
        mlp_layers.append(nn.Linear(prev_dim, num_classes))
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # ====================================================================
        # INITIALIZATION
        # ====================================================================
        
        self.reset_parameters()
        
        logger.info(f"BrainStateGNN ({conv_type.upper()}) initialized with {self.count_parameters()} parameters")
    
    def reset_parameters(self):
        """Initialize model parameters."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data: Batch) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            data: Batch of PyTorch Geometric Data objects
            
        Returns:
            Log probabilities for each class (batch_size, num_classes)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Handle graph_attr - ensure it's [batch_size, num_graph_features]
        if hasattr(data, 'graph_attr') and data.graph_attr is not None:
            graph_attr = data.graph_attr
            if graph_attr.dim() == 1:
                num_features = self.num_graph_features or graph_attr.shape[0]
                if num_features > 0 and graph_attr.numel() % num_features == 0:
                    graph_attr = graph_attr.view(-1, num_features)
                else:
                    graph_attr = graph_attr.unsqueeze(0)
        else:
            graph_attr = None
        
        # Encode edge attributes (only for GATv2)
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Graph convolution layers with residual connections
        h = x
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            # Forward through conv layer
            if self.conv_type == 'gatv2':
                h_new = conv(h, edge_index, edge_attr=edge_attr)
            elif self.conv_type == 'cheby':
                h_new = conv(h, edge_index)
            
            h_new = bn(h_new)
            h_new = F.elu(h_new)
            h_new = self.dropout(h_new)
            
            # Residual connection (if dimensions match)
            if i > 0 and h.shape[-1] == h_new.shape[-1]:
                h = h + h_new
            else:
                h = h_new
        
        # Global pooling
        if self.pooling_method == "attention":
            h_graph = self.pool(h, batch)
        elif self.pooling_method == "set2set":
            h_graph = self.pool(h, batch)
        elif self.pooling_method == "mean":
            h_graph = global_mean_pool(h, batch)
        elif self.pooling_method == "max":
            h_graph = global_max_pool(h, batch)
        elif self.pooling_method == "add":
            h_graph = global_add_pool(h, batch)
        else:
            # This should not happen with config pooling="mean"
            logger.error(f"Unexpected pooling_method: '{self.pooling_method}' - using mean+max fallback")
            # Combined mean + max pooling
            h_mean = global_mean_pool(h, batch)
            h_max = global_max_pool(h, batch)
            h_graph = torch.cat([h_mean, h_max], dim=1)
        
        # Concatenate with graph-level features
        if graph_attr is not None:
            h_graph = torch.cat([h_graph, graph_attr], dim=1)
        
        # MLP classifier
        out = self.mlp(h_graph)
        
        return F.log_softmax(out, dim=1)
    
    def get_attention_weights(self, data: Batch, layer_idx: int = 0):
        """
        Extract attention weights from a specific GAT layer (only for GATv2).
        
        Args:
            data: Batch of PyTorch Geometric Data objects
            layer_idx: Which GAT layer to extract attention from (0 = first layer)
            
        Returns:
            Tuple of (edge_index, attention_weights)
        """
        if self.conv_type != 'gatv2':
            logger.warning("Attention weights only available for GATv2Conv")
            return None, None
        
        if layer_idx >= len(self.conv_layers):
            raise ValueError(f"Layer index {layer_idx} out of range (model has {len(self.conv_layers)} layers)")
        
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Encode edge attributes
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Forward through layers up to layer_idx
        h = x
        for i in range(layer_idx + 1):
            if i == layer_idx:
                # Get attention weights from target layer
                _, (edge_index_att, alpha) = self.conv_layers[i](
                    h, edge_index, edge_attr=edge_attr, return_attention_weights=True
                )
            else:
                # Normal forward pass
                if self.conv_type == 'gatv2':
                    h = self.conv_layers[i](h, edge_index, edge_attr=edge_attr)
                else:
                    h = self.conv_layers[i](h, edge_index)
                
                h = self.batch_norms[i](h)
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=False)
        
        return edge_index_att, alpha
    
    def get_all_attention_weights(self, data: Batch):
        """
        Extract attention weights from ALL GAT layers (only for GATv2).
        
        Args:
            data: Batch of PyTorch Geometric Data objects
            
        Returns:
            List of tuples (edge_index, attention_weights) for each layer
        """
        if self.conv_type != 'gatv2':
            logger.warning("Attention weights only available for GATv2Conv")
            return []
        
        all_attentions = []
        
        for layer_idx in range(len(self.conv_layers)):
            edge_index_att, alpha = self.get_attention_weights(data, layer_idx)
            all_attentions.append((edge_index_att, alpha))
        
        return all_attentions


def create_model_from_config(config: Dict[str, Any],
                             num_node_features: int,
                             num_edge_features: int,
                             num_graph_features: int,
                             num_classes: int) -> BrainStateGAT:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Configuration dictionary
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        num_graph_features: Number of graph-level features
        num_classes: Number of output classes
        
    Returns:
        Initialized BrainStateGAT model
    """
    model = BrainStateGAT(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes,
        config=config
    )
    
    return model

