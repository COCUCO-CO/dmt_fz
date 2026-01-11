"""
Multi-output regression model for predicting subjective experiences.

Supports MLP, GNN (GATv2, GCN, GraphSAGE, Chebyshev), and hybrid architectures.
"""

import logging
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Batch

logger = logging.getLogger(__name__)


class AttentionPooling(nn.Module):
    """Attention-based global pooling."""
    
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


class MLPRegressor(nn.Module):
    """Multi-layer perceptron for multi-output regression."""
    
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dims: list = [256, 128, 64],
                 activation: str = "relu",
                 use_batch_norm: bool = True,
                 dropout: float = 0.4):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU(0.2))
            elif activation == "gelu":
                layers.append(nn.GELU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)


class GNNRegressor(nn.Module):
    """
    Graph Neural Network for multi-output regression.
    
    Supports multiple convolution types:
    - GATv2: Graph Attention Networks v2
    - GCN: Graph Convolutional Networks
    - SAGE: GraphSAGE
    - Cheby: Chebyshev spectral convolutions
    """
    
    def __init__(self,
                 num_node_features: int,
                 output_dim: int,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 num_heads: int = 4,
                 conv_type: str = "gatv2",
                 dropout: float = 0.3,
                 cheby_k: int = 3,
                 use_edge_attr: bool = True,
                 pooling: str = "mean+max",
                 mlp_hidden_dims: list = [128, 64],
                 mlp_dropout: float = 0.4,
                 num_edge_features: int = 1):
        super().__init__()
        
        self.conv_type = conv_type.lower()
        self.num_layers = num_layers
        self.use_edge_attr = use_edge_attr and conv_type.lower() == "gatv2"
        self.pooling_method = pooling
        
        # Edge encoder for GATv2
        if self.use_edge_attr and num_edge_features > 0:
            self.edge_encoder = nn.Linear(num_edge_features, num_heads)
        else:
            self.edge_encoder = None
        
        # Graph conv layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        if self.conv_type == "gatv2":
            self.convs.append(GATv2Conv(
                num_node_features, hidden_dim, 
                heads=num_heads, concat=True, dropout=dropout,
                edge_dim=num_heads if self.use_edge_attr else None
            ))
            current_dim = hidden_dim * num_heads
        elif self.conv_type == "gcn":
            self.convs.append(GCNConv(num_node_features, hidden_dim))
            current_dim = hidden_dim
        elif self.conv_type == "sage":
            self.convs.append(SAGEConv(num_node_features, hidden_dim))
            current_dim = hidden_dim
        elif self.conv_type == "cheby":
            self.convs.append(ChebConv(num_node_features, hidden_dim, K=cheby_k))
            current_dim = hidden_dim
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            if self.conv_type == "gatv2":
                self.convs.append(GATv2Conv(
                    current_dim, hidden_dim,
                    heads=num_heads, concat=True, dropout=dropout,
                    edge_dim=num_heads if self.use_edge_attr else None
                ))
                current_dim = hidden_dim * num_heads
            elif self.conv_type == "gcn":
                self.convs.append(GCNConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            elif self.conv_type == "sage":
                self.convs.append(SAGEConv(current_dim, hidden_dim))
                current_dim = hidden_dim
            elif self.conv_type == "cheby":
                self.convs.append(ChebConv(current_dim, hidden_dim, K=cheby_k))
                current_dim = hidden_dim
            
            self.batch_norms.append(nn.BatchNorm1d(current_dim))
        
        self.dropout = nn.Dropout(dropout)
        
        # Pooling
        if pooling == "attention":
            self.pool = AttentionPooling(current_dim)
            pooled_dim = current_dim
        elif pooling == "mean+max":
            self.pool = None
            pooled_dim = current_dim * 2
        else:
            self.pool = None
            pooled_dim = current_dim
        
        # MLP head
        self.mlp = MLPRegressor(
            input_dim=pooled_dim,
            output_dim=output_dim,
            hidden_dims=mlp_hidden_dims,
            dropout=mlp_dropout
        )
        
        logger.info(f"GNNRegressor ({conv_type}) with {self._count_params()} parameters")
    
    def _count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, data: Batch):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Encode edge attributes
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        # Graph convolutions
        for conv, bn in zip(self.convs, self.batch_norms):
            if self.conv_type == "gatv2" and self.use_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
            x = self.dropout(x)
        
        # Global pooling
        if self.pooling_method == "attention":
            x = self.pool(x, batch)
        elif self.pooling_method == "mean+max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        elif self.pooling_method == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_method == "max":
            x = global_max_pool(x, batch)
        elif self.pooling_method == "add":
            x = global_add_pool(x, batch)
        
        # MLP regressor
        out = self.mlp(x)
        
        return out
    
    def get_graph_embedding(self, data: Batch) -> torch.Tensor:
        """Extract graph-level embedding before MLP."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        for conv, bn in zip(self.convs, self.batch_norms):
            if self.conv_type == "gatv2" and self.use_edge_attr:
                x = conv(x, edge_index, edge_attr=edge_attr)
            else:
                x = conv(x, edge_index)
            x = bn(x)
            x = F.elu(x)
        
        if self.pooling_method == "attention":
            x = self.pool(x, batch)
        elif self.pooling_method == "mean+max":
            x_mean = global_mean_pool(x, batch)
            x_max = global_max_pool(x, batch)
            x = torch.cat([x_mean, x_max], dim=1)
        elif self.pooling_method == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling_method == "max":
            x = global_max_pool(x, batch)
        
        return x


class ExperienceRegressor(nn.Module):
    """
    Wrapper model supporting MLP, GNN, or hybrid architectures.
    """
    
    def __init__(self, config: Dict[str, Any], 
                 input_dim: int = None, 
                 num_node_features: int = None,
                 num_edge_features: int = 1):
        super().__init__()
        
        model_type = config['model']['type'].lower()
        output_dim = config['model']['output_dim']
        mlp_config = config['model']['mlp']
        
        self.model_type = model_type
        
        if model_type == "mlp":
            assert input_dim is not None, "input_dim required for MLP"
            self.model = MLPRegressor(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=mlp_config['hidden_dims'],
                activation=mlp_config['activation'],
                use_batch_norm=mlp_config['use_batch_norm'],
                dropout=mlp_config['dropout']
            )
            self.is_gnn = False
            
        elif model_type == "gnn":
            assert num_node_features is not None, "num_node_features required for GNN"
            gnn_config = config['model']['gnn']
            self.model = GNNRegressor(
                num_node_features=num_node_features,
                output_dim=output_dim,
                hidden_dim=gnn_config['hidden_dim'],
                num_layers=gnn_config['num_layers'],
                num_heads=gnn_config['num_heads'],
                conv_type=gnn_config['conv_type'],
                dropout=gnn_config['dropout'],
                cheby_k=gnn_config.get('cheby_k', 3),
                use_edge_attr=gnn_config.get('use_edge_attr', True),
                pooling=gnn_config.get('pooling', 'mean+max'),
                mlp_hidden_dims=mlp_config['hidden_dims'],
                mlp_dropout=mlp_config['dropout'],
                num_edge_features=num_edge_features
            )
            self.is_gnn = True
            
        elif model_type == "hybrid":
            # GNN for graph + additional MLP
            assert num_node_features is not None, "num_node_features required for hybrid"
            gnn_config = config['model']['gnn']
            self.gnn = GNNRegressor(
                num_node_features=num_node_features,
                output_dim=output_dim,
                hidden_dim=gnn_config['hidden_dim'],
                num_layers=gnn_config['num_layers'],
                num_heads=gnn_config['num_heads'],
                conv_type=gnn_config['conv_type'],
                dropout=gnn_config['dropout'],
                cheby_k=gnn_config.get('cheby_k', 3),
                use_edge_attr=gnn_config.get('use_edge_attr', True),
                pooling=gnn_config.get('pooling', 'mean+max'),
                mlp_hidden_dims=mlp_config['hidden_dims'],
                mlp_dropout=mlp_config['dropout'],
                num_edge_features=num_edge_features
            )
            self.model = self.gnn
            self.is_gnn = True
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        logger.info(f"Created ExperienceRegressor ({model_type}) with {self.count_parameters()} parameters")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, x):
        return self.model(x)


def create_model_from_config(config: Dict[str, Any], 
                            input_dim: int = None,
                            num_node_features: int = None,
                            num_edge_features: int = 1) -> ExperienceRegressor:
    """Factory function to create model from config."""
    return ExperienceRegressor(
        config, 
        input_dim=input_dim, 
        num_node_features=num_node_features,
        num_edge_features=num_edge_features
    )
