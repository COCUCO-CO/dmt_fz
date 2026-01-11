"""
SimCLR: Contrastive Learning for EEG Graph Representations.

Encoder-only model with NT-Xent (InfoNCE) contrastive loss for learning
discriminative latent representations of brain states without reconstruction.

Key features:
- GAT-based encoder for graph data
- MLP projection head for contrastive learning
- Graph augmentations (node dropout, edge dropout, feature perturbation)
- NT-Xent loss with temperature scaling
- Supports visualization of latent space for brain state analysis

Author: Claude
"""

import logging
from typing import Dict, Any, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, ChebConv
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GlobalAttention
from torch_geometric.data import Data, Batch
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# GRAPH AUGMENTATIONS
# =============================================================================

class GraphAugmentor:
    """
    Graph augmentation strategies for contrastive learning.
    
    Implements multiple augmentation types:
    - Node feature masking/noise
    - Edge dropout
    - Node dropout
    - Feature permutation across nodes
    """
    
    def __init__(self, 
                 node_drop_prob: float = 0.1,
                 edge_drop_prob: float = 0.2,
                 feature_mask_prob: float = 0.15,
                 feature_noise_std: float = 0.1):
        """
        Initialize augmentor.
        
        Args:
            node_drop_prob: Probability of dropping each node
            edge_drop_prob: Probability of dropping each edge
            feature_mask_prob: Probability of masking each feature dimension
            feature_noise_std: Standard deviation of Gaussian noise
        """
        self.node_drop_prob = node_drop_prob
        self.edge_drop_prob = edge_drop_prob
        self.feature_mask_prob = feature_mask_prob
        self.feature_noise_std = feature_noise_std
    
    def __call__(self, data: Data) -> Data:
        """Apply random augmentations to a graph."""
        data = data.clone()
        
        # Apply augmentations with some randomness in which are applied
        augmentations = [
            (0.5, self._add_feature_noise),
            (0.5, self._mask_features),
            (0.4, self._drop_edges),
            # Node dropping can be problematic for small graphs, use sparingly
            # (0.2, self._drop_nodes),
        ]
        
        for prob, aug_fn in augmentations:
            if torch.rand(1).item() < prob:
                data = aug_fn(data)
        
        return data
    
    def _add_feature_noise(self, data: Data) -> Data:
        """Add Gaussian noise to node features."""
        if self.feature_noise_std > 0:
            # randn_like preserves device
            noise = torch.randn_like(data.x) * self.feature_noise_std
            data.x = data.x + noise
        return data
    
    def _mask_features(self, data: Data) -> Data:
        """Randomly mask feature dimensions."""
        if self.feature_mask_prob > 0:
            device = data.x.device
            # Create mask on same device as data
            mask = torch.rand(data.x.shape[1], device=device) > self.feature_mask_prob
            mask = mask.float().unsqueeze(0).expand_as(data.x)
            data.x = data.x * mask
        return data
    
    def _drop_edges(self, data: Data) -> Data:
        """Randomly drop edges."""
        if self.edge_drop_prob > 0 and data.edge_index.shape[1] > 0:
            num_edges = data.edge_index.shape[1]
            device = data.edge_index.device
            
            # Create mask on same device
            keep_mask = torch.rand(num_edges, device=device) > self.edge_drop_prob
            
            # Ensure we keep at least some edges
            if keep_mask.sum() < num_edges * 0.3:
                # Keep at least 30% of edges
                n_keep = int(num_edges * 0.3)
                keep_indices = torch.randperm(num_edges, device=device)[:n_keep]
                keep_mask[keep_indices] = True
            
            data.edge_index = data.edge_index[:, keep_mask]
            
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[keep_mask]
        
        return data
    
    def _drop_nodes(self, data: Data) -> Data:
        """Randomly drop nodes (with edge updates)."""
        if self.node_drop_prob > 0 and data.num_nodes > 2:
            num_nodes = data.num_nodes
            keep_mask = torch.rand(num_nodes) > self.node_drop_prob
            
            # Ensure we keep at least half the nodes
            if keep_mask.sum() < num_nodes * 0.5:
                n_keep = int(num_nodes * 0.5)
                keep_indices = torch.randperm(num_nodes)[:n_keep]
                keep_mask = torch.zeros(num_nodes, dtype=torch.bool)
                keep_mask[keep_indices] = True
            
            # Reindex nodes
            node_idx = torch.arange(num_nodes)
            new_idx = torch.zeros(num_nodes, dtype=torch.long) - 1
            new_idx[keep_mask] = torch.arange(keep_mask.sum())
            
            # Filter nodes
            data.x = data.x[keep_mask]
            
            # Filter and reindex edges
            edge_mask = keep_mask[data.edge_index[0]] & keep_mask[data.edge_index[1]]
            data.edge_index = new_idx[data.edge_index[:, edge_mask]]
            
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                data.edge_attr = data.edge_attr[edge_mask]
        
        return data


# =============================================================================
# GAT ENCODER
# =============================================================================

class GATEncoder(nn.Module):
    """
    GAT-based encoder for contrastive learning.
    
    Produces graph-level embeddings from node features using:
    - Multiple GATv2Conv layers with skip connections
    - Global pooling (mean, max, or attention)
    - Final projection to embedding space
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 hidden_dim: int = 64,
                 embedding_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 attention_dropout: float = 0.1,
                 pooling: str = 'mean',
                 use_edge_attr: bool = True,
                 use_skip_connections: bool = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.use_skip = use_skip_connections
        self.pooling_method = pooling
        
        # Edge encoder
        if use_edge_attr and num_edge_features > 0:
            self.edge_encoder = nn.Linear(num_edge_features, num_heads)
        else:
            self.edge_encoder = None
        
        # GAT layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        
        # First layer: input -> hidden
        out_dim = hidden_dim * num_heads
        self.conv_layers.append(
            GATv2Conv(
                num_node_features, hidden_dim,
                heads=num_heads, concat=True,
                dropout=attention_dropout,
                edge_dim=num_heads if use_edge_attr and num_edge_features > 0 else None
            )
        )
        self.batch_norms.append(nn.BatchNorm1d(out_dim))
        
        # Skip projection for first layer
        if use_skip_connections and num_node_features != out_dim:
            self.skip_projs.append(nn.Linear(num_node_features, out_dim))
        else:
            self.skip_projs.append(None)
        
        # Hidden layers
        for _ in range(num_layers - 1):
            self.conv_layers.append(
                GATv2Conv(
                    out_dim, hidden_dim,
                    heads=num_heads, concat=True,
                    dropout=attention_dropout,
                    edge_dim=num_heads if use_edge_attr and num_edge_features > 0 else None
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(out_dim))
            self.skip_projs.append(None)  # Same dimension
        
        self.output_dim = out_dim
        self.dropout = nn.Dropout(dropout)
        
        # Attention pooling (if used)
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.ReLU(),
                nn.Linear(out_dim // 2, 1)
            )
            self.global_attention = GlobalAttention(self.attention_pool)
        
        # Final embedding projection
        if pooling == 'mean+max':
            self.embed_proj = nn.Sequential(
                nn.Linear(out_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
        else:
            self.embed_proj = nn.Sequential(
                nn.Linear(out_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim)
            )
        
        # Storage for activations
        self._layer_activations = {}
        self._attention_weights = {}
    
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None,
                store_activations: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Node features [num_nodes, num_features]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes [num_edges, num_edge_features]
            batch: Batch assignment [num_nodes]
            store_activations: Whether to store intermediate activations
            
        Returns:
            Graph embeddings [batch_size, embedding_dim]
        """
        if store_activations:
            self._layer_activations = {}
            self._attention_weights = {}
        
        # Encode edge attributes
        if self.edge_encoder is not None and edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        
        h = x
        
        # Pass through GAT layers
        for i, (conv, bn) in enumerate(zip(self.conv_layers, self.batch_norms)):
            h_prev = h
            
            # GAT convolution
            if store_activations:
                h_new, (edge_idx, alpha) = conv(
                    h, edge_index, edge_attr=edge_attr,
                    return_attention_weights=True
                )
                self._attention_weights[f'layer_{i}'] = {
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
                self._layer_activations[f'layer_{i}'] = h.detach().cpu()
        
        # Global pooling
        if self.pooling_method == 'attention':
            h_graph = self.global_attention(h, batch)
        elif self.pooling_method == 'mean':
            h_graph = global_mean_pool(h, batch)
        elif self.pooling_method == 'max':
            h_graph = global_max_pool(h, batch)
        elif self.pooling_method == 'sum':
            h_graph = global_add_pool(h, batch)
        elif self.pooling_method == 'mean+max':
            h_graph = torch.cat([
                global_mean_pool(h, batch),
                global_max_pool(h, batch)
            ], dim=1)
        else:
            h_graph = global_mean_pool(h, batch)
        
        if store_activations:
            self._layer_activations['pooled'] = h_graph.detach().cpu()
        
        # Project to embedding space
        embedding = self.embed_proj(h_graph)
        
        if store_activations:
            self._layer_activations['embedding'] = embedding.detach().cpu()
        
        return embedding
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get stored layer activations."""
        return self._layer_activations
    
    def get_attention_weights(self) -> Dict:
        """Get stored attention weights."""
        return self._attention_weights


# =============================================================================
# PROJECTION HEAD
# =============================================================================

class ProjectionHead(nn.Module):
    """
    MLP projection head for contrastive learning.
    
    Maps embeddings to a space where contrastive loss is applied.
    Following SimCLR paper, this is a 2-layer MLP with ReLU.
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 128,
                 output_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings."""
        return self.net(x)


# =============================================================================
# NT-XENT LOSS
# =============================================================================

class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) loss.
    
    Also known as InfoNCE loss. For each positive pair (i, j), treats
    all other samples in the batch as negatives.
    
    Loss = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
    """
    
    def __init__(self, temperature: float = 0.5):
        """
        Args:
            temperature: Temperature scaling parameter (τ)
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for paired embeddings.
        
        Args:
            z1: First view embeddings [batch_size, dim]
            z2: Second view embeddings [batch_size, dim]
            
        Returns:
            Scalar loss value
        """
        batch_size = z1.shape[0]
        
        # Normalize embeddings
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        
        # Concatenate both views: [2*batch_size, dim]
        z = torch.cat([z1, z2], dim=0)
        
        # Compute similarity matrix: [2*batch_size, 2*batch_size]
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Mask out self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim.masked_fill_(mask, float('-inf'))
        
        # Create labels: positive pairs are at positions (i, i+batch_size) and (i+batch_size, i)
        # For sample i in first view, positive is at i+batch_size
        # For sample i in second view (at i+batch_size), positive is at i
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device)
        ])
        
        # Cross entropy loss
        loss = F.cross_entropy(sim, labels)
        
        return loss


# =============================================================================
# SIMCLR MODEL
# =============================================================================

class BrainStateSimCLR(nn.Module):
    """
    SimCLR model for contrastive learning on brain state graphs.
    
    Architecture:
        1. Graph Augmentation (2 views per sample)
        2. GAT Encoder: Graph -> Graph embedding
        3. Projection Head: Embedding -> Contrastive space
        4. NT-Xent Loss
        
    After training, the projection head is discarded and
    the encoder embeddings are used for downstream tasks.
    """
    
    def __init__(self,
                 num_node_features: int,
                 num_edge_features: int,
                 config: Dict[str, Any]):
        super().__init__()
        
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.config = config
        
        model_config = config['model']
        enc_config = model_config['encoder']
        
        # Dimensions
        self.embedding_dim = model_config['embedding']['dim']
        projection_dim = model_config['projection']['dim']
        projection_hidden = model_config['projection'].get('hidden_dim', self.embedding_dim)
        
        # Encoder
        self.encoder = GATEncoder(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=enc_config['hidden_dim'],
            embedding_dim=self.embedding_dim,
            num_layers=enc_config['num_gat_layers'],
            num_heads=enc_config.get('num_attention_heads', 4),
            dropout=enc_config['dropout'],
            attention_dropout=enc_config.get('attention_dropout', 0.1),
            pooling=model_config['pooling']['method'],
            use_edge_attr=enc_config.get('use_edge_attr', True),
            use_skip_connections=enc_config.get('use_skip_connections', True)
        )
        
        # Projection head (only used during training)
        self.projection = ProjectionHead(
            input_dim=self.embedding_dim,
            hidden_dim=projection_hidden,
            output_dim=projection_dim
        )
        
        # Loss function
        temperature = config['loss'].get('temperature', 0.5)
        self.criterion = NTXentLoss(temperature=temperature)
        
        # Augmentor
        aug_config = config.get('augmentation', {})
        self.augmentor = GraphAugmentor(
            node_drop_prob=aug_config.get('node_drop_prob', 0.1),
            edge_drop_prob=aug_config.get('edge_drop_prob', 0.2),
            feature_mask_prob=aug_config.get('feature_mask_prob', 0.15),
            feature_noise_std=aug_config.get('feature_noise_std', 0.1)
        )
        
        # Storage for embeddings
        self._last_embeddings = None
        self._last_projections = None
        
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(f"BrainStateSimCLR: {n_params:,} parameters")
        logger.info(f"  Embedding dim: {self.embedding_dim}")
        logger.info(f"  Projection dim: {projection_dim}")
        logger.info(f"  Temperature: {temperature}")
    
    def augment_batch(self, batch: Batch) -> Tuple[Batch, Batch]:
        """
        Create two augmented views of the batch.
        
        Args:
            batch: PyG Batch object
            
        Returns:
            Tuple of (view1_batch, view2_batch)
        """
        device = batch.x.device
        
        # Unbatch to get individual graphs
        data_list = batch.to_data_list()
        
        # Create two augmented views for each graph
        view1_list = []
        view2_list = []
        
        for data in data_list:
            view1_list.append(self.augmentor(data))
            view2_list.append(self.augmentor(data))
        
        # Re-batch and ensure on correct device
        view1_batch = Batch.from_data_list(view1_list)
        view2_batch = Batch.from_data_list(view2_list)
        
        # Ensure tensors are on original device (in case augmentations moved them)
        if view1_batch.x.device != device:
            view1_batch = view1_batch.to(device)
            view2_batch = view2_batch.to(device)
        
        return view1_batch, view2_batch
    
    def encode(self, data: Batch, store_activations: bool = False) -> torch.Tensor:
        """
        Encode graphs to embeddings.
        
        Args:
            data: PyG Batch
            store_activations: Whether to store intermediate activations
            
        Returns:
            Graph embeddings [batch_size, embedding_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        embeddings = self.encoder(
            x, edge_index, edge_attr, batch,
            store_activations=store_activations
        )
        
        return embeddings
    
    def project(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Project embeddings to contrastive space."""
        return self.projection(embeddings)
    
    def forward(self, batch: Batch, 
                augment: bool = True,
                store_activations: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            batch: PyG Batch
            augment: Whether to create augmented views (training mode)
            store_activations: Whether to store intermediate activations
            
        Returns:
            Dict with embeddings, projections, and loss (if augmented)
        """
        if augment:
            # Create two augmented views
            view1, view2 = self.augment_batch(batch)
            
            # Encode both views
            z1 = self.encode(view1, store_activations=store_activations)
            z2 = self.encode(view2, store_activations=False)
            
            # Project to contrastive space
            p1 = self.project(z1)
            p2 = self.project(z2)
            
            # Compute contrastive loss
            loss = self.criterion(p1, p2)
            
            # Store for logging
            self._last_embeddings = z1.detach().cpu()
            self._last_projections = p1.detach().cpu()
            
            return {
                'embeddings': z1,
                'embeddings_view2': z2,
                'projections': p1,
                'projections_view2': p2,
                'loss': loss
            }
        else:
            # Single forward pass (inference)
            embeddings = self.encode(batch, store_activations=store_activations)
            projections = self.project(embeddings)
            
            self._last_embeddings = embeddings.detach().cpu()
            self._last_projections = projections.detach().cpu()
            
            return {
                'embeddings': embeddings,
                'projections': projections
            }
    
    def get_embeddings(self, batch: Batch) -> torch.Tensor:
        """
        Get embeddings for downstream tasks (no augmentation, no projection).
        
        Args:
            batch: PyG Batch
            
        Returns:
            Graph embeddings [batch_size, embedding_dim]
        """
        with torch.no_grad():
            return self.encode(batch, store_activations=False)
    
    def get_all_activations(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """
        Extract all intermediate activations for analysis.
        
        Args:
            batch: PyG Batch
            
        Returns:
            Dict with activations from each layer
        """
        with torch.no_grad():
            self.forward(batch, augment=False, store_activations=True)
            
            activations = self.encoder.get_activations()
            activations['attention'] = self.encoder.get_attention_weights()
            
            return activations
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_simclr_from_config(config: Dict[str, Any],
                               num_node_features: int,
                               num_edge_features: int) -> BrainStateSimCLR:
    """
    Factory function to create SimCLR model from configuration.
    
    Args:
        config: Configuration dictionary
        num_node_features: Number of node features
        num_edge_features: Number of edge features
        
    Returns:
        Initialized BrainStateSimCLR model
    """
    return BrainStateSimCLR(
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        config=config
    )

