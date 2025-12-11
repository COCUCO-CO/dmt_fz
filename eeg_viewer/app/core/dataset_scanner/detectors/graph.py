"""
Graph dataset detector.

Handles PyTorch Geometric files, NetworkX formats, and project-specific
phases-*.pkl format.
"""

from pathlib import Path
from typing import Set, List, Dict, Any, Optional
from dataclasses import dataclass, field
import pickle

from .base import BaseDetector, DetectionResult
from ..models import GraphSpecificInfo


@dataclass
class PhasesInfo:
    """Information from phases-*.pkl files."""
    bands: List[str] = field(default_factory=list)
    has_eeg: bool = False
    has_stc: bool = False
    n_channels_eeg: int = 0
    n_parcels: int = 0
    n_epochs_per_file: int = 0
    n_timepoints: int = 0  # Timepoints per epoch
    available_keys: List[str] = field(default_factory=list)
    # Feature types available (phases, amplitudes, syncros, kuramoto, filtered)
    eeg_feature_types: List[str] = field(default_factory=list)
    stc_feature_types: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'bands': self.bands,
            'has_eeg': self.has_eeg,
            'has_stc': self.has_stc,
            'n_channels_eeg': self.n_channels_eeg,
            'n_parcels': self.n_parcels,
            'n_epochs_per_file': self.n_epochs_per_file,
            'n_timepoints': self.n_timepoints,
            'available_keys': self.available_keys,
            'eeg_feature_types': self.eeg_feature_types,
            'stc_feature_types': self.stc_feature_types,
        }


class GraphDetector(BaseDetector):
    """Detector for graph datasets."""
    
    EXTENSIONS: Set[str] = {
        '.pt', '.pth',  # PyTorch
        '.pkl', '.pickle',  # Pickle (including phases-*.pkl)
        '.gpickle',  # NetworkX pickle
        '.graphml', '.gml',  # Standard formats
        '.edgelist', '.edges',  # Edge lists
        '.adj', '.mtx',  # Adjacency matrices
    }
    
    # Patterns for phases files
    PHASES_PATTERNS = ['phases-', 'syncro-', 'order-']
    
    def detect(self, path: Path, analyze_samples: bool = False,
               sample_size: int = 10) -> DetectionResult:
        """Detect graph dataset."""
        result = DetectionResult()
        
        if not path.exists():
            result.warnings.append(f"Path does not exist: {path}")
            return result
        
        # Find all potential graph files
        files = self._find_files(path, self.EXTENSIONS)
        
        if not files:
            return result
        
        # Separate phases files from other graph files
        phases_files = [f for f in files if any(p in f.name for p in self.PHASES_PATTERNS)]
        pt_files = [f for f in files if f.suffix in {'.pt', '.pth'} and f not in phases_files]
        other_files = [f for f in files if f not in phases_files and f not in pt_files]
        
        # Prioritize detection based on file types found
        if phases_files:
            result = self._detect_phases_dataset(path, phases_files, analyze_samples, sample_size)
        elif pt_files:
            result = self._detect_pyg_dataset(path, pt_files, analyze_samples, sample_size)
        elif other_files:
            result = self._detect_generic_graph_dataset(path, other_files, analyze_samples, sample_size)
        
        return result
    
    def _detect_phases_dataset(self, path: Path, files: List[Path],
                                analyze_samples: bool, sample_size: int) -> DetectionResult:
        """Detect phases-*.pkl format dataset."""
        result = DetectionResult()
        result.is_detected = True
        result.is_phases_format = True
        result.file_count = len(files)
        result.sample_files = files[:sample_size]
        result.total_size_bytes = self._calculate_total_size(files)
        result.extensions_found = ['.pkl']
        result.confidence = 0.95
        
        # Detect conditions (DMT, EC, EO folders)
        has_conditions, conditions, condition_counts = self._detect_conditions(path)
        result.has_conditions = has_conditions
        result.conditions_found = conditions
        result.condition_counts = condition_counts
        
        # Also set as classes for compatibility
        if has_conditions:
            result.has_classes = True
            result.classes_found = conditions
            result.class_counts = condition_counts
        
        # Detect subjects from filenames
        has_subjects, subjects, subject_counts = self._detect_subjects_from_files(files)
        result.has_subjects = has_subjects
        result.subjects_found = subjects
        result.subject_counts = subject_counts
        
        # Analyze sample files - prefer phases-*.pkl over order-*.pkl or syncro-*.pkl
        if analyze_samples and files:
            # Find a phases-*.pkl file specifically (not order or syncro)
            phases_only = [f for f in files if f.name.startswith('phases-')]
            sample_file = phases_only[0] if phases_only else files[0]
            result.phases_info = self._analyze_phases_file(sample_file)
            
            if result.phases_info:
                # Calculate total potential graphs
                n_files = len(files)
                n_epochs = result.phases_info.n_epochs_per_file
                n_bands = len(result.phases_info.bands)
                result.total_graphs = n_files * n_epochs * n_bands
        
        # Generate suggestions
        result.suggestions = [
            "Use custom DataLoader for phases format",
            "Available data: phases, syncros, kuramoto, amplitudes",
        ]
        if result.phases_info and result.phases_info.has_stc:
            result.suggestions.append("Source-localized (STC) data available")
        
        result.suggested_loader = "Custom phases loader or PyG DataLoader"
        
        return result
    
    def _analyze_phases_file(self, filepath: Path) -> Optional[PhasesInfo]:
        """Analyze a phases-*.pkl file."""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, dict):
                return None
            
            info = PhasesInfo()
            info.available_keys = list(data.keys())
            
            # Extract feature types for EEG and STC
            feature_type_prefixes = ['phases', 'amplitudes', 'syncros', 'kuramoto', 'filtered']
            for prefix in feature_type_prefixes:
                if f'{prefix}_eeg' in data:
                    info.eeg_feature_types.append(prefix)
                if f'{prefix}_stc' in data:
                    info.stc_feature_types.append(prefix)
            
            # Check for EEG data
            if 'phases_eeg' in data:
                info.has_eeg = True
                phases_eeg = data['phases_eeg']
                if isinstance(phases_eeg, dict):
                    info.bands = list(phases_eeg.keys())
                    # Get channel count and timepoints from first band, first epoch
                    first_band = info.bands[0] if info.bands else None
                    if first_band and phases_eeg[first_band]:
                        first_epoch = phases_eeg[first_band][0]
                        if hasattr(first_epoch, 'shape'):
                            info.n_channels_eeg = first_epoch.shape[0]
                            if len(first_epoch.shape) > 1:
                                info.n_timepoints = first_epoch.shape[1]
                        info.n_epochs_per_file = len(phases_eeg[first_band])
            
            # Check for STC data
            if 'phases_stc' in data:
                info.has_stc = True
                phases_stc = data['phases_stc']
                if isinstance(phases_stc, dict):
                    if not info.bands:
                        info.bands = list(phases_stc.keys())
                    first_band = info.bands[0] if info.bands else None
                    if first_band and phases_stc[first_band]:
                        first_epoch = phases_stc[first_band][0]
                        if hasattr(first_epoch, 'shape'):
                            info.n_parcels = first_epoch.shape[0]
                        if info.n_epochs_per_file == 0:
                            info.n_epochs_per_file = len(phases_stc[first_band])
            
            return info
            
        except Exception as e:
            return None
    
    def _detect_pyg_dataset(self, path: Path, files: List[Path],
                            analyze_samples: bool, sample_size: int) -> DetectionResult:
        """Detect PyTorch Geometric dataset."""
        result = DetectionResult()
        result.is_detected = True
        result.file_count = len(files)
        result.sample_files = files[:sample_size]
        result.total_size_bytes = self._calculate_total_size(files)
        result.extensions_found = ['.pt']
        result.confidence = 0.90
        
        # Detect structure
        has_splits, splits, split_counts = self._detect_splits(path)
        result.has_splits = has_splits
        result.splits_found = splits
        result.split_counts = split_counts
        
        has_classes, classes, class_counts = self._detect_classes(path)
        result.has_classes = has_classes
        result.classes_found = classes
        result.class_counts = class_counts
        
        # Analyze samples
        if analyze_samples and files:
            result.graph_info = self._analyze_pyg_files(files[:sample_size])
        
        result.suggestions = [
            "Use torch_geometric.loader.DataLoader",
            "Consider using InMemoryDataset for better performance",
        ]
        result.suggested_loader = "torch_geometric.loader.DataLoader"
        
        return result
    
    def _analyze_pyg_files(self, files: List[Path]) -> GraphSpecificInfo:
        """Analyze PyTorch Geometric files."""
        info = GraphSpecificInfo()
        
        nodes_list = []
        edges_list = []
        node_features = set()
        edge_features = set()
        labels = set()
        
        for f in files:
            try:
                try:
                    import torch
                    data = torch.load(f, map_location='cpu')
                except ImportError:
                    with open(f, 'rb') as file:
                        data = pickle.load(file)
                
                if isinstance(data, dict):
                    # Dict format
                    if 'x' in data and hasattr(data['x'], 'shape'):
                        nodes_list.append(data['x'].shape[0])
                        node_features.add(data['x'].shape[1] if len(data['x'].shape) > 1 else 1)
                    if 'num_nodes' in data:
                        nodes_list.append(data['num_nodes'])
                    if 'edge_index' in data and hasattr(data['edge_index'], 'shape'):
                        edges_list.append(data['edge_index'].shape[1])
                    if 'edge_attr' in data and hasattr(data['edge_attr'], 'shape'):
                        edge_features.add(data['edge_attr'].shape[1] if len(data['edge_attr'].shape) > 1 else 1)
                    if 'y' in data:
                        y = data['y']
                        if hasattr(y, 'item'):
                            labels.add(y.item())
                        elif hasattr(y, 'tolist'):
                            labels.update(y.tolist() if hasattr(y.tolist(), '__iter__') else [y.tolist()])
                else:
                    # PyG Data object
                    if hasattr(data, 'x') and data.x is not None:
                        nodes_list.append(data.x.shape[0])
                        node_features.add(data.x.shape[1] if len(data.x.shape) > 1 else 1)
                    if hasattr(data, 'num_nodes'):
                        nodes_list.append(data.num_nodes)
                    if hasattr(data, 'edge_index') and data.edge_index is not None:
                        edges_list.append(data.edge_index.shape[1])
                    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                        edge_features.add(data.edge_attr.shape[1] if len(data.edge_attr.shape) > 1 else 1)
                    if hasattr(data, 'y') and data.y is not None:
                        info.has_labels = True
                        y = data.y
                        if hasattr(y, 'item'):
                            labels.add(y.item())
                        elif hasattr(y, 'tolist'):
                            labels.update(y.flatten().tolist())
                            
            except Exception:
                continue
        
        if nodes_list:
            info.num_nodes_range = (min(nodes_list), max(nodes_list))
        if edges_list:
            info.num_edges_range = (min(edges_list), max(edges_list))
        if node_features:
            info.node_features_dim = max(node_features)
        if edge_features:
            info.edge_features_dim = max(edge_features)
        if labels:
            info.has_labels = True
            info.num_classes = len(labels)
        
        return info
    
    def _detect_generic_graph_dataset(self, path: Path, files: List[Path],
                                       analyze_samples: bool, sample_size: int) -> DetectionResult:
        """Detect generic graph format dataset."""
        result = DetectionResult()
        result.is_detected = True
        result.file_count = len(files)
        result.sample_files = files[:sample_size]
        result.total_size_bytes = self._calculate_total_size(files)
        
        extensions = set(f.suffix.lower() for f in files)
        result.extensions_found = sorted(extensions)
        result.confidence = 0.75
        
        # Try to analyze if NetworkX available
        if analyze_samples:
            try:
                import networkx as nx
                result.graph_info = self._analyze_networkx_files(files[:sample_size])
            except ImportError:
                pass
        
        result.suggestions = [
            "Consider converting to PyTorch Geometric format for ML",
            "Use networkx for graph analysis",
        ]
        
        return result
    
    def _analyze_networkx_files(self, files: List[Path]) -> GraphSpecificInfo:
        """Analyze NetworkX graph files."""
        import networkx as nx
        
        info = GraphSpecificInfo()
        nodes_list = []
        edges_list = []
        
        for f in files:
            try:
                if f.suffix == '.gpickle':
                    G = nx.read_gpickle(f)
                elif f.suffix == '.graphml':
                    G = nx.read_graphml(f)
                elif f.suffix == '.gml':
                    G = nx.read_gml(f)
                else:
                    continue
                
                nodes_list.append(G.number_of_nodes())
                edges_list.append(G.number_of_edges())
                info.is_directed = G.is_directed()
                
            except Exception:
                continue
        
        if nodes_list:
            info.num_nodes_range = (min(nodes_list), max(nodes_list))
        if edges_list:
            info.num_edges_range = (min(edges_list), max(edges_list))
        
        return info

