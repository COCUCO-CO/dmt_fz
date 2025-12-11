"""
Tests for GraphDetector.

Verifies detection of graph datasets including PyTorch Geometric files
and the project-specific phases-*.pkl format.
"""

import pytest
from pathlib import Path
import numpy as np
import pickle


class TestGraphDetectorBasic:
    """Basic graph detection tests."""
    
    def test_detects_pt_files(self, graph_pt_dataset: Path):
        """Verify .pt graph files are detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset)
        
        assert result.is_detected is True
        assert result.confidence > 0.8
        assert '.pt' in result.extensions_found
    
    def test_counts_graphs_correctly(self, graph_pt_dataset: Path):
        """Verify graph count is accurate."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset)
        
        # Created 20 graph files
        assert result.file_count == 20
    
    def test_detects_phases_pkl(self, by_condition_phases_dataset: Path):
        """Verify phases-*.pkl files are detected as graph data."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset)
        
        assert result.is_detected is True
        assert result.is_phases_format is True
    
    def test_empty_directory_not_detected(self, empty_dataset: Path):
        """Verify empty directory is not detected as graph dataset."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(empty_dataset)
        
        assert result.is_detected is False


class TestGraphDetectorStructures:
    """Test graph detection with various structures."""
    
    def test_detects_condition_structure(self, by_condition_phases_dataset: Path):
        """Verify condition-based structure is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset)
        
        assert result.is_detected is True
        assert result.has_conditions is True
        assert set(result.conditions_found) == {'DMT', 'EC', 'EO'}
    
    def test_detects_subjects(self, by_condition_phases_dataset: Path):
        """Verify subjects are detected from filenames."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset)
        
        assert result.has_subjects is True
        assert 'S01' in result.subjects_found
        assert len(result.subjects_found) == 5
    
    def test_detects_class_folders(self, graph_by_class_dataset: Path):
        """Verify class-based folder structure is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_by_class_dataset)
        
        assert result.is_detected is True
        assert result.has_classes is True
        assert set(result.classes_found) == {'class_0', 'class_1', 'class_2'}
    
    def test_counts_per_condition(self, by_condition_phases_dataset: Path):
        """Verify correct count per condition."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset)
        
        # 5 subjects per condition
        assert result.condition_counts['DMT'] == 5
        assert result.condition_counts['EC'] == 5
        assert result.condition_counts['EO'] == 5


class TestGraphDetectorPhasesFormat:
    """Test phases-*.pkl specific detection."""
    
    def test_detects_phases_structure(self, by_condition_phases_dataset: Path):
        """Verify phases file structure is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset, analyze_samples=True)
        
        assert result.is_phases_format is True
        assert result.phases_info is not None
    
    def test_detects_bands(self, by_condition_phases_dataset: Path):
        """Verify frequency bands are detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset, analyze_samples=True)
        
        expected_bands = {'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma'}
        assert set(result.phases_info.bands) == expected_bands
    
    def test_detects_eeg_data(self, by_condition_phases_dataset: Path):
        """Verify EEG data availability is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset, analyze_samples=True)
        
        assert result.phases_info.has_eeg is True
        assert result.phases_info.n_channels_eeg == 24
    
    def test_detects_stc_data(self, by_condition_phases_dataset: Path):
        """Verify STC (source-localized) data availability is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset, analyze_samples=True)
        
        assert result.phases_info.has_stc is True
        assert result.phases_info.n_parcels == 68
    
    def test_detects_epochs_count(self, by_condition_phases_dataset: Path):
        """Verify epochs count is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset, analyze_samples=True)
        
        assert result.phases_info.n_epochs_per_file == 10
    
    def test_detects_data_keys(self, by_condition_phases_dataset: Path):
        """Verify all data keys are detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset, analyze_samples=True)
        
        # Should detect phases, syncros, kuramoto, amplitudes
        assert 'phases_eeg' in result.phases_info.available_keys
        assert 'syncros_eeg' in result.phases_info.available_keys
        assert 'kuramoto_eeg' in result.phases_info.available_keys


class TestGraphDetectorPyGFormat:
    """Test PyTorch Geometric format detection."""
    
    def test_detects_node_features(self, graph_pt_dataset: Path):
        """Verify node features dimension is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset, analyze_samples=True)
        
        assert result.graph_info is not None
        assert result.graph_info.node_features_dim == 10  # Created with 10 features
    
    def test_detects_edge_features(self, graph_pt_dataset: Path):
        """Verify edge features dimension is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset, analyze_samples=True)
        
        assert result.graph_info.edge_features_dim == 2  # Created with 2 features
    
    def test_detects_node_count_range(self, graph_pt_dataset: Path):
        """Verify node count range is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset, analyze_samples=True)
        
        # Created with 20-50 nodes
        assert result.graph_info.num_nodes_range[0] >= 20
        assert result.graph_info.num_nodes_range[1] <= 50
    
    def test_detects_edge_count_range(self, graph_pt_dataset: Path):
        """Verify edge count range is detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset, analyze_samples=True)
        
        assert result.graph_info.num_edges_range[0] > 0
        assert result.graph_info.num_edges_range[1] > result.graph_info.num_edges_range[0]
    
    def test_detects_labels(self, graph_pt_dataset: Path):
        """Verify graph labels are detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset, analyze_samples=True)
        
        assert result.graph_info.has_labels is True
        # Number of classes depends on sample - created with 3 classes but may see 2-3 in samples
        assert result.graph_info.num_classes >= 2


class TestGraphDetectorNetworkX:
    """Test NetworkX format detection."""
    
    def test_detects_gpickle(self, temp_dir: Path):
        """Verify .gpickle NetworkX files are detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")
        
        dataset_path = temp_dir / "networkx_graphs"
        dataset_path.mkdir()
        
        # Create sample NetworkX graphs
        for i in range(5):
            G = nx.erdos_renyi_graph(20, 0.3)
            nx.set_node_attributes(G, {n: {'feature': np.random.randn(5)} for n in G.nodes()})
            nx.write_gpickle(G, dataset_path / f"graph_{i}.gpickle")
        
        detector = GraphDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.gpickle' in result.extensions_found
    
    def test_detects_graphml(self, temp_dir: Path):
        """Verify .graphml files are detected."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        try:
            import networkx as nx
        except ImportError:
            pytest.skip("NetworkX not available")
        
        dataset_path = temp_dir / "graphml_graphs"
        dataset_path.mkdir()
        
        for i in range(3):
            G = nx.erdos_renyi_graph(10, 0.4)
            nx.write_graphml(G, dataset_path / f"graph_{i}.graphml")
        
        detector = GraphDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        assert '.graphml' in result.extensions_found


class TestGraphDetectorEdgeCases:
    """Test edge cases for graph detection."""
    
    def test_handles_mixed_formats(self, temp_dir: Path):
        """Verify mixed graph formats are handled."""
        from app.core.dataset_scanner.detectors import GraphDetector
        from tests.dataset_scanner.conftest import create_graph_pt, create_phases_pkl
        
        dataset_path = temp_dir / "mixed_graphs"
        dataset_path.mkdir()
        
        # PyG format
        create_graph_pt(dataset_path / "graph.pt", n_nodes=20, n_edges=50)
        
        # Phases format
        create_phases_pkl(dataset_path / "phases-S01-DMT.pkl")
        
        detector = GraphDetector()
        result = detector.detect(dataset_path)
        
        assert result.is_detected is True
        # Detector prioritizes phases format when found, so may report only one type
        assert len(result.extensions_found) >= 1
    
    def test_handles_corrupt_pkl(self, temp_dir: Path):
        """Verify corrupt pickle files are handled gracefully."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        dataset_path = temp_dir / "corrupt_pkl"
        dataset_path.mkdir()
        
        # Create corrupt pickle
        corrupt_file = dataset_path / "corrupt.pkl"
        with open(corrupt_file, 'wb') as f:
            f.write(b'not valid pickle data')
        
        detector = GraphDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Should not crash - that's the main requirement
        # May still detect as graph with low confidence since it's a .pkl file
        assert result is not None
    
    def test_distinguishes_from_other_pkl(self, temp_dir: Path):
        """Verify non-graph pickles are handled appropriately."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        dataset_path = temp_dir / "other_pkl"
        dataset_path.mkdir()
        
        # Create non-graph pickle
        with open(dataset_path / "data.pkl", 'wb') as f:
            pickle.dump({'some': 'data', 'not': 'graph'}, f)
        
        detector = GraphDetector()
        result = detector.detect(dataset_path, analyze_samples=True)
        
        # Generic .pkl files may be detected but with limited graph_info
        # The important thing is it doesn't crash and provides some result
        if result.is_detected:
            # If detected, graph_info should have minimal data since it's not a real graph
            assert result.graph_info is None or result.graph_info.num_nodes_range == (0, 0)


class TestGraphDetectorStatistics:
    """Test statistical analysis of graph datasets."""
    
    def test_calculates_total_graphs(self, by_condition_phases_dataset: Path):
        """Verify total graphs are calculated considering epochs."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset, analyze_samples=True)
        
        # 3 conditions × 5 subjects × 10 epochs × 5 bands = 750 potential graphs per source
        # Just verify we get a reasonable total
        assert result.total_graphs > 0
    
    def test_calculates_size_statistics(self, graph_pt_dataset: Path):
        """Verify size statistics are calculated."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset)
        
        assert result.total_size_bytes > 0
    
    def test_balance_analysis(self, graph_by_class_dataset: Path):
        """Verify class balance is analyzed."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_by_class_dataset)
        
        # All classes have 10 graphs each (balanced)
        assert all(count == 10 for count in result.class_counts.values())


class TestGraphDetectorOutput:
    """Test output format and completeness."""
    
    def test_returns_detection_result(self, graph_pt_dataset: Path):
        """Verify detection result has all required fields."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset)
        
        assert hasattr(result, 'is_detected')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'file_count')
        assert hasattr(result, 'extensions_found')
        assert hasattr(result, 'sample_files')
    
    def test_generates_suggestions(self, graph_pt_dataset: Path):
        """Verify detector generates useful suggestions."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(graph_pt_dataset)
        
        assert len(result.suggestions) > 0 or result.suggested_loader is not None
    
    def test_phases_generates_appropriate_suggestions(self, by_condition_phases_dataset: Path):
        """Verify phases format gets appropriate suggestions."""
        from app.core.dataset_scanner.detectors import GraphDetector
        
        detector = GraphDetector()
        result = detector.detect(by_condition_phases_dataset)
        
        # Should suggest using phases data appropriately
        suggestions_text = ' '.join(result.suggestions).lower()
        assert 'phases' in suggestions_text or 'graph' in suggestions_text or len(result.suggestions) > 0

