"""
Tests for Kuramoto proxy calculation module.
"""

import pytest
import numpy as np
import torch

from app.core.kuramoto_proxy import (
    compute_kuramoto_proxy_from_edges,
    build_sync_matrix,
    compute_kuramoto_comparison,
    batch_kuramoto_comparison
)


class TestBuildSyncMatrix:
    """Tests for build_sync_matrix function."""
    
    def test_basic_matrix_construction(self):
        """Test basic sync matrix construction from edges."""
        # 3 nodes, 3 edges (triangle)
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        edge_weights = np.array([0.5, 0.8, 0.3])
        num_nodes = 3
        
        matrix = build_sync_matrix(edge_index, edge_weights, num_nodes)
        
        assert matrix.shape == (3, 3)
        # Check symmetry
        assert matrix[0, 1] == matrix[1, 0]
        assert matrix[1, 2] == matrix[2, 1]
        assert matrix[0, 2] == matrix[2, 0]
        # Check values
        assert matrix[0, 1] == 0.5
        assert matrix[1, 2] == 0.8
        assert matrix[0, 2] == 0.3
    
    def test_empty_edges(self):
        """Test with no edges."""
        edge_index = np.array([[], []]).astype(int)
        edge_weights = np.array([])
        num_nodes = 5
        
        matrix = build_sync_matrix(edge_index, edge_weights, num_nodes)
        
        assert matrix.shape == (5, 5)
        assert np.all(matrix == 0)
    
    def test_fully_connected(self):
        """Test fully connected graph."""
        num_nodes = 4
        # All pairs
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        edge_weights = np.ones(len(src)) * 0.7
        
        matrix = build_sync_matrix(edge_index, edge_weights, num_nodes)
        
        assert matrix.shape == (num_nodes, num_nodes)
        # Diagonal should be 0
        assert np.all(np.diag(matrix) == 0)
        # Off-diagonal should be 0.7
        off_diag = matrix[~np.eye(num_nodes, dtype=bool)]
        assert np.allclose(off_diag, 0.7)


class TestComputeKuramotoProxy:
    """Tests for compute_kuramoto_proxy_from_edges function."""
    
    def test_mean_plv_method(self):
        """Test mean_plv method."""
        # Simple case: all PLV = 0.6
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        edge_attr = np.array([0.6, 0.6, 0.6])
        
        proxy = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes=3, method='mean_plv'
        )
        
        assert proxy == pytest.approx(0.6)
    
    def test_mean_plv_varied_values(self):
        """Test mean_plv with varied edge weights."""
        edge_index = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
        edge_attr = np.array([0.2, 0.4, 0.6, 0.8])
        
        proxy = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes=4, method='mean_plv'
        )
        
        assert proxy == pytest.approx(0.5)  # Mean of [0.2, 0.4, 0.6, 0.8]
    
    def test_spectral_method_high_sync(self):
        """Test spectral method with high synchronization."""
        num_nodes = 4
        # Fully connected with high PLV
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        edge_attr = np.ones(len(src)) * 0.9  # High sync
        
        proxy = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes, method='spectral'
        )
        
        # Spectral method returns max_eigenvalue/N, for fully connected with weight w:
        # max eigenvalue ≈ w*(N-1), so result ≈ w*(N-1)/N
        # For N=4, w=0.9: expect ≈ 0.9*3/4 = 0.675
        assert 0.6 < proxy < 0.8
    
    def test_spectral_method_low_sync(self):
        """Test spectral method with low synchronization."""
        num_nodes = 4
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        edge_attr = np.ones(len(src)) * 0.1  # Low sync
        
        proxy = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes, method='spectral'
        )
        
        # Should be low
        assert proxy < 0.3
    
    def test_tensor_input(self):
        """Test with PyTorch tensor input."""
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        edge_attr = torch.tensor([0.5, 0.5, 0.5])
        
        proxy = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes=3, method='mean_plv'
        )
        
        assert proxy == pytest.approx(0.5)
    
    def test_2d_edge_attr(self):
        """Test with 2D edge attributes (first column is PLV)."""
        edge_index = np.array([[0, 1], [1, 0]])
        # Edge attr with PLV in first column, other features in rest
        edge_attr = np.array([[0.7, 0.1], [0.7, 0.2]])
        
        proxy = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes=2, method='mean_plv'
        )
        
        assert proxy == pytest.approx(0.7)
    
    def test_clipping_out_of_range(self):
        """Test that values outside [0, 1] are clipped."""
        edge_index = np.array([[0, 1], [1, 0]])
        edge_attr = np.array([1.5, -0.2])  # Out of range
        
        proxy = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes=2, method='mean_plv'
        )
        
        # Should clip to [0, 1] -> mean of [1.0, 0.0] = 0.5
        assert proxy == pytest.approx(0.5)
    
    def test_invalid_method_raises(self):
        """Test that invalid method raises ValueError."""
        edge_index = np.array([[0], [1]])
        edge_attr = np.array([0.5])
        
        with pytest.raises(ValueError, match="Unknown method"):
            compute_kuramoto_proxy_from_edges(
                edge_attr, edge_index, num_nodes=2, method='invalid'
            )


class TestComputeKuramotoComparison:
    """Tests for compute_kuramoto_comparison function."""
    
    def test_comparison_output_structure(self):
        """Test that comparison returns all expected keys."""
        graph_attr = np.array([0.6, 0.55, 2.0, 0.01])  # kuramoto, global_sync, degree, var
        edge_index = np.array([[0, 1], [1, 0]])
        edge_attr = np.array([0.55, 0.55])
        
        result = compute_kuramoto_comparison(
            graph_attr, edge_attr, edge_index, num_nodes=2
        )
        
        expected_keys = [
            'kuramoto_original', 'kuramoto_proxy',
            'global_sync_original', 'global_sync_proxy',
            'absolute_error', 'relative_error'
        ]
        for key in expected_keys:
            assert key in result
    
    def test_perfect_reconstruction(self):
        """Test with perfect edge reconstruction."""
        # Original: kuramoto=0.7, global_sync=0.7
        graph_attr = np.array([0.7, 0.7, 2.0, 0.0])
        edge_index = np.array([[0, 1], [1, 0]])
        # Perfect reconstruction: PLV = 0.7
        edge_attr = np.array([0.7, 0.7])
        
        result = compute_kuramoto_comparison(
            graph_attr, edge_attr, edge_index, num_nodes=2
        )
        
        assert result['kuramoto_original'] == pytest.approx(0.7)
        assert result['global_sync_proxy'] == pytest.approx(0.7)
        assert result['absolute_error'] == pytest.approx(0.0)
    
    def test_imperfect_reconstruction(self):
        """Test with imperfect reconstruction."""
        graph_attr = np.array([0.8, 0.75, 2.0, 0.01])
        edge_index = np.array([[0, 1, 2], [1, 2, 0]])
        # Reconstruction has some error
        edge_attr = np.array([0.6, 0.65, 0.55])  # Mean = 0.6
        
        result = compute_kuramoto_comparison(
            graph_attr, edge_attr, edge_index, num_nodes=3
        )
        
        assert result['kuramoto_original'] == pytest.approx(0.8)
        assert result['kuramoto_proxy'] == pytest.approx(0.6)
        assert result['absolute_error'] == pytest.approx(0.2)
        assert result['relative_error'] == pytest.approx(0.25)  # 0.2 / 0.8
    
    def test_tensor_inputs(self):
        """Test with tensor inputs."""
        graph_attr = torch.tensor([0.5, 0.5, 1.0, 0.0])
        edge_index = torch.tensor([[0, 1], [1, 0]])
        edge_attr = torch.tensor([0.5, 0.5])
        
        result = compute_kuramoto_comparison(
            graph_attr, edge_attr, edge_index, num_nodes=2
        )
        
        assert result['kuramoto_original'] == pytest.approx(0.5)
        assert result['kuramoto_proxy'] == pytest.approx(0.5)


class TestBatchKuramotoComparison:
    """Tests for batch_kuramoto_comparison function."""
    
    def test_missing_edge_attr_recon(self):
        """Test handling of missing edge_attr_recon."""
        model_output = {'x_recon': torch.randn(10, 5)}  # No edge_attr_recon
        
        class MockBatch:
            pass
        
        result = batch_kuramoto_comparison(model_output, MockBatch())
        
        assert len(result['kuramoto_original']) == 0
        assert len(result['kuramoto_proxy']) == 0
    
    def test_single_graph(self):
        """Test with single graph (no batch)."""
        # Create mock single graph - batch=None means single graph mode
        class MockGraph:
            def __init__(self):
                self.edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
                self.edge_attr = torch.tensor([[0.6], [0.6], [0.6]])
                self.graph_attr = torch.tensor([0.6, 0.6, 2.0, 0.0])
                self.num_nodes = 3
                self.batch = None  # No batch = single graph
        
        graph = MockGraph()
        model_output = {
            'edge_attr_recon': torch.tensor([[0.55], [0.58], [0.52]])
        }
        
        result = batch_kuramoto_comparison(model_output, graph)
        
        assert len(result['kuramoto_original']) == 1
        assert result['kuramoto_original'][0] == pytest.approx(0.6)


class TestKuramotoProxyAccuracy:
    """Integration tests comparing proxy to known Kuramoto values."""
    
    def test_high_sync_approximation(self):
        """
        When all PLV values are high (close to 1), 
        the proxy should be close to what we'd expect from Kuramoto.
        """
        num_nodes = 10
        # Fully connected graph with high sync
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        
        # High synchronization: PLV ≈ 0.9
        edge_attr = np.random.normal(0.9, 0.02, len(src))
        edge_attr = np.clip(edge_attr, 0, 1)
        
        proxy_mean = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes, 'mean_plv'
        )
        proxy_spectral = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes, 'spectral'
        )
        
        # Mean should be close to 0.9
        assert proxy_mean > 0.85
        # Spectral: max_eigenvalue/N ≈ w*(N-1)/N for fully connected
        # For N=10, w=0.9: expect ≈ 0.9*9/10 = 0.81
        assert proxy_spectral > 0.75
    
    def test_low_sync_approximation(self):
        """
        When all PLV values are low (close to 0),
        the proxy should indicate low synchronization.
        """
        num_nodes = 10
        src, dst = [], []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    src.append(i)
                    dst.append(j)
        edge_index = np.array([src, dst])
        
        # Low synchronization: PLV ≈ 0.1
        edge_attr = np.random.normal(0.1, 0.02, len(src))
        edge_attr = np.clip(edge_attr, 0, 1)
        
        proxy_mean = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes, 'mean_plv'
        )
        proxy_spectral = compute_kuramoto_proxy_from_edges(
            edge_attr, edge_index, num_nodes, 'spectral'
        )
        
        # Both should indicate low sync
        assert proxy_mean < 0.15
        assert proxy_spectral < 0.20


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

