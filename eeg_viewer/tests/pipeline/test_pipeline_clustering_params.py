"""
Tests for clustering step parameter persistence.

These tests verify that:
- Clustering parameters (min_k, max_k, min_comps, max_comps, bands) persist in PS
- Parameters are restored when returning to pipeline page
- Changes to parameters update PS state
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state import PS
from app.state.global_state import PipelineState


class TestClusteringParamsInPS:
    """Verify PS has all clustering parameter attributes."""
    
    def test_ps_has_min_k(self):
        """PS should have min_k attribute."""
        assert hasattr(PS, 'min_k')
    
    def test_ps_has_max_k(self):
        """PS should have max_k attribute."""
        assert hasattr(PS, 'max_k')
    
    def test_ps_has_min_comps(self):
        """PS should have min_comps attribute."""
        assert hasattr(PS, 'min_comps')
    
    def test_ps_has_max_comps(self):
        """PS should have max_comps attribute."""
        assert hasattr(PS, 'max_comps')
    
    def test_ps_has_bands(self):
        """PS should have bands attribute."""
        assert hasattr(PS, 'bands')
    
    def test_ps_has_clustering_search_mode(self):
        """PS should have clustering_search_mode attribute."""
        assert hasattr(PS, 'clustering_search_mode')


class TestClusteringParamDefaults:
    """Verify default values for clustering params."""
    
    def test_min_k_default(self):
        """min_k should have sensible default."""
        ps = PipelineState()
        assert ps.min_k >= 2
    
    def test_max_k_default(self):
        """max_k should have sensible default."""
        ps = PipelineState()
        assert ps.max_k > ps.min_k
    
    def test_min_comps_default(self):
        """min_comps should have sensible default."""
        ps = PipelineState()
        assert ps.min_comps >= 2
    
    def test_max_comps_default(self):
        """max_comps should have sensible default."""
        ps = PipelineState()
        assert ps.max_comps > ps.min_comps
    
    def test_bands_default_all(self):
        """bands should default to all frequency bands."""
        ps = PipelineState()
        assert 'Delta' in ps.bands
        assert 'Theta' in ps.bands
        assert 'Alpha' in ps.bands
        assert 'Beta' in ps.bands
        assert 'Gamma' in ps.bands


class TestClusteringParamPersistence:
    """Test clustering parameter persistence."""
    
    def test_min_k_persists(self):
        """min_k should persist after setting."""
        original = PS.min_k
        try:
            PS.min_k = 5
            assert PS.min_k == 5
            
            # Simulate re-import (page navigation)
            from app.state import PS as PS2
            assert PS2.min_k == 5
        finally:
            PS.min_k = original
    
    def test_max_k_persists(self):
        """max_k should persist after setting."""
        original = PS.max_k
        try:
            PS.max_k = 20
            assert PS.max_k == 20
            
            from app.state import PS as PS2
            assert PS2.max_k == 20
        finally:
            PS.max_k = original
    
    def test_min_comps_persists(self):
        """min_comps should persist after setting."""
        original = PS.min_comps
        try:
            PS.min_comps = 4
            assert PS.min_comps == 4
            
            from app.state import PS as PS2
            assert PS2.min_comps == 4
        finally:
            PS.min_comps = original
    
    def test_max_comps_persists(self):
        """max_comps should persist after setting."""
        original = PS.max_comps
        try:
            PS.max_comps = 12
            assert PS.max_comps == 12
            
            from app.state import PS as PS2
            assert PS2.max_comps == 12
        finally:
            PS.max_comps = original
    
    def test_bands_persists(self):
        """bands selection should persist."""
        original = list(PS.bands)
        try:
            PS.bands = ['Alpha', 'Beta']
            assert PS.bands == ['Alpha', 'Beta']
            
            from app.state import PS as PS2
            assert PS2.bands == ['Alpha', 'Beta']
        finally:
            PS.bands = original
    
    def test_clustering_search_mode_persists(self):
        """clustering_search_mode should persist."""
        original = getattr(PS, 'clustering_search_mode', 'Quick')
        try:
            PS.clustering_search_mode = 'Full'
            assert PS.clustering_search_mode == 'Full'
            
            from app.state import PS as PS2
            assert PS2.clustering_search_mode == 'Full'
        finally:
            PS.clustering_search_mode = original


class TestClusteringNavigationWorkflow:
    """Test clustering params through navigation workflow."""
    
    def test_set_all_params_navigate_return(self):
        """All clustering params should persist through navigation."""
        # Save originals
        orig_min_k = PS.min_k
        orig_max_k = PS.max_k
        orig_min_comps = PS.min_comps
        orig_max_comps = PS.max_comps
        orig_bands = list(PS.bands)
        orig_mode = getattr(PS, 'clustering_search_mode', 'Quick')
        
        try:
            # Step 1: User sets clustering params
            PS.min_k = 3
            PS.max_k = 12
            PS.min_comps = 3
            PS.max_comps = 8
            PS.bands = ['Delta', 'Alpha', 'Gamma']
            PS.clustering_search_mode = 'Full'
            
            # Step 2: User navigates to viewer (state persists)
            # ...
            
            # Step 3: User returns to pipeline
            # All params should still be set
            assert PS.min_k == 3
            assert PS.max_k == 12
            assert PS.min_comps == 3
            assert PS.max_comps == 8
            assert PS.bands == ['Delta', 'Alpha', 'Gamma']
            assert PS.clustering_search_mode == 'Full'
            
        finally:
            # Restore
            PS.min_k = orig_min_k
            PS.max_k = orig_max_k
            PS.min_comps = orig_min_comps
            PS.max_comps = orig_max_comps
            PS.bands = orig_bands
            PS.clustering_search_mode = orig_mode
    
    def test_partial_band_selection_persists(self):
        """Selecting only some bands should persist."""
        original = list(PS.bands)
        try:
            # User unchecks some bands
            PS.bands = ['Theta', 'Beta']
            
            # Navigate and return
            from app.state import PS as PS2
            assert 'Theta' in PS2.bands
            assert 'Beta' in PS2.bands
            assert 'Delta' not in PS2.bands
            assert 'Alpha' not in PS2.bands
            assert 'Gamma' not in PS2.bands
        finally:
            PS.bands = original








