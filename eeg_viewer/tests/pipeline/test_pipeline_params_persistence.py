"""
Tests for pipeline parameters persistence across page navigation.

These tests verify that:
- Pipeline parameters (max_subjects, conditions, workers, etc.) persist in PS
- Parameters are restored when returning to pipeline page
- Changes to parameters update PS state
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.state import PS
from app.state.global_state import PipelineState


class TestPipelineStateHasAllParams:
    """Verify PipelineState has all required parameter attributes."""
    
    def test_ps_has_max_subjects(self):
        """PS should have max_subjects attribute."""
        ps = PipelineState()
        assert hasattr(ps, 'max_subjects')
    
    def test_ps_has_conditions(self):
        """PS should have conditions attribute."""
        ps = PipelineState()
        assert hasattr(ps, 'conditions')
    
    def test_ps_has_jobs(self):
        """PS should have jobs attribute."""
        ps = PipelineState()
        assert hasattr(ps, 'jobs')
    
    def test_ps_has_workers(self):
        """PS should have workers attribute."""
        ps = PipelineState()
        assert hasattr(ps, 'workers')
    
    def test_ps_has_max_epochs(self):
        """PS should have max_epochs attribute."""
        ps = PipelineState()
        assert hasattr(ps, 'max_epochs')
    
    def test_ps_has_bands(self):
        """PS should have bands attribute for filtering steps."""
        ps = PipelineState()
        assert hasattr(ps, 'bands')
    
    def test_ps_has_clustering_params(self):
        """PS should have clustering params (min_k, max_k, min_comps, max_comps)."""
        ps = PipelineState()
        assert hasattr(ps, 'min_k')
        assert hasattr(ps, 'max_k')
        assert hasattr(ps, 'min_comps')
        assert hasattr(ps, 'max_comps')
    
    def test_ps_has_input_dir(self):
        """PS should have input_dir for input path persistence."""
        ps = PipelineState()
        assert hasattr(ps, 'input_dir')


class TestParamDefaultValues:
    """Verify default values are sensible."""
    
    def test_max_subjects_default_zero(self):
        """max_subjects should default to 0 (all)."""
        ps = PipelineState()
        assert ps.max_subjects == 0
    
    def test_conditions_default_all(self):
        """conditions should default to all three."""
        ps = PipelineState()
        assert 'DMT' in ps.conditions
        assert 'EC' in ps.conditions
        assert 'EO' in ps.conditions
    
    def test_workers_default_positive(self):
        """workers should default to a positive number."""
        ps = PipelineState()
        assert ps.workers > 0
    
    def test_jobs_default_zero(self):
        """jobs should default to 0 (auto)."""
        ps = PipelineState()
        assert ps.jobs == 0
    
    def test_max_epochs_default_zero(self):
        """max_epochs should default to 0 (all)."""
        ps = PipelineState()
        assert ps.max_epochs == 0


class TestParamPersistence:
    """Test parameter persistence across simulated page navigations."""
    
    def test_max_subjects_persists(self):
        """max_subjects should persist after setting."""
        ps = PipelineState()
        ps.max_subjects = 5
        assert ps.max_subjects == 5
        
        # Simulate accessing again
        assert ps.max_subjects == 5
    
    def test_conditions_persists(self):
        """conditions should persist after modification."""
        ps = PipelineState()
        ps.conditions = ['DMT']
        assert ps.conditions == ['DMT']
        
        # Add condition
        ps.conditions = ['DMT', 'EC']
        assert ps.conditions == ['DMT', 'EC']
    
    def test_workers_persists(self):
        """workers should persist after setting."""
        ps = PipelineState()
        ps.workers = 12
        assert ps.workers == 12
    
    def test_jobs_persists(self):
        """jobs should persist after setting."""
        ps = PipelineState()
        ps.jobs = 8
        assert ps.jobs == 8
    
    def test_max_epochs_persists(self):
        """max_epochs should persist after setting."""
        ps = PipelineState()
        ps.max_epochs = 50
        assert ps.max_epochs == 50


class TestGlobalPSPersistence:
    """Test persistence using the global PS instance."""
    
    def test_global_ps_max_subjects_persists(self):
        """Global PS max_subjects should persist."""
        original = PS.max_subjects
        try:
            PS.max_subjects = 10
            assert PS.max_subjects == 10
            
            # Re-import (simulates page navigation)
            from app.state import PS as PS2
            assert PS2.max_subjects == 10
        finally:
            PS.max_subjects = original
    
    def test_global_ps_conditions_persists(self):
        """Global PS conditions should persist."""
        original = list(PS.conditions)
        try:
            PS.conditions = ['DMT', 'EO']
            assert PS.conditions == ['DMT', 'EO']
            
            from app.state import PS as PS2
            assert PS2.conditions == ['DMT', 'EO']
        finally:
            PS.conditions = original
    
    def test_global_ps_workers_persists(self):
        """Global PS workers should persist."""
        original = PS.workers
        try:
            PS.workers = 16
            assert PS.workers == 16
            
            from app.state import PS as PS2
            assert PS2.workers == 16
        finally:
            PS.workers = original
    
    def test_global_ps_jobs_persists(self):
        """Global PS jobs should persist."""
        original = PS.jobs
        try:
            PS.jobs = 4
            assert PS.jobs == 4
            
            from app.state import PS as PS2
            assert PS2.jobs == 4
        finally:
            PS.jobs = original
    
    def test_global_ps_max_epochs_persists(self):
        """Global PS max_epochs should persist."""
        original = PS.max_epochs
        try:
            PS.max_epochs = 100
            assert PS.max_epochs == 100
            
            from app.state import PS as PS2
            assert PS2.max_epochs == 100
        finally:
            PS.max_epochs = original
    
    def test_global_ps_input_dir_persists(self):
        """Global PS input_dir should persist."""
        original = getattr(PS, 'input_dir', None)
        try:
            PS.input_dir = '/test/path/to/data'
            assert PS.input_dir == '/test/path/to/data'
            
            from app.state import PS as PS2
            assert PS2.input_dir == '/test/path/to/data'
        finally:
            PS.input_dir = original


class TestNavigationWorkflowWithParams:
    """Test complete navigation workflow with parameter changes."""
    
    def test_set_params_navigate_away_return(self):
        """Parameters should persist through navigation workflow."""
        original_max_subj = PS.max_subjects
        original_conditions = list(PS.conditions)
        original_workers = PS.workers
        original_max_epochs = PS.max_epochs
        
        try:
            # Step 1: User is on pipeline page, sets parameters
            PS.max_subjects = 3
            PS.conditions = ['DMT']
            PS.workers = 8
            PS.max_epochs = 25
            
            # Step 2: User navigates to viewer page
            # ... (state persists in PS)
            
            # Step 3: User returns to pipeline page
            # Parameters should still be there
            assert PS.max_subjects == 3
            assert PS.conditions == ['DMT']
            assert PS.workers == 8
            assert PS.max_epochs == 25
            
        finally:
            PS.max_subjects = original_max_subj
            PS.conditions = original_conditions
            PS.workers = original_workers
            PS.max_epochs = original_max_epochs
    
    def test_params_survive_multiple_navigations(self):
        """Parameters should survive multiple page navigations."""
        original_max_subj = PS.max_subjects
        
        try:
            PS.max_subjects = 7
            
            # Simulate multiple navigations
            for _ in range(5):
                # Navigate away
                pass
                # Navigate back
                assert PS.max_subjects == 7
                
        finally:
            PS.max_subjects = original_max_subj


class TestClusteringParamsPersistence:
    """Test clustering-specific parameters persistence."""
    
    def test_min_k_persists(self):
        """min_k should persist."""
        original = PS.min_k
        try:
            PS.min_k = 3
            assert PS.min_k == 3
        finally:
            PS.min_k = original
    
    def test_max_k_persists(self):
        """max_k should persist."""
        original = PS.max_k
        try:
            PS.max_k = 20
            assert PS.max_k == 20
        finally:
            PS.max_k = original
    
    def test_min_comps_persists(self):
        """min_comps should persist."""
        original = PS.min_comps
        try:
            PS.min_comps = 5
            assert PS.min_comps == 5
        finally:
            PS.min_comps = original
    
    def test_max_comps_persists(self):
        """max_comps should persist."""
        original = PS.max_comps
        try:
            PS.max_comps = 15
            assert PS.max_comps == 15
        finally:
            PS.max_comps = original


class TestBandsPersistence:
    """Test bands selection persistence."""
    
    def test_bands_default_all(self):
        """bands should default to all frequency bands."""
        ps = PipelineState()
        assert 'Delta' in ps.bands
        assert 'Theta' in ps.bands
        assert 'Alpha' in ps.bands
        assert 'Beta' in ps.bands
        assert 'Gamma' in ps.bands
    
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


class TestInputDirPersistence:
    """Test input directory persistence."""
    
    def test_input_dir_persists_string(self):
        """input_dir should persist as string path."""
        original = getattr(PS, 'input_dir', None)
        try:
            PS.input_dir = '/media/storage/test/EEG_CLEAN'
            assert PS.input_dir == '/media/storage/test/EEG_CLEAN'
        finally:
            PS.input_dir = original
    
    def test_input_dir_persists_path(self):
        """input_dir should persist if set as Path."""
        original = getattr(PS, 'input_dir', None)
        try:
            from pathlib import Path
            PS.input_dir = Path('/media/storage/test/EEG_CLEAN')
            # Should be convertible back to string
            assert str(PS.input_dir) == '/media/storage/test/EEG_CLEAN'
        finally:
            PS.input_dir = original

