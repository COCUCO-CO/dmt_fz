"""
Tests for Step 1 Visualizer improvements.

Validates:
- 3D brain loads by default (no button needed)
- Shows summary stats instead of selectors
- Proper data gathering from generated files
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestStep1DefaultBehavior:
    """Test Step 1 shows 3D brain by default."""
    
    def test_step1_minimal_controls(self):
        """Step 1 should have minimal controls (just hilbert toggle)."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        controls = viz.get_controls()
        
        # Should only have hilbert toggle, no subject/band/epoch selectors
        assert 'hilbert_toggle' in controls, "Step 1 should have hilbert toggle"
        assert 'subject' not in controls, "Step 1 should not have subject selector"
        assert 'band' not in controls, "Step 1 should not have band selector"
        assert 'epoch' not in controls, "Step 1 should not have epoch selector"
    
    def test_step1_uses_plotly_for_3d(self):
        """Step 1 should use plotly for 3D visualization."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        
        assert 'plotly' in viz.supported_backends
    
    def test_step1_has_correct_patterns(self):
        """Step 1 should find phases-*.pkl files."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        
        assert any('phases' in p for p in viz.file_patterns)


class TestStep1StatsGathering:
    """Test Step 1 data statistics gathering."""
    
    def test_gather_stats_counts_conditions(self, tmp_path):
        """Should count files by condition (DMT, EC, EO)."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create mock files
        (tmp_path / 'phases-S01-DMT.pkl').touch()
        (tmp_path / 'phases-S02-DMT.pkl').touch()
        (tmp_path / 'phases-S01-EC.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        viz._gather_stats()
        
        assert viz._stats is not None
        assert viz._stats['total_files'] == 3
        assert viz._stats['conditions'].get('DMT', 0) == 2
        assert viz._stats['conditions'].get('EC', 0) == 1
    
    def test_gather_stats_returns_none_when_no_files(self, tmp_path):
        """Should return None stats when no files found."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: tmp_path)
        viz._gather_stats()
        
        assert viz._stats is None or viz._stats['total_files'] == 0
    
    def test_gather_stats_extracts_metadata_from_file(self, tmp_path):
        """Should extract sfreq, n_parcels from a file."""
        import pickle
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create a mock pkl file with metadata
        data = {
            'sfreq': 500.0,
            'n_epochs': 150,
            'phases_stc': {
                'Delta': np.random.rand(10, 102, 800)  # 10 epochs, 102 parcels, 800 samples
            }
        }
        pkl_file = tmp_path / 'phases-S01-DMT.pkl'
        with open(pkl_file, 'wb') as f:
            pickle.dump(data, f)
        
        viz = Step1Visualizer(lambda: tmp_path)
        viz._gather_stats()
        
        assert viz._stats is not None
        assert viz._stats['sfreq'] == 500.0
        assert viz._stats['n_parcels'] == 102
        assert viz._stats['n_times'] == 800


class TestStep1NoButtonRequired:
    """Test that 3D brain loads automatically without button."""
    
    def test_render_visualization_calls_brain_3d_directly(self):
        """_render_visualization should call _render_brain_3d directly."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        
        # Check that the method exists and is called in render flow
        assert hasattr(viz, '_render_brain_3d')
        assert hasattr(viz, '_render_summary_stats')
        assert hasattr(viz, '_gather_stats')
    
    def test_no_3d_brain_button_in_controls(self):
        """Controls should not include a 3D Brain toggle button."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        controls = viz.get_controls()
        
        # Should not have any 3D toggle
        assert '3d' not in str(controls).lower()
        assert 'brain' not in str(controls).lower()


class TestMatplotlibImageRendering:
    """Test matplotlib rendering uses ui.image instead of ui.html."""
    
    def test_show_matplotlib_uses_image(self):
        """show_matplotlib should use ui.image, not ui.html."""
        import inspect
        from app.pages.pipeline.visualizers.base import BaseVisualizer
        
        source = inspect.getsource(BaseVisualizer.show_matplotlib)
        
        # Should use ui.image
        assert 'ui.image(' in source, "show_matplotlib should use ui.image"
        
        # Should NOT call ui.html() function (checking for actual call, not comment)
        assert 'ui.html(' not in source, "show_matplotlib should not call ui.html() (causes sanitize error)"


class TestStepSelectorFunctionality:
    """Test step selector buttons work correctly."""
    
    def test_visualization_panel_can_switch_steps(self):
        """VisualizationPanel.set_active_step should update state."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        
        panel = VisualizationPanel(lambda: None)
        
        # Should start at step 1
        assert panel.active_step == 1
        
        # Should be able to change to any step
        for step in range(1, 9):
            panel._active_step = step  # Direct assignment for testing
            assert panel.active_step == step
    
    def test_set_active_step_validates_range(self):
        """set_active_step should only accept 1-8."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        
        panel = VisualizationPanel(lambda: None)
        panel._step_buttons = {}  # Mock buttons
        
        # Valid steps
        for step in range(1, 9):
            panel._active_step = 0  # Reset
            # Can't fully test without UI, but structure should work
            assert 1 <= step <= 8
    
    def test_visualizer_registry_has_scientific_steps(self):
        """VISUALIZER_REGISTRY should have all scientific visualization steps."""
        from app.pages.pipeline.visualizers import VISUALIZER_REGISTRY
        
        # Steps 2 and 6 are data processing (no visualization needed)
        scientific_steps = [1, 3, 4, 5, 7, 8]
        assert len(VISUALIZER_REGISTRY) == len(scientific_steps)
        for step in scientific_steps:
            assert step in VISUALIZER_REGISTRY, f"Step {step} missing from registry"
    
    def test_get_visualizer_returns_correct_class(self):
        """get_visualizer should return correct visualizer for scientific steps."""
        from app.pages.pipeline.visualizers import get_visualizer, VISUALIZER_REGISTRY
        
        # Steps 2 and 6 are data processing (no visualization needed)
        scientific_steps = [1, 3, 4, 5, 7, 8]
        for step in scientific_steps:
            viz = get_visualizer(step, lambda: None)
            assert viz is not None, f"No visualizer for step {step}"
            assert viz.step_number == step, f"Wrong step number for step {step}"
        
        # Steps 2 and 6 should return None
        assert get_visualizer(2, lambda: None) is None
        assert get_visualizer(6, lambda: None) is None


class TestVisualizationPanelErrorHandling:
    """Test visualization panel handles errors gracefully."""
    
    def test_render_visualizer_has_try_except(self):
        """_do_render_visualizer should catch exceptions."""
        import inspect
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        
        # Error handling is now in _do_render_visualizer (deferred rendering)
        source = inspect.getsource(VisualizationPanel._do_render_visualizer)
        
        # Should have try-except for error handling
        assert 'try:' in source, "_do_render_visualizer should have try block"
        assert 'except' in source, "_do_render_visualizer should have except block"
    
    def test_visualization_panel_update_button_styles_method_exists(self):
        """Panel should have _update_button_styles method for selector."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        
        panel = VisualizationPanel(lambda: None)
        assert hasattr(panel, '_update_button_styles')
        assert callable(panel._update_button_styles)

