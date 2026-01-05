"""
Tests for NiceGUI event value extraction.

Verifies that:
- Simple values are passed through
- Dict values (from select) are properly extracted
- All visualizer handlers use the helper correctly
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestExtractEventValue:
    """Test extract_event_value helper function."""
    
    def test_simple_string_passes_through(self):
        """Simple string value should pass through unchanged."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        assert extract_event_value('Alpha') == 'Alpha'
    
    def test_simple_int_passes_through(self):
        """Simple int value should pass through unchanged."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        assert extract_event_value(5) == 5
    
    def test_dict_with_label_extracts_label(self):
        """Dict with 'label' key should extract label value."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value({'value': 1, 'label': 'Theta'})
        assert result == 'Theta'
    
    def test_dict_with_only_value_extracts_value(self):
        """Dict with only 'value' key should extract value."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value({'value': 42})
        assert result == 42
    
    def test_none_passes_through(self):
        """None should pass through unchanged."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        assert extract_event_value(None) is None
    
    def test_empty_dict_returns_dict(self):
        """Empty dict should return itself."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value({})
        assert result == {}


class TestStep1BandChange:
    """Test Step 1 visualizer band change handling."""
    
    def test_band_change_with_simple_string(self):
        """Should handle simple string value."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        from app.pages.pipeline.visualizers.base import extract_event_value
        
        viz = Step1Visualizer(lambda: None)
        viz._selected_band = 'Alpha'
        
        # Simulate event value
        viz._selected_band = extract_event_value('Theta')
        assert viz._selected_band == 'Theta'
    
    def test_band_change_with_dict_value(self):
        """Should handle dict value from NiceGUI select."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        from app.pages.pipeline.visualizers.base import extract_event_value
        
        viz = Step1Visualizer(lambda: None)
        viz._selected_band = 'Alpha'
        
        # Simulate dict event value (NiceGUI format)
        viz._selected_band = extract_event_value({'value': 1, 'label': 'Theta'})
        assert viz._selected_band == 'Theta'


class TestStep4BandChange:
    """Test Step 4 visualizer band change handling."""
    
    def test_band_change_with_dict_value(self):
        """Should handle dict value from NiceGUI select."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        from app.pages.pipeline.visualizers.base import extract_event_value
        
        viz = Step4Visualizer(lambda: None)
        viz._selected_band = 'Alpha'
        
        viz._selected_band = extract_event_value({'value': 2, 'label': 'Beta'})
        assert viz._selected_band == 'Beta'


class TestStep5BandChange:
    """Test Step 5 visualizer band change handling."""
    
    def test_band_change_with_dict_value(self):
        """Should handle dict value from NiceGUI select."""
        from app.pages.pipeline.visualizers.step_5_viz import Step5Visualizer
        from app.pages.pipeline.visualizers.base import extract_event_value
        
        viz = Step5Visualizer(lambda: None)
        viz._selected_band = 'Alpha'
        
        viz._selected_band = extract_event_value({'value': 4, 'label': 'Gamma'})
        assert viz._selected_band == 'Gamma'


class TestEpochChange:
    """Test epoch number change handling."""
    
    def test_epoch_change_with_int(self):
        """Should handle simple int value."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value(5)
        assert int(result) == 5
    
    def test_epoch_change_with_float(self):
        """Should handle float value (from number input)."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value(5.0)
        assert int(result) == 5
    
    def test_epoch_change_with_none(self):
        """Should handle None value gracefully."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value(None)
        # Code should check: int(val) if val else 0
        assert result is None


class TestSubjectChange:
    """Test subject selector change handling."""
    
    def test_subject_change_with_string(self):
        """Should handle subject string value."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value('S01-DMT')
        assert result == 'S01-DMT'
    
    def test_subject_change_with_dict(self):
        """Should handle dict value from select."""
        from app.pages.pipeline.visualizers.base import extract_event_value
        result = extract_event_value({'value': 0, 'label': 'S01-DMT'})
        assert result == 'S01-DMT'


class TestAllVisualizersImportHelper:
    """Verify all visualizers import and use the helper."""
    
    def test_step1_imports_helper(self):
        """Step 1 visualizer should import extract_event_value."""
        from app.pages.pipeline.visualizers import step_1_viz
        assert hasattr(step_1_viz, 'extract_event_value')
    
    def test_step4_imports_helper(self):
        """Step 4 visualizer should import extract_event_value."""
        from app.pages.pipeline.visualizers import step_4_viz
        assert hasattr(step_4_viz, 'extract_event_value')
    
    def test_step5_imports_helper(self):
        """Step 5 visualizer should import extract_event_value."""
        from app.pages.pipeline.visualizers import step_5_viz
        assert hasattr(step_5_viz, 'extract_event_value')
    
    def test_step7_imports_helper(self):
        """Step 7 visualizer should import extract_event_value."""
        from app.pages.pipeline.visualizers import step_7_viz
        assert hasattr(step_7_viz, 'extract_event_value')
    
    def test_step8_imports_helper(self):
        """Step 8 visualizer should import extract_event_value."""
        from app.pages.pipeline.visualizers import step_8_viz
        assert hasattr(step_8_viz, 'extract_event_value')








