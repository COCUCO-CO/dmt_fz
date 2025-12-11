"""
Tests for visualization panel state persistence.
"""
import pytest
import inspect


class TestVisualizationPanelPersistence:
    """Test that visualization panel state persists."""
    
    def test_panel_restores_active_step_from_ps(self):
        """Panel should restore active step from PS on init."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        from app.state import PS
        
        # Set a specific step in PS
        PS.viz_active_step = 5
        
        # Create panel - should restore from PS
        panel = VisualizationPanel(lambda: None)
        assert panel._active_step == 5
        
        # Reset PS
        PS.viz_active_step = 1
    
    def test_set_active_step_persists_to_ps(self):
        """set_active_step should persist to PS."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        
        source = inspect.getsource(VisualizationPanel.set_active_step)
        
        # Should persist to PS
        assert 'PS.viz_active_step' in source
    
    def test_ps_has_viz_panel_state(self):
        """PS should have visualization panel state fields."""
        from app.state import PS
        
        assert hasattr(PS, 'viz_active_step')
        assert hasattr(PS, 'viz_panel_height')


class TestStep4MatrixSize:
    """Test Step 4 PLV matrix has appropriate size."""
    
    def test_step_4_heatmap_height_reduced(self):
        """Step 4 heatmap should have reduced height."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        source = inspect.getsource(Step4Visualizer._render_plotly_heatmap)
        
        # Height should be 220 (reduced from 280)
        assert 'height=220' in source
    
    def test_step_4_colorbar_has_light_text(self):
        """Step 4 colorbar should have light colored text."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        source = inspect.getsource(Step4Visualizer._render_plotly_heatmap)
        
        # Should have light color for visibility against dark bg
        assert '#cccccc' in source or '#aaaaaa' in source


class TestAnimationGeneratorSubfolders:
    """Test animation generator handles subfolders."""
    
    def test_animation_generator_has_subfolder_select(self):
        """Animation generator should have subfolder selector."""
        from app.pages.pipeline.components.animation_generator import AnimationGenerator
        
        gen = AnimationGenerator(lambda: None)
        assert hasattr(gen, '_subfolder_select')
        assert hasattr(gen, '_subfolder')
    
    def test_load_subjects_checks_subfolders(self):
        """_load_subjects should check for subfolders."""
        from app.pages.pipeline.components.animation_generator import AnimationGenerator
        
        source = inspect.getsource(AnimationGenerator._load_subjects)
        
        # Should iterate over subdirectories
        assert 'iterdir()' in source or 'subfolders' in source.lower()
    
    def test_has_subfolder_change_handler(self):
        """Should have handler for subfolder change."""
        from app.pages.pipeline.components.animation_generator import AnimationGenerator
        
        assert hasattr(AnimationGenerator, '_on_subfolder_change')
    
    def test_has_load_subjects_from_folder(self):
        """Should have method to load subjects from specific folder."""
        from app.pages.pipeline.components.animation_generator import AnimationGenerator
        
        assert hasattr(AnimationGenerator, '_load_subjects_from_folder')


