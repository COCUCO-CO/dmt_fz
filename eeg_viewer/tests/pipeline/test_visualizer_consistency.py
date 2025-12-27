"""
Tests for visualizer consistency: size, quality, and loading behavior.
"""
import inspect


class TestVisualizerRegistry:
    """Test that visualizer registry is correct."""
    
    def test_step_2_not_in_registry(self):
        """Step 2 should not be in registry (data processing only)."""
        from app.pages.pipeline.visualizers import VISUALIZER_REGISTRY
        
        assert 2 not in VISUALIZER_REGISTRY
    
    def test_step_6_not_in_registry(self):
        """Step 6 should not be in registry (data processing only)."""
        from app.pages.pipeline.visualizers import VISUALIZER_REGISTRY
        
        assert 6 not in VISUALIZER_REGISTRY
    
    def test_scientific_steps_in_registry(self):
        """Scientific visualization steps should be in registry."""
        from app.pages.pipeline.visualizers import VISUALIZER_REGISTRY
        
        # Steps with scientific visualization
        assert 1 in VISUALIZER_REGISTRY  # Source localization
        assert 3 in VISUALIZER_REGISTRY  # Network filter
        assert 4 in VISUALIZER_REGISTRY  # Synchronization
        assert 5 in VISUALIZER_REGISTRY  # Kuramoto
        assert 7 in VISUALIZER_REGISTRY  # Pearson
        assert 8 in VISUALIZER_REGISTRY  # Clustering


class TestVisualizerChartSizes:
    """Test that visualizers use consistent chart sizes."""
    
    def test_step_3_uses_plotly(self):
        """Step 3 should use Plotly for better quality."""
        from app.pages.pipeline.visualizers.step_3_viz import Step3Visualizer
        
        source = inspect.getsource(Step3Visualizer)
        assert 'plotly' in source.lower() or 'go.Figure' in source or 'go.Pie' in source
    
    def test_step_3_has_fixed_height(self):
        """Step 3 chart should have fixed height."""
        from app.pages.pipeline.visualizers.step_3_viz import Step3Visualizer
        
        source = inspect.getsource(Step3Visualizer)
        assert 'height=' in source or 'height:' in source


class TestStep4MultiCondition:
    """Test Step 4 multi-condition support."""
    
    def test_step_4_has_condition_checkboxes(self):
        """Step 4 should have condition checkboxes."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        source = inspect.getsource(Step4Visualizer)
        
        assert 'DMT' in source
        assert 'EC' in source
        assert 'EO' in source
        assert 'checkbox' in source.lower() or '_toggle_condition' in source
    
    def test_step_4_has_selected_conditions(self):
        """Step 4 should track selected conditions."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        viz = Step4Visualizer(lambda: None)
        assert hasattr(viz, '_selected_conditions')
        assert isinstance(viz._selected_conditions, list)
    
    def test_step_4_has_compact_heatmap(self):
        """Step 4 heatmaps should be compact."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        source = inspect.getsource(Step4Visualizer)
        
        # Should have compact height (220 is current value)
        assert 'height=220' in source


class TestLoadingSpinner:
    """Test that visualization panel has loading state."""
    
    def test_visualization_panel_has_spinner(self):
        """Panel should have a loading spinner mechanism."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        
        source = inspect.getsource(VisualizationPanel)
        
        has_loading = (
            'spinner' in source.lower() or
            'loading' in source.lower() or
            'cargando' in source.lower()
        )
        
        assert has_loading, "VisualizationPanel should have loading indicator"


class TestChartQuality:
    """Test chart quality parameters."""
    
    def test_step_3_uses_dark_theme(self):
        """Step 3 charts should use dark theme."""
        from app.pages.pipeline.visualizers.step_3_viz import Step3Visualizer
        
        source = inspect.getsource(Step3Visualizer)
        assert 'plotly_dark' in source or 'template' in source
    
    def test_step_4_uses_dark_theme(self):
        """Step 4 charts should use dark theme."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        
        source = inspect.getsource(Step4Visualizer)
        assert 'plotly_dark' in source or 'template' in source


class TestStep5KuramotoVisualization:
    """Test Step 5 Kuramoto visualization."""
    
    def test_step_5_shows_debug_on_no_data(self):
        """Step 5 should show debug info when no data found."""
        from app.pages.pipeline.visualizers.step_5_viz import Step5Visualizer
        
        source = inspect.getsource(Step5Visualizer)
        
        # Should have debug info method
        assert '_render_no_data_with_debug' in source
    
    def test_step_5_has_kuramoto_display(self):
        """Step 5 should display Kuramoto R(t) timeline."""
        from app.pages.pipeline.visualizers.step_5_viz import Step5Visualizer
        
        source = inspect.getsource(Step5Visualizer)
        
        # Should have R(t) timeline
        assert 'R(t)' in source
        assert '_render_r_timeline' in source
    
    def test_step_5_has_oscillator_plot(self):
        """Step 5 should display Kuramoto oscillators."""
        from app.pages.pipeline.visualizers.step_5_viz import Step5Visualizer
        
        source = inspect.getsource(Step5Visualizer)
        
        # Should have oscillator visualization
        assert '_render_oscillator_plot' in source
        assert 'Scatterpolar' in source
    
    def test_step_5_has_network_comparison(self):
        """Step 5 should compare R across brain networks."""
        from app.pages.pipeline.visualizers import step_5_viz
        
        source = inspect.getsource(step_5_viz)
        
        # Should have network comparison
        assert '_render_network_comparison' in source
        assert 'FPN' in source  # Frontoparietal Network
        assert 'DMN' in source  # Default Mode Network
    
    def test_step_5_has_band_comparison(self):
        """Step 5 should compare R across frequency bands."""
        from app.pages.pipeline.visualizers.step_5_viz import Step5Visualizer
        
        source = inspect.getsource(Step5Visualizer)
        
        # Should have band comparison
        assert '_render_band_comparison' in source
        assert 'Delta' in source
        assert 'Alpha' in source


class TestVisualizationPanelStepSelector:
    """Test that visualization panel maps display numbers to actual steps."""
    
    def test_step_selector_maps_display_to_actual(self):
        """Step selector should map consecutive display numbers to actual steps."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        
        source = inspect.getsource(VisualizationPanel._render_header)
        
        # Should have mapping from display (1-6) to actual steps (skipping 2 and 6)
        assert 'DISPLAY_TO_STEP' in source or 'data processing only' in source

