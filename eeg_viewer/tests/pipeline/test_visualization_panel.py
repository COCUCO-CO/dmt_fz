"""
Tests for contextual visualization panel.

Verifies that:
- Visualization panel detects active step correctly
- Auto-switch works based on file detection
- Each step has appropriate visualizer
- Controls update based on selected step
"""
import pytest
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestVisualizationPanelStructure:
    """Test visualization panel structure and components."""
    
    def test_visualization_panel_class_exists(self):
        """VisualizationPanel class should exist."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        assert VisualizationPanel is not None
    
    def test_visualization_panel_has_step_selector(self):
        """Panel should have step selector method."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        panel = VisualizationPanel(lambda: None)
        assert hasattr(panel, 'set_active_step')
    
    def test_visualization_panel_has_refresh(self):
        """Panel should have refresh method."""
        from app.pages.pipeline.components.visualization_panel import VisualizationPanel
        panel = VisualizationPanel(lambda: None)
        assert hasattr(panel, 'refresh')


class TestStepDetection:
    """Test automatic step detection from files."""
    
    def test_detect_step_from_phases_file(self):
        """Should detect step 1 from phases-*.pkl file."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("phases-S01-DMT.pkl") == 1
    
    def test_detect_step_from_subject_phases_file(self):
        """Should detect step 2 from subject_phases_*.pkl file."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("subject_phases_DMT.pkl") == 2
    
    def test_detect_step_from_syncro_file(self):
        """Should detect step 4 from syncro-*.pkl file."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("syncro-S01-DMT.pkl") == 4
    
    def test_detect_step_from_order_file(self):
        """Should detect step 5 from order-*.pkl file."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("order-S01-DMT.pkl") == 5
    
    def test_detect_step_from_order_all_file(self):
        """Should detect step 6 from order_all*.pkl file."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("order_all_DMT.pkl") == 6
    
    def test_detect_step_from_pearson_dir(self):
        """Should detect step 7 from pearson_results path."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("pearson_results/corr.pkl") == 7
    
    def test_detect_step_from_clustering_dir(self):
        """Should detect step 8 from clustering_results path."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("clustering_results/kmeans.pkl") == 8
    
    def test_detect_step_unknown_file(self):
        """Should return None for unknown files."""
        from app.pages.pipeline.components.visualization_panel import detect_step_from_file
        assert detect_step_from_file("random_file.txt") is None


class TestVisualizerRegistry:
    """Test visualizer registry and loading."""
    
    def test_all_steps_have_visualizers(self):
        """Each step 1-8 should have a registered visualizer."""
        from app.pages.pipeline.visualizers import VISUALIZER_REGISTRY
        for step_num in range(1, 9):
            assert step_num in VISUALIZER_REGISTRY, f"Step {step_num} missing visualizer"
    
    def test_visualizers_are_callable(self):
        """All visualizers should be classes or callables."""
        from app.pages.pipeline.visualizers import VISUALIZER_REGISTRY
        for step_num, viz_class in VISUALIZER_REGISTRY.items():
            assert callable(viz_class), f"Step {step_num} visualizer not callable"
    
    def test_visualizers_have_render_method(self):
        """All visualizer instances should have render method."""
        from app.pages.pipeline.visualizers import VISUALIZER_REGISTRY
        for step_num, viz_class in VISUALIZER_REGISTRY.items():
            instance = viz_class(lambda: None)
            assert hasattr(instance, 'render'), f"Step {step_num} missing render()"


class TestBaseVisualizer:
    """Test base visualizer class."""
    
    def test_base_visualizer_exists(self):
        """BaseVisualizer class should exist."""
        from app.pages.pipeline.visualizers.base import BaseVisualizer
        assert BaseVisualizer is not None
    
    def test_base_visualizer_has_required_methods(self):
        """Base visualizer should have required methods."""
        from app.pages.pipeline.visualizers.base import BaseVisualizer
        assert hasattr(BaseVisualizer, 'render')
        assert hasattr(BaseVisualizer, 'load_data')
        assert hasattr(BaseVisualizer, 'get_controls')
    
    def test_base_visualizer_step_info(self):
        """Base visualizer should provide step info."""
        from app.pages.pipeline.visualizers.base import BaseVisualizer
        assert hasattr(BaseVisualizer, 'step_number')
        assert hasattr(BaseVisualizer, 'step_name')


class TestVisualizerDataLoading:
    """Test data loading for visualizers."""
    
    def test_step1_finds_phases_files(self):
        """Step 1 visualizer should find phases-*.pkl files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create mock files
            (tmp / "DMT").mkdir()
            (tmp / "DMT" / "phases-S01-DMT.pkl").touch()
            (tmp / "DMT" / "phases-S02-DMT.pkl").touch()
            
            from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
            viz = Step1Visualizer(lambda: tmp)
            files = viz.find_data_files()
            # May find duplicates due to multiple patterns, but should find at least 2 unique
            unique_files = set(files)
            assert len(unique_files) >= 2
    
    def test_step4_finds_syncro_files(self):
        """Step 4 visualizer should find syncro-*.pkl files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "syncro-S01-DMT.pkl").touch()
            (tmp / "syncro-S02-DMT.pkl").touch()
            
            from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
            viz = Step4Visualizer(lambda: tmp)
            files = viz.find_data_files()
            # May find duplicates due to multiple patterns, but should find at least 2 unique
            unique_files = set(files)
            assert len(unique_files) >= 2


class TestVisualizationControls:
    """Test visualization control rendering."""
    
    def test_step4_has_band_selector(self):
        """Step 4 (Syncro) should have band selector."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        viz = Step4Visualizer(lambda: None)
        controls = viz.get_controls()
        assert 'band' in controls
    
    def test_step4_has_epoch_selector(self):
        """Step 4 (Syncro) should have epoch selector."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        viz = Step4Visualizer(lambda: None)
        controls = viz.get_controls()
        assert 'epoch' in controls
    
    def test_step5_has_band_selector(self):
        """Step 5 (Order) should have band selector."""
        from app.pages.pipeline.visualizers.step_5_viz import Step5Visualizer
        viz = Step5Visualizer(lambda: None)
        controls = viz.get_controls()
        assert 'band' in controls


class TestPlotTypes:
    """Test different plot type support."""
    
    def test_step4_supports_matplotlib(self):
        """Step 4 should support matplotlib plots."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        viz = Step4Visualizer(lambda: None)
        assert 'matplotlib' in viz.supported_backends
    
    def test_step4_supports_plotly(self):
        """Step 4 should support plotly plots."""
        from app.pages.pipeline.visualizers.step_4_viz import Step4Visualizer
        viz = Step4Visualizer(lambda: None)
        assert 'plotly' in viz.supported_backends
    
    def test_brain_3d_uses_plotly(self):
        """Brain 3D visualization should use plotly."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        viz = Step1Visualizer(lambda: None)
        assert viz.get_3d_backend() == 'plotly'


class TestAnimationGenerator:
    """Test animation generator functionality."""
    
    def test_animation_generator_exists(self):
        """AnimationGenerator class should exist."""
        from app.pages.pipeline.components.animation_generator import AnimationGenerator
        assert AnimationGenerator is not None
    
    def test_animation_generator_has_render(self):
        """Animation generator should have render method."""
        from app.pages.pipeline.components.animation_generator import AnimationGenerator
        gen = AnimationGenerator(lambda: None)
        assert hasattr(gen, 'render')


class TestPearsonGallery:
    """Test Pearson results gallery."""
    
    def test_pearson_gallery_exists(self):
        """PearsonGallery class should exist."""
        from app.pages.pipeline.components.pearson_gallery import PearsonGallery
        assert PearsonGallery is not None
    
    def test_pearson_gallery_finds_images(self):
        """Gallery should find PNG/SVG images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "pearson_results").mkdir()
            (tmp / "pearson_results" / "corr_DMT_vs_EC.png").touch()
            (tmp / "pearson_results" / "corr_DMT_vs_EO.svg").touch()
            
            from app.pages.pipeline.components.pearson_gallery import PearsonGallery
            gallery = PearsonGallery(lambda: tmp)
            images = gallery.find_images()
            assert len(images) == 2
    
    def test_pearson_gallery_filter_by_condition(self):
        """Gallery should filter by condition."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            (tmp / "pearson_results").mkdir()
            (tmp / "pearson_results" / "corr_DMT_vs_EC.png").touch()
            (tmp / "pearson_results" / "corr_EO_vs_EC.png").touch()
            
            from app.pages.pipeline.components.pearson_gallery import PearsonGallery
            gallery = PearsonGallery(lambda: tmp)
            dmt_images = gallery.find_images(condition='DMT')
            assert len(dmt_images) == 1

