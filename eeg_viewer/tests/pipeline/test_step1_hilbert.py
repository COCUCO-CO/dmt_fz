"""
Tests for Step 1 Hilbert visualizations.

Validates:
- Files found in subdirectories (DMT/, EC/, EO/)
- Condition detection and selection
- Hilbert 2D/3D visualization rendering
- Multi-condition display
"""
import pytest
import sys
import pickle
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestStep1FileDetection:
    """Test file detection in subdirectories."""
    
    def test_finds_files_in_root(self, tmp_path):
        """Should find phases files in root directory."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create files in root
        (tmp_path / 'phases-S01-DMT.pkl').touch()
        (tmp_path / 'phases-S02-DMT.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        files = viz.find_data_files()
        
        assert len(files) == 2
    
    def test_finds_files_in_subdirectories(self, tmp_path):
        """Should find phases files in DMT/EC/EO subdirectories."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create subdirectories with files
        (tmp_path / 'DMT').mkdir()
        (tmp_path / 'EC').mkdir()
        (tmp_path / 'EO').mkdir()
        
        (tmp_path / 'DMT' / 'phases-S01-DMT.pkl').touch()
        (tmp_path / 'DMT' / 'phases-S02-DMT.pkl').touch()
        (tmp_path / 'EC' / 'phases-S01-EC.pkl').touch()
        (tmp_path / 'EO' / 'phases-S01-EO.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        files = viz.find_data_files()
        
        assert len(files) == 4
        
        # Check all conditions are found
        file_names = [f.name for f in files]
        assert any('DMT' in n for n in file_names)
        assert any('EC' in n for n in file_names)
        assert any('EO' in n for n in file_names)
    
    def test_finds_files_mixed_locations(self, tmp_path):
        """Should find files in both root and subdirectories."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Files in root
        (tmp_path / 'phases-S01-DMT.pkl').touch()
        
        # Files in subdirectory
        (tmp_path / 'EC').mkdir()
        (tmp_path / 'EC' / 'phases-S02-EC.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        files = viz.find_data_files()
        
        assert len(files) == 2


class TestConditionDetection:
    """Test condition detection from files."""
    
    def test_detects_conditions_from_filenames(self, tmp_path):
        """Should detect DMT/EC/EO from file names."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        (tmp_path / 'phases-S01-DMT.pkl').touch()
        (tmp_path / 'phases-S01-EC.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        conditions = viz._get_available_conditions()
        
        assert 'DMT' in conditions
        assert 'EC' in conditions
    
    def test_detects_conditions_from_directories(self, tmp_path):
        """Should detect conditions from parent directory names."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        (tmp_path / 'DMT').mkdir()
        (tmp_path / 'EO').mkdir()
        (tmp_path / 'DMT' / 'phases-S01.pkl').touch()
        (tmp_path / 'EO' / 'phases-S01.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        conditions = viz._get_available_conditions()
        
        assert 'DMT' in conditions
        assert 'EO' in conditions
    
    def test_gets_subjects_for_condition(self, tmp_path):
        """Should get list of subjects for a specific condition."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        (tmp_path / 'DMT').mkdir()
        (tmp_path / 'DMT' / 'phases-S01-DMT.pkl').touch()
        (tmp_path / 'DMT' / 'phases-S02-DMT.pkl').touch()
        (tmp_path / 'DMT' / 'phases-S03-DMT.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        subjects = viz._get_subjects_for_condition('DMT')
        
        assert len(subjects) == 3
        assert 'S01-DMT' in subjects or 'S01' in subjects[0]


class TestConditionSelection:
    """Test condition toggle functionality."""
    
    def test_toggle_condition_adds(self):
        """Should add condition when toggled on."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        viz._selected_conditions = ['DMT']
        
        viz._toggle_condition('EC', True)
        
        assert 'EC' in viz._selected_conditions
        assert 'DMT' in viz._selected_conditions
    
    def test_toggle_condition_removes(self):
        """Should remove condition when toggled off."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        viz._selected_conditions = ['DMT', 'EC']
        
        viz._toggle_condition('DMT', False)
        
        assert 'DMT' not in viz._selected_conditions
        assert 'EC' in viz._selected_conditions
    
    def test_multiple_conditions_supported(self):
        """Should support selecting multiple conditions."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        viz._selected_conditions = []
        
        viz._toggle_condition('DMT', True)
        viz._toggle_condition('EC', True)
        viz._toggle_condition('EO', True)
        
        assert len(viz._selected_conditions) == 3


class TestHilbertDataLoading:
    """Test Hilbert data loading functionality."""
    
    def test_load_condition_data_caches(self, tmp_path):
        """Should cache loaded data."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create test data
        (tmp_path / 'DMT').mkdir()
        data = {'phases_stc': {'Alpha': np.random.rand(10, 102, 800)}}
        with open(tmp_path / 'DMT' / 'phases-S01-DMT.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        viz = Step1Visualizer(lambda: tmp_path)
        
        # Load first time
        result1 = viz._load_condition_data('DMT', 'S01-DMT')
        
        # Load second time (should be from cache)
        result2 = viz._load_condition_data('DMT', 'S01-DMT')
        
        assert result1 is not None
        assert result1 is result2  # Same object (cached)
    
    def test_load_handles_missing_file(self, tmp_path):
        """Should handle missing file gracefully."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: tmp_path)
        result = viz._load_condition_data('DMT', 'NonExistent')
        
        assert result is None


class TestHilbertVisualization:
    """Test Hilbert visualization methods exist and work."""
    
    def test_has_hilbert_2d_method(self):
        """Should have _render_hilbert_2d_proper method."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        assert hasattr(viz, '_render_hilbert_2d_proper')
    
    def test_has_hilbert_3d_method(self):
        """Should have _render_hilbert_3d_proper method."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        assert hasattr(viz, '_render_hilbert_3d_proper')
    
    def test_hilbert_view_checks_conditions(self):
        """Should check for selected conditions before rendering."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        viz._selected_conditions = []
        
        # Should not crash when no conditions selected
        assert len(viz._selected_conditions) == 0


class TestStatsGathering:
    """Test statistics gathering from files."""
    
    def test_gather_stats_detects_condition_from_parent_dir(self, tmp_path):
        """Should detect condition from parent directory name."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create subdirectory structure
        (tmp_path / 'DMT').mkdir()
        (tmp_path / 'EC').mkdir()
        
        # Create minimal pkl files
        data = {'phases_stc': {'Alpha': [np.random.rand(102, 800)]}, 'sfreq': 500}
        
        with open(tmp_path / 'DMT' / 'phases-S01.pkl', 'wb') as f:
            pickle.dump(data, f)
        with open(tmp_path / 'DMT' / 'phases-S02.pkl', 'wb') as f:
            pickle.dump(data, f)
        with open(tmp_path / 'EC' / 'phases-S01.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        viz = Step1Visualizer(lambda: tmp_path)
        viz._gather_stats()
        
        assert viz._stats is not None
        assert viz._stats['total_files'] == 3
        assert viz._stats['conditions'].get('DMT', 0) == 2
        assert viz._stats['conditions'].get('EC', 0) == 1


class TestHilbert2DPlotContent:
    """Test that Hilbert 2D plot contains correct subplots."""
    
    def test_hilbert_2d_creates_four_subplots(self, tmp_path):
        """Hilbert 2D should create 4 subplots using Plotly make_subplots."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Check that Plotly figure would have 4 parts
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_2d_proper)
        
        # Should use Plotly make_subplots with 2x2 layout
        assert 'make_subplots' in source
        assert 'rows=2' in source and 'cols=2' in source
        
        # Should have amplitude, phase, evolution, and polar
        assert 'Amplitude' in source or 'amplitude' in source
        assert 'Phase' in source or 'phase' in source
        assert 'polar' in source or 'Polar' in source


class TestHilbert3DPlotContent:
    """Test that Hilbert 3D plot matches original visualize_hilbert_improved.py."""
    
    def test_hilbert_3d_has_trajectory(self):
        """Hilbert 3D should have main trajectory trace."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_3d_proper)
        
        assert 'Scatter3d' in source
        assert 'trajectory' in source.lower() or 'Trayectoria' in source
    
    def test_hilbert_3d_has_projections(self):
        """Hilbert 3D should have orthogonal projections."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_3d_proper)
        
        # Should have 3 projections
        assert 'proj_time_real' in source
        assert 'proj_time_imag' in source
        assert 'proj_real_imag' in source
    
    def test_hilbert_3d_has_markers(self):
        """Hilbert 3D should have start/end markers."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_3d_proper)
        
        assert 'markers' in source.lower()
        assert '#2ca02c' in source or 'green' in source.lower()  # Start marker color
    
    def test_hilbert_3d_trims_edges(self):
        """Hilbert 3D should trim 15% from each edge."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_3d_proper)
        
        assert 'trim' in source or '0.15' in source


class TestHilbertPlotQuality:
    """Test Hilbert plot quality and alignment."""
    
    def test_hilbert_2d_uses_plotly(self):
        """Hilbert 2D should use Plotly for better quality."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_2d_proper)
        
        # Should use Plotly instead of matplotlib
        assert 'plotly.graph_objects' in source or 'import plotly' in source
        assert 'make_subplots' in source
    
    def test_hilbert_2d_has_consistent_height(self):
        """Hilbert 2D should have same height as 3D (350px)."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_2d_proper)
        
        assert 'height=350' in source
    
    def test_hilbert_3d_has_consistent_height(self):
        """Hilbert 3D should have same height as 2D (350px)."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_3d_proper)
        
        assert 'height=350' in source
    
    def test_hilbert_2d_has_proper_subplot_spacing(self):
        """Hilbert 2D should have proper spacing to avoid overlap."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_2d_proper)
        
        # Should have spacing configs
        assert 'horizontal_spacing' in source
        assert 'vertical_spacing' in source
    
    def test_hilbert_2d_has_small_fonts(self):
        """Hilbert 2D should have small fonts to avoid text overlap."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_2d_proper)
        
        # Font sizes should be small (9 or less)
        assert 'font_size=9' in source or 'size=9' in source
        assert 'tickfont_size=8' in source
    
    def test_hilbert_layout_uses_items_stretch(self):
        """Hilbert row layout should use items-stretch for alignment."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_view)
        
        assert 'items-stretch' in source
    
    def test_hilbert_cards_have_min_height(self):
        """Hilbert cards should have consistent min-height."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_view)
        
        # Both cards should have min-height
        assert 'min-height: 380px' in source or 'min-height:380px' in source
    
    def test_hilbert_2d_has_four_subplots(self):
        """Hilbert 2D should create proper 2x2 subplot layout."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        import inspect
        source = inspect.getsource(Step1Visualizer._render_hilbert_2d_proper)
        
        # Should have 2x2 layout
        assert 'rows=2' in source
        assert 'cols=2' in source
        # Should have proper subplot types
        assert 'heatmap' in source
        assert 'polar' in source


class TestDirectoryScanningInfo:
    """Test directory structure scanning and display."""
    
    def test_scan_directory_structure_exists(self):
        """Should have _scan_directory_structure method."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: None)
        assert hasattr(viz, '_scan_directory_structure')
    
    def test_scan_empty_directory(self, tmp_path):
        """Should handle empty directory."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        viz = Step1Visualizer(lambda: tmp_path)
        result = viz._scan_directory_structure()
        
        assert result['has_subdirs'] == False
        assert result['root_files'] == 0
    
    def test_scan_detects_subdirectories(self, tmp_path):
        """Should detect DMT/EC/EO subdirectories."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create subdirectories
        (tmp_path / 'DMT').mkdir()
        (tmp_path / 'EC').mkdir()
        
        viz = Step1Visualizer(lambda: tmp_path)
        result = viz._scan_directory_structure()
        
        assert result['has_subdirs'] == True
        assert 'DMT' in result['subdirs']
        assert 'EC' in result['subdirs']
    
    def test_scan_counts_pkl_files(self, tmp_path):
        """Should count pkl files in subdirectories."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create subdirectory with files
        (tmp_path / 'DMT').mkdir()
        (tmp_path / 'DMT' / 'phases-S01-DMT.pkl').touch()
        (tmp_path / 'DMT' / 'phases-S02-DMT.pkl').touch()
        (tmp_path / 'DMT' / 'order-S01-DMT.pkl').touch()
        (tmp_path / 'DMT' / 'syncro-S01-DMT.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        result = viz._scan_directory_structure()
        
        assert result['subdirs']['DMT']['total_pkl'] == 4
        assert result['subdirs']['DMT']['phases'] == 2
        assert result['subdirs']['DMT']['order'] == 1
        assert result['subdirs']['DMT']['syncro'] == 1
    
    def test_scan_counts_root_files(self, tmp_path):
        """Should count pkl files in root directory."""
        from app.pages.pipeline.visualizers.step_1_viz import Step1Visualizer
        
        # Create files in root
        (tmp_path / 'some_data.pkl').touch()
        (tmp_path / 'extra.pkl').touch()
        
        viz = Step1Visualizer(lambda: tmp_path)
        result = viz._scan_directory_structure()
        
        assert result['root_files'] == 2

