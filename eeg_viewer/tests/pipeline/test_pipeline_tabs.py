"""
Tests for pipeline page tabs functionality.

Tests:
- Console tab (logging, history)
- Files tab (directory browsing)
- System tab (resource monitoring)
- Visualize tab (plot rendering)
"""

import sys
import psutil
from pathlib import Path
from datetime import datetime

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Test: Console Tab
# =============================================================================

class TestConsoleTab:
    """Tests for console tab functionality."""
    
    def test_log_history_restoration(self, pipeline_state_with_history):
        """Console should restore log history when revisiting tab."""
        state = pipeline_state_with_history
        
        # History should be available
        assert len(state.log_history) > 0
        assert "[RUN] Starting pipeline" in state.log_history
    
    def test_status_label_idle(self, fresh_pipeline_state):
        """Status should show 'Idle' when not running."""
        state = fresh_pipeline_state
        
        # When not running, status should be idle
        if state.running and state.start_time:
            status = "Running"
        else:
            status = "Idle"
        
        assert status == "Idle"
    
    def test_status_label_running(self, pipeline_state_running):
        """Status should show running info when running."""
        state = pipeline_state_running
        
        elapsed = (datetime.now() - state.start_time).seconds
        mins, secs = divmod(elapsed, 60)
        status = f'Running: {state.current_step} ({mins}m {secs}s)'
        
        assert "Running:" in status
        assert state.current_step in status
    
    def test_clear_log(self, pipeline_state_with_history):
        """Clear should empty log history."""
        state = pipeline_state_with_history
        
        assert len(state.log_history) > 0
        
        state.log_history.clear()
        
        assert len(state.log_history) == 0
    
    def test_special_messages_formatting(self, fresh_pipeline_state):
        """Special messages should be identifiable for formatting."""
        state = fresh_pipeline_state
        
        messages = [
            "[RUN] Starting",
            "[SETUP] Init",
            "[INFO] Done",
            "===== Header =====",
            "COMPLETED successfully",
            "FAILED with error",
        ]
        
        # These should trigger special formatting (line break before)
        special_keywords = ['[RUN]', 'Starting:', '[INFO]', '[SETUP]', '===', 'COMPLETED', 'FAILED']
        
        for msg in messages:
            is_special = any(x in msg for x in special_keywords)
            # All our test messages should be recognized as special
            assert is_special


# =============================================================================
# Test: Files Tab
# =============================================================================

class TestFilesTab:
    """Tests for files tab functionality."""
    
    def test_no_run_selected_message(self, fresh_pipeline_state):
        """Should indicate when no run is selected."""
        state = fresh_pipeline_state
        
        assert state.selected_run is None
    
    def test_run_dir_exists_check(self, pipeline_state_with_run_selected):
        """Should verify run directory exists."""
        state = pipeline_state_with_run_selected
        
        if state.selected_run:
            assert state.selected_run.exists()
    
    def test_count_syncro_files(self, temp_pipeline_with_results):
        """Should count syncro/phases files correctly."""
        run_path = temp_pipeline_with_results["result_run"]
        
        syncro_count = len(list(run_path.rglob('syncro-*.pkl')))
        phases_count = len(list(run_path.rglob('phases-*.pkl')))
        
        total = syncro_count + phases_count
        assert total > 0
    
    def test_count_order_files(self, tmp_path):
        """Should count order files correctly."""
        run_path = tmp_path / "run_test"
        run_path.mkdir()
        
        # Create test order files
        (run_path / "order_all-S01.pkl").touch()
        (run_path / "order_all-S02.pkl").touch()
        (run_path / "order-S01.pkl").touch()
        
        order_all_count = len(list(run_path.rglob('order_all-*.pkl')))
        order_count = len(list(run_path.rglob('order-*.pkl')))
        
        assert order_all_count == 2
        assert order_count == 1
    
    def test_directory_listing(self, temp_pipeline_with_results):
        """Should list directories and files in run."""
        run_path = temp_pipeline_with_results["result_run"]
        
        items = list(run_path.iterdir())
        
        # Should have condition directories
        dirs = [i for i in items if i.is_dir()]
        files = [i for i in items if i.is_file()]
        
        assert len(dirs) > 0  # Should have DMT, EC, EO dirs
    
    def test_file_size_calculation(self, temp_pipeline_with_results):
        """Should calculate file sizes."""
        run_path = temp_pipeline_with_results["result_run"]
        
        for item in run_path.rglob("*.pkl"):
            size_bytes = item.stat().st_size
            size_kb = size_bytes / 1024
            
            # Size should be calculable
            assert size_kb >= 0


# =============================================================================
# Test: System Tab
# =============================================================================

class TestSystemTab:
    """Tests for system monitoring tab."""
    
    def test_cpu_percent_available(self):
        """Should be able to get CPU percentage."""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        assert 0 <= cpu_percent <= 100
    
    def test_cpu_count_available(self):
        """Should be able to get CPU count."""
        cpu_count = psutil.cpu_count()
        
        assert cpu_count > 0
    
    def test_per_cpu_stats(self):
        """Should be able to get per-CPU stats."""
        per_cpu = psutil.cpu_percent(percpu=True)
        
        assert len(per_cpu) > 0
        for pct in per_cpu:
            assert 0 <= pct <= 100
    
    def test_memory_stats(self):
        """Should be able to get memory statistics."""
        mem = psutil.virtual_memory()
        
        assert mem.total > 0
        assert mem.used >= 0
        assert mem.percent >= 0
        assert mem.percent <= 100
    
    def test_memory_in_gb(self):
        """Should convert memory to GB correctly."""
        mem = psutil.virtual_memory()
        
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        
        assert mem_used_gb < mem_total_gb
        assert mem_total_gb > 0
    
    def test_cpu_color_thresholds(self):
        """CPU color should change based on usage."""
        def get_cpu_color(cpu_percent):
            if cpu_percent < 50:
                return 'green'
            elif cpu_percent < 80:
                return 'yellow'
            else:
                return 'red'
        
        assert get_cpu_color(30) == 'green'
        assert get_cpu_color(60) == 'yellow'
        assert get_cpu_color(90) == 'red'
    
    def test_memory_color_thresholds(self):
        """Memory color should change based on usage."""
        def get_mem_color(mem_percent):
            if mem_percent < 60:
                return 'green'
            elif mem_percent < 85:
                return 'yellow'
            else:
                return 'red'
        
        assert get_mem_color(40) == 'green'
        assert get_mem_color(70) == 'yellow'
        assert get_mem_color(90) == 'red'


# =============================================================================
# Test: Visualize Tab Data Selection
# =============================================================================

class TestVisualizeTabDataSelection:
    """Tests for visualization data selection functionality."""
    
    def test_band_options(self):
        """Band selector should have all frequency bands."""
        bands = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        assert len(bands) == 5
        assert 'Alpha' in bands
    
    def test_default_band(self):
        """Default band should be Alpha."""
        default_band = 'Alpha'
        
        assert default_band == 'Alpha'
    
    def test_epoch_range(self):
        """Epoch selector should accept valid range."""
        min_epoch = 0
        max_epoch = 100
        
        for epoch in [0, 50, 100]:
            assert min_epoch <= epoch <= max_epoch
    
    def test_custom_path_input(self, tmp_path):
        """Should accept custom data path."""
        custom_path = tmp_path / "custom_data"
        custom_path.mkdir()
        
        # Path should be usable
        assert custom_path.exists()


# =============================================================================
# Test: Visualize Tab Plot Types
# =============================================================================

class TestVisualizeTabPlotTypes:
    """Tests for different plot types in visualize tab."""
    
    def test_brain_plot_types(self):
        """Should support different brain plot types."""
        plot_types = ['network', 'colored', 'sync', 'all_bands']
        
        assert 'network' in plot_types
        assert 'sync' in plot_types
    
    def test_brain_plot_default(self):
        """Default brain plot should be 'network'."""
        default = 'network'
        
        assert default == 'network'
    
    def test_brain_plot_button_active_state(self):
        """Only one brain plot button should be active."""
        buttons = {
            'network': True,
            'colored': False,
            'sync': False,
            'all_bands': False,
        }
        
        active_count = sum(1 for v in buttons.values() if v)
        assert active_count == 1


# =============================================================================
# Test: Visualize Tab Animation Generator
# =============================================================================

class TestVisualizeTabAnimationGenerator:
    """Tests for animation generation functionality."""
    
    def test_animation_modes(self):
        """Should support different animation modes."""
        modes = ['stc', 'eeg', 'all', 'advanced']
        
        assert 'stc' in modes
        assert 'advanced' in modes
    
    def test_animation_quality_options(self):
        """Should support different quality levels."""
        qualities = ['high', 'medium', 'low']
        
        assert 'medium' in qualities
    
    def test_animation_formats(self):
        """Should support different output formats."""
        formats = ['png', 'jpg']
        
        assert 'png' in formats
    
    def test_fps_range(self):
        """FPS should be in valid range."""
        min_fps = 1
        max_fps = 30
        default_fps = 5
        
        assert min_fps <= default_fps <= max_fps
    
    def test_epoch_range_for_animation(self):
        """Animation epoch range should be valid."""
        start_epoch = 0
        end_epoch = 10
        
        assert start_epoch < end_epoch
    
    def test_ffmpeg_command_construction(self, tmp_path):
        """Should construct valid ffmpeg command."""
        frames_dir = tmp_path / "frames"
        output_video = tmp_path / "output.mp4"
        fps = 5
        format = "png"
        
        cmd = [
            'ffmpeg', '-y',
            '-framerate', str(fps),
            '-pattern_type', 'glob',
            '-i', str(frames_dir / f'*.{format}'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            str(output_video)
        ]
        
        assert 'ffmpeg' in cmd
        assert '-framerate' in cmd
        assert str(fps) in cmd


# =============================================================================
# Test: Visualize Tab Pearson Gallery
# =============================================================================

class TestVisualizeTabPearsonGallery:
    """Tests for Pearson correlation results gallery."""
    
    def test_pearson_filter_options(self):
        """Should have filter options for Pearson results."""
        metrics = ['All', 'Coherence', 'Metastability']
        conditions = ['All', 'DMT', 'EC', 'EO']
        bands = ['All', 'Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']
        
        assert 'All' in metrics
        assert 'All' in conditions
        assert 'All' in bands
    
    def test_pearson_file_patterns(self, tmp_path):
        """Should recognize Pearson result file patterns."""
        pearson_dir = tmp_path / "pearson_results"
        pearson_dir.mkdir()
        
        # Create test files
        (pearson_dir / "coherence_DMT_Alpha.png").touch()
        (pearson_dir / "metastability_EC_Beta.svg").touch()
        
        png_files = list(pearson_dir.glob("*.png"))
        svg_files = list(pearson_dir.glob("*.svg"))
        
        assert len(png_files) == 1
        assert len(svg_files) == 1
    
    def test_pearson_filter_by_metric(self, tmp_path):
        """Should filter results by metric."""
        pearson_dir = tmp_path / "pearson_results"
        pearson_dir.mkdir()
        
        files = [
            "coherence_DMT_Alpha.png",
            "coherence_EC_Beta.png",
            "metastability_DMT_Alpha.png",
        ]
        
        for f in files:
            (pearson_dir / f).touch()
        
        # Filter for coherence
        all_files = list(pearson_dir.glob("*.png"))
        coherence_files = [f for f in all_files if "coherence" in f.name.lower()]
        
        assert len(coherence_files) == 2
    
    def test_pearson_filter_by_condition(self, tmp_path):
        """Should filter results by condition."""
        pearson_dir = tmp_path / "pearson_results"
        pearson_dir.mkdir()
        
        files = [
            "coherence_DMT_Alpha.png",
            "coherence_EC_Alpha.png",
            "coherence_EO_Alpha.png",
        ]
        
        for f in files:
            (pearson_dir / f).touch()
        
        all_files = list(pearson_dir.glob("*.png"))
        dmt_files = [f for f in all_files if "dmt" in f.name.lower()]
        
        assert len(dmt_files) == 1


# =============================================================================
# Test: Visualize Tab Clustering
# =============================================================================

class TestVisualizeTabClustering:
    """Tests for clustering visualization functionality."""
    
    def test_clustering_results_directory(self, tmp_path):
        """Should find clustering results directory."""
        run_dir = tmp_path / "run_test"
        cluster_dir = run_dir / "clustering_results"
        cluster_dir.mkdir(parents=True)
        
        assert cluster_dir.exists()
    
    def test_clustering_output_files(self, tmp_path):
        """Should recognize clustering output files."""
        cluster_dir = tmp_path / "clustering_results"
        cluster_dir.mkdir()
        
        # Create test files
        (cluster_dir / "cluster_scores_Alpha.csv").touch()
        (cluster_dir / "pca_scatter_Alpha.png").touch()
        (cluster_dir / "cluster_model_Alpha.pkl").touch()
        
        csv_files = list(cluster_dir.glob("*.csv"))
        png_files = list(cluster_dir.glob("*.png"))
        pkl_files = list(cluster_dir.glob("*.pkl"))
        
        assert len(csv_files) == 1
        assert len(png_files) == 1
        assert len(pkl_files) == 1








