"""
Smoke tests - Run BEFORE and AFTER refactoring.
These verify the app still works correctly.

Run with: pytest tests/test_smoke.py -v
"""
import numpy as np


class TestSignalProcessing:
    """Verify signal processing functions work."""
    
    def test_apply_notch(self):
        """Test notch filter application."""
        from main import apply_notch
        data = np.random.randn(10, 1000)
        result = apply_notch(data, 256.0, 50.0)
        assert result.shape == data.shape
        assert not np.allclose(result, data)  # Should be different
    
    def test_apply_bandpass(self):
        """Test bandpass filter application."""
        from main import apply_bandpass
        data = np.random.randn(10, 1000)
        result = apply_bandpass(data, 256.0, 1.0, 45.0)
        assert result.shape == data.shape
    
    def test_compute_fft(self):
        """Test FFT computation."""
        from main import compute_fft
        data = np.random.randn(10, 1000)
        freqs, power = compute_fft(data, 256.0)
        assert len(freqs) > 0
        assert power.shape[0] == 10
        assert power.shape[1] == len(freqs)
    
    def test_compute_hilbert(self):
        """Test Hilbert transform computation."""
        from main import compute_hilbert
        data = np.random.randn(1000)
        amp, phase = compute_hilbert(data)
        assert amp.shape == data.shape
        assert phase.shape == data.shape
        assert np.all(amp >= 0)  # Amplitude should be non-negative
    
    def test_process_data_no_filters(self):
        """Test process_data with no filters enabled."""
        from main import process_data, S
        # Save original state
        orig_notch = S.notch_enabled
        orig_bp = S.bandpass_enabled
        
        # Disable all filters
        S.notch_enabled = False
        S.bandpass_enabled = False
        
        data = np.random.randn(10, 1000)
        result = process_data(data, 256.0)
        
        # Should return copy of original
        assert result.shape == data.shape
        assert np.allclose(result, data)
        
        # Restore state
        S.notch_enabled = orig_notch
        S.bandpass_enabled = orig_bp
    
    def test_process_data_with_notch(self):
        """Test process_data with notch filter enabled."""
        from main import process_data, S
        orig_notch = S.notch_enabled
        orig_freq = S.notch_freq
        
        S.notch_enabled = True
        S.notch_freq = 50.0
        
        data = np.random.randn(10, 1000)
        result = process_data(data, 256.0)
        
        assert result.shape == data.shape
        assert not np.allclose(result, data)
        
        S.notch_enabled = orig_notch
        S.notch_freq = orig_freq


class TestFigureCreation:
    """Verify Plotly figures can be created."""
    
    def test_eeg_figure(self):
        """Test EEG figure creation."""
        from main import make_eeg_fig
        fig = make_eeg_fig()
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_eeg_figure_secondary(self):
        """Test EEG figure creation with secondary style."""
        from main import make_eeg_fig
        fig = make_eeg_fig(use_eeg2=True)
        assert fig is not None
    
    def test_fft_figure(self):
        """Test FFT figure creation."""
        from main import make_fft_fig
        fig = make_fft_fig()
        assert fig is not None
        # FFT figure should have frequency band rectangles
        assert len(fig.layout.annotations) > 0
    
    def test_hilbert_figure(self):
        """Test Hilbert figure creation."""
        from main import make_hilbert_fig
        fig = make_hilbert_fig()
        assert fig is not None
    
    def test_brain_figure(self):
        """Test brain topography figure creation."""
        from main import make_brain_fig
        fig = make_brain_fig()
        assert fig is not None
        # Should have head outline traces
        assert len(fig.data) >= 4  # Head, nose, 2 ears


class TestStateCreation:
    """Verify state objects can be created."""
    
    def test_viewer_state(self):
        """Test viewer State class."""
        from main import State
        s = State()
        assert s.eeg_data is None
        assert s.eeg_data2 is None
        assert s.compare_mode == False
        assert s.selected_channels == []
        assert s.view_start == 0.0
        assert s.view_duration == 5.0
        assert s.is_playing == False
        assert s.scale_factor == 1.0
    
    def test_pipeline_state(self):
        """Test PipelineState class."""
        from main import PipelineState
        ps = PipelineState()
        assert ps.running == False
        assert ps.current_step == ""
        assert ps.progress == 0
        assert ps.workers == 7
    
    def test_model_state(self):
        """Test ModelState class."""
        from main import ModelState
        ms = ModelState()
        assert ms.training == False
        assert ms.status == 'idle'
        assert ms.history == {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': [], 'epoch': []}
    
    def test_analysis_state(self):
        """Test AnalysisState class."""
        from main import AnalysisState
        ast = AnalysisState()
        assert ast.model is None
        assert ast.dataset is None
        assert ast.playing == False


class TestImports:
    """Verify all imports work."""
    
    def test_main_imports(self):
        """Test that main module can be imported."""
        from main import (
            State, apply_notch, ELECTRODE_POSITIONS, STYLE
        )
        assert State is not None
        assert apply_notch is not None
        assert ELECTRODE_POSITIONS is not None
        assert len(STYLE) > 0
    
    def test_config_imports(self):
        """Test that config module can be imported."""
        from config import (
            THEME_PRIMARY, FREQ_BANDS, SIGNAL_COLORS
        )
        assert THEME_PRIMARY == "#00ff88"
        assert len(SIGNAL_COLORS) >= 10
        assert "δ" in FREQ_BANDS or "Delta" in str(FREQ_BANDS)
    
    def test_eeg_loader_imports(self):
        """Test that eeg_loader module can be imported."""
        from eeg_loader import load_eeg_file, EEGData
        assert load_eeg_file is not None
        assert EEGData is not None


class TestConstants:
    """Verify constants are correctly defined."""
    
    def test_electrode_positions(self):
        """Test electrode positions dictionary."""
        from main import ELECTRODE_POSITIONS
        assert len(ELECTRODE_POSITIONS) > 20
        assert 'Cz' in ELECTRODE_POSITIONS
        assert 'Fp1' in ELECTRODE_POSITIONS
        assert 'O1' in ELECTRODE_POSITIONS
        
        # Check position format
        for ch, pos in ELECTRODE_POSITIONS.items():
            assert isinstance(pos, tuple)
            assert len(pos) == 2
            assert -2 <= pos[0] <= 2
            assert -2 <= pos[1] <= 2
    
    def test_style_css(self):
        """Test CSS style string."""
        from main import STYLE
        assert ':root' in STYLE
        assert '--primary' in STYLE
        assert 'JetBrains Mono' in STYLE


class TestHelperFunctions:
    """Test helper/utility functions."""
    
    def test_update_functions_exist(self):
        """Verify update functions are defined."""
        from main import (
            update_eeg, update_all
        )
        # Just verify they exist and are callable
        assert callable(update_eeg)
        assert callable(update_all)
    
    def test_navigation_functions_exist(self):
        """Verify navigation functions are defined."""
        from main import nav_start, nav_back, nav_fwd, nav_end, set_win
        assert callable(nav_start)
        assert callable(nav_back)
        assert callable(nav_fwd)
        assert callable(nav_end)
        assert callable(set_win)
    
    def test_channel_functions_exist(self):
        """Verify channel management functions are defined."""
        from main import refresh_channels, toggle_ch, clear_ch
        assert callable(refresh_channels)
        assert callable(toggle_ch)
        assert callable(clear_ch)
    
    def test_epoch_functions_exist(self):
        """Verify epoch functions are defined."""
        from main import gen_epochs, save_epochs
        assert callable(gen_epochs)
        assert callable(save_epochs)


class TestDatasetDetection:
    """Test dataset detection function."""
    
    def test_detect_dataset_type_exists(self):
        """Verify detect_dataset_type is defined."""
        from main import detect_dataset_type
        assert callable(detect_dataset_type)
    
    def test_detect_nonexistent_path(self):
        """Test detection with non-existent path."""
        from main import detect_dataset_type
        from pathlib import Path
        
        result = detect_dataset_type(Path('/nonexistent/path'))
        assert result['type'] == 'unknown'
        assert result['error'] is not None


class TestModelFunctions:
    """Test model-related functions exist."""
    
    def test_model_functions_exist(self):
        """Verify model functions are defined."""
        from main import (
            detect_dataset_type,
            create_default_config,
            model_log
        )
        assert callable(detect_dataset_type)
        assert callable(create_default_config)
        assert callable(model_log)


class TestAnalysisFunctions:
    """Test analysis-related functions exist."""
    
    def test_analysis_functions_exist(self):
        """Verify analysis functions are defined."""
        from main import (
            analysis_log,
            load_trained_model,
            process_sample,
            compute_latent_pca
        )
        assert callable(analysis_log)
        assert callable(load_trained_model)
        assert callable(process_sample)
        assert callable(compute_latent_pca)

