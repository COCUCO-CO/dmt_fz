"""
Tests for app/visualization module.

Tests Plotly figure creation and styling.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import plotly.graph_objects as go

from app.visualization import (
    make_eeg_fig,
    make_fft_fig,
    make_hilbert_fig,
    make_brain_fig,
    FigureFactory,
)
from app.visualization.figures import (
    EEGFigure,
    FFTFigure,
    HilbertFigure,
    BrainFigure,
    BaseFigure,
)
from app.visualization.styles import (
    THEME_PRIMARY,
    THEME_SECONDARY,
    SIGNAL_COLORS,
    EEG_HEIGHT,
    FFT_HEIGHT,
    HILBERT_HEIGHT,
    BRAIN_HEIGHT,
)


class TestEEGFigure:
    """Tests for EEG figure creation."""
    
    def test_make_eeg_fig_returns_figure(self):
        """Test make_eeg_fig returns Plotly Figure."""
        fig = make_eeg_fig()
        assert isinstance(fig, go.Figure)
    
    def test_eeg_fig_height(self):
        """Test EEG figure has correct height."""
        fig = make_eeg_fig()
        assert fig.layout.height == EEG_HEIGHT
    
    def test_eeg_fig_dark_theme(self):
        """Test EEG figure uses dark theme."""
        fig = make_eeg_fig()
        assert fig.layout.template.layout.paper_bgcolor is not None or 'dark' in str(fig.layout.template)
    
    def test_eeg_fig_primary_style(self):
        """Test primary style uses green color."""
        fig = make_eeg_fig(use_eeg2=False)
        # Check y-axis uses primary color
        assert THEME_PRIMARY in str(fig.layout.yaxis.tickfont.color)
    
    def test_eeg_fig_secondary_style(self):
        """Test secondary style uses pink color."""
        fig = make_eeg_fig(use_eeg2=True)
        # Secondary uses pink
        assert '#f472b6' in str(fig.layout.yaxis.tickfont.color)
    
    def test_eeg_figure_class(self):
        """Test EEGFigure class directly."""
        eeg_fig = EEGFigure(use_secondary_style=False)
        fig = eeg_fig.create()
        assert isinstance(fig, go.Figure)


class TestFFTFigure:
    """Tests for FFT figure creation."""
    
    def test_make_fft_fig_returns_figure(self):
        """Test make_fft_fig returns Plotly Figure."""
        fig = make_fft_fig()
        assert isinstance(fig, go.Figure)
    
    def test_fft_fig_height(self):
        """Test FFT figure has correct height."""
        fig = make_fft_fig()
        assert fig.layout.height == FFT_HEIGHT
    
    def test_fft_fig_has_band_annotations(self):
        """Test FFT figure has frequency band annotations."""
        fig = make_fft_fig()
        # Should have 5 band annotations (δ, θ, α, β, γ)
        assert len(fig.layout.annotations) == 5
    
    def test_fft_fig_has_legend(self):
        """Test FFT figure shows legend."""
        fig = make_fft_fig()
        assert fig.layout.showlegend == True
    
    def test_fft_fig_x_range(self):
        """Test FFT figure has correct frequency range."""
        fig = make_fft_fig()
        assert list(fig.layout.xaxis.range) == [0, 60]


class TestHilbertFigure:
    """Tests for Hilbert figure creation."""
    
    def test_make_hilbert_fig_returns_figure(self):
        """Test make_hilbert_fig returns Plotly Figure."""
        fig = make_hilbert_fig()
        assert isinstance(fig, go.Figure)
    
    def test_hilbert_fig_height(self):
        """Test Hilbert figure has correct height."""
        fig = make_hilbert_fig()
        assert fig.layout.height == HILBERT_HEIGHT
    
    def test_hilbert_fig_has_subplots(self):
        """Test Hilbert figure has 2 subplot titles."""
        fig = make_hilbert_fig()
        # Should have ENVELOPE and PHASE annotations
        assert len(fig.layout.annotations) == 2
        titles = [a.text for a in fig.layout.annotations]
        assert '<b>ENVELOPE</b>' in titles
        assert '<b>PHASE</b>' in titles


class TestBrainFigure:
    """Tests for brain topography figure creation."""
    
    def test_make_brain_fig_returns_figure(self):
        """Test make_brain_fig returns Plotly Figure."""
        fig = make_brain_fig()
        assert isinstance(fig, go.Figure)
    
    def test_brain_fig_height(self):
        """Test brain figure has correct height."""
        fig = make_brain_fig()
        assert fig.layout.height == BRAIN_HEIGHT
    
    def test_brain_fig_has_head_outline(self):
        """Test brain figure has head outline traces."""
        fig = make_brain_fig()
        # Should have: head circle, nose, left ear, right ear = 4 traces
        assert len(fig.data) == 4
    
    def test_brain_fig_no_axis_labels(self):
        """Test brain figure hides axis labels."""
        fig = make_brain_fig()
        assert fig.layout.xaxis.showticklabels == False
        assert fig.layout.yaxis.showticklabels == False
    
    def test_brain_fig_primary_color(self):
        """Test primary style uses green head outline."""
        fig = make_brain_fig(use_eeg2=False)
        # First trace is head outline
        assert 'rgba(0,255,136' in fig.data[0].line.color
    
    def test_brain_fig_secondary_color(self):
        """Test secondary style uses pink head outline."""
        fig = make_brain_fig(use_eeg2=True)
        assert 'rgba(244,114,182' in fig.data[0].line.color


class TestFigureFactory:
    """Tests for FigureFactory pattern."""
    
    def test_factory_create_eeg(self):
        """Test factory creates EEG figure."""
        fig = FigureFactory.create('eeg')
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == EEG_HEIGHT
    
    def test_factory_create_fft(self):
        """Test factory creates FFT figure."""
        fig = FigureFactory.create('fft')
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == FFT_HEIGHT
    
    def test_factory_create_hilbert(self):
        """Test factory creates Hilbert figure."""
        fig = FigureFactory.create('hilbert')
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == HILBERT_HEIGHT
    
    def test_factory_create_brain(self):
        """Test factory creates brain figure."""
        fig = FigureFactory.create('brain')
        assert isinstance(fig, go.Figure)
        assert fig.layout.height == BRAIN_HEIGHT
    
    def test_factory_with_secondary_style(self):
        """Test factory passes kwargs correctly."""
        fig = FigureFactory.create('eeg', use_secondary_style=True)
        assert '#f472b6' in str(fig.layout.yaxis.tickfont.color)
    
    def test_factory_unknown_type_raises(self):
        """Test factory raises for unknown figure type."""
        with pytest.raises(ValueError):
            FigureFactory.create('unknown_type')


class TestThemeConstants:
    """Tests for theme constants."""
    
    def test_theme_colors_defined(self):
        """Test theme colors are defined."""
        assert THEME_PRIMARY == '#00ff88'
        assert THEME_SECONDARY == '#00d4ff'
    
    def test_signal_colors_list(self):
        """Test signal colors is a list."""
        assert isinstance(SIGNAL_COLORS, list)
        assert len(SIGNAL_COLORS) >= 10
    
    def test_heights_defined(self):
        """Test figure heights are defined."""
        assert EEG_HEIGHT == 250
        assert FFT_HEIGHT == 180
        assert HILBERT_HEIGHT == 180
        assert BRAIN_HEIGHT == 200

