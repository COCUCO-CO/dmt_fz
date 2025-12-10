"""
Base figure classes for Plotly visualizations.

Provides factory pattern for consistent figure creation with shared styling.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
import plotly.graph_objects as go

from ..styles.theme import (
    THEME_TEXT,
    THEME_TEXT_DIM,
    THEME_CARD,
    PLOT_BG,
    PLOT_GRID,
    FONT_FAMILY,
    FONT_SIZE_NORMAL,
    SECONDARY_COLOR,
)


class BaseFigure(ABC):
    """
    Abstract base class for all Plotly figures in the EEG Viewer.
    
    Provides:
    - Consistent dark theme styling
    - Primary/secondary color schemes
    - Common layout configuration
    
    Usage:
        class MyFigure(BaseFigure):
            def create(self) -> go.Figure:
                fig = go.Figure()
                self.apply_base_layout(fig)
                return fig
    """
    
    def __init__(self, use_secondary_style: bool = False):
        """
        Initialize figure.
        
        Args:
            use_secondary_style: Use pink color scheme (for EEG2)
        """
        self.use_secondary = use_secondary_style
    
    @property
    def primary_color(self) -> str:
        """Primary color for this figure."""
        return SECONDARY_COLOR if self.use_secondary else self._get_default_primary()
    
    @abstractmethod
    def _get_default_primary(self) -> str:
        """Return the default primary color for this figure type."""
        pass
    
    @abstractmethod
    def create(self) -> go.Figure:
        """Create and return the configured figure."""
        pass
    
    def apply_base_layout(
        self,
        fig: go.Figure,
        height: int,
        margins: Optional[dict] = None,
        show_legend: bool = False,
        hover_mode: str = 'x unified'
    ) -> None:
        """
        Apply consistent base layout to a figure.
        
        Args:
            fig: Plotly figure to configure
            height: Figure height in pixels
            margins: Custom margins dict (l, r, t, b)
            show_legend: Whether to show legend
            hover_mode: Hover interaction mode
        """
        default_margins = dict(l=60, r=10, t=10, b=50)
        if margins:
            default_margins.update(margins)
        
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor=PLOT_BG,
            margin=default_margins,
            height=height,
            font=dict(
                family=FONT_FAMILY,
                size=FONT_SIZE_NORMAL,
                color=THEME_TEXT
            ),
            showlegend=show_legend,
            hovermode=hover_mode,
            hoverlabel=dict(
                bgcolor=THEME_CARD,
                font=dict(family=FONT_FAMILY, size=FONT_SIZE_NORMAL)
            )
        )
    
    def configure_axis(
        self,
        fig: go.Figure,
        axis: str = 'x',
        title: Optional[str] = None,
        fixed_range: bool = True,
        show_grid: bool = True,
        axis_range: Optional[list] = None
    ) -> None:
        """
        Configure an axis with consistent styling.
        
        Args:
            fig: Plotly figure
            axis: 'x' or 'y'
            title: Axis title text
            fixed_range: Whether to fix the axis range
            show_grid: Whether to show grid lines
            axis_range: Optional [min, max] range
        """
        axis_config = dict(
            gridcolor=PLOT_GRID if show_grid else 'rgba(0,0,0,0)',
            tickfont=dict(size=9, color=THEME_TEXT_DIM),
            fixedrange=fixed_range
        )
        
        if title:
            axis_config['title'] = dict(
                text=title,
                font=dict(size=9, color=self.primary_color)
            )
        
        if axis_range:
            axis_config['range'] = axis_range
        
        if axis == 'x':
            fig.update_xaxes(**axis_config)
        else:
            fig.update_yaxes(**axis_config)


class FigureFactory:
    """
    Factory for creating visualization figures.
    
    Usage:
        factory = FigureFactory()
        eeg_fig = factory.create('eeg', use_secondary_style=False)
        fft_fig = factory.create('fft', use_secondary_style=True)
    """
    
    _registry: dict = {}
    
    @classmethod
    def register(cls, name: str, figure_class: type):
        """Register a figure class with a name."""
        cls._registry[name] = figure_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> go.Figure:
        """
        Create a figure by name.
        
        Args:
            name: Figure type name ('eeg', 'fft', 'hilbert', 'brain')
            **kwargs: Arguments passed to figure constructor
            
        Returns:
            Configured Plotly figure
        """
        if name not in cls._registry:
            raise ValueError(f"Unknown figure type: {name}. Available: {list(cls._registry.keys())}")
        return cls._registry[name](**kwargs).create()

