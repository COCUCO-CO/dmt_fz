"""
EEG Viewer Application Package.

Modular architecture following SOLID principles.

Modules:
- state: Reactive state management with Observer pattern
- core: Business logic (signal processing, etc.)
- visualization: Plotly figures and UI components
- pages: NiceGUI page definitions
- utils: Utility functions and constants
"""
from . import state
from . import core

__all__ = ['state', 'core']

