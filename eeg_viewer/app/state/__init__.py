"""
State management module for EEG Viewer.

Provides reactive state classes with Observer pattern support,
and global state instances used by the application.
"""
from .base import BaseState, StateHolder
from .viewer_state import ViewerState
from .global_state import (
    State, 
    PipelineState, 
    ModelState, 
    AnalysisState,
    DebugState,
    S, PS, MS, AS, DS,  # Global instances
)

# Aliases for backwards compatibility with tests
# The tests expect certain attributes that are in global_state versions

__all__ = [
    # Base classes
    'BaseState',
    'StateHolder',
    'ViewerState',
    # State classes (from global_state, used by main.py)
    'State',
    'PipelineState',
    'ModelState',
    'AnalysisState',
    'DebugState',
    # Global instances
    'S', 'PS', 'MS', 'AS', 'DS',
]

