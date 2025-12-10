"""
UI Pages for EEG Viewer.

Each page is a separate module that registers with @ui.page.
"""
from . import viewer
from . import pipeline
from . import model
from . import analysis

__all__ = ['viewer', 'pipeline', 'model', 'analysis']
