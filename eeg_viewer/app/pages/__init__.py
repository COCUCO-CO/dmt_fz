"""
UI Pages for EEG Viewer.

Each page is a separate module that registers with @ui.page.

Note: pipeline is now a package (pipeline/) instead of a single file.
The old pipeline.py has been renamed to pipeline_old.py for backup.
"""
from . import viewer
from . import pipeline  # Now a package with modular structure
from . import model
from . import analysis

__all__ = ['viewer', 'pipeline', 'model', 'analysis']
