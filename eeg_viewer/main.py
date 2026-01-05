#!/usr/bin/env python3
"""
EEG Viewer - Powerful EEG Processing Interface

This is the main entry point. All functionality is in app/ modules.
"""
from nicegui import ui

# Re-export everything for backwards compatibility with tests
from app.visualization.styles.css import STYLE

# Re-export from pages

# Import pages - each page registers itself with @ui.page
import app.pages.viewer  # noqa: F401 - registers /
import app.pages.pipeline.page  # noqa: F401 - registers /pipeline
import app.pages.model  # noqa: F401 - registers /model
import app.pages.analysis  # noqa: F401 - registers /analysis
from cleaning.cleaner_page import cleaner_page


@ui.page('/cleaner')
def cleaner_page_route():
    """EEG Cleaning Pipeline page."""
    ui.add_head_html(f'<style>{STYLE}</style>')
    cleaner_page()


if __name__ in {"__main__", "__mp_main__"}:
    print("EEG VIEWER - http://localhost:8080")
    print("  - Viewer:   http://localhost:8080/")
    print("  - Cleaner:  http://localhost:8080/cleaner")
    print("  - Pipeline: http://localhost:8080/pipeline")
    print("  - Model:    http://localhost:8080/model")
    print("  - Analysis: http://localhost:8080/analysis")
    ui.run(title='EEG Viewer', port=8080, reload=False, show=False, dark=True)
