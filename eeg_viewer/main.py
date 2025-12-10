#!/usr/bin/env python3
"""
EEG Viewer - Powerful EEG Processing Interface

This is the main entry point. All functionality is in app/ modules.
"""
from nicegui import ui
import plotly.graph_objects as go

# Re-export everything for backwards compatibility with tests
from app.state import S, PS, MS, AS, State, PipelineState, ModelState, AnalysisState
from app.visualization import make_eeg_fig, make_fft_fig, make_hilbert_fig, make_brain_fig
from app.visualization.styles.css import STYLE
from app.core.signal import apply_notch, apply_bandpass, compute_fft, compute_hilbert

# Re-export from pages
from app.pages.viewer import (
    ELECTRODE_POSITIONS, process_data,
    update_eeg, update_fft, update_hilbert, update_brain, update_all,
    update_eeg2, update_fft2, update_hilbert2, update_brain2,
    nav_start, nav_back, nav_fwd, nav_end, set_win, toggle_play,
    refresh_channels, toggle_ch, select_all_ch, select_10_ch, clear_ch,
    refresh_hilbert_select, refresh_info, gen_epochs, save_epochs
)
from app.pages.model import detect_dataset_type, model_log, update_status_indicator, create_default_config
from app.pages.analysis import analysis_log, load_trained_model, register_activation_hooks, process_sample, compute_latent_pca

# Import pages - each page registers itself with @ui.page
from app.pages import viewer, pipeline, model, analysis
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
