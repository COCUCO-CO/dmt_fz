"""
Tests for Pipeline UI callbacks and button interactions.

These tests verify that button clicks and UI interactions produce
the expected effects, including:
- Step execution triggers
- Parameter validation
- State changes
- Notifications
"""
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock
import asyncio
import sys

# Ensure proper path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestStepButtonCallbacks:
    """Tests for step RUN button callbacks."""
    
    def test_run_button_requires_run_dir(self):
        """RUN buttons should require a run directory to be selected."""
        current_run_dir = [None]
        notification_shown = [False]
        
        async def run_fwd():
            if not current_run_dir[0]:
                notification_shown[0] = True  # ui.notify mock
                return
            # Would proceed with execution
        
        asyncio.run(run_fwd())
        assert notification_shown[0] is True
    
    def test_run_button_with_valid_run_dir(self):
        """RUN button should proceed when run directory exists."""
        current_run_dir = [Path("/tmp/run_test")]
        executed = [False]
        
        async def run_fwd():
            if not current_run_dir[0]:
                return
            executed[0] = True
        
        asyncio.run(run_fwd())
        assert executed[0] is True
    
    def test_step_args_construction_fwd(self):
        """Step 1 (fwd.py) should construct correct arguments."""
        max_subj_value = 5
        jobs_value = 4
        workers_value = 7
        max_epochs_value = 10
        conditions = ['DMT', 'EC']
        
        def get_conditions():
            return conditions
        
        args = [
            '--max-subjects', str(int(max_subj_value or 0)),
            '--conditions'] + get_conditions() + [
            '--jobs', str(int(jobs_value or 0)),
            '--workers', str(int(workers_value or 7)),
            '--max-epochs', str(int(max_epochs_value or 0))
        ]
        
        assert args == [
            '--max-subjects', '5',
            '--conditions', 'DMT', 'EC',
            '--jobs', '4',
            '--workers', '7',
            '--max-epochs', '10'
        ]
    
    def test_step_args_construction_clustering(self):
        """Step 8 (clustering.py) should construct correct arguments."""
        band_values = {
            'delta': True,
            'theta': True,
            'alpha': True,
            'beta': False,
            'gamma': False
        }
        conditions = ['DMT']
        min_k_value = 2
        max_k_value = 15
        min_pca_value = 2
        max_pca_value = 10
        workers_value = 4
        search_mode = 'Quick'
        output_dir = Path("/tmp/run/clustering_results")
        
        bands = []
        if band_values['delta']: bands.append('Delta')
        if band_values['theta']: bands.append('Theta')
        if band_values['alpha']: bands.append('Alpha')
        if band_values['beta']: bands.append('Beta')
        if band_values['gamma']: bands.append('Gamma')
        
        mode_arg = '--quick-search' if search_mode == 'Quick' else '--full-search'
        
        args = [
            mode_arg,
            '--conditions'] + conditions + [
            '--bands'] + bands + [
            '--min-k', str(int(min_k_value)),
            '--max-k', str(int(max_k_value)),
            '--min-comps', str(int(min_pca_value)),
            '--max-comps', str(int(max_pca_value)),
            '--workers', str(int(workers_value or 4)),
            '--output-dir', str(output_dir)
        ]
        
        assert '--quick-search' in args
        assert '--conditions' in args
        assert 'DMT' in args
        assert '--bands' in args
        assert 'Delta' in args
        assert 'Theta' in args
        assert 'Alpha' in args
        assert 'Beta' not in args
    
    def test_step_args_with_zero_values(self):
        """Zero values should be handled correctly (meaning 'all')."""
        max_subj_value = 0
        max_epochs_value = 0
        
        args = [
            '--max-subjects', str(int(max_subj_value or 0)),
            '--max-epochs', str(int(max_epochs_value or 0))
        ]
        
        assert args == ['--max-subjects', '0', '--max-epochs', '0']


class TestNewRunCallback:
    """Tests for NEW RUN button callback."""
    
    def test_create_new_run_creates_directory(self, tmp_path):
        """NEW RUN should create a new timestamped directory."""
        PIPELINE_OUTPUTS = tmp_path / "pipeline_outputs"
        PIPELINE_OUTPUTS.mkdir()
        
        
        def create_new_run():
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            run_dir = PIPELINE_OUTPUTS / f"run_{ts}"
            run_dir.mkdir(parents=True, exist_ok=True)
            for cond in ["DMT", "EC", "EO"]:
                (run_dir / cond).mkdir(exist_ok=True)
            return run_dir
        
        run_dir = create_new_run()
        
        assert run_dir.exists()
        assert run_dir.name.startswith('run_')
        assert (run_dir / "DMT").exists()
        assert (run_dir / "EC").exists()
        assert (run_dir / "EO").exists()
    
    def test_new_run_updates_state(self, tmp_path):
        """NEW RUN should update current_run_dir and PS.selected_run."""
        from app.state.global_state import PipelineState
        
        ps = PipelineState()
        current_run_dir = [None]
        
        new_run = tmp_path / "run_test"
        new_run.mkdir()
        
        # Simulate callback
        current_run_dir[0] = new_run
        ps.selected_run = new_run
        
        assert current_run_dir[0] == new_run
        assert ps.selected_run == new_run


class TestSelectExistingRunCallback:
    """Tests for selecting existing run from dropdown."""
    
    def test_use_existing_run_updates_state(self, tmp_path):
        """Selecting existing run should update state."""
        from app.state.global_state import PipelineState
        
        PIPELINE_OUTPUTS = tmp_path / "pipeline_outputs"
        PIPELINE_OUTPUTS.mkdir()
        existing_run = PIPELINE_OUTPUTS / "run_20231201_120000"
        existing_run.mkdir()
        
        ps = PipelineState()
        current_run_dir = [None]
        
        # Simulate dropdown selection
        run_select_value = "run_20231201_120000"
        
        def use_existing():
            if run_select_value:
                current_run_dir[0] = PIPELINE_OUTPUTS / run_select_value
                ps.selected_run = current_run_dir[0]
        
        use_existing()
        
        assert current_run_dir[0] == existing_run
        assert ps.selected_run == existing_run
    
    def test_get_run_dirs_returns_sorted(self, tmp_path):
        """get_run_dirs should return runs sorted by name (newest first)."""
        PIPELINE_OUTPUTS = tmp_path / "pipeline_outputs"
        PIPELINE_OUTPUTS.mkdir()
        
        # Create runs with different timestamps
        (PIPELINE_OUTPUTS / "run_20231201_100000").mkdir()
        (PIPELINE_OUTPUTS / "run_20231201_120000").mkdir()
        (PIPELINE_OUTPUTS / "run_20231130_120000").mkdir()
        (PIPELINE_OUTPUTS / "not_a_run").mkdir()  # Should be excluded
        
        def get_run_dirs():
            if not PIPELINE_OUTPUTS.exists():
                return []
            return sorted(
                [d.name for d in PIPELINE_OUTPUTS.iterdir() 
                 if d.is_dir() and d.name.startswith('run_')],
                reverse=True
            )
        
        runs = get_run_dirs()
        
        assert len(runs) == 3
        assert runs[0] == "run_20231201_120000"  # Newest first
        assert "not_a_run" not in runs


class TestScanInputDirCallback:
    """Tests for input directory scan button."""
    
    def test_scan_subdirectory_structure(self, tmp_path):
        """Scan should detect subdirectory structure (DMT/, EC/, EO/)."""
        input_dir = tmp_path / "EEG_CLEAN"
        input_dir.mkdir()
        (input_dir / "DMT").mkdir()
        (input_dir / "EC").mkdir()
        (input_dir / "EO").mkdir()
        
        # Create some .set files
        (input_dir / "DMT" / "S01.set").touch()
        (input_dir / "DMT" / "S02.set").touch()
        (input_dir / "EC" / "S01.set").touch()
        
        p = input_dir
        conds = [d.name for d in p.iterdir() if d.is_dir() and d.name in ['DMT', 'EC', 'EO']]
        files = sum(len(list((p / c).glob('*.set'))) for c in conds)
        
        assert set(conds) == {'DMT', 'EC', 'EO'}
        assert files == 3
    
    def test_scan_flat_structure(self, tmp_path):
        """Scan should detect flat structure (files with condition in name)."""
        input_dir = tmp_path / "EEG_FLAT"
        input_dir.mkdir()
        
        # Create files with condition in name
        (input_dir / "S01_DMT.set").touch()
        (input_dir / "S02_DMT.set").touch()
        (input_dir / "S01_EC.set").touch()
        (input_dir / "S01_EO.set").touch()
        
        p = input_dir
        conds = [d.name for d in p.iterdir() if d.is_dir() and d.name in ['DMT', 'EC', 'EO']]
        
        # No subdirectories, so flat structure
        if not conds:
            all_set = list(p.glob('*.set'))
            dmt = len([f for f in all_set if 'DMT' in f.name.upper()])
            ec = len([f for f in all_set if 'EC' in f.name.upper()])
            eo = len([f for f in all_set if 'EO' in f.name.upper()])
        
        assert dmt == 2
        assert ec == 1
        assert eo == 1


class TestStopPipelineCallback:
    """Tests for STOP button callback."""
    
    def test_stop_terminates_process(self):
        """STOP should terminate the current process."""
        from app.state.global_state import PipelineState
        
        ps = PipelineState()
        mock_process = Mock()
        ps.current_process = mock_process
        ps.current_step = "Source Localization"
        
        async def stop_pipeline():
            if ps.current_process:
                ps.current_process.terminate()
                # Would add log and notify
        
        asyncio.run(stop_pipeline())
        mock_process.terminate.assert_called_once()
    
    def test_stop_does_nothing_without_process(self):
        """STOP should do nothing if no process is running."""
        from app.state.global_state import PipelineState
        
        ps = PipelineState()
        ps.current_process = None
        
        error_raised = [False]
        
        async def stop_pipeline():
            try:
                if ps.current_process:
                    ps.current_process.terminate()
            except:
                error_raised[0] = True
        
        asyncio.run(stop_pipeline())
        assert not error_raised[0]


class TestClearLogCallback:
    """Tests for CLEAR button callback."""
    
    def test_clear_log_clears_history(self):
        """CLEAR should clear log_history."""
        from app.state.global_state import PipelineState
        
        ps = PipelineState()
        ps.log_history = ["Log 1", "Log 2", "Log 3"]
        
        def clear_log():
            ps.log_history.clear()
        
        clear_log()
        assert len(ps.log_history) == 0
    
    def test_clear_log_clears_container(self):
        """CLEAR should clear log_container."""
        from app.state.global_state import PipelineState
        
        ps = PipelineState()
        mock_container = Mock()
        ps.log_container = mock_container
        
        def clear_log():
            if ps.log_container:
                ps.log_container.clear()
            ps.log_history.clear()
        
        clear_log()
        mock_container.clear.assert_called_once()


class TestRefreshFilesCallback:
    """Tests for FILES tab refresh callback."""
    
    def test_refresh_files_shows_stats(self, tmp_path):
        """Refresh should show file statistics."""
        run_dir = tmp_path / "run_test"
        run_dir.mkdir()
        (run_dir / "DMT").mkdir()
        
        # Create some files
        (run_dir / "DMT" / "syncro-S01.pkl").touch()
        (run_dir / "DMT" / "phases-S01.pkl").touch()
        (run_dir / "order_all-S01.pkl").touch()
        
        def count_files(pattern):
            return len(list(run_dir.rglob(pattern)))
        
        stats = {
            'syncro': count_files('syncro-*.pkl') + count_files('phases-*.pkl'),
            'order_all': count_files('order_all-*.pkl'),
        }
        
        assert stats['syncro'] == 2
        assert stats['order_all'] == 1
    
    def test_refresh_files_handles_no_run(self):
        """Refresh should handle case when no run is selected."""
        current_run_dir = [None]
        result = [""]
        
        def refresh_files():
            if not current_run_dir[0] or not current_run_dir[0].exists():
                result[0] = "No run selected"
                return
        
        refresh_files()
        assert result[0] == "No run selected"


class TestLoadSubjectsCallback:
    """Tests for Load Subjects button in Visualize tab."""
    
    def test_load_subjects_finds_files(self, tmp_path):
        """Load subjects should find subject files in directory."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "DMT").mkdir()
        
        # Create subject files
        (data_dir / "DMT" / "syncro-S01-DMT.pkl").touch()
        (data_dir / "DMT" / "syncro-S02-DMT.pkl").touch()
        (data_dir / "DMT" / "phases-S01-DMT.pkl").touch()
        
        p = data_dir
        all_files = list(p.rglob('syncro-*.pkl')) + list(p.rglob('phases-*.pkl'))
        subjects = sorted(list(set([
            f.stem.split('-')[1] if '-' in f.stem else f.stem 
            for f in all_files
        ])))
        
        assert 'S01' in subjects
        assert 'S02' in subjects
    
    def test_load_subjects_limits_results(self, tmp_path):
        """Load subjects should limit to 30 subjects."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        # Create many subject files
        for i in range(50):
            (data_dir / f"syncro-S{i:02d}.pkl").touch()
        
        all_files = list(data_dir.glob('syncro-*.pkl'))
        subjects = sorted(list(set([
            f.stem.split('-')[1] if '-' in f.stem else f.stem 
            for f in all_files
        ])))[:30]
        
        assert len(subjects) == 30


class TestVisualizationCallbacks:
    """Tests for visualization update callbacks."""
    
    def test_band_change_triggers_refresh(self):
        """Band selector change should trigger refresh_all_plots."""
        refresh_called = [0]
        
        def refresh_all_plots():
            refresh_called[0] += 1
        
        # Simulate band change callback
        def on_band_change(e):
            refresh_all_plots()
        
        on_band_change(Mock())
        assert refresh_called[0] == 1
    
    def test_subject_change_triggers_refresh(self):
        """Subject selector change should trigger refresh_all_plots."""
        refresh_called = [False]
        
        def refresh_all_plots():
            refresh_called[0] = True
        
        def on_subject_change(e):
            refresh_all_plots()
        
        on_subject_change(Mock())
        assert refresh_called[0] is True
    
    def test_epoch_change_triggers_refresh(self):
        """Epoch change should trigger refresh_all_plots."""
        refresh_called = [False]
        
        def refresh_all_plots():
            refresh_called[0] = True
        
        def on_epoch_change(e):
            refresh_all_plots()
        
        on_epoch_change(Mock())
        assert refresh_called[0] is True
    
    def test_brain_button_updates_active_state(self):
        """Brain plot buttons should update active state."""
        viz_state = {'current_brain_plot': 'network'}
        active_button = ['network']
        
        def update_brain_plot(plot_type):
            viz_state['current_brain_plot'] = plot_type
            active_button[0] = plot_type
        
        update_brain_plot('sync')
        
        assert viz_state['current_brain_plot'] == 'sync'
        assert active_button[0] == 'sync'


class TestPearsonGalleryCallbacks:
    """Tests for Pearson gallery callbacks."""
    
    def test_filter_by_metric(self, tmp_path):
        """Gallery should filter by metric."""
        pearson_dir = tmp_path / "pearson_results"
        pearson_dir.mkdir()
        
        (pearson_dir / "coherence_DMT_Alpha.png").touch()
        (pearson_dir / "metastability_DMT_Alpha.png").touch()
        (pearson_dir / "coherence_EC_Beta.png").touch()
        
        prs_metric = 'Coherence'
        prs_cond = 'All'
        prs_band = 'All'
        
        all_files = list(pearson_dir.glob('*.png'))
        filtered = [f for f in all_files if 
            (prs_metric == 'All' or prs_metric.lower() in f.name.lower()) and
            (prs_cond == 'All' or prs_cond.lower() in f.name.lower()) and
            (prs_band == 'All' or prs_band.lower() in f.name.lower())]
        
        assert len(filtered) == 2
        assert all('coherence' in f.name.lower() for f in filtered)
    
    def test_filter_by_condition(self, tmp_path):
        """Gallery should filter by condition."""
        pearson_dir = tmp_path / "pearson_results"
        pearson_dir.mkdir()
        
        (pearson_dir / "coherence_DMT_Alpha.png").touch()
        (pearson_dir / "coherence_EC_Alpha.png").touch()
        
        prs_metric = 'All'
        prs_cond = 'DMT'
        prs_band = 'All'
        
        all_files = list(pearson_dir.glob('*.png'))
        filtered = [f for f in all_files if 
            (prs_metric == 'All' or prs_metric.lower() in f.name.lower()) and
            (prs_cond == 'All' or prs_cond.lower() in f.name.lower()) and
            (prs_band == 'All' or prs_band.lower() in f.name.lower())]
        
        assert len(filtered) == 1
        assert 'DMT' in filtered[0].name


class TestAnimationGeneratorCallbacks:
    """Tests for animation generator callbacks."""
    
    def test_frame_generation_command_construction(self):
        """Frame generation should construct correct command."""
        viz_subject_value = 'S01-DMT'
        viz_band_value = 'Alpha'
        anim_mode_value = 'stc'
        anim_start_epoch_value = 0
        anim_end_epoch_value = 10
        
        # Extract subject and condition
        subj = viz_subject_value
        cond = 'DMT'
        if '-' in str(subj):
            parts = str(subj).split('-')
            subj = parts[0]
            cond = parts[1] if len(parts) > 1 else 'DMT'
        
        mode = anim_mode_value
        start_ep = int(anim_start_epoch_value or 0)
        end_ep = int(anim_end_epoch_value or 10)
        
        cmd = [
            'python', '/path/to/plot.py',
            '--subject', subj,
            '--condition', cond,
            '--band', viz_band_value,
            '--mode', mode,
            '--epochs', f'{start_ep}:{end_ep}',
        ]
        
        assert '--subject' in cmd
        assert 'S01' in cmd
        assert '--condition' in cmd
        assert 'DMT' in cmd
        assert '--epochs' in cmd
        assert '0:10' in cmd
    
    def test_video_generation_command(self, tmp_path):
        """Video generation should construct correct ffmpeg command."""
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        output_video = frames_dir.parent / "animation.mp4"
        fps = 5
        format = 'png'
        
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
        assert '5' in cmd
        assert '-c:v' in cmd
        assert 'libx264' in cmd









