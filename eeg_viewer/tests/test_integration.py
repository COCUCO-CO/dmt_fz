"""
Integration tests for the complete EEG cleaning pipeline.

These tests run the full pipeline from raw EEG to cleaned epochs,
simulating a real user workflow.

Uses real EEG data from: /media/storage_hdd/dmt_fz/EEG/DMT/S01-DMT.bdf
Limited to 30 seconds for speed.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import mne

from cleaning.state import CleaningState, CleaningStep
from cleaning.filters import apply_filter_preset, FilterPreset
from cleaning.bad_channels import detect_bad_channels, interpolate_channels
from cleaning.rereferencing import apply_reference, ReferenceType
from cleaning.ica import compute_ica, detect_eog_components, apply_ica_exclusion
from cleaning.epochs import create_epochs, detect_bad_epochs, apply_epoch_rejection, EpochRejectionCriteria
from cleaning.export import create_export_bundle


class TestFullPipeline:
    """
    Integration test running the complete cleaning pipeline.
    
    Pipeline steps:
    1. Load raw EEG (30 seconds)
    2. Apply bandpass filter
    3. Detect and interpolate bad channels
    4. Apply average reference
    5. Compute ICA and remove artifacts
    6. Create epochs
    7. Detect and reject bad epochs
    8. Visualize final results (NEW)
    9. Export cleaned data
    """
    
    @pytest.fixture
    def pipeline_state(self, sample_raw):
        """Initialize state with loaded data."""
        state = CleaningState()
        state.load_raw(sample_raw, Path("/test/S01-DMT.bdf"))
        return state
    
    def test_step1_load(self, pipeline_state):
        """Step 1: Verify data is loaded correctly."""
        assert pipeline_state.is_loaded
        assert pipeline_state.sfreq > 0
        assert pipeline_state.n_channels > 0
        assert pipeline_state.duration > 0
        assert CleaningStep.LOAD in pipeline_state.completed_steps
        print(f"✓ Loaded: {pipeline_state.n_channels} channels, {pipeline_state.sfreq} Hz, {pipeline_state.duration:.1f}s")
    
    def test_step2_filter(self, pipeline_state):
        """Step 2: Apply bandpass filter."""
        original_data = pipeline_state.raw.get_data().copy()
        
        # Apply standard filter
        raw_filtered, params = apply_filter_preset(pipeline_state.raw, FilterPreset.STANDARD)
        
        pipeline_state.update_raw(
            raw_filtered,
            'apply_filter',
            CleaningStep.FILTER,
            params.__dict__,
            'Standard filter applied'
        )
        pipeline_state.complete_step(CleaningStep.FILTER)
        
        # Verify filter was applied
        assert not np.allclose(pipeline_state.raw.get_data(), original_data)
        assert CleaningStep.FILTER in pipeline_state.completed_steps
        print(f"✓ Filter applied: {params.highpass}-{params.lowpass} Hz, notch {params.notch} Hz")
    
    def test_step3_bad_channels(self, pipeline_state, sample_raw):
        """Step 3: Detect and handle bad channels."""
        # First apply filter (step 2)
        raw_filtered, _ = apply_filter_preset(sample_raw, FilterPreset.STANDARD)
        pipeline_state.update_raw(raw_filtered, 'filter', CleaningStep.FILTER, {}, 'Filtered')
        
        # Detect bad channels
        result = detect_bad_channels(pipeline_state.raw, std_threshold=4.0)
        pipeline_state.bad_channels = result.all_bad
        pipeline_state.bad_channels_auto = result.all_bad.copy()
        
        print(f"✓ Bad channel detection: {len(result.all_bad)} channels flagged")
        print(f"  - Flat: {result.flat}")
        print(f"  - Noisy: {result.noisy}")
        print(f"  - Uncorrelated: {result.uncorrelated}")
        
        # Interpolate if any bad channels found
        if pipeline_state.bad_channels:
            raw_interp = interpolate_channels(pipeline_state.raw, pipeline_state.bad_channels)
            pipeline_state.interpolated_channels = pipeline_state.bad_channels.copy()
            pipeline_state.update_raw(
                raw_interp,
                'interpolate',
                CleaningStep.BAD_CHANNELS,
                {'channels': pipeline_state.bad_channels},
                f'Interpolated {len(pipeline_state.bad_channels)} channels'
            )
            pipeline_state.bad_channels = []
            print(f"✓ Interpolated: {pipeline_state.interpolated_channels}")
        
        pipeline_state.complete_step(CleaningStep.BAD_CHANNELS)
        assert CleaningStep.BAD_CHANNELS in pipeline_state.completed_steps
    
    def test_step4_rereference(self, pipeline_state, sample_raw):
        """Step 4: Apply average reference."""
        # Setup: filter first
        raw_filtered, _ = apply_filter_preset(sample_raw, FilterPreset.STANDARD)
        pipeline_state.update_raw(raw_filtered, 'filter', CleaningStep.FILTER, {}, 'Filtered')
        
        # Apply average reference
        raw_ref, desc = apply_reference(pipeline_state.raw, ReferenceType.AVERAGE)
        
        pipeline_state.update_raw(
            raw_ref,
            'rereference',
            CleaningStep.REREFERENCE,
            {'type': 'average'},
            desc
        )
        pipeline_state.reference_type = desc
        pipeline_state.complete_step(CleaningStep.REREFERENCE)
        
        assert CleaningStep.REREFERENCE in pipeline_state.completed_steps
        print(f"✓ Re-reference applied: {desc}")
    
    def test_step5_ica(self, pipeline_state, sample_raw):
        """Step 5: Compute ICA and detect artifacts."""
        # Setup: filter and rereference
        raw_filtered, _ = apply_filter_preset(sample_raw, FilterPreset.STANDARD)
        raw_ref, _ = apply_reference(raw_filtered, ReferenceType.AVERAGE)
        pipeline_state.update_raw(raw_ref, 'prep', CleaningStep.REREFERENCE, {}, 'Prepared')
        
        # Compute ICA (limited components for speed)
        n_components = min(10, pipeline_state.n_channels - 1)
        ica_result = compute_ica(
            pipeline_state.raw,
            n_components=n_components,
            method='fastica',
            max_iter=200
        )
        
        print(f"✓ ICA computed: {ica_result.n_components} components")
        
        # Detect EOG artifacts
        eog_indices, _ = detect_eog_components(ica_result, pipeline_state.raw)
        print(f"✓ EOG detection: {eog_indices}")
        
        # Mark detected components for exclusion
        for idx in eog_indices:
            if idx not in ica_result.excluded:
                ica_result.excluded.append(idx)
        
        # Apply ICA if any components to exclude
        if ica_result.excluded:
            raw_clean = apply_ica_exclusion(pipeline_state.raw, ica_result)
            pipeline_state.update_raw(
                raw_clean,
                'apply_ica',
                CleaningStep.ICA,
                {'excluded': ica_result.excluded},
                f'Excluded {len(ica_result.excluded)} components'
            )
            pipeline_state.ica_excluded = ica_result.excluded.copy()
            print(f"✓ ICA applied: excluded components {ica_result.excluded}")
        
        pipeline_state.complete_step(CleaningStep.ICA)
        assert CleaningStep.ICA in pipeline_state.completed_steps
    
    def test_step6_epochs(self, pipeline_state, sample_raw):
        """Step 6: Create epochs."""
        # Setup: minimal preprocessing
        raw_filtered, _ = apply_filter_preset(sample_raw, FilterPreset.STANDARD)
        pipeline_state.update_raw(raw_filtered, 'prep', CleaningStep.FILTER, {}, 'Prepared')
        
        # Create epochs
        epoch_result = create_epochs(
            pipeline_state.raw,
            duration=2.0,
            overlap=0.0
        )
        
        pipeline_state.epoch_duration = 2.0
        pipeline_state.epochs_total = epoch_result.n_total
        pipeline_state.complete_step(CleaningStep.EPOCHS)
        
        assert epoch_result.n_total > 0
        assert epoch_result.epochs is not None
        assert CleaningStep.EPOCHS in pipeline_state.completed_steps
        print(f"✓ Epochs created: {epoch_result.n_total} epochs of 2.0s")
        
        return epoch_result
    
    def test_step7_rejection(self, pipeline_state, sample_raw):
        """Step 7: Detect and reject bad epochs."""
        # Setup
        raw_filtered, _ = apply_filter_preset(sample_raw, FilterPreset.STANDARD)
        pipeline_state.update_raw(raw_filtered, 'prep', CleaningStep.FILTER, {}, 'Prepared')
        
        # Create epochs
        epoch_result = create_epochs(pipeline_state.raw, duration=2.0)
        
        # Detect bad epochs
        criteria = EpochRejectionCriteria.default()
        epoch_result = detect_bad_epochs(epoch_result, criteria)
        
        print(f"✓ Epoch rejection detection:")
        print(f"  - Total: {epoch_result.n_total}")
        print(f"  - Good: {epoch_result.n_good}")
        print(f"  - Rejected: {epoch_result.n_rejected}")
        print(f"  - Rate: {epoch_result.rejection_rate:.1f}%")
        
        # Apply rejection
        if epoch_result.n_rejected > 0:
            epoch_result = apply_epoch_rejection(epoch_result)
            print(f"✓ Rejection applied: {len(epoch_result.epochs)} epochs remaining")
        
        pipeline_state.epochs = epoch_result.epochs
        pipeline_state.epochs_rejected = epoch_result.rejected_indices
        pipeline_state.rejection_criteria = criteria.to_dict()
        pipeline_state.complete_step(CleaningStep.REJECT)
        
        assert CleaningStep.REJECT in pipeline_state.completed_steps
        return epoch_result
    
    def test_step8_visualize(self, pipeline_state, sample_raw):
        """Step 8: Visualize final results."""
        # Setup: run minimal pipeline
        raw_filtered, _ = apply_filter_preset(sample_raw, FilterPreset.STANDARD)
        pipeline_state.update_raw(raw_filtered, 'filter', CleaningStep.FILTER, {}, 'Filtered')
        
        epoch_result = create_epochs(pipeline_state.raw, duration=2.0)
        pipeline_state.epochs = epoch_result.epochs
        pipeline_state.epochs_total = epoch_result.n_total
        
        # Visualize step - just get the preprocessing summary
        summary = pipeline_state.get_preprocessing_summary()
        
        assert 'filename' in summary
        assert 'operations' in summary
        print(f"✓ Visualization step: {len(summary['operations'])} operations applied")
        
        pipeline_state.complete_step(CleaningStep.VISUALIZE)
        assert CleaningStep.VISUALIZE in pipeline_state.completed_steps
    
    def test_step9_export(self, pipeline_state, sample_raw):
        """Step 9: Export cleaned data."""
        # Setup: run minimal pipeline
        raw_filtered, _ = apply_filter_preset(sample_raw, FilterPreset.STANDARD)
        pipeline_state.update_raw(raw_filtered, 'filter', CleaningStep.FILTER, {}, 'Filtered')
        
        epoch_result = create_epochs(pipeline_state.raw, duration=2.0)
        pipeline_state.epochs = epoch_result.epochs
        pipeline_state.epochs_total = epoch_result.n_total
        
        # Export
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = create_export_bundle(
                pipeline_state,
                Path(tmpdir),
                include_raw=True,
                include_epochs=True
            )
            
            print(f"✓ Export completed:")
            for name, path in outputs.items():
                size = path.stat().st_size / 1024  # KB
                print(f"  - {name}: {path.name} ({size:.1f} KB)")
            
            # Verify files exist
            assert 'raw' in outputs
            assert 'log' in outputs
            assert outputs['raw'].exists()
            assert outputs['log'].exists()
        
        pipeline_state.complete_step(CleaningStep.EXPORT)
        assert CleaningStep.EXPORT in pipeline_state.completed_steps
    
    def test_full_pipeline_end_to_end(self, sample_raw):
        """
        Complete end-to-end pipeline test.
        
        Runs all steps in sequence simulating a real user workflow.
        """
        print("\n" + "="*60)
        print("FULL PIPELINE TEST - END TO END")
        print("="*60)
        
        # Initialize
        state = CleaningState()
        state.load_raw(sample_raw, Path("/test/S01-DMT.bdf"))
        print(f"\n[1/9] LOAD: {state.n_channels} ch, {state.sfreq} Hz, {state.duration:.1f}s")
        
        # Step 2: Filter
        raw_filtered, params = apply_filter_preset(state.raw, FilterPreset.STANDARD)
        state.update_raw(raw_filtered, 'filter', CleaningStep.FILTER, params.__dict__, 'Filtered')
        state.complete_step(CleaningStep.FILTER)
        print(f"[2/9] FILTER: {params.highpass}-{params.lowpass} Hz")
        
        # Step 3: Bad channels
        bad_result = detect_bad_channels(state.raw, std_threshold=4.0)
        state.bad_channels = bad_result.all_bad
        if state.bad_channels:
            raw_interp = interpolate_channels(state.raw, state.bad_channels)
            state.interpolated_channels = state.bad_channels.copy()
            state.update_raw(raw_interp, 'interp', CleaningStep.BAD_CHANNELS, {}, 'Interpolated')
            state.bad_channels = []
        state.complete_step(CleaningStep.BAD_CHANNELS)
        print(f"[3/9] BAD_CH: {len(state.interpolated_channels)} interpolated")
        
        # Step 4: Re-reference
        raw_ref, ref_desc = apply_reference(state.raw, ReferenceType.AVERAGE)
        state.update_raw(raw_ref, 'reref', CleaningStep.REREFERENCE, {}, ref_desc)
        state.reference_type = ref_desc
        state.complete_step(CleaningStep.REREFERENCE)
        print(f"[4/9] RE-REF: {ref_desc}")
        
        # Step 5: ICA
        n_comp = min(8, state.n_channels - 1)
        ica_result = compute_ica(state.raw, n_components=n_comp, max_iter=150)
        eog_idx, _ = detect_eog_components(ica_result, state.raw)
        for idx in eog_idx:
            ica_result.excluded.append(idx)
        if ica_result.excluded:
            raw_ica = apply_ica_exclusion(state.raw, ica_result)
            state.update_raw(raw_ica, 'ica', CleaningStep.ICA, {}, 'ICA applied')
            state.ica_excluded = ica_result.excluded
        state.complete_step(CleaningStep.ICA)
        print(f"[5/9] ICA: {ica_result.n_components} comp, {len(ica_result.excluded)} excluded")
        
        # Step 6: Epochs
        epoch_result = create_epochs(state.raw, duration=2.0, overlap=0.0)
        state.epochs_total = epoch_result.n_total
        state.epoch_duration = 2.0
        state.complete_step(CleaningStep.EPOCHS)
        print(f"[6/9] EPOCHS: {epoch_result.n_total} epochs")
        
        # Step 7: Rejection
        # Use lenient criteria to avoid rejecting all epochs in short real data
        criteria = EpochRejectionCriteria.lenient()
        epoch_result = detect_bad_epochs(epoch_result, criteria)
        # Only apply rejection if it leaves some epochs
        if epoch_result.n_rejected > 0 and epoch_result.n_good > 0:
            epoch_result = apply_epoch_rejection(epoch_result)
        state.epochs = epoch_result.epochs
        state.epochs_rejected = epoch_result.rejected_indices
        state.complete_step(CleaningStep.REJECT)
        print(f"[7/9] REJECT: {epoch_result.n_good}/{epoch_result.n_total} good ({epoch_result.rejection_rate:.1f}% rejected)")
        
        # Step 8: Visualize
        summary = state.get_preprocessing_summary()
        state.complete_step(CleaningStep.VISUALIZE)
        print(f"[8/9] VISUALIZE: {len(summary['operations'])} operations logged")
        
        # Step 9: Export
        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = create_export_bundle(state, Path(tmpdir))
            state.complete_step(CleaningStep.EXPORT)
            print(f"[9/9] EXPORT: {len(outputs)} files")
        
        # Final verification
        print("\n" + "-"*60)
        print("PIPELINE SUMMARY:")
        summary = state.get_preprocessing_summary()
        print(f"  - Operations applied: {len(summary['operations'])}")
        print(f"  - Final channels: {summary['current_channels']}")
        print(f"  - Epochs: {summary['epochs_total']} total, {summary['epochs_rejected']} rejected")
        print(f"  - All steps completed: {len(state.completed_steps) == len(CleaningStep)}")
        print("="*60 + "\n")
        
        # Assert all steps completed
        assert len(state.completed_steps) == len(CleaningStep)
        assert state.epochs is not None
        assert len(state.epochs) > 0


class TestPipelineWithRealData:
    """
    Test pipeline specifically with real EEG data.
    
    Marked as slow - can be skipped with: pytest -m "not slow"
    """
    
    @pytest.mark.slow
    def test_real_eeg_pipeline(self, sample_raw):
        """Test with actual EEG data from S01-DMT.bdf."""
        state = CleaningState()
        state.load_raw(sample_raw, Path("/test/S01-DMT.bdf"))
        
        # Quick pipeline
        raw_filtered, _ = apply_filter_preset(state.raw, FilterPreset.STANDARD)
        raw_ref, _ = apply_reference(raw_filtered, ReferenceType.AVERAGE)
        
        epochs = create_epochs(raw_ref, duration=2.0)
        epochs = detect_bad_epochs(epochs, EpochRejectionCriteria.default())
        
        assert epochs.n_total > 0
        assert epochs.epochs is not None
        
        print(f"\nReal data pipeline:")
        print(f"  - Sample rate: {sample_raw.info['sfreq']} Hz")
        print(f"  - Channels: {len(sample_raw.ch_names)}")
        print(f"  - Duration: {sample_raw.n_times / sample_raw.info['sfreq']:.1f}s")
        print(f"  - Epochs created: {epochs.n_total}")
        print(f"  - Good epochs: {epochs.n_good}")

