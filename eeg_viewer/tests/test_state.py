"""
Tests for cleaning/state.py - CleaningState and CleaningStep.
"""

import pytest
from pathlib import Path

from cleaning.state import CleaningState, CleaningStep, OperationRecord


class TestCleaningStep:
    """Tests for CleaningStep enum."""
    
    def test_all_steps_exist(self):
        """Verify all expected steps exist."""
        steps = list(CleaningStep)
        assert len(steps) == 9  # Updated: now includes VISUALIZE
        assert CleaningStep.LOAD in steps
        assert CleaningStep.FILTER in steps
        assert CleaningStep.BAD_CHANNELS in steps
        assert CleaningStep.REREFERENCE in steps
        assert CleaningStep.ICA in steps
        assert CleaningStep.EPOCHS in steps
        assert CleaningStep.REJECT in steps
        assert CleaningStep.VISUALIZE in steps  # New step
        assert CleaningStep.EXPORT in steps
    
    def test_display_name(self):
        """Test display names are set correctly."""
        assert "Load" in CleaningStep.LOAD.display_name
        assert "Filter" in CleaningStep.FILTER.display_name
        assert "Bad" in CleaningStep.BAD_CHANNELS.display_name
    
    def test_short_name(self):
        """Test short names are set correctly."""
        assert CleaningStep.LOAD.short_name == "LOAD"
        assert CleaningStep.FILTER.short_name == "FILTER"
        assert CleaningStep.ICA.short_name == "ICA"
        assert CleaningStep.VISUALIZE.short_name == "VIEW"
        assert CleaningStep.EXPORT.short_name == "EXPORT"


class TestCleaningState:
    """Tests for CleaningState class."""
    
    def test_initial_state(self, cleaning_state):
        """Test initial state after creation."""
        assert cleaning_state.raw is None
        assert not cleaning_state.is_loaded
        assert cleaning_state.current_step == CleaningStep.LOAD
        assert len(cleaning_state.completed_steps) == 0
        assert len(cleaning_state.operations) == 0
        assert cleaning_state.sfreq == 0.0
        assert cleaning_state.n_channels == 0
        assert cleaning_state.duration == 0.0
    
    def test_load_raw(self, cleaning_state, synthetic_raw):
        """Test loading raw data."""
        test_path = Path("/fake/path/test.bdf")
        cleaning_state.load_raw(synthetic_raw, test_path)
        
        assert cleaning_state.is_loaded
        assert cleaning_state.raw is not None
        assert cleaning_state.filename == "test.bdf"
        assert cleaning_state.filepath == test_path
        assert cleaning_state.sfreq == synthetic_raw.info['sfreq']
        assert cleaning_state.n_channels == len(synthetic_raw.ch_names)
        assert CleaningStep.LOAD in cleaning_state.completed_steps
        assert len(cleaning_state.operations) == 1
    
    def test_properties_with_data(self, loaded_state):
        """Test properties return correct values when loaded."""
        assert loaded_state.is_loaded
        assert loaded_state.sfreq > 0
        assert loaded_state.n_channels > 0
        assert loaded_state.duration > 0
        assert len(loaded_state.ch_names) > 0
    
    def test_update_raw(self, loaded_state):
        """Test updating raw data."""
        original_raw = loaded_state.raw.copy()
        new_raw = loaded_state.raw.copy()
        
        loaded_state.update_raw(
            new_raw,
            'test_operation',
            CleaningStep.FILTER,
            {'param': 'value'},
            'Test description'
        )
        
        assert loaded_state.can_undo
        assert len(loaded_state.operations) == 2  # load + update
        assert loaded_state.operations[-1].operation == 'test_operation'
    
    def test_undo(self, loaded_state):
        """Test undo functionality."""
        # Apply an operation
        original_sfreq = loaded_state.sfreq
        new_raw = loaded_state.raw.copy()
        
        loaded_state.update_raw(
            new_raw,
            'test',
            CleaningStep.FILTER,
            {},
            'Test'
        )
        
        assert loaded_state.can_undo
        
        # Undo
        result = loaded_state.undo()
        assert result == True
        assert loaded_state.sfreq == original_sfreq
    
    def test_undo_empty(self, cleaning_state):
        """Test undo when nothing to undo."""
        assert not cleaning_state.can_undo
        result = cleaning_state.undo()
        assert result == False
    
    def test_reset_to_original(self, loaded_state):
        """Test reset to original data."""
        # Apply some operations
        loaded_state.bad_channels = ['Fp1', 'Fp2']
        loaded_state.ica_excluded = [0, 1]
        
        loaded_state.update_raw(
            loaded_state.raw.copy(),
            'test',
            CleaningStep.FILTER,
            {},
            'Test'
        )
        
        # Reset
        loaded_state.reset_to_original()
        
        assert loaded_state.is_loaded
        assert len(loaded_state.bad_channels) == 0
        assert len(loaded_state.ica_excluded) == 0
        assert loaded_state.current_step == CleaningStep.LOAD
        assert len(loaded_state.completed_steps) == 1
    
    def test_step_navigation(self, loaded_state):
        """Test step navigation."""
        loaded_state.go_to_step(CleaningStep.ICA)
        assert loaded_state.current_step == CleaningStep.ICA
        
        loaded_state.go_to_step(CleaningStep.FILTER)
        assert loaded_state.current_step == CleaningStep.FILTER
    
    def test_complete_step(self, loaded_state):
        """Test completing a step."""
        loaded_state.go_to_step(CleaningStep.FILTER)
        loaded_state.complete_step(CleaningStep.FILTER)
        
        assert CleaningStep.FILTER in loaded_state.completed_steps
        assert loaded_state.current_step == CleaningStep.BAD_CHANNELS  # Auto-advance
    
    def test_preprocessing_summary(self, loaded_state):
        """Test getting preprocessing summary."""
        loaded_state.bad_channels = ['Fp1']
        loaded_state.reference_type = "Average"
        
        summary = loaded_state.get_preprocessing_summary()
        
        assert 'filename' in summary
        assert 'bad_channels' in summary
        assert 'operations' in summary
        assert summary['bad_channels'] == ['Fp1']
        assert summary['reference_type'] == "Average"


class TestOperationRecord:
    """Tests for OperationRecord class."""
    
    def test_creation(self):
        """Test creating an operation record."""
        record = OperationRecord(
            step=CleaningStep.FILTER,
            operation='apply_filter',
            parameters={'hp': 0.1, 'lp': 45},
            description='Filter applied'
        )
        
        assert record.step == CleaningStep.FILTER
        assert record.operation == 'apply_filter'
        assert record.parameters['hp'] == 0.1
        assert record.timestamp is not None
    
    def test_to_dict(self):
        """Test converting to dictionary."""
        record = OperationRecord(
            step=CleaningStep.FILTER,
            operation='test',
            parameters={'key': 'value'},
            description='Test'
        )
        
        d = record.to_dict()
        
        assert d['step'] == 'FILTER'
        assert d['operation'] == 'test'
        assert d['parameters'] == {'key': 'value'}
        assert 'timestamp' in d

