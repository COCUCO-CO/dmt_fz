"""
Tests for app/state module.

Tests the BaseState Observer pattern and all state classes.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.state import BaseState, StateHolder, ViewerState, PipelineState, ModelState, AnalysisState


class TestBaseState:
    """Tests for BaseState Observer pattern."""
    
    def test_subscribe_and_notify(self):
        """Test that subscribers receive notifications."""
        state = ViewerState()
        notifications = []
        
        state.subscribe(lambda attr, old, new: notifications.append((attr, old, new)))
        state.view_duration = 10.0
        
        assert len(notifications) == 1
        assert notifications[0] == ('view_duration', 5.0, 10.0)
    
    def test_multiple_subscribers(self):
        """Test multiple subscribers receive notifications."""
        state = ViewerState()
        count1 = [0]
        count2 = [0]
        
        state.subscribe(lambda a, o, n: count1.__setitem__(0, count1[0] + 1))
        state.subscribe(lambda a, o, n: count2.__setitem__(0, count2[0] + 1))
        
        state.view_start = 5.0
        
        assert count1[0] == 1
        assert count2[0] == 1
    
    def test_unsubscribe(self):
        """Test unsubscribe stops notifications."""
        state = ViewerState()
        notifications = []
        
        unsubscribe = state.subscribe(lambda a, o, n: notifications.append(1))
        state.view_duration = 10.0
        assert len(notifications) == 1
        
        unsubscribe()
        state.view_duration = 15.0
        assert len(notifications) == 1  # No new notification
    
    def test_no_notification_if_value_unchanged(self):
        """Test no notification when value doesn't change."""
        state = ViewerState()
        notifications = []
        
        state.subscribe(lambda a, o, n: notifications.append(1))
        state.view_duration = 5.0  # Same as default
        
        assert len(notifications) == 0
    
    def test_batch_update(self):
        """Test batch_update triggers single notification."""
        state = ViewerState()
        notifications = []
        
        state.subscribe(lambda a, o, n: notifications.append((a, n)))
        state.batch_update(view_start=10.0, view_duration=20.0, is_playing=True)
        
        # Should be single batch_update notification
        assert len(notifications) == 1
        assert notifications[0][0] == 'batch_update'
        changes = notifications[0][1]
        assert 'view_start' in changes
        assert 'view_duration' in changes
        assert 'is_playing' in changes
    
    def test_snapshot(self):
        """Test snapshot returns copy of state."""
        state = ViewerState()
        state.view_duration = 15.0
        state.is_playing = True
        
        snap = state.snapshot()
        
        assert snap['view_duration'] == 15.0
        assert snap['is_playing'] == True
        assert 'raw1' not in snap or snap['raw1'] is None


class TestStateHolder:
    """Tests for StateHolder singleton."""
    
    def setup_method(self):
        """Reset singleton before each test."""
        StateHolder.reset_instance()
    
    def test_singleton_pattern(self):
        """Test StateHolder is singleton."""
        h1 = StateHolder()
        h2 = StateHolder()
        assert h1 is h2
    
    def test_register_and_get_state(self):
        """Test registering and retrieving states."""
        holder = StateHolder()
        viewer = ViewerState()
        
        holder.register('viewer', viewer)
        
        assert holder.get('viewer') is viewer
        assert holder.viewer is viewer
    
    def test_attribute_style_access(self):
        """Test attribute-style state access."""
        holder = StateHolder()
        holder.viewer = ViewerState()
        holder.pipeline = PipelineState()
        
        assert isinstance(holder.viewer, ViewerState)
        assert isinstance(holder.pipeline, PipelineState)
    
    def test_get_nonexistent_returns_default(self):
        """Test get() returns default for missing state."""
        holder = StateHolder()
        
        result = holder.get('nonexistent', 'default')
        assert result == 'default'
    
    def test_attribute_error_for_missing(self):
        """Test AttributeError for missing state attribute."""
        holder = StateHolder()
        
        with pytest.raises(AttributeError):
            _ = holder.nonexistent


class TestViewerState:
    """Tests for ViewerState properties and methods."""
    
    def test_default_values(self):
        """Test default values are set correctly."""
        state = ViewerState()
        
        assert state.view_duration == 5.0
        assert state.view_start == 0.0
        assert state.is_playing == False
        assert state.notch_enabled == False
        assert state.bandpass_enabled == False
    
    def test_view_end_property(self):
        """Test view_end computed property."""
        state = ViewerState()
        state.view_start = 10.0
        state.view_duration = 5.0
        
        assert state.view_end == 15.0
    
    def test_has_data_property(self):
        """Test has_data property."""
        state = ViewerState()
        assert state.has_data == False
        
        # Can't easily test with real data, but property works
    
    def test_reset(self):
        """Test reset() restores defaults."""
        state = ViewerState()
        state.view_start = 100.0
        state.view_duration = 50.0
        state.is_playing = True
        
        state.reset()
        
        assert state.view_start == 0.0
        assert state.view_duration == 5.0
        assert state.is_playing == False
    
    def test_get_view_samples(self):
        """Test get_view_samples calculation."""
        state = ViewerState()
        state.view_start = 1.0
        state.view_duration = 2.0
        
        start, end = state.get_view_samples(sfreq=100.0)
        
        assert start == 100  # 1.0 * 100
        assert end == 300    # 3.0 * 100


class TestPipelineState:
    """Tests for PipelineState."""
    
    def test_default_values(self):
        """Test default values."""
        state = PipelineState()
        
        assert state.running == False
        assert state.workers == 7
        assert state.progress == 0
        assert state.current_step == ""
    
    def test_pipeline_parameters(self):
        """Test pipeline parameters exist."""
        state = PipelineState()
        
        assert state.max_subjects == 0
        assert "DMT" in state.conditions
        assert state.min_k == 2
        assert state.max_k == 15


class TestModelState:
    """Tests for ModelState."""
    
    def test_default_values(self):
        """Test default values."""
        state = ModelState()
        
        assert state.training == False
        assert state.model_type == "vae"
        assert state.status == 'idle'
        assert state.dataset_path == ""


class TestAnalysisState:
    """Tests for AnalysisState."""
    
    def test_default_values(self):
        """Test default values."""
        state = AnalysisState()
        
        assert state.device == "cpu"
        assert state.model_type == "graph"
        assert state.model is None
        assert state.dataset is None
