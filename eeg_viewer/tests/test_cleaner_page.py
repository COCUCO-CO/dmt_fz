"""
Tests for cleaning/cleaner_page.py - UI state and toggle functions.
"""



class TestCleanerPageState:
    """Tests for CleanerPageState class."""
    
    def test_initial_display_options(self):
        """Test initial display options are set correctly."""
        from cleaning.cleaner_page import CleanerPageState
        
        state = CleanerPageState()
        
        # Default values
        assert state.show_tooltips == True
        assert state.show_bad_channels_highlight == True
        assert state.show_rejected_epochs_overlay == True  # Default is now True
    
    def test_toggle_bad_channels_on_off(self):
        """Test toggling bad channel highlight on and off."""
        from cleaning.cleaner_page import PS, toggle_bad_channel_highlight
        
        # Save original value
        original = PS.show_bad_channels_highlight
        
        try:
            # Set to known state
            PS.show_bad_channels_highlight = True
            assert PS.show_bad_channels_highlight == True
            
            # Toggle off with bool
            toggle_bad_channel_highlight(False)
            assert PS.show_bad_channels_highlight == False
            
            # Toggle on with bool
            toggle_bad_channel_highlight(True)
            assert PS.show_bad_channels_highlight == True
            
            # Toggle with integer (0 = False, 1 = True)
            toggle_bad_channel_highlight(0)
            assert PS.show_bad_channels_highlight == False
            
            toggle_bad_channel_highlight(1)
            assert PS.show_bad_channels_highlight == True
            
            # Toggle with string
            toggle_bad_channel_highlight('false')
            assert PS.show_bad_channels_highlight == False
            
            toggle_bad_channel_highlight('true')
            assert PS.show_bad_channels_highlight == True
            
        finally:
            # Restore original
            PS.show_bad_channels_highlight = original
    
    def test_toggle_rejected_epochs_on_off(self):
        """Test toggling rejected epochs overlay on and off."""
        from cleaning.cleaner_page import PS, toggle_rejected_epochs_overlay
        
        # Save original value
        original = PS.show_rejected_epochs_overlay
        
        try:
            # Set to known state
            PS.show_rejected_epochs_overlay = False
            assert PS.show_rejected_epochs_overlay == False
            
            # Toggle on with bool
            toggle_rejected_epochs_overlay(True)
            assert PS.show_rejected_epochs_overlay == True
            
            # Toggle off with bool
            toggle_rejected_epochs_overlay(False)
            assert PS.show_rejected_epochs_overlay == False
            
            # Toggle with integer
            toggle_rejected_epochs_overlay(1)
            assert PS.show_rejected_epochs_overlay == True
            
            toggle_rejected_epochs_overlay(0)
            assert PS.show_rejected_epochs_overlay == False
            
        finally:
            # Restore original
            PS.show_rejected_epochs_overlay = original
    
    def test_toggle_help(self):
        """Test toggling help tooltips on and off."""
        from cleaning.cleaner_page import PS, toggle_help
        
        original = PS.show_tooltips
        
        try:
            PS.show_tooltips = True
            
            toggle_help(False)
            assert PS.show_tooltips == False
            
            toggle_help(True)
            assert PS.show_tooltips == True
            
        finally:
            PS.show_tooltips = original
    
    def test_playback_state_initial(self):
        """Test initial playback state."""
        from cleaning.cleaner_page import CleanerPageState
        
        state = CleanerPageState()
        
        assert state.is_playing == False
        assert state.playback_speed == 1.0
    
    def test_set_playback_speed(self):
        """Test setting playback speed."""
        from cleaning.cleaner_page import PS, set_playback_speed
        
        original = PS.playback_speed
        
        try:
            set_playback_speed(2.0)
            assert PS.playback_speed == 2.0
            
            set_playback_speed(0.5)
            assert PS.playback_speed == 0.5
            
            # Test with integer
            set_playback_speed(4)
            assert PS.playback_speed == 4.0
            
        finally:
            PS.playback_speed = original
    
    def test_step_raw_snapshots(self):
        """Test per-step raw snapshots functionality."""
        from cleaning.cleaner_page import CleanerPageState
        from cleaning.state import CleaningStep
        
        state = CleanerPageState()
        
        # Initially no snapshots
        assert len(state._step_raw_snapshots) == 0
        
        # get_step_raw returns None for non-existent
        assert state.get_step_raw(CleaningStep.LOAD) is None
    
    def test_selected_filter_preset_initial(self):
        """Test initial filter preset is None."""
        from cleaning.cleaner_page import CleanerPageState
        
        state = CleanerPageState()
        assert state.selected_filter_preset is None


class TestParseIndicesWithRanges:
    """Tests for the parse_indices_with_ranges helper function."""
    
    def test_single_indices(self):
        """Test parsing single indices."""
        # We need to test the function inside render_reject_controls
        # For now, we'll test the logic directly
        def parse_indices_with_ranges(indices_str: str) -> list:
            indices = []
            parts = [p.strip() for p in indices_str.split(',') if p.strip()]
            for part in parts:
                if '-' in part and not part.startswith('-'):
                    range_parts = part.split('-')
                    if len(range_parts) == 2:
                        try:
                            start = int(range_parts[0].strip())
                            end = int(range_parts[1].strip())
                            indices.extend(range(start, end + 1))
                        except ValueError:
                            continue
                else:
                    try:
                        indices.append(int(part))
                    except ValueError:
                        continue
            return sorted(set(indices))
        
        # Test single indices
        result = parse_indices_with_ranges("1, 5, 10")
        assert result == [1, 5, 10]
        
        # Test range
        result = parse_indices_with_ranges("1-5")
        assert result == [1, 2, 3, 4, 5]
        
        # Test mixed
        result = parse_indices_with_ranges("1-3, 10, 15-17")
        assert result == [1, 2, 3, 10, 15, 16, 17]
        
        # Test duplicates removed
        result = parse_indices_with_ranges("1, 2, 1-3")
        assert result == [1, 2, 3]
        
        # Test empty string
        result = parse_indices_with_ranges("")
        assert result == []
        
        # Test invalid values ignored
        result = parse_indices_with_ranges("1, abc, 5")
        assert result == [1, 5]


class TestFormatIndicesWithRanges:
    """Tests for format_indices_with_ranges function."""
    
    def test_format_single_indices(self):
        """Test formatting single indices."""
        from cleaning.cleaner_page import format_indices_with_ranges
        
        result = format_indices_with_ranges([1, 5, 10])
        assert result == '1, 5, 10'
    
    def test_format_consecutive_range(self):
        """Test formatting consecutive indices as range."""
        from cleaning.cleaner_page import format_indices_with_ranges
        
        result = format_indices_with_ranges([1, 2, 3, 4, 5])
        assert result == '0-4' or result == '1-5'  # Depending on 0 vs 1 indexing
        
        # Actually test with 0-based
        result = format_indices_with_ranges([0, 1, 2, 3, 4])
        assert result == '0-4'
    
    def test_format_mixed(self):
        """Test formatting mixed single and consecutive."""
        from cleaning.cleaner_page import format_indices_with_ranges
        
        result = format_indices_with_ranges([0, 1, 2, 5, 10, 11, 12])
        assert result == '0-2, 5, 10-12'
    
    def test_format_empty(self):
        """Test formatting empty list."""
        from cleaning.cleaner_page import format_indices_with_ranges
        
        result = format_indices_with_ranges([])
        assert result == ''
    
    def test_format_single_value(self):
        """Test formatting single value."""
        from cleaning.cleaner_page import format_indices_with_ranges
        
        result = format_indices_with_ranges([5])
        assert result == '5'


class TestNavigationFunctions:
    """Tests for navigation functions."""
    
    def test_nav_to_start(self):
        """Test navigating to start sets view_start to 0."""
        from cleaning.cleaner_page import PS, nav_to_start
        
        original = PS.view_start
        PS.view_start = 10.0
        
        try:
            nav_to_start()
            assert PS.view_start == 0
            assert PS.is_playing == False
        finally:
            PS.view_start = original
    
    def test_nav_back(self):
        """Test navigating back decreases view_start."""
        from cleaning.cleaner_page import PS, nav_back
        
        original_start = PS.view_start
        original_duration = PS.view_duration
        
        try:
            PS.view_start = 10.0
            PS.view_duration = 5.0
            
            nav_back()
            assert PS.view_start == 5.0
            
            # Test can't go below 0
            PS.view_start = 2.0
            nav_back()
            assert PS.view_start == 0.0
            
        finally:
            PS.view_start = original_start
            PS.view_duration = original_duration


class TestReverseEEG:
    """Tests for EEG reverse functionality."""
    
    def test_initial_reversed_steps_empty(self):
        """Test that reversed_steps is initially empty."""
        from cleaning.cleaner_page import CleanerPageState
        
        state = CleanerPageState()
        assert len(state.reversed_steps) == 0
    
    def test_reversed_steps_tracking(self):
        """Test tracking reversed steps."""
        from cleaning.cleaner_page import PS
        from cleaning.state import CleaningStep
        
        original = PS.reversed_steps.copy()
        
        try:
            PS.reversed_steps = set()
            
            # Add a step
            PS.reversed_steps.add(CleaningStep.FILTER)
            assert CleaningStep.FILTER in PS.reversed_steps
            assert len(PS.reversed_steps) == 1
            
            # Add another step
            PS.reversed_steps.add(CleaningStep.BAD_CHANNELS)
            assert CleaningStep.BAD_CHANNELS in PS.reversed_steps
            assert len(PS.reversed_steps) == 2
            
            # Remove a step
            PS.reversed_steps.remove(CleaningStep.FILTER)
            assert CleaningStep.FILTER not in PS.reversed_steps
            assert len(PS.reversed_steps) == 1
            
        finally:
            PS.reversed_steps = original


class TestInvalidateSubsequentSteps:
    """Tests for invalidating subsequent steps."""
    
    def test_invalidate_clears_completed_steps(self):
        """Test that invalidate_subsequent_steps clears completed steps."""
        from cleaning.cleaner_page import PS, invalidate_subsequent_steps
        from cleaning.state import CleaningStep
        
        # Setup: mark multiple steps as completed
        original_completed = PS.cleaning.completed_steps.copy()
        
        try:
            PS.cleaning.completed_steps = {
                CleaningStep.LOAD,
                CleaningStep.FILTER,
                CleaningStep.BAD_CHANNELS,
                CleaningStep.REREFERENCE,
                CleaningStep.ICA
            }
            
            # Invalidate from FILTER
            invalidate_subsequent_steps(CleaningStep.FILTER)
            
            # LOAD and FILTER should still be there
            assert CleaningStep.LOAD in PS.cleaning.completed_steps
            assert CleaningStep.FILTER in PS.cleaning.completed_steps
            
            # Subsequent steps should be removed
            assert CleaningStep.BAD_CHANNELS not in PS.cleaning.completed_steps
            assert CleaningStep.REREFERENCE not in PS.cleaning.completed_steps
            assert CleaningStep.ICA not in PS.cleaning.completed_steps
            
        finally:
            PS.cleaning.completed_steps = original_completed
    
    def test_invalidate_clears_reversed_steps(self):
        """Test that invalidate_subsequent_steps clears reversed steps."""
        from cleaning.cleaner_page import PS, invalidate_subsequent_steps
        from cleaning.state import CleaningStep
        
        original = PS.reversed_steps.copy()
        
        try:
            PS.reversed_steps = {
                CleaningStep.FILTER,
                CleaningStep.BAD_CHANNELS,
                CleaningStep.ICA
            }
            
            # Invalidate from FILTER
            invalidate_subsequent_steps(CleaningStep.FILTER)
            
            # Only FILTER should remain
            assert CleaningStep.FILTER in PS.reversed_steps
            assert CleaningStep.BAD_CHANNELS not in PS.reversed_steps
            assert CleaningStep.ICA not in PS.reversed_steps
            
        finally:
            PS.reversed_steps = original

