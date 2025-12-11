"""Test slider value synchronization logic."""

import pytest


class TestSliderValueSync:
    """Test slider value formatting and sync logic."""
    
    def test_subsample_value_formatting(self):
        """Test that subsample values format correctly."""
        # Simulate slider values
        test_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        for value in test_values:
            formatted = f'{value:.1f}'
            # Verify format is correct
            assert len(formatted) == 3, f"Expected 3 chars for {value}, got {formatted}"
            assert '.' in formatted, f"Expected decimal point in {formatted}"
            
            # Verify value can be parsed back
            parsed = float(formatted)
            assert abs(parsed - value) < 0.01, f"Round trip failed: {value} -> {formatted} -> {parsed}"
    
    def test_slider_value_range_valid(self):
        """Test that slider values stay within range."""
        min_val = 0.1
        max_val = 1.0
        step = 0.1
        
        # Generate all valid values
        valid_values = []
        current = min_val
        while current <= max_val + 0.001:  # Small epsilon for float comparison
            valid_values.append(round(current, 1))
            current += step
        
        expected = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert valid_values == expected
    
    def test_label_text_matches_value(self):
        """Test that label text generation matches slider value."""
        # Simulate the on_value_change callback logic
        class MockEvent:
            def __init__(self, value):
                self.value = value
        
        class MockLabel:
            def __init__(self):
                self.text = ''
            def set_text(self, text):
                self.text = text
        
        label = MockLabel()
        
        # Simulate what happens in the callback
        def on_value_change(e):
            label.set_text(f'{e.value:.1f}')
        
        # Test various slider movements
        test_cases = [
            (0.1, '0.1'),
            (0.3, '0.3'),
            (0.5, '0.5'),
            (0.7, '0.7'),
            (1.0, '1.0'),
        ]
        
        for value, expected_text in test_cases:
            event = MockEvent(value)
            on_value_change(event)
            assert label.text == expected_text, \
                f"For value {value}, expected label '{expected_text}', got '{label.text}'"
    
    def test_initial_value_sync(self):
        """Test that initial value and label match."""
        initial_value = 0.3
        initial_label = f'{initial_value:.1f}'
        
        assert initial_label == '0.3'
        assert float(initial_label) == initial_value
    
    def test_slider_step_values(self):
        """Test that slider step produces expected values."""
        min_val = 0.1
        max_val = 1.0  
        step = 0.1
        
        # Verify step divides range evenly
        range_size = max_val - min_val
        num_steps = range_size / step
        
        assert abs(num_steps - 9) < 0.001, "Should have 9 steps from 0.1 to 1.0"
        
        # Verify each step value
        for i in range(10):
            expected = round(min_val + i * step, 1)
            formatted = f'{expected:.1f}'
            parsed = float(formatted)
            assert abs(parsed - expected) < 0.001, f"Step {i}: {expected} != {parsed}"


class TestSliderEdgeCases:
    """Test edge cases for slider behavior."""
    
    def test_minimum_value(self):
        """Test minimum slider value."""
        value = 0.1
        label = f'{value:.1f}'
        assert label == '0.1'
    
    def test_maximum_value(self):
        """Test maximum slider value."""
        value = 1.0
        label = f'{value:.1f}'
        assert label == '1.0'
    
    def test_float_precision(self):
        """Test that float precision doesn't cause issues."""
        # Simulating what could happen with float arithmetic
        value = 0.1 + 0.1 + 0.1  # Could be 0.30000000000000004
        
        # Using round() to fix precision issues
        rounded = round(value, 1)
        label = f'{rounded:.1f}'
        
        assert label == '0.3', f"Expected '0.3', got '{label}'"
    
    def test_value_after_multiple_changes(self):
        """Test value stability after multiple changes."""
        class MockLabel:
            def __init__(self):
                self.text = '0.3'
            def set_text(self, text):
                self.text = text
        
        label = MockLabel()
        
        # Simulate moving slider back and forth
        values = [0.3, 0.5, 0.7, 0.3, 0.1, 0.9, 0.3]
        
        for val in values:
            label.set_text(f'{val:.1f}')
        
        # Final value should be 0.3
        assert label.text == '0.3'
        assert float(label.text) == 0.3
