"""
Tests for cleaning/ica.py - ICA decomposition and artifact detection.
"""

import numpy as np

from cleaning.ica import (
    ICAResult, compute_ica, detect_eog_components, detect_ecg_components,
    detect_muscle_components, get_component_properties, apply_ica_exclusion,
    plot_component_topomap, get_component_timeseries
)


class TestICAResult:
    """Tests for ICAResult dataclass."""
    
    def test_empty_result(self):
        """Test empty result."""
        result = ICAResult()
        
        assert result.ica is None
        assert result.n_components == 0
        assert len(result.excluded) == 0
        assert len(result.labels) == 0
    
    def test_all_bad_property(self):
        """Test all_bad property."""
        result = ICAResult(excluded=[0, 2, 5])
        
        assert result.all_bad == [0, 2, 5]
    
    def test_set_label(self):
        """Test setting component label."""
        result = ICAResult()
        result.set_label(0, "EOG")
        
        assert result.labels[0] == "EOG"
    
    def test_get_label_custom(self):
        """Test getting custom label."""
        result = ICAResult(labels={0: "BLINK"})
        
        assert result.get_label(0) == "BLINK"
    
    def test_get_label_auto_eog(self):
        """Test getting auto-detected EOG label."""
        result = ICAResult(eog_indices=[1, 3])
        
        assert result.get_label(1) == "EOG"
        assert result.get_label(3) == "EOG"
    
    def test_get_label_auto_ecg(self):
        """Test getting auto-detected ECG label."""
        result = ICAResult(ecg_indices=[2])
        
        assert result.get_label(2) == "ECG"
    
    def test_get_label_empty(self):
        """Test getting label for unlabeled component."""
        result = ICAResult()
        
        assert result.get_label(0) == ""


class TestComputeICA:
    """Tests for compute_ica function."""
    
    def test_compute_ica_basic(self, synthetic_raw):
        """Test basic ICA computation."""
        result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        
        assert result.ica is not None
        assert result.n_components == 5
    
    def test_compute_ica_default_components(self, synthetic_raw):
        """Test ICA with default component count."""
        result = compute_ica(synthetic_raw, max_iter=100)
        
        assert result.ica is not None
        assert result.n_components > 0
        assert result.n_components <= len(synthetic_raw.ch_names)
    
    def test_compute_ica_methods(self, synthetic_raw):
        """Test different ICA methods."""
        for method in ['fastica']:  # Only test fastica for speed
            result = compute_ica(synthetic_raw, n_components=3, method=method, max_iter=100)
            assert result.ica is not None


class TestDetectEOGComponents:
    """Tests for detect_eog_components function."""
    
    def test_detect_eog_no_ica(self):
        """Test with no ICA computed."""
        result = ICAResult()
        indices, scores = detect_eog_components(result, None)
        
        assert indices == []
        assert len(scores) == 0
    
    def test_detect_eog_with_ica(self, synthetic_raw_with_artifacts):
        """Test EOG detection with computed ICA."""
        ica_result = compute_ica(synthetic_raw_with_artifacts, n_components=5, max_iter=100)
        indices, scores = detect_eog_components(ica_result, synthetic_raw_with_artifacts)
        
        # Should return lists (may or may not find components)
        assert isinstance(indices, list)
    
    def test_updates_ica_result(self, synthetic_raw_with_artifacts):
        """Test that detection updates ICA result."""
        ica_result = compute_ica(synthetic_raw_with_artifacts, n_components=5, max_iter=100)
        detect_eog_components(ica_result, synthetic_raw_with_artifacts)
        
        # Should have updated eog_indices
        assert hasattr(ica_result, 'eog_indices')


class TestDetectECGComponents:
    """Tests for detect_ecg_components function."""
    
    def test_detect_ecg_no_ica(self):
        """Test with no ICA computed."""
        result = ICAResult()
        indices, scores = detect_ecg_components(result, None)
        
        assert indices == []
    
    def test_detect_ecg_with_ica(self, synthetic_raw):
        """Test ECG detection with computed ICA."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        indices, scores = detect_ecg_components(ica_result, synthetic_raw)
        
        # Should return lists (may not find ECG in synthetic data)
        assert isinstance(indices, list)


class TestDetectMuscleComponents:
    """Tests for detect_muscle_components function."""
    
    def test_detect_muscle_no_ica(self):
        """Test with no ICA computed."""
        result = ICAResult()
        indices = detect_muscle_components(result, None)
        
        assert indices == []
    
    def test_detect_muscle_with_ica(self, synthetic_raw):
        """Test muscle detection with computed ICA."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        indices = detect_muscle_components(ica_result, synthetic_raw)
        
        # Should return list (may not find muscle artifacts in synthetic data)
        assert isinstance(indices, list)


class TestApplyICAExclusion:
    """Tests for apply_ica_exclusion function."""
    
    def test_apply_no_ica(self, synthetic_raw):
        """Test with no ICA computed."""
        result = ICAResult()
        raw_clean = apply_ica_exclusion(synthetic_raw, result)
        
        # Should return copy of original
        assert np.allclose(raw_clean.get_data(), synthetic_raw.get_data())
    
    def test_apply_no_exclusions(self, synthetic_raw):
        """Test with ICA but no exclusions."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        ica_result.excluded = []
        
        raw_clean = apply_ica_exclusion(synthetic_raw, ica_result)
        
        # Data should be similar (ICA reconstruction not perfect)
        assert raw_clean is not None
    
    def test_apply_with_exclusions(self, synthetic_raw):
        """Test with ICA and exclusions."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        ica_result.excluded = [0]  # Exclude first component
        
        raw_clean = apply_ica_exclusion(synthetic_raw, ica_result)
        
        assert raw_clean is not None
        # Data should change when excluding component
        assert not np.allclose(raw_clean.get_data(), synthetic_raw.get_data())
    
    def test_updates_excluded_list(self, synthetic_raw):
        """Test that exclusion list is updated."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        
        apply_ica_exclusion(synthetic_raw, ica_result, exclude=[1, 2])
        
        assert ica_result.excluded == [1, 2]


class TestGetComponentProperties:
    """Tests for get_component_properties function."""
    
    def test_no_ica(self):
        """Test with no ICA computed."""
        result = ICAResult()
        props = get_component_properties(result, None, 0)
        
        assert props == {}
    
    def test_get_properties(self, synthetic_raw):
        """Test getting component properties."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        props = get_component_properties(ica_result, synthetic_raw, 0)
        
        assert 'index' in props
        assert 'variance' in props
        assert 'peak_frequency' in props
        assert props['index'] == 0


class TestPlotComponentTopomap:
    """Tests for plot_component_topomap function."""
    
    def test_no_ica(self):
        """Test with no ICA computed."""
        result = ICAResult()
        topo = plot_component_topomap(result, None, 0)
        
        assert len(topo) == 0
    
    def test_get_topomap(self, synthetic_raw):
        """Test getting topomap data."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        topo = plot_component_topomap(ica_result, synthetic_raw, 0)
        
        # Should return array with weights for each channel
        assert len(topo) == len(synthetic_raw.ch_names)


class TestGetComponentTimeseries:
    """Tests for get_component_timeseries function."""
    
    def test_no_ica(self):
        """Test with no ICA computed."""
        result = ICAResult()
        times, data = get_component_timeseries(result, None, 0)
        
        assert len(times) == 0
        assert len(data) == 0
    
    def test_get_timeseries(self, synthetic_raw):
        """Test getting component timeseries."""
        ica_result = compute_ica(synthetic_raw, n_components=5, max_iter=100)
        times, data = get_component_timeseries(ica_result, synthetic_raw, 0, start=0, duration=2)
        
        assert len(times) > 0
        assert len(data) > 0
        assert len(times) == len(data)










