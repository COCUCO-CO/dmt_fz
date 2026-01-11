"""
Tests for cleaning/rereferencing.py - Re-referencing functions.
"""

import pytest
import numpy as np

from cleaning.rereferencing import (
    ReferenceType, get_available_references, find_mastoid_channels,
    apply_average_reference, apply_single_reference,
    apply_reference, get_current_reference
)


class TestReferenceType:
    """Tests for ReferenceType enum."""
    
    def test_all_types_exist(self):
        """Verify all reference types exist."""
        types = list(ReferenceType)
        assert ReferenceType.AVERAGE in types
        assert ReferenceType.LINKED_MASTOIDS in types
        assert ReferenceType.SINGLE_ELECTRODE in types
        assert ReferenceType.REST in types
        assert ReferenceType.ORIGINAL in types
    
    def test_display_names(self):
        """Test display names are set."""
        assert "Average" in ReferenceType.AVERAGE.display_name
        assert "Mastoid" in ReferenceType.LINKED_MASTOIDS.display_name
    
    def test_descriptions(self):
        """Test descriptions are set."""
        assert len(ReferenceType.AVERAGE.description) > 0
        assert "average" in ReferenceType.AVERAGE.description.lower()


class TestGetAvailableReferences:
    """Tests for get_available_references function."""
    
    def test_average_always_available(self):
        """Test AVERAGE is always available."""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        available = get_available_references(ch_names)
        
        assert ReferenceType.AVERAGE in available
    
    def test_original_always_available(self):
        """Test ORIGINAL is always available."""
        ch_names = ['Fp1', 'Fp2']
        available = get_available_references(ch_names)
        
        assert ReferenceType.ORIGINAL in available
    
    def test_linked_mastoids_with_a1_a2(self):
        """Test LINKED_MASTOIDS available when A1/A2 present."""
        ch_names = ['Fp1', 'Fp2', 'A1', 'A2']
        available = get_available_references(ch_names)
        
        assert ReferenceType.LINKED_MASTOIDS in available
    
    def test_linked_mastoids_with_m1_m2(self):
        """Test LINKED_MASTOIDS available when M1/M2 present."""
        ch_names = ['Fp1', 'Fp2', 'M1', 'M2']
        available = get_available_references(ch_names)
        
        assert ReferenceType.LINKED_MASTOIDS in available
    
    def test_linked_mastoids_not_available(self):
        """Test LINKED_MASTOIDS not available without mastoids."""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        available = get_available_references(ch_names)
        
        assert ReferenceType.LINKED_MASTOIDS not in available
    
    def test_single_electrode_available(self):
        """Test SINGLE_ELECTRODE available when channels exist."""
        ch_names = ['Fp1', 'Fp2']
        available = get_available_references(ch_names)
        
        assert ReferenceType.SINGLE_ELECTRODE in available


class TestFindMastoidChannels:
    """Tests for find_mastoid_channels function."""
    
    def test_find_a1_a2(self):
        """Test finding A1/A2 pair."""
        ch_names = ['Fp1', 'A1', 'A2', 'Fp2']
        pair = find_mastoid_channels(ch_names)
        
        assert pair == ('A1', 'A2')
    
    def test_find_m1_m2(self):
        """Test finding M1/M2 pair."""
        ch_names = ['Fp1', 'M1', 'M2', 'Fp2']
        pair = find_mastoid_channels(ch_names)
        
        assert pair == ('M1', 'M2')
    
    def test_find_tp9_tp10(self):
        """Test finding TP9/TP10 pair."""
        ch_names = ['Fp1', 'TP9', 'TP10', 'Fp2']
        pair = find_mastoid_channels(ch_names)
        
        assert pair == ('TP9', 'TP10')
    
    def test_no_mastoids(self):
        """Test when no mastoids present."""
        ch_names = ['Fp1', 'Fp2', 'F3', 'F4']
        pair = find_mastoid_channels(ch_names)
        
        assert pair is None
    
    def test_only_one_mastoid(self):
        """Test when only one mastoid present."""
        ch_names = ['Fp1', 'A1', 'Fp2']  # Only A1, no A2
        pair = find_mastoid_channels(ch_names)
        
        assert pair is None


class TestApplyAverageReference:
    """Tests for apply_average_reference function."""
    
    def test_apply_average_reference(self, synthetic_raw):
        """Test applying average reference."""
        raw_ref = apply_average_reference(synthetic_raw)
        
        assert raw_ref is not None
        assert len(raw_ref.ch_names) == len(synthetic_raw.ch_names)
    
    def test_average_reference_changes_data(self, synthetic_raw):
        """Test that average reference changes the data."""
        original_data = synthetic_raw.get_data()
        raw_ref = apply_average_reference(synthetic_raw)
        ref_data = raw_ref.get_data()
        
        assert not np.allclose(original_data, ref_data)
    
    def test_returns_copy(self, synthetic_raw):
        """Test that original data is not modified."""
        original_data = synthetic_raw.get_data().copy()
        apply_average_reference(synthetic_raw)
        
        assert np.allclose(synthetic_raw.get_data(), original_data)


class TestApplySingleReference:
    """Tests for apply_single_reference function."""
    
    def test_apply_single_reference(self, synthetic_raw):
        """Test applying single electrode reference."""
        raw_ref = apply_single_reference(synthetic_raw, 'Fp1')
        
        assert raw_ref is not None
    
    def test_invalid_channel_raises(self, synthetic_raw):
        """Test that invalid channel raises error."""
        with pytest.raises(ValueError):
            apply_single_reference(synthetic_raw, 'InvalidChannel')


class TestApplyReference:
    """Tests for apply_reference function."""
    
    def test_apply_average(self, synthetic_raw):
        """Test applying average reference via main function."""
        raw_ref, desc = apply_reference(synthetic_raw, ReferenceType.AVERAGE)
        
        assert raw_ref is not None
        assert "Average" in desc
    
    def test_apply_original(self, synthetic_raw):
        """Test keeping original reference."""
        raw_ref, desc = apply_reference(synthetic_raw, ReferenceType.ORIGINAL)
        
        assert raw_ref is not None
        assert "Original" in desc
        # Data should be unchanged (copy of original)
        assert np.allclose(raw_ref.get_data(), synthetic_raw.get_data())
    
    def test_apply_single(self, synthetic_raw):
        """Test applying single electrode reference via main function."""
        raw_ref, desc = apply_reference(
            synthetic_raw,
            ReferenceType.SINGLE_ELECTRODE,
            ref_channels=['Fp1']
        )
        
        assert raw_ref is not None
        assert "Fp1" in desc
    
    def test_single_requires_channels(self, synthetic_raw):
        """Test that SINGLE_ELECTRODE requires ref_channels."""
        with pytest.raises(ValueError):
            apply_reference(synthetic_raw, ReferenceType.SINGLE_ELECTRODE)


class TestGetCurrentReference:
    """Tests for get_current_reference function."""
    
    def test_unknown_reference(self, synthetic_raw):
        """Test default returns unknown."""
        ref = get_current_reference(synthetic_raw)
        
        # Should return some string
        assert len(ref) > 0
    
    def test_after_average_reference(self, synthetic_raw):
        """Test after applying average reference."""
        raw_ref = apply_average_reference(synthetic_raw)
        ref = get_current_reference(raw_ref)
        
        # Should indicate some reference was applied
        assert len(ref) > 0










