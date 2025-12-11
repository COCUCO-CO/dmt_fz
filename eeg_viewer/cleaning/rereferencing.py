"""
EEG Re-Referencing Module

Provides different re-referencing schemes for EEG data.
"""

from enum import Enum
from typing import List, Optional, Tuple
import numpy as np
import mne


class ReferenceType(Enum):
    """Available reference types."""
    AVERAGE = "average"
    LINKED_MASTOIDS = "linked_mastoids"
    SINGLE_ELECTRODE = "single"
    REST = "rest"
    ORIGINAL = "original"
    
    @property
    def display_name(self) -> str:
        names = {
            ReferenceType.AVERAGE: "Average Reference",
            ReferenceType.LINKED_MASTOIDS: "Linked Mastoids (A1+A2)/2",
            ReferenceType.SINGLE_ELECTRODE: "Single Electrode",
            ReferenceType.REST: "REST (Infinity)",
            ReferenceType.ORIGINAL: "Keep Original",
        }
        return names.get(self, self.name)
    
    @property
    def description(self) -> str:
        descriptions = {
            ReferenceType.AVERAGE: "Uses the average of all EEG channels as reference. Standard for source analysis.",
            ReferenceType.LINKED_MASTOIDS: "Uses the average of A1 and A2 (or M1/M2) electrodes. Common in clinical EEG.",
            ReferenceType.SINGLE_ELECTRODE: "Uses a single electrode as reference (e.g., Cz, A1).",
            ReferenceType.REST: "Reference Electrode Standardization Technique. Projects to infinity reference.",
            ReferenceType.ORIGINAL: "Keep the original reference from recording.",
        }
        return descriptions.get(self, "")


def get_available_references(ch_names: List[str]) -> List[ReferenceType]:
    """
    Get available reference types based on available channels.
    
    Args:
        ch_names: List of channel names
        
    Returns:
        List of available ReferenceType options
    """
    available = [ReferenceType.AVERAGE, ReferenceType.ORIGINAL]
    
    # Check for mastoid/earlobe electrodes
    mastoid_pairs = [
        ('A1', 'A2'),
        ('M1', 'M2'),
        ('TP9', 'TP10'),
    ]
    
    for m1, m2 in mastoid_pairs:
        if m1 in ch_names and m2 in ch_names:
            available.append(ReferenceType.LINKED_MASTOIDS)
            break
    
    # Single electrode is always available if we have channels
    if len(ch_names) > 0:
        available.append(ReferenceType.SINGLE_ELECTRODE)
    
    return available


def find_mastoid_channels(ch_names: List[str]) -> Optional[Tuple[str, str]]:
    """
    Find mastoid/earlobe electrode pair in channel list.
    
    Returns:
        Tuple of (left, right) channel names or None if not found
    """
    pairs = [
        ('A1', 'A2'),
        ('M1', 'M2'),
        ('TP9', 'TP10'),
    ]
    
    for left, right in pairs:
        if left in ch_names and right in ch_names:
            return (left, right)
    
    return None


def apply_average_reference(raw: mne.io.Raw,
                            exclude_bads: bool = True) -> mne.io.Raw:
    """
    Apply average reference.
    
    Args:
        raw: MNE Raw object
        exclude_bads: Exclude bad channels from average calculation
        
    Returns:
        Re-referenced Raw object (copy)
    """
    raw_ref = raw.copy()
    
    # MNE's set_eeg_reference handles bad channels automatically
    raw_ref.set_eeg_reference(
        ref_channels='average',
        verbose=False
    )
    
    return raw_ref


def apply_linked_mastoids(raw: mne.io.Raw,
                          left_channel: Optional[str] = None,
                          right_channel: Optional[str] = None) -> mne.io.Raw:
    """
    Apply linked mastoids reference.
    
    The reference is (left + right) / 2.
    
    Args:
        raw: MNE Raw object
        left_channel: Left mastoid channel name (auto-detect if None)
        right_channel: Right mastoid channel name (auto-detect if None)
        
    Returns:
        Re-referenced Raw object (copy)
    """
    raw_ref = raw.copy()
    
    if left_channel is None or right_channel is None:
        pair = find_mastoid_channels(raw.ch_names)
        if pair is None:
            raise ValueError("No mastoid channels found (A1/A2, M1/M2, or TP9/TP10)")
        left_channel, right_channel = pair
    
    # Apply linked mastoids reference
    raw_ref.set_eeg_reference(
        ref_channels=[left_channel, right_channel],
        verbose=False
    )
    
    return raw_ref


def apply_single_reference(raw: mne.io.Raw,
                           ref_channel: str) -> mne.io.Raw:
    """
    Apply single electrode reference.
    
    Args:
        raw: MNE Raw object
        ref_channel: Channel name to use as reference
        
    Returns:
        Re-referenced Raw object (copy)
    """
    if ref_channel not in raw.ch_names:
        raise ValueError(f"Reference channel '{ref_channel}' not found")
    
    raw_ref = raw.copy()
    
    raw_ref.set_eeg_reference(
        ref_channels=[ref_channel],
        verbose=False
    )
    
    return raw_ref


def apply_rest_reference(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply REST (Reference Electrode Standardization Technique) reference.
    
    Projects data to infinity reference using a lead field matrix.
    Requires electrode positions (montage).
    
    Args:
        raw: MNE Raw object with montage set
        
    Returns:
        Re-referenced Raw object (copy)
    """
    raw_ref = raw.copy()
    
    # Ensure montage is set
    if raw_ref.get_montage() is None:
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw_ref.set_montage(montage, on_missing='ignore')
        except Exception as e:
            raise ValueError(f"Cannot apply REST: no montage and auto-set failed: {e}")
    
    # Apply REST reference
    try:
        raw_ref.set_eeg_reference(
            ref_channels='REST',
            verbose=False
        )
    except Exception as e:
        raise ValueError(f"REST reference failed: {e}")
    
    return raw_ref


def apply_reference(raw: mne.io.Raw,
                    ref_type: ReferenceType,
                    ref_channels: Optional[List[str]] = None,
                    exclude_bads: bool = True) -> Tuple[mne.io.Raw, str]:
    """
    Apply the specified reference type.
    
    Args:
        raw: MNE Raw object
        ref_type: Type of reference to apply
        ref_channels: Channels for single electrode reference
        exclude_bads: Exclude bad channels from average
        
    Returns:
        Tuple of (re-referenced Raw, description string)
    """
    if ref_type == ReferenceType.ORIGINAL:
        return raw.copy(), "Original reference (unchanged)"
    
    elif ref_type == ReferenceType.AVERAGE:
        raw_ref = apply_average_reference(raw, exclude_bads)
        return raw_ref, "Average reference"
    
    elif ref_type == ReferenceType.LINKED_MASTOIDS:
        raw_ref = apply_linked_mastoids(raw)
        pair = find_mastoid_channels(raw.ch_names)
        desc = f"Linked mastoids ({pair[0]}+{pair[1]})/2" if pair else "Linked mastoids"
        return raw_ref, desc
    
    elif ref_type == ReferenceType.SINGLE_ELECTRODE:
        if not ref_channels or len(ref_channels) == 0:
            raise ValueError("Single electrode reference requires ref_channels")
        raw_ref = apply_single_reference(raw, ref_channels[0])
        return raw_ref, f"Single reference ({ref_channels[0]})"
    
    elif ref_type == ReferenceType.REST:
        raw_ref = apply_rest_reference(raw)
        return raw_ref, "REST (infinity) reference"
    
    else:
        raise ValueError(f"Unknown reference type: {ref_type}")


def get_current_reference(raw: mne.io.Raw) -> str:
    """
    Get description of current reference.
    
    Args:
        raw: MNE Raw object
        
    Returns:
        String description of current reference
    """
    # Check if custom reference was applied
    if 'custom_ref_applied' in raw.info:
        if raw.info['custom_ref_applied']:
            return "Custom reference applied"
    
    # Check for projection
    if raw.info.get('projs'):
        for proj in raw.info['projs']:
            if 'average' in proj.get('desc', '').lower():
                return "Average reference"
    
    return "Original/Unknown"



