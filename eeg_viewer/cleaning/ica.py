"""
ICA (Independent Component Analysis) Module

Provides ICA decomposition and artifact detection for EEG.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import mne
from mne.preprocessing import ICA


@dataclass
class ICAResult:
    """Results from ICA computation and artifact detection."""
    ica: Optional[ICA] = None
    n_components: int = 0
    
    # Auto-detected components
    eog_indices: List[int] = field(default_factory=list)
    eog_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    ecg_indices: List[int] = field(default_factory=list)
    ecg_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    muscle_indices: List[int] = field(default_factory=list)
    
    # User-excluded components
    excluded: List[int] = field(default_factory=list)
    
    # Labels for components (user-assigned or auto-detected)
    labels: Dict[int, str] = field(default_factory=dict)
    
    @property
    def all_bad(self) -> List[int]:
        """All components marked as bad (excluded)."""
        return sorted(set(self.excluded))
    
    def set_label(self, component: int, label: str):
        """Set label for a component."""
        self.labels[component] = label
    
    def get_label(self, component: int) -> str:
        """Get label for a component."""
        if component in self.labels:
            return self.labels[component]
        if component in self.eog_indices:
            return "EOG"
        if component in self.ecg_indices:
            return "ECG"
        if component in self.muscle_indices:
            return "MUSCLE"
        return ""


def compute_ica(raw: mne.io.Raw,
                n_components: Optional[int] = None,
                method: str = 'fastica',
                max_iter: int = 500,
                random_state: int = 42,
                fit_params: Optional[Dict] = None) -> ICAResult:
    """
    Compute ICA decomposition.
    
    Args:
        raw: MNE Raw object (will be filtered internally for ICA)
        n_components: Number of components (None = n_channels - n_bads)
        method: ICA method ('fastica', 'infomax', 'picard')
        max_iter: Maximum iterations
        random_state: Random seed for reproducibility
        fit_params: Additional parameters for fit
        
    Returns:
        ICAResult with computed ICA
    """
    # Filter for ICA (1 Hz highpass recommended)
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    
    # Determine n_components
    if n_components is None:
        n_eeg = len(mne.pick_types(raw.info, eeg=True, exclude='bads'))
        n_components = min(n_eeg - 1, 20)  # Cap at 20 for speed
    
    # Create ICA object
    ica = ICA(
        n_components=n_components,
        method=method,
        max_iter=max_iter,
        random_state=random_state,
        verbose=False
    )
    
    # Fit ICA
    fit_params = fit_params or {}
    ica.fit(raw_filtered, **fit_params)
    
    result = ICAResult(
        ica=ica,
        n_components=ica.n_components_
    )
    
    return result


def detect_eog_components(ica_result: ICAResult,
                          raw: mne.io.Raw,
                          ch_name: Optional[str] = None,
                          threshold: float = 3.0) -> Tuple[List[int], np.ndarray]:
    """
    Detect EOG (eye movement) components.
    
    If no EOG channel is available, uses frontal EEG channels (Fp1, Fp2).
    
    Args:
        ica_result: ICAResult with computed ICA
        raw: Original Raw object
        ch_name: Channel to use for EOG detection (None = auto)
        threshold: Z-score threshold for detection
        
    Returns:
        Tuple of (indices, scores)
    """
    if ica_result.ica is None:
        return [], np.array([])
    
    # Try to find EOG channel or use frontal electrodes
    if ch_name is None:
        # Check for EOG channels
        eog_chs = mne.pick_types(raw.info, eog=True)
        if len(eog_chs) > 0:
            ch_name = raw.ch_names[eog_chs[0]]
        else:
            # Use frontal channels
            frontal = ['Fp1', 'Fp2', 'Fpz', 'AF7', 'AF8']
            for ch in frontal:
                if ch in raw.ch_names:
                    ch_name = ch
                    break
    
    if ch_name is None or ch_name not in raw.ch_names:
        return [], np.array([])
    
    try:
        # Filter raw for ICA matching
        raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        
        eog_indices, eog_scores = ica_result.ica.find_bads_eog(
            raw_filtered,
            ch_name=ch_name,
            threshold=threshold,
            verbose=False
        )
        
        ica_result.eog_indices = list(eog_indices)
        ica_result.eog_scores = eog_scores
        
        # Set labels
        for idx in eog_indices:
            ica_result.labels[idx] = "EOG"
        
        return list(eog_indices), eog_scores
        
    except Exception as e:
        print(f"EOG detection failed: {e}")
        return [], np.array([])


def detect_ecg_components(ica_result: ICAResult,
                          raw: mne.io.Raw,
                          ch_name: Optional[str] = None,
                          threshold: float = 3.0) -> Tuple[List[int], np.ndarray]:
    """
    Detect ECG (heartbeat) components.
    
    Args:
        ica_result: ICAResult with computed ICA
        raw: Original Raw object
        ch_name: ECG channel name (None = auto-detect from ICA)
        threshold: Z-score threshold
        
    Returns:
        Tuple of (indices, scores)
    """
    if ica_result.ica is None:
        return [], np.array([])
    
    try:
        # Filter raw for ICA matching
        raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        
        ecg_indices, ecg_scores = ica_result.ica.find_bads_ecg(
            raw_filtered,
            ch_name=ch_name,
            threshold=threshold,
            verbose=False
        )
        
        ica_result.ecg_indices = list(ecg_indices)
        ica_result.ecg_scores = ecg_scores
        
        # Set labels
        for idx in ecg_indices:
            if idx not in ica_result.labels:  # Don't overwrite EOG
                ica_result.labels[idx] = "ECG"
        
        return list(ecg_indices), ecg_scores
        
    except Exception as e:
        print(f"ECG detection failed: {e}")
        return [], np.array([])


def detect_muscle_components(ica_result: ICAResult,
                             raw: mne.io.Raw,
                             threshold: float = 0.9) -> List[int]:
    """
    Detect muscle artifact components based on high-frequency content.
    
    Muscle artifacts typically have high power in gamma band (>30 Hz).
    
    Args:
        ica_result: ICAResult with computed ICA
        raw: Original Raw object
        threshold: Ratio threshold (high_freq_power / total_power)
        
    Returns:
        List of muscle component indices
    """
    if ica_result.ica is None:
        return []
    
    try:
        # Get ICA sources
        raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
        sources = ica_result.ica.get_sources(raw_filtered)
        source_data = sources.get_data()
        sfreq = raw.info['sfreq']
        
        muscle_indices = []
        
        for i in range(source_data.shape[0]):
            # Compute power spectrum
            from scipy import signal as sig
            freqs, psd = sig.welch(source_data[i], sfreq, nperseg=int(sfreq * 2))
            
            # High frequency power (> 30 Hz)
            high_freq_mask = freqs > 30
            low_freq_mask = (freqs > 1) & (freqs <= 30)
            
            if np.sum(high_freq_mask) > 0 and np.sum(low_freq_mask) > 0:
                high_power = np.mean(psd[high_freq_mask])
                low_power = np.mean(psd[low_freq_mask])
                
                ratio = high_power / (low_power + 1e-10)
                
                if ratio > threshold:
                    muscle_indices.append(i)
        
        ica_result.muscle_indices = muscle_indices
        
        # Set labels
        for idx in muscle_indices:
            if idx not in ica_result.labels:
                ica_result.labels[idx] = "MUSCLE"
        
        return muscle_indices
        
    except Exception as e:
        print(f"Muscle detection failed: {e}")
        return []


def get_component_properties(ica_result: ICAResult,
                             raw: mne.io.Raw,
                             component: int) -> Dict[str, Any]:
    """
    Get detailed properties of an ICA component.
    
    Args:
        ica_result: ICAResult with computed ICA
        raw: Original Raw object
        component: Component index
        
    Returns:
        Dictionary with component properties
    """
    if ica_result.ica is None:
        return {}
    
    ica = ica_result.ica
    
    # Get component data
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    sources = ica.get_sources(raw_filtered)
    source_data = sources.get_data()[component]
    
    # Get topomap data
    topo = ica.get_components()[:, component]
    
    # Compute spectrum
    from scipy import signal as sig
    freqs, psd = sig.welch(source_data, raw.info['sfreq'], nperseg=int(raw.info['sfreq'] * 2))
    
    # Peak frequency
    peak_freq = freqs[np.argmax(psd)]
    
    # Variance explained
    var_explained = np.var(source_data)
    
    return {
        'index': component,
        'label': ica_result.get_label(component),
        'is_excluded': component in ica_result.excluded,
        'topomap': topo,
        'variance': float(var_explained),
        'peak_frequency': float(peak_freq),
        'eog_score': float(ica_result.eog_scores[component]) if len(ica_result.eog_scores) > component else None,
        'ecg_score': float(ica_result.ecg_scores[component]) if len(ica_result.ecg_scores) > component else None,
    }


def apply_ica_exclusion(raw: mne.io.Raw,
                        ica_result: ICAResult,
                        exclude: Optional[List[int]] = None) -> mne.io.Raw:
    """
    Apply ICA and exclude specified components.
    
    Args:
        raw: Original Raw object
        ica_result: ICAResult with computed ICA
        exclude: Components to exclude (None = use ica_result.excluded)
        
    Returns:
        Cleaned Raw object (copy)
    """
    if ica_result.ica is None:
        return raw.copy()
    
    if exclude is None:
        exclude = ica_result.excluded
    
    ica_result.ica.exclude = list(exclude)
    ica_result.excluded = list(exclude)
    
    # Apply ICA to raw data
    raw_clean = ica_result.ica.apply(raw.copy(), verbose=False)
    
    return raw_clean


def plot_component_topomap(ica_result: ICAResult,
                           raw: mne.io.Raw,
                           component: int) -> np.ndarray:
    """
    Get topomap data for a component.
    
    Args:
        ica_result: ICAResult with computed ICA
        raw: Raw object with montage
        component: Component index
        
    Returns:
        Topomap weights array
    """
    if ica_result.ica is None:
        return np.array([])
    
    return ica_result.ica.get_components()[:, component]


def get_component_timeseries(ica_result: ICAResult,
                             raw: mne.io.Raw,
                             component: int,
                             start: float = 0,
                             duration: float = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get timeseries data for a component.
    
    Args:
        ica_result: ICAResult with computed ICA
        raw: Raw object
        component: Component index
        start: Start time in seconds
        duration: Duration in seconds
        
    Returns:
        Tuple of (times, data)
    """
    if ica_result.ica is None:
        return np.array([]), np.array([])
    
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=None, verbose=False)
    sources = ica_result.ica.get_sources(raw_filtered)
    
    start_sample = int(start * raw.info['sfreq'])
    end_sample = int((start + duration) * raw.info['sfreq'])
    
    data = sources.get_data()[component, start_sample:end_sample]
    times = np.arange(len(data)) / raw.info['sfreq'] + start
    
    return times, data

