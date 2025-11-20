
from nice.markers import PowerSpectralDensityEstimator
from nice.markers import PowerSpectralDensity

import mne

import numpy as np
from scipy.stats import trim_mean

from tqdm import tqdm
from pathlib import Path

from paths import EEG_CLEAN_DIR, RESULTS_DIR, ensure_dir


#%%

def trim_mean80(a, axis=0):
    return trim_mean(a, proportiontocut=.1, axis=axis)

def entropy(a, axis=0):  # noqa
    return -np.nansum(a * np.log(a), axis=axis) / np.log(a.shape[axis])

def all_markers(epochs, tmin, tmax):
   
        descriptors = {}

        markers=[["delta", 1.0,   4.0],
                 ["theta", 4.0,   8.0],
                 ["alpha", 8.0,  13.0],
                 ["beta",  13.0, 30.0],
                 ["gamma", 30.0, 45.0]]
        
        psds_params = dict(n_fft=4096, n_overlap=100, n_jobs='auto', nperseg=128)
        
        base_psd = PowerSpectralDensityEstimator(psd_method='welch',
                                                 tmin=tmin, tmax=tmax,
                                                 fmin=1., fmax=45.,
                                                 psd_params=psds_params, comment='default')
                
        for marker in markers:
            for option in [True, False]:
                name = marker[0]+"_norm" if option else marker[0]
    
                marker_data = PowerSpectralDensity(estimator=base_psd,
                                                 fmin=marker[1], fmax=marker[2],
                                                 normalize=option, comment=name)
           
                print(name)
        
                marker_data.fit(epochs)
                descriptors[name] = marker_data._prepare_data(target=None, picks=None).sum(axis=2)

        return descriptors

#%%

import os

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

data_root = EEG_CLEAN_DIR
eo_dir = data_root / "EO"
ec_dir = data_root / "EC"
dmt_dir = data_root / "DMT"

eyes_open_files_list = sorted(eo_dir.glob("*.set"))
eyes_closed_files_list = sorted(ec_dir.glob("*.set"))
dmt_files_list = sorted(dmt_dir.glob("*.set"))

subjects = []
for file_name in tqdm(dmt_files_list):
  dmt_epochs = mne.io.read_epochs_eeglab(str(file_name), montage_units='dm', verbose=False)
  descriptors = all_markers(dmt_epochs, 0, 1000)
  subjects.append(descriptors)
  
#%%

import pickle

output_folder = ensure_dir(RESULTS_DIR)

with open(output_folder / 'subjects_psd.pickle', 'wb') as handle:
    pickle.dump(subjects, handle, protocol=pickle.HIGHEST_PROTOCOL)