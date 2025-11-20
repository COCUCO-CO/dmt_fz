# Kuramoto
import pickle

folder = "C:\\Users\\xochipilli\\Desktop\\DMT\\fwd-inv-stc\\"

with open(folder+'subjects_phases_eeg.pickle', 'rb') as handle:
    subjects_phases_eeg = pickle.load(handle)

with open(folder+'subjects_phases_stc.pickle', 'rb') as handle:
    subjects_phases_stc = pickle.load(handle)

#%%

import numpy as np

euler_notation = np.vectorize(lambda x: np.exp(1j*x))

def order_parameter(phase):
    n_osc = phase.shape[0]
    r = np.abs(euler_notation(phase).sum(axis=0)) / n_osc
    return r

#%%
from tqdm import tqdm

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

subjects_r_eeg = []
subjects_r_stc = []

for subject in range(len(subjects_phases_eeg)):

  r_eeg = {}
  r_stc = {}
  
  for band in band_list:
    
    r_eeg[band] = []
    r_stc[band] = []

    num_epochs = len(subjects_phases_eeg[subject][band])

    for epoch in tqdm(range(num_epochs)):
        eeg_phase_mat = subjects_phases_eeg[subject][band][epoch]
        r = order_parameter(eeg_phase_mat)       
        r_eeg[band].append(r)
        
        stc_phase_mat = subjects_phases_stc[subject][band][epoch]
        r = order_parameter(stc_phase_mat)
        r_stc[band].append(r)
        
  subjects_r_eeg.append(r_eeg)
  subjects_r_stc.append(r_stc)

#%%

folder = "C:\\Users\\xochipilli\\Desktop\\DMT\\fwd-inv-stc\\"

with open(folder+'subjects_r_eeg.pickle', 'wb') as handle:
    pickle.dump(subjects_r_eeg, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open(folder+'subjects_r_stc.pickle', 'wb') as handle:
    pickle.dump(subjects_r_stc, handle, protocol=pickle.HIGHEST_PROTOCOL)
  