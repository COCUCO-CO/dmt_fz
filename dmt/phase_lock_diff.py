import os
import mne
import numpy as np

subject = 0
band = "Alpha"
epoch = 100

channel1 = 7
channel2 = 15

#%%

from mne.filter import filter_data
from scipy.signal import hilbert

freq_bands = {
    "Delta": {"l_freq":1, "h_freq":4},
    "Theta": {"l_freq":4, "h_freq":8},
    "Alpha": {"l_freq":8, "h_freq":12},
    "Beta": {"l_freq":12, "h_freq":30},
    "Gamma": {"l_freq":30, "h_freq":40}
    }

def filtered(signal, band, sfreq=500):
  l_freq = freq_bands[band]["l_freq"]
  h_freq = freq_bands[band]["h_freq"]
  filtered_signal = filter_data(data=signal, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False, filter_length="auto")
  return filtered_signal

#%%
import pickle

folder = "C:\\Users\\xochipilli\\Desktop\\DMT\\fwd-inv-stc\\"

with open(folder+'subjects_phases_eeg.pickle', 'rb') as handle:
    subjects_phases_eeg = pickle.load(handle)

#%%

folder = "C:\\Users\\xochipilli\\Desktop\\DMT\\EEG\\Clean EEGLAB\\DMT\\"

dmt_files = sorted([x for x in os.listdir(folder) if x[-3:]=="set"])
dmt_epochs_raw = mne.io.read_epochs_eeglab(folder+dmt_files[subject], montage_units='dm', verbose=False)


#%%

dmt_epochs = dmt_epochs_raw.get_data()

signal1 = np.hstack(dmt_epochs[epoch][channel1])
filtered_signal1 = filtered(signal1, "Alpha")
analytic_signal1 = hilbert(filtered_signal1)

signal2 = np.hstack(dmt_epochs[epoch][channel2])
filtered_signal2 = filtered(signal2, "Alpha")
analytic_signal2 = hilbert(filtered_signal2)

diff1 = np.abs((np.angle(analytic_signal1) - np.angle(analytic_signal2)).mean())
diff2 = np.abs((np.angle(analytic_signal2) - np.angle(analytic_signal1)).mean())

print(diff1)
print(diff2)


#%%

phase_mat = subjects_phases_eeg[subject][band][epoch]

phases_1 = phase_mat[channel1,:]
phases_2 = phase_mat[channel2,:]

diff1 = np.abs((phases_1 - phases_2).mean())
diff2 = np.abs((phases_2 - phases_1).mean())

print(diff1)
print(diff2)

#%%

def diff_ang(theta1, theta2, full_p=2*np.pi, abso=True):
  half_p = 0.5 * full_p
  fmod1 = np.fmod(theta2 - theta1 + half_p, full_p)
  fmod2 = np.fmod(fmod1 + full_p, full_p) - half_p
  if abso==True:
    return abs(fmod2) #abs(np.fmod(np.fmod(theta2 - theta1 + half_p, full_p) + full_p, full_p) - half_p)
  else:
    return fmod2

max_diff = np.pi * 1000
dist = diff_ang(phases_1, phases_2).sum()
print(dist/max_diff)
print(1 - (dist/max_diff))

#%%

print(1-diff1)