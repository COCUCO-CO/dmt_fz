
import numpy as np

import mne
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator
from mne.minimum_norm import apply_inverse_epochs

from mne.filter import filter_data
from scipy.signal import hilbert

from tqdm import tqdm

from itertools import combinations as combs_without
from itertools import combinations_with_replacement as combs_with


#%%

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#%%

import pickle

def save_file(data, folder, file):
    with open(folder+file+".pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

#%%

[node_colors, label_names, \
 label_names_short, stc_coords_3d, \
 ch_names, mapping, eeg_coords_2d] = load_file("E:\\FedeZ\\DMT\\fwd-inv-stc\\extra.pkl")

label_network = [x[:6] for x in label_names]

ch_region = []
for name in ch_names:
    if name.find("F") != -1:
        region = "Frontal"
    elif name.find("C") != -1:
        region = "Central"
    elif name.find("O") != -1:
        region = "Occipital"
    elif name.find("P") != -1:
        region = "Parietal"
    else:
        region = "Temporal"
    ch_region.append(region)

#%%
import pandas as pd

df_labels_stc = pd.DataFrame()
df_labels_stc["label"] = [x[:6] for x in label_names]
df_labels_stc["hemi"] = [x[:2] for x in df_labels_stc["label"]]
df_labels_stc["net"] = [x[3:] for x in df_labels_stc["label"]]

#%%


import os.path as op

verbose = False

# fsaverage

fs_dir = fetch_fsaverage(verbose=verbose) # Download fsaverage files
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'
trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
src = op.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # Source Space
bem = op.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif') # Boundary Element Method (BEM) for forward modeling

labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', verbose=verbose)
#labels_aparc = mne.read_labels_from_annot('fsaverage', parc='aparc', verbose=verbose)


#%%

method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

# Define frequencies of interest
fmin, fmax = 0., 70.
bandwidth = 4.  # bandwidth of the windows in Hz

def fwd_inv_stc(epochs):
    epochs = epochs.pick('eeg').apply_baseline((None, None), verbose=verbose)
    epochs = epochs.set_montage('standard_1020')
    epochs = epochs.set_eeg_reference(projection=True, verbose=verbose).apply_proj()
    
    noise_cov = mne.compute_covariance(epochs, tmax=0., method=['shrunk', 'empirical'], rank=None, verbose=verbose)
    fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None, verbose=verbose)
    inverse_operator = make_inverse_operator(epochs.info, fwd, noise_cov, loose=0.2, depth=0.8, verbose=verbose)
    
    stc = apply_inverse_epochs(epochs, inverse_operator, lambda2,
                               method=method, pick_ori=None, label=None,
                               return_generator=True, verbose=verbose)
    return stc




#%%

from mne.time_frequency import psd_array_welch

hemi_list = ["RH", "LH", "both"]

net_list = ["FPN", "DMN", "DAN", "LN " , "SVA", "SMN", "VN "]

freq_bands = {
    "Delta": [1, 4],
    "Theta": [4, 8],
    "Alpha": [8, 13],
    "Beta":  [13, 30],
    "Gamma": [30, 45]
    }

band_list = list(freq_bands.keys())


def welch_psd(epochs_data, band, sfreq=500, n_fft=4096, n_overlap=100, nperseg=128):
    fmin = freq_bands[band][0]
    fmax = freq_bands[band][1]
    psd, freqs = psd_array_welch(epochs_data, sfreq=sfreq, n_fft=n_fft,
                                 fmin=fmin, fmax=fmax, n_overlap=n_overlap,
                                 n_per_seg=nperseg, window='hamming')#, n_jobs=None, average='mean')
    return psd.sum(axis=2)


def filtered(signal, band, sfreq=500):
  l_freq = freq_bands[band][0]
  h_freq = freq_bands[band][1]
  filtered_signal = filter_data(data=signal, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, verbose=False, filter_length="auto")
  return filtered_signal


def diff_ang(theta1, theta2, full_p=2*np.pi, abso=True):
  half_p = 0.5 * full_p
  fmod1 = np.fmod(theta2 - theta1 + half_p, full_p)
  fmod2 = np.fmod(fmod1 + full_p, full_p) - half_p
  if abso==True:
    return abs(fmod2) #abs(np.fmod(np.fmod(theta2 - theta1 + half_p, full_p) + full_p, full_p) - half_p)
  else:
    return fmod2


def hilbert_transform(band_signals, trim=False):
  signal_num = len(band_signals)
  samples = band_signals[0].shape[0]
  #envelope_mat = np.zeros((signal_num, samples))
  phase_mat = np.zeros((signal_num, samples))

  for i, filtered_signal in enumerate(band_signals):
    analytic_signal = hilbert(filtered_signal)
    #envelope = np.abs(analytic_signal)
    inst_phase = np.angle(analytic_signal)

    #envelope_mat[i,:] = envelope
    phase_mat[i,:] = inst_phase
  
  if trim != False:
    #envelope_mat = envelope_mat[:,trim:(samples-trim)]
    phase_mat = phase_mat[:,trim:(samples-trim)]
      
  return phase_mat #, envelope_mat


def calculate_syncro(phase_mat):
  max_diff = np.pi * phase_mat.shape[1]
  size = phase_mat.shape[0]
  syncro_mat = np.zeros((size,size))
  for i, j in combs_without(range(size), 2):
        signal1 = phase_mat[i,:]
        signal2 = phase_mat[j,:]
        value = 1 - (diff_ang(signal1,signal2).sum()/max_diff)
        syncro_mat[i,j] = value
        syncro_mat[j,i] = value
  eigen = np.diag(np.linalg.eigh(syncro_mat)[1])
  return syncro_mat, eigen


# def calculate_entropy(amplitude_mat):
#     median = np.median(amplitude_mat, axis=1)
#     thresholded = np.where(amplitude_mat/median[:,None]>=1, 1, 0)
#     for channel in range(num_channels):
#           sequence = "".join(thresholded[channel,:].astype("str"))
#           lempel_ziv(sequence)


def lempel_ziv(seq):
    keys_dict = {}
    ind = 0
    inc = 1
    while len(seq) >= ind+inc:
        sub_str = seq[ind:ind+inc]
        #print(sub_str)
        if sub_str in keys_dict:
            inc += 1
        else:
            keys_dict[sub_str] = (ind,inc)
            ind += inc
            inc = 1
            #print("Adding", sub_str)
    return len(keys_dict.keys()) / len(seq)


def eeg_pre(epochs, epoch, num_channels=24):
    eeg_signals = []
    for channel in range(num_channels):
      signal = epochs[epoch][channel]
      eeg_signals.append(np.hstack(signal))
    return np.asarray(eeg_signals)


def stc_pre(epoch_data, labels):
    stc_signals = []
    for label in labels:
        if not label.name.startswith("unknown"):#'Background'):
            try:
              label_data = epoch_data.in_label(label).data
              stc_signals.append(label_data.mean(axis=0))
            except:
              pass
    return np.asarray(stc_signals)


def order_parameter(phase):
    r = np.abs(np.exp(1j*phase).mean(axis=0))
    return r


def network_filter(phase, df_labels, hemi="both", net="all"):
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    df = (df[df["hemi"]==hemi] if hemi != "both" else df)
    df = (df[df["net"]==net] if net != "all" else df)
    df = df.drop(columns=["hemi","label","net"])
    return df
       

def kuramoto_net(epoch_data, df_labels, r_dict,):

    if r_dict == {}:
        for (net1,net2) in combs_with(deepcopy(net_list), 2):
            r_dict[net1] = r_dict.get(net1,{})
            r_dict[net1][net2] = []
        r_dict["all"] = []

    
    for (net1,net2) in combs_with(deepcopy(net_list), 2):
        if net1 != net2:
            data_net1 = network_filter(epoch_data, df_labels, net=net1)
            data_net2 = network_filter(epoch_data, df_labels, net=net2)
            r = order_parameter(pd.concat([data_net1, data_net2]))
        else:
            data_net1 = network_filter(epoch_data, df_labels, net=net1)
            r = order_parameter(data_net1)
            
        r_dict[net1][net2].append(r) #.mean()
        
    data_net_all = network_filter(epoch_data, net="all")
    r_all = order_parameter(data_net_all)
    r_dict["all"].append(r_all) #.mean()

    return r_dict

#%%

from copy import deepcopy

markers_list1 = ["psd_eeg", "phases_eeg", #"amplitudes_eeg", "filtered_eeg", 
                 "psd_stc", "phases_stc"] #"amplitudes_stc", "filtered_stc"

markers_list2 = ["syncros_eeg", "kuramoto_eeg", "eigen_eeg",
                 "syncros_stc", "kuramoto_stc", "eigen_stc"]

markers_list3 = ["coherence_eeg", "metastability_eeg",
                 "coherence_stc", "metastability_stc"]

band_dict = {band:[] for band in deepcopy(band_list)}

subject_dict1 = {marker:deepcopy(band_dict) for marker in deepcopy(markers_list1)}
subject_dict2 = {marker:deepcopy(band_dict) for marker in deepcopy(markers_list2)}
subject_dict3 = {marker:deepcopy(band_dict) for marker in deepcopy(markers_list3)}


def do_the_math(file_name):
    epochs_raw = mne.io.read_epochs_eeglab(file_name, montage_units='dm', verbose=False)
    epochs_data = epochs_raw.get_data()
    (num_epochs, num_channels, num_samples) = epochs_data.shape
    
    fname = file_name[file_name.rfind("\\")+1:]
    subject_id = fname[:fname.rfind("ICA")-1].replace("_","-")
    condition = file_name.replace(path,"").replace(fname,"")
    output_folder = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"+condition

    subject_data1 = deepcopy(subject_dict1)
    subject_data2 = deepcopy(subject_dict2)
    subject_data3 = deepcopy(subject_dict3)
    
    for band in band_list:
        stc = fwd_inv_stc(epochs_raw)
        
        #cohe_eeg, meta_eeg = {}, {}
        #cohe_stc, meta_stc = {}, {}
        
        for epoch in range(num_epochs):
            # EEG
            #psd_eeg = welch_psd(epochs_data, band)
            signals_eeg = eeg_pre(epochs_data, epoch)
            filtered_eeg = filtered(signals_eeg, band)
            amplitude_eeg, phases_eeg = hilbert_transform(filtered_eeg, trim=100)   
            syncro_eeg, eigen_eeg = calculate_syncro(phases_eeg)
            #r_eeg = kuramoto_net(phases_eeg, df_labels_eeg, r_eeg)
            
            #subject_data1["psd_eeg"][band].append(psd_eeg)
            subject_data1["phases_eeg"][band].append(phases_eeg)
            
            subject_data2["syncros_eeg"][band].append(syncro_eeg)
            subject_data2["eigen_eeg"][band].append(eigen_eeg)
 
            # Sources
            signals_stc = stc_pre(next(stc), labels)
            #psd_stc = welch_psd(signals_stc, band)
            filtered_stc = filtered(signals_stc, band)
            phases_stc = hilbert_transform(filtered_stc, trim=100)
            syncro_stc, eigen_stc = calculate_syncro(phases_stc)
            #r_stc = kuramoto_net(phases_stc, df_labels_stc, r_stc)
            
            #subject_data1["psd_stc"][band].append(psd_stc)
            subject_data1["phases_stc"][band].append(phases_stc)
            
            subject_data2["syncros_stc"][band].append(syncro_stc)
            subject_data2["eigen_stc"][band].append(eigen_stc)

        #subject_data3["coherence_eeg"][band] = cohe_eeg
        #subject_data3["metastability_eeg"][band] = meta_eeg
        
        #subject_data3["coherence_stc"][band] = cohe_stc
        #subject_data3["metastability_stc"][band] = meta_stc
            
    save_file(subject_data1, output_folder, "phases-"+subject_id)
    save_file(subject_data2, output_folder, "syncro-"+subject_id)
    save_file(subject_data3, output_folder, "order-"+subject_id)


#%%

import os

path = "E:\\FedeZ\\DMT\\EEG\\Clean EEGLAB\\"

eyes_open_files_list = sorted([path+"EO\\"+x for x in os.listdir(path+"EO\\") if x[-3:]=="set"])
eyes_closed_files_list = sorted([path+"EC\\"+x for x in os.listdir(path+"EC\\") if x[-3:]=="set"])
dmt_files_list = sorted([path+"DMT\\"+x for x in os.listdir(path+"DMT\\") if x[-3:]=="set"])


#%%

from multiprocessing import Pool

if __name__ == '__main__':
    with Pool(7) as p:
        p.map(do_the_math, eyes_open_files_list)
        p.map(do_the_math, eyes_closed_files_list)
        p.map(do_the_math, dmt_files_list)

#%%

# from multiprocessing import Pool

# if __name__ == '__main__':
#     with Pool(1) as p:
#         p.map(do_the_math, [eyes_open_files_list[1]])

#%%

# subj_data = load_file("E:\\FedeZ\\DMT\\fwd-inv-stc\\EO\\phases-S01-EO.pkl")

# file_name = eyes_open_files_list[0]
# epochs_raw = mne.io.read_epochs_eeglab(file_name, montage_units='dm', verbose=False)
# epochs_data = epochs_raw.get_data()
# num_epochs = epochs_data.shape[0]
# num_samples = epochs_data.shape[2]

#%%

# file_name = eyes_open_files_list[1]
# fname = file_name[file_name.rfind("\\")+1:]
# subject_id = fname[:fname.rfind("ICA")-1].replace("_","-")
# condition = file_name.replace(path,"").replace(fname,"")
# output_folder = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"+condition

#%%

# for splits in range(0,2):
#     ruler = np.linspace(0, num_samples, splits+2, dtype="int")
#     for i in range(len(ruler)-1):
# phases_eeg = phase_eeg[:,ruler[i]:ruler[i+1]]
# phases_stc = phase_stc[:,ruler[i]:ruler[i+1]]

#%%