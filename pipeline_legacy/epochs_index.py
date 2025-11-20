
import pickle

def save_file(data, folder, file):
    with open(folder+file+".pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)


#%%

import os
from tqdm import tqdm

files_list = []
path = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"
epochs_len = {}

for cond in ["DMT\\", "EO\\", "EC\\"]:
    files_names = sorted([x for x in os.listdir(path+cond) if x.startswith('syncro')])
    #files_list += [path+cond+x for x in files_names]

    for file in tqdm(files_names):
        subject_data = load_file(path+cond+file)
        subj = file.replace("syncro-","")[:6]
        epochs_num = len(subject_data["syncros_stc"]["Alpha"][0])
        epochs_len[subj] = epochs_num


#%%

from pymatreader import read_mat
import numpy as np

rejected_subjects = [2, 5, 8, 16, 23, 31]
subjects = np.array([x for x in range(35) if x not in rejected_subjects])

folder = "E:\\FedeZ\\DMT\\EEG\\Clean EEGLAB\\"

bad_epochs = read_mat(folder+"rejected_epochs.mat")
subject = bad_epochs["rejected_epochs"][0]
epochs = bad_epochs["rejected_epochs"][1]
rejected_epochs = {k[:6].replace("_","-"): [x-1 for x in v.tolist()] for k, v in zip(subject, epochs)}


#%%

epochs_index = {}

for key, value in epochs_len.items():
    try:
        largo = len(rejected_epochs[key])+value
        epochs_index[key] = [x for x in range(largo) if x not in rejected_epochs[key]]
    except:
        largo = value
        epochs_index[key] = [x for x in range(largo)]


#%%

# files_list = []
# path = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"
# cond = "DMT\\" #"EO\\", "EC\\",

# files_names = sorted([x for x in os.listdir(path+cond) if x.startswith('syncro')])
# files_list += [path+cond+x for x in files_names]

# split = 0
# band = "Alpha"
# syncros_stc = []

# for file in tqdm(files_list):
#     subject_data = load_file(file)
#     syncros_stc.append(subject_data["syncros_stc"][band][split])
        
        # all_epochs = []
        # for epoch_list in eigen_stc_list:
        #     for epoch in epoch_list:
        #         all_epochs.append(epoch)