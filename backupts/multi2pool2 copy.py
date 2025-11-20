
import pickle
from pathlib import Path

from paths import RESULTS_DIR


def save_file(data, folder, file_stem):
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)
    with open(folder_path / f"{file_stem}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


#%%

BASE_PATH = RESULTS_DIR

[node_colors, label_names, \
 label_names_short, stc_coords_3d, \
 ch_names, mapping, eeg_coords_2d] = load_file(BASE_PATH / "extra.pkl")

label_network = [x[:6] for x in label_names]


#%%
import pandas as pd
from copy import deepcopy

df_labels = pd.DataFrame()
df_labels["label"] = [x[:6] for x in label_names]
df_labels["hemi"] = [x[:2] for x in df_labels["label"]]
df_labels["net"] = [x[3:] for x in df_labels["label"]]

def network_filter(phase, df_labels, hemi="both", net="all"):
    df = pd.concat([pd.DataFrame(phase), df_labels], axis=1)
    df = (df[df["hemi"]==hemi] if hemi != "both" else df)
    df = (df[df["net"]==net] if net != "all" else df)
    df = df.drop(columns=["hemi","label","net"])
    return df

def order_parameter(phase):
    r = np.abs(np.exp(1j*phase).mean(axis=0))
    return r

def kuramoto_net(data_dict):   
    r_dict = {}
    
    for band in deepcopy(band_list):
        epoch_list = data_dict["phases_stc"][band]
        r_dict[band] = {}
        
        for hemi in deepcopy(hemi_list):
            r_dict[band][hemi] = {}
            
            for net in deepcopy(net_list):
                r_dict[band][hemi][net] = []
            
                for epoch_data in tqdm(epoch_list):
                    filtered = network_filter(epoch_data, df_labels, hemi=hemi, net=net)
                    r = order_parameter(filtered)
                    r_dict[band][hemi][net].append(r) #.mean()
    return r_dict


#%%

import numpy as np
from tqdm import tqdm

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
hemi_list = ["RH", "LH", "both"]
net_list = ["FPN", "DMN", "DAN", "LN " , "SVA", "SMN", "VN "]
#cond_list = ["DMT\\", "EC\\", "EO\\"]

def do_the_math2(file_name):
    
    data_dict = load_file(file_name)
    file_path = Path(file_name)
    subject_id = file_path.name.replace("phases","order_all").replace(".pkl","")
    cond = file_path.parent.name
    output_folder = BASE_PATH / cond
    
    r_dict = {}
    r_dict[cond] = {}
    
    for band in band_list:
        epoch_list = data_dict["phases_stc"][band]
        r_dict[cond][band] = {}
        print(cond, band)
        
        for hemi in hemi_list:
            r_dict[cond][band][hemi] = {}
            
            for net in net_list:
                r_dict[cond][band][hemi][net] = []
            
                for epoch_data in tqdm(epoch_list):
                    r = network_filter(epoch_data, df_labels, hemi=hemi, net=net)
                    #r = order_parameter(r)
                    r_dict[cond][band][hemi][net].append(r) #.mean()
                        
    save_file(r_dict, output_folder, subject_id)
    #return r_dict
                        
#%%


from multiprocessing import Pool

dmt_files_names = sorted((BASE_PATH / "DMT").glob("phases*.pkl"))
eo_files_names = sorted((BASE_PATH / "EO").glob("phases*.pkl"))
ec_files_names = sorted((BASE_PATH / "EC").glob("phases*.pkl"))

#%%
if __name__ == '__main__':
    with Pool(20) as p:
        p.map(do_the_math2, dmt_files_names)
        p.map(do_the_math2, eo_files_names)
        p.map(do_the_math2, ec_files_names)

#%%

# file_name = dmt_files_names[0]
# fname = file_name[file_name.rfind("\\")+1:]
# subject_id = fname.replace("phases","order").replace(".pkl","")
# cond = file_name.replace(path,"").replace(fname,"").replace("\\","")
# output_folder = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"+cond+"\\"
