

import pickle
from pathlib import Path
import sys

# Add parent directory to path to import from pipeline
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from paths import RESULTS_DIR, ensure_dir


def save_file(data, folder, file):
    folder = ensure_dir(folder)
    with open(Path(folder) / f"{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(Path(file), 'rb') as handle:
        return pickle.load(handle)
    

#%%

import mne
from mne.datasets import fetch_fsaverage

verbose = False
fs_dir = Path(fetch_fsaverage(verbose=verbose))  # Download fsaverage files
subjects_dir = fs_dir
src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"  # Source Space
labels = mne.read_labels_from_annot('fsaverage', parc='Schaefer2018_100Parcels_7Networks_order', verbose=verbose)


#%%

from mne.bem import _ico_downsample as ico_downsampler

lh, rh = mne.read_source_spaces(src)
lh_verts, lh_tris = lh['rr'], lh['tris']
rh_verts, rh_tris = rh['rr'], rh['tris']

lh_ico4 = ico_downsampler(lh, dest_grade=4)
rh_ico4 = ico_downsampler(rh, dest_grade=4)

lh_verts_ico4, lh_tris_ico4 = lh_ico4['rr'], lh_ico4['tris']
rh_verts_ico4, rh_tris_ico4 = rh_ico4['rr'], rh_ico4['tris']


#%%

import pandas as pd

lh_vert_idx, lh_vert_color, lh_roi = [], [], []
rh_vert_idx, rh_vert_color, rh_roi = [], [], []

for i, label in enumerate(labels):
  if label.name[-2:] == "lh":
    lh_vert_idx += list(label.vertices)
    lh_vert_color += [label.color]*len(label.vertices)
    lh_roi += [i]*len(label.vertices)

  if label.name[-2:] == "rh":
    rh_vert_idx += list(label.vertices)
    rh_vert_color += [label.color]*len(label.vertices)
    rh_roi += [i]*len(label.vertices)

lh_df = pd.DataFrame({"vertices":lh_vert_idx,"colors":lh_vert_color,"roi":lh_roi}).sort_values("vertices").reset_index(drop=True)
lh_df[["x","y","z"]] = lh_verts
lh_colors_dict = lh_df.groupby("roi")["colors"].max().to_dict()

rh_df = pd.DataFrame({"vertices":rh_vert_idx,"colors":rh_vert_color,"roi":rh_roi}).sort_values("vertices").reset_index(drop=True)
rh_df[["x","y","z"]] = rh_verts
rh_colors_dict = rh_df.groupby("roi")["colors"].max().to_dict()

#%%

#@title Replace dictionary
replace_dict = {"7Networks_":"",
"RH_": "RH ", "LH_": "LH ", "-rh":"", "-lh":"",

"DorsAttn_": "DAN ", #"Dorsal Attention ",
"Default_": "DMN ", #"Default ",
"Limbic_": "LN ", #"Limbic ",
"SalVentAttn_": "SVAN ", #"Salience/Ventral Attention ",
"SomMot_": "SMN ", #"Somatomotor ",
"Vis_": "VN ", #"Visual ",
"Cont_": "FPN ", #"Frontoparietal ",

"Post_":  "Posterior ",
"Temp_": "Temporal ",
"Par_": "Parietal ",

"Cing_": "Cingulate ",
"Med_": "Medial ",

"PFC_": "PFC ", #"Prefrontal Cortex ",
"PFCv_": "Prefrontal Ventral ",
"PFCl_": "Lateral PFC ", #"Lateral Prefrontal Cortex ",
"PFCmp_": "Medial PFC ", #"Medial Posterior Prefrontal Cortex ",
"PFCdPFCm_": "Prefrontal Dorsal Medial ",
"OFC_": "Orbito-Frontal ", #Cortex

"pCun_": "Precuneus ",
"FrOperIns_": "Frontal Operculum Insula ",
"ParOper_": "Parietal Operculum ",

"pCunPCC_": "Precuneus/Posterior Cingulate ", #Cortex
"PcunCing": "Precuneus Cingulate ",

"TempOccPar_": "Tempro-Occipital-Parietal ",
"TempPole_": "Temporal Pole ",
"TempPar_": "Tempro-Parietal ",

"FEF_": "Frontal Eye Fields ",
"PrCv_": "Precentral Ventral ",

"_":""}

replace_dict_short = {"7Networks_":"",
"RH_": "", "LH_": "", "-rh":"", "-lh":"",

"DorsAttn_": "", #"Dorsal Attention ",
"Default_": "", #"Default ",
"Limbic_": "", #"Limbic ",
"SalVentAttn_": "", #"Salience/Ventral Attention ",
"SomMot_": "SMN", #"Somatomotor ",
"Vis_": "VN", #"Visual ",
"Cont_": "", #"Frontoparietal ",

"_":""}

dict_networks = {
  "DAN": "Dorsal Attention Network (DAN)",
  "DMN": "Default Mode Network (DMN)",
  "LN": "Limbic Network (LN)",
  "SVA": "Salience/Ventral Attention Network (SVAN)",
  "SMN": "Somatomotor Network (SMN)",
  "VN": "Visual Network (VN)",
  "FPN": "Frontoparietal Network (FPN)"}

def replacer(string, dictio):
  for i in dictio.items():
      string = string.replace(i[0], i[1])
  return string

#%%

import numpy as np

label_names = [label.name for label in labels if not label.name.startswith('Background')]
node_colors = [label.color for label in labels if not label.name.startswith('Background')]
stc_coords_3d = np.asarray([label.pos.mean(axis=0) for label in labels if not label.name.startswith('Background')])

label_names = [replacer(label.name, replace_dict) for label in labels if not label.name.startswith('Background')]
label_names_short = [replacer(label.name, replace_dict_short) for label in labels if not label.name.startswith('Background')]

label_network = [x[:6] for x in label_names]


#%%


import pandas as pd
from itertools import combinations

nodes = pd.DataFrame(stc_coords_3d, index=label_names_short, columns=["x","y","z"]).reset_index(names="roi")
nodes["color"] = ["rgba"+str(color) for color in node_colors]
nodes["hemi"] = [x[:2] for x in label_network]
nodes["hemi-net"] = label_network
nodes["net"] = [x[3:] for x in label_network]

networks_list = list(set([x[3:] for x in label_network]))

edges = {}
for net in networks_list:
  edges[net] = {}
  net_data = nodes[nodes["net"]==net]
  for ax in ["x", "y", "z"]:
    edges[net][ax] = []
    for i,j in combinations(net_data.index, 2):
      edges[net][ax] += [net_data[ax][i], net_data[ax][j], None]
      
      
#%%

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "browser"

fig = px.scatter_3d(nodes, x='x', y='y', z='z', text="roi", hover_data=[])
fig.update_traces(marker=dict(size=8, color=nodes["color"].values), hovertemplate=None, hoverinfo='skip')

for net in networks_list:
  color = nodes[nodes["net"]==net]["color"].values[0]
  fig.add_trace(go.Scatter3d(x=edges[net]["x"], y=edges[net]["y"], z=edges[net]["z"],
                             mode='lines', name=replacer(net,dict_networks), hoverinfo='skip',
                             line=dict(color=color, width=2), opacity=0.5))

fig.add_trace(go.Mesh3d(x=lh_verts_ico4[:,0], y=lh_verts_ico4[:,1], z=lh_verts_ico4[:,2],
                        i=lh_tris_ico4[:,0], j=lh_tris_ico4[:,1], k=lh_tris_ico4[:,2],
                        color="pink", opacity=0.1,  hoverinfo='skip', name='LH'))

fig.add_trace(go.Mesh3d(x=rh_verts_ico4[:,0], y=rh_verts_ico4[:,1], z=rh_verts_ico4[:,2],
                        i=rh_tris_ico4[:,0], j=rh_tris_ico4[:,1], k=rh_tris_ico4[:,2],
                        color="pink", opacity=0.1,  hoverinfo='skip', name='RH'))

fig.update_layout(scene=dict(xaxis=dict(title='', showticklabels=False, backgroundcolor="white", showspikes=False),
                             yaxis=dict(title='', showticklabels=False, backgroundcolor="white", showspikes=False),
                             zaxis=dict(title='', showticklabels=False, backgroundcolor="white", showspikes=False)),
                  margin=dict(l=0, r=0, b=0, t=0),
                  legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))

fig.show(config={'displayModeBar':False})


#%%

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1, weights='distance')

knn.fit(lh_df[["x","y","z"]].values, lh_df["roi"])
lh_colors = [lh_colors_dict[pred] for pred in knn.predict(lh_verts_ico4)]

knn.fit(rh_df[["x","y","z"]].values, rh_df["roi"])
rh_colors = [rh_colors_dict[pred] for pred in knn.predict(rh_verts_ico4)]


#%%

import plotly.graph_objects as go
pio.renderers.default = "browser"

fig = go.Figure()

fig.add_trace(go.Mesh3d(x=lh_verts_ico4[:,0], y=lh_verts_ico4[:,1], z=lh_verts_ico4[:,2],
                        i=lh_tris_ico4[:,0], j=lh_tris_ico4[:,1], k=lh_tris_ico4[:,2],
                        vertexcolor=lh_colors, opacity=1, hoverinfo='skip', name='LH'))

fig.add_trace(go.Mesh3d(x=rh_verts_ico4[:,0], y=rh_verts_ico4[:,1], z=rh_verts_ico4[:,2],
                        i=rh_tris_ico4[:,0], j=rh_tris_ico4[:,1], k=rh_tris_ico4[:,2],
                        vertexcolor=rh_colors, opacity=1,  hoverinfo='skip', name='RH'))

fig.update_layout(scene=dict(xaxis=dict(title='', showticklabels=False, backgroundcolor="white", showspikes=False),
                             yaxis=dict(title='', showticklabels=False, backgroundcolor="white", showspikes=False),
                             zaxis=dict(title='', showticklabels=False, backgroundcolor="white", showspikes=False)),
                  margin=dict(l=0, r=0, b=0, t=0),
                  legend=dict(yanchor="top",y=0.99,xanchor="left",x=0.01))

fig.show(config={'displayModeBar':False})


con_mats = load_file(RESULTS_DIR / "con_mat.pkl")

