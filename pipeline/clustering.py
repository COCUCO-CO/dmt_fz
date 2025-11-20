
import pickle
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "fwd-inv-stc"
SPECTRAL_DIR = BASE_DIR / "spectral_sources"


def save_file(data, folder, file):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{file}.pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(Path(file), 'rb') as handle:
        return pickle.load(handle)


def cond_path(cond):
    return RESULTS_DIR / cond.replace("\\", "")
    
#%%

import numpy as np
from tqdm import tqdm

#%%

# folder = "E:\\FedeZ\\DMT\\fwd-inv-stc\\old\\"
# syncros = load_file(folder+"subjects_syncros_stc.pickle")

# diags_list = []
# for subj in tqdm(range(29)):
#     for band in ["Alpha"]:
#         for epoch in syncros[subj][band]:      
#             diag_eigenvalues = np.diag(np.linalg.eigh(epoch)[1])
#             diags_list.append(diag_eigenvalues)

# diags = np.asarray(diags_list)
# print(diags.shape)

#%%

import os
from copy import deepcopy


band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
cond_list = ["DMT", "EC", "EO"]

eigen_dict = {cond: {band:[] for band in deepcopy(band_list)} for cond in deepcopy(cond_list)}
kuramoto_dict = {cond: {band:[] for band in deepcopy(band_list)} for cond in deepcopy(cond_list)}

path = RESULTS_DIR

for cond in cond_list:
    print(cond)
    files_names = sorted([x for x in os.listdir(cond_path(cond)) if x.startswith('syncro-')])
    for file in tqdm(files_names):
        data_dict = load_file(cond_path(cond) / file)
        for band in band_list:
            kuramoto_dict[cond][band] += data_dict["kuramoto_stc"][band][0]
            for epoch in data_dict["syncros_stc"][band][0]:
                eigen_dict[cond][band].append(np.diag(np.linalg.eigh(epoch)[1]))

#save_file(eigen_dict, path, "eigen_corrected")


#%%

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

max_k = 15
min_k = 2
max_comps = 10
min_comps = 2

for band in band_list:
    
    x_dmt = np.asarray(eigen_dict["DMT"][band])
    x_ec = np.asarray(eigen_dict["EC"][band])
    x_eo = np.asarray(eigen_dict["EO"][band])
    
    r_dmt = np.asarray(kuramoto_dict["DMT"][band]).mean(axis=1)
    r_ec = np.asarray(kuramoto_dict["EC"][band]).mean(axis=1)
    r_eo = np.asarray(kuramoto_dict["EO"][band]).mean(axis=1)
    
    x_dmt = x_dmt[reject_outliers(r_dmt)]
    x_ec = x_ec[reject_outliers(r_ec)]
    x_eo = x_eo[reject_outliers(r_eo)]
    
    diags = np.concatenate((x_dmt,x_ec,x_eo))
    
    heatmap = np.zeros((max_comps-2,max_k-2))

    for comps in range(min_comps, max_comps):
        pca = PCA(n_components=comps)
        data = pca.fit_transform(deepcopy(diags))

        for k in range(min_k, max_k):
            kmeans = KMeans(n_clusters=k, n_init="auto").fit(data) #, n_init="auto"
            lista_cluster = kmeans.labels_
            simi = pairwise_distances(data, metric="cosine")
            silhouette = silhouette_score(simi, lista_cluster)
            heatmap[comps-min_comps,k-min_k] = silhouette
            print(comps, k, silhouette)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_title(band)
    ax.imshow(heatmap)
    ax.set_xticks(range(0,max_k-min_comps), labels=range(min_k,max_k))
    ax.set_yticks(range(0,max_comps-min_comps), labels=range(min_comps,max_comps))
    for i in range(max_comps-min_comps):
        for j in range(max_k-min_k):
            text = ax.text(j, i, str(heatmap[i, j])[:5],
                          ha="center", va="center", color="k", fontsize=8)



#%%

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score            
import matplotlib.pyplot as plt

max_k = 15
max_comps = 10

for band in band_list:
heatmap = np.zeros((max_comps-2,max_k-2))
diags = np.asarray(eigen_dict["DMT"]["Theta"])

for i in range(2,max_comps):
    pca = PCA(n_components=i)
    data = pca.fit_transform(deepcopy(diags))

    for j in range(2,max_k):
        kmeans = KMeans(n_clusters=j, n_init="auto").fit(data) #, n_init="auto"
        lista_cluster = kmeans.labels_
        simi = pairwise_distances(data, metric="cosine")
        silhouette = silhouette_score(simi, lista_cluster)
        heatmap[i-2,j-2] = silhouette
        print(i, j, silhouette)

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(heatmap)
ax.set_xticks(range(0,max_k-2), labels=range(2,max_k))
ax.set_yticks(range(0,max_comps-2), labels=range(2,max_comps))
for i in range(max_comps-2):
    for j in range(max_k-2):
        text = ax.text(j, i, str(heatmap[i, j])[:5],
                      ha="center", va="center", color="k", fontsize=8)

plt.show()



#%%

from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score            
import matplotlib.pyplot as plt


max_k = 15
max_comps = 10
heatmap = np.zeros((max_comps-2,max_k-2))

for i in [2]:#range(2,max_comps):
  pca = PCA(n_components=i)
  data = pca.fit_transform(deepcopy(diags))
  
  for j in [3]:#range(2,max_k):
    kmeans = KMeans(n_clusters=j, n_init="auto").fit(data) #, n_init="auto"
    lista_cluster = kmeans.labels_
    simi = pairwise_distances(data, metric="cosine")
    silhouette = silhouette_score(simi, lista_cluster)
    heatmap[i-2,j-2] = silhouette
    print(i,j,silhouette)

#%%

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

mats = []
for i in range(j):
    center = kmeans.cluster_centers_[i,:]
    reverse = pca.inverse_transform(center).reshape(-1,1)
    eigen_vals = scaler.fit_transform(reverse)
    mat = eigen_vals @ np.transpose(eigen_vals)
    mats.append(mat)
    fig, ax = plt.subplots(figsize=(8,8))
    ax.imshow(mat)
    plt.show()

#%%

import networkx as nx
import matplotlib as mpl

pos = stc_coords_2d

matdata = mats[2]
np.fill_diagonal(matdata, 0)
syncro_graph = nx.from_numpy_array(matdata)
edge_dict = nx.get_edge_attributes(syncro_graph,'weight')
sorted_dict = dict(sorted(edge_dict.items(), key=lambda item: item[1])) # = {k: v for k, v in sorted(edge_dict.items(), key=lambda item: item[1])}
edges, weights = zip(*sorted_dict.items())
weights = scaler.fit_transform(np.asarray(weights).reshape(-1,1))
colors = [mpl.colormaps["Blues"](x) for x in weights]

fig, ax = plt.subplots(figsize=(10,10))
nx.draw_networkx_nodes(syncro_graph, ax=ax, pos=pos)
nx.draw_networkx_labels(syncro_graph, ax=ax, pos=pos)
nx.draw_networkx_edges(syncro_graph, ax=ax, pos=pos, edgelist=edges, edge_color=colors)
ax.axis('off')
plt.show()



#%%

import mne

labels = mne.read_labels_from_annot('fsaverage', parc='aparc', verbose=False)
stc_coords_3d = np.asarray([label.pos.mean(axis=0) for label in labels if not label.name.startswith('Background')])
stc_coords_2d = stc_coords_3d[:,:2]



#%%


import matplotlib.pyplot as plt
import pandas as pd

#diags_df = pd.DataFrame(np.asarray(diags))
pca = PCA(n_components=2)
#pca.fit_transform(diags)

#random_state = 64

from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def clustering(data_cluster):
  fig, axs = plt.subplots(nrows=1,ncols=5,figsize=(40,8))

  for i in range(0,5):
    data_cluster = pd.DataFrame(np.asarray(diags))
    data_cluster[["PCA1","PCA2"]] = pca.fit_transform(data_cluster.values)

    kmeans = KMeans(n_clusters=5+i).fit(data_cluster[["PCA1","PCA2"]])
    data_cluster["Cluster"] = kmeans.labels_
    data_sorted = data_cluster.sort_values(by="Cluster", ascending=False)
    lista_cluster = list(data_sorted["Cluster"])
    data_sorted = data_sorted[["PCA1","PCA2"]]

    simi = pairwise_distances(data_sorted, metric="euclidean")
    simi = pd.DataFrame(simi, index=data_cluster.index, columns=data_cluster.index)

    corte = []
    for j in range(1,len(lista_cluster)):
      if lista_cluster[j] != lista_cluster[j-1]:
        corte.append(j)

    axs[i].imshow(simi*-1)
    for cut in corte:
      axs[i].axhline(y=cut, color="red", linestyle="--", linewidth=2)
      axs[i].axvline(x=cut, color="red", linestyle="--", linewidth=2)

    silhouette = silhouette_score(simi, lista_cluster)
    title = "Silhouette coefficient = "+"{:.3f}".format(silhouette)
    axs[i-1].set_title(title, fontdict={"fontsize":14})

  plt.show()


#%%
subject_seq = pd.DataFrame({"subject":subj_idx, "label":kmeans.labels_})

#%%

states = set(subject_seq["label"])

from collections import Counter
import networkx as nx

def markov_model(n, seq, threshold=None):
    # Model n-grams
    model = {}
    grams = []
    seq = list(seq[:])# + [None]
    for i in range(len(seq)-1):
        gram = seq[i] #tuple(seq[i:i+n])
        next_item = seq[i+n]
        if gram not in model:
            model[gram] = []
            grams.append(gram)
        model[gram].append(next_item)

    # Model stats
    model_stats = {}
    for gram in model.keys():
      counts = Counter(model[gram])
      total = sum(counts.values())
      for key in counts.keys():
        counts[key] /= total
      model_stats[gram] = dict(counts)

    # Model graph
    node_list = sorted(grams)
    num_nodes = len(node_list)
    adj_matrix = np.zeros((num_nodes, num_nodes))

    for origin in range(num_nodes):
      for destination in range(num_nodes):
          try:
              adj_matrix[origin,destination] = round(model_stats[origin][destination],3)
          except:
              pass

    node_index = {node_list[x]:x for x in range(num_nodes)}
    relabel = {v:k for k,v in node_index.items()}

    if threshold != None:
        adj_matrix[adj_matrix < np.percentile(adj_matrix, threshold)] = 0

    markov_graph = nx.from_numpy_array(adj_matrix, create_using=nx.DiGraph)
    markov_graph = nx.relabel_nodes(markov_graph, relabel)

    return model, model_stats, markov_graph, adj_matrix

#%%

subj_concat = []
for i in range(29):
    subj_data = list(subject_seq[subject_seq["subject"]==i]["label"])
    subj_concat.append(subj_data)

#%%

adj_concat = []
for subj in subj_concat:
    model, model_stats, markov_graph, adj_matrix = markov_model(1, subj_data, 50)
    adj_concat.append(adj_matrix.flatten())


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
cmap = mpl.colormaps["Oranges"]

G = markov_graph.copy()
pos = nx.circular_layout(markov_graph)

fig, ax = plt.subplots(figsize=(12,12))

nx.draw_networkx_nodes(G, pos, ax=ax, node_size=750, node_color="orange")
nx.draw_networkx_labels(G, pos, ax=ax)

# curved_edges = [edge for edge in G.edges() if reversed(edge) in G.edges()]
# straight_edges = list(set(G.edges()) - set(curved_edges))
# nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
# nx.draw_networkx_edges(G, pos, ax=ax, edgelist=curved_edges, connectionstyle='arc3, rad = 0.1')

edge_dict = nx.get_edge_attributes(markov_graph,'weight')
sorted_dict = dict(sorted(edge_dict.items(), key=lambda item: item[1])) # = {k: v for k, v in sorted(edge_dict.items(), key=lambda item: item[1])}
edges, weights = zip(*sorted_dict.items())

nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, edge_color=weights,
                       width=3, edge_cmap=cmap, connectionstyle='arc3, rad = 0.1')

plt.show()


#%%

print(kmeans.cluster_centers_)

#%%
import mne

labels = mne.read_labels_from_annot('fsaverage', parc='aparc')



#%%
from mne.datasets import fetch_fsaverage

fs_dir = fetch_fsaverage() # Download fsaverage files
subjects_dir = fs_dir
src = fs_dir+"/bem/fsaverage-ico-5-src.fif" # Source Space

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

label_names = [label.name for label in labels if not label.name.startswith('unknown')]
node_colors = [label.color for label in labels if not label.name.startswith('unknown')]
stc_coords_3d = np.asarray([label.pos.mean(axis=0) for label in labels if not label.name.startswith('unknown')])

#label_names = [replacer(label.name, replace_dict) for label in labels if not label.name.startswith('Background')]
#label_names_short = [replacer(label.name, replace_dict_short) for label in labels if not label.name.startswith('Background')]
#label_network = [x[:6] for x in label_names]

#%%
print(label_names)

#%%
import pandas as pd
from itertools import combinations

nodes = pd.DataFrame(stc_coords_3d, index=label_names, columns=["x","y","z"]).reset_index(names="roi")
nodes["color"] = ["rgba"+str(color) for color in node_colors]

#%%

import pandas as pd

lh_vert_idx, lh_vert_color, lh_roi = [], [], []
rh_vert_idx, rh_vert_color, rh_roi = [], [], []

#[label for label in labels if not label.name.startswith('unknown')]):
for i, label in enumerate(labels):
      print(i)
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

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

knn.fit(lh_df[["x","y","z"]].values, lh_df["roi"])
lh_colors = [lh_colors_dict[pred] for pred in knn.predict(lh_verts_ico4)]

knn.fit(rh_df[["x","y","z"]].values, rh_df["roi"])
rh_colors = [rh_colors_dict[pred] for pred in knn.predict(rh_verts_ico4)]


#%%

targets = pd.read_csv(SPECTRAL_DIR / "target.csv", header=None)
labels = pd.read_csv(SPECTRAL_DIR / "target_labels.txt", header=None)[0].to_list()

X = np.asarray(adj_concat)

#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

reg = RandomForestRegressor(n_estimators=500)
scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

for i in range(23):
    y = targets.iloc[:,i]
    print(np.mean(cross_val_score(reg, X_scaled, y, scoring="r2", cv=5)))
    
#%%

fig, axs = plt.subplots(figsize=(32,16), nrows=2, ncols=4)
ax = axs.flatten()

for i, center in enumerate(kmeans.cluster_centers_):
    diag_values = pca.inverse_transform(center)
    color = ["red" if x>0 else "blue" for x in diag_values]
    y_pos = np.arange(len(label_names))
    
    ax[i].barh(y_pos, diag_values, align='center', color=color)
    ax[i].set_yticks(y_pos, labels=label_names)
    ax[i].set_ylim((-1,100))
    ax[i].invert_yaxis()
plt.show()