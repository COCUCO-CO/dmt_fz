from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


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

stc_coords_2d = stc_coords_3d[:,:2]

#%%
import os
from copy import deepcopy
from tqdm import tqdm
import numpy as np

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
cond_list = ["DMT", "EC", "EO"]

eigen_dict = {cond: {band:[] for band in deepcopy(band_list)} for cond in deepcopy(cond_list)}
kuramoto_dict = {cond: {band:[] for band in deepcopy(band_list)} for cond in deepcopy(cond_list)}

path = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"


# for cond in cond_list:
#     print(cond)
#     files_names = sorted([x for x in os.listdir(path+cond) if x.startswith('syncro-')])
#     for file in tqdm(files_names):
#         data_dict = load_file(path+cond+"\\"+file)
#         for band in band_list:
#             kuramoto_dict[cond][band] += data_dict["kuramoto_stc"][band][0]
#             for epoch in data_dict["syncros_stc"][band][0]:
#                 eigen_dict[cond][band].append(np.diag(np.linalg.eigh(epoch)[1]))

# #save_file(eigen_dict, path, "eigen_corrected")

#%%
# for cond in cond_list:
#     print(cond)
#     files_names = sorted([x for x in os.listdir(path+cond) if x.startswith('syncro')])
#     for file in tqdm(files_names):
#         data_dict = load_file(path+cond+"\\"+file)
#         for band in band_list:
#             eigen_dict[cond][band] += data_dict["eigen_stc"][band][0]
#             kuramoto_dict[cond][band] += data_dict["kuramoto_stc"][band][0]

# save_file(eigen_dict, path, "eigen_all")
# save_file(kuramoto_dict, path, "kuramoto_all")

#%%

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

# band = "Beta"

# eigen_dict = load_file(path+"eigen_all.pkl")

# x_dmt = np.asarray(eigen_dict["DMT"][band])
# x_ec = np.asarray(eigen_dict["EC"][band])
# x_eo = np.asarray(eigen_dict["EO"][band])

# kuramoto_dict = load_file(path+"kuramoto_all.pkl")

# r_dmt = np.asarray(kuramoto_dict["DMT"][band]).mean(axis=1)
# r_ec = np.asarray(kuramoto_dict["EC"][band]).mean(axis=1)
# r_eo = np.asarray(kuramoto_dict["EO"][band]).mean(axis=1)

# x_dmt = x_dmt[reject_outliers(r_dmt)]
# x_ec = x_ec[reject_outliers(r_ec)]
# x_eo = x_eo[reject_outliers(r_eo)]

# x_np = np.concatenate((x_dmt,x_ec,x_eo))

#%%

import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
#from sklearn.metrics import pairwise_distances
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score

from fast_pytorch_kmeans import KMeans
from torchclustermetrics import silhouette         

def reject_outliers(data, m=2):
    return abs(data - np.mean(data)) < m * np.std(data)

max_k = 15
min_k = 2

device = torch.device("cuda")


eigen_dict = load_file(path+"eigen_correct.pkl")
kuramoto_dict = load_file(path+"kuramoto_all.pkl")

fig, ax = plt.subplots(figsize=(8*5,8), ncols=5)

for i, band in enumerate(band_list):
    print(band)
    
    x_dmt = np.asarray(eigen_dict["DMT"][band])
    x_ec = np.asarray(eigen_dict["EC"][band])
    x_eo = np.asarray(eigen_dict["EO"][band])
    
    r_dmt = np.asarray(kuramoto_dict["DMT"][band]).mean(axis=1)
    r_ec = np.asarray(kuramoto_dict["EC"][band]).mean(axis=1)
    r_eo = np.asarray(kuramoto_dict["EO"][band]).mean(axis=1)
    
    x_dmt = x_dmt[reject_outliers(r_dmt)]
    x_ec = x_ec[reject_outliers(r_ec)]
    x_eo = x_eo[reject_outliers(r_eo)]
    
    x_all = np.concatenate((x_dmt,x_ec,x_eo))

    x = torch.from_numpy(deepcopy(x_all)).float()
    x_cuda = x.to(device=device)
    
    scores = []
    for j in range(20):
        scores.append([])
        for k in range(min_k, max_k):
            kmeans = KMeans(n_clusters=k, mode='cosine', verbose=0)
            labels = kmeans.fit_predict(x_cuda)
            score = silhouette.score(x_cuda, labels)
            scores[j].append(score)
        ax[i].plot(scores[j])
    title = ("DMT" if case==0 else "All")
    ax[i].set_title(band+" - "+title)
    ax[i].set_xticks(range(len(scores)))

        
plt.show()


#%%

scores


#%%

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


band = "Delta"

x_dmt = np.asarray(eigen_dict["DMT"][band])
x_ec = np.asarray(eigen_dict["EC"][band])
x_eo = np.asarray(eigen_dict["EO"][band])

r_dmt = np.asarray(kuramoto_dict["DMT"][band]).mean(axis=1)
r_ec = np.asarray(kuramoto_dict["EC"][band]).mean(axis=1)
r_eo = np.asarray(kuramoto_dict["EO"][band]).mean(axis=1)

x_dmt = x_dmt[reject_outliers(r_dmt)]
x_ec = x_ec[reject_outliers(r_ec)]
x_eo = x_eo[reject_outliers(r_eo)]

x_all = np.concatenate((x_dmt,x_ec,x_eo))

diags = deepcopy(x_all)
k = 3

x = torch.from_numpy(diags).float()
x_cuda = x.to(device=device)

kmeans = KMeans(n_clusters=k, mode='cosine', verbose=0)
labels = kmeans.fit_predict(x_cuda)
# score = silhouette.score(x_cuda, labels)

fig, ax = plt.subplots(figsize=(8,8), nrows=2, ncols=k)

import matplotlib as mpl
import networkx as nx

pos = stc_coords_2d

for i in range(k):
    center = np.asarray(kmeans.centroids[i,:].cpu()).reshape(-1,1)
    mat = center @ np.transpose(center)
    ax[0][i].imshow(scaler.fit_transform(mat))
    
    np.fill_diagonal(mat, 0)
    syncro_graph = nx.from_numpy_array(mat)
    edge_dict = nx.get_edge_attributes(syncro_graph,'weight')
    sorted_dict = dict(sorted(edge_dict.items(), key=lambda item: item[1])) # = {k: v for k, v in sorted(edge_dict.items(), key=lambda item: item[1])}
    edges, weights = zip(*sorted_dict.items())
    weights = scaler.fit_transform(np.asarray(weights).reshape(-1,1))
    colors = [mpl.colormaps["Blues"](x) for x in weights]

    nx.draw_networkx_nodes(syncro_graph, ax=ax[1][i], pos=pos)
    nx.draw_networkx_labels(syncro_graph, ax=ax[1][i], pos=pos)
    nx.draw_networkx_edges(syncro_graph, ax=ax[1][i], pos=pos, edgelist=edges, edge_color=colors)
    ax[1][i].axis('off')

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



    
        # fig, ax = plt.subplots(figsize=(8,8))
        # ax.set_title(band)
        # ax.imshow(heatmap)

        # ax.set_yticks(range(0,max_comps-min_comps), labels=range(min_comps,max_comps))
        # for i in range(max_comps-min_comps):
        #     for j in range(max_k-min_k):
        #         text = ax.text(j, i, str(heatmap[i, j])[:5],
        #                       ha="center", va="center", color="k", fontsize=8)
        # plt.show()


%%

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# x_np = scaler.fit_transform(x_np)

#%%

# band = "Alpha"
# hemi = "both"
# net = "DMN"

# order_dict = load_file(path+"r_kuramoto_nets_epochs.pkl")

# x_dmt = np.asarray(order_dict["DMT\\"][band][hemi][net])
# x_ec = np.asarray(order_dict["EC\\"][band][hemi][net])
# x_eo = np.asarray(order_dict["EO\\"][band][hemi][net])

#%%

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()

from fast_pytorch_kmeans import KMeans
import numpy as np
import torch
from torchclustermetrics import silhouette
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

#pca = PCA(n_components=3)
#pca_data = pca.fit_transform(x_np)

device = torch.device("cuda")
x_np = x_dmt#np.concatenate((x_dmt,x_ec,x_eo))
x = torch.from_numpy(scaler.fit_transform(x_np)).float()
x_cuda = x.to(device=device)

con_mats = []
for k in range(2,13):
    kmeans = KMeans(n_clusters=k, mode='euclidean', verbose=0)
    labels = kmeans.fit_predict(x_cuda)
    score = silhouette.score(x_cuda, labels)
    print(score)
    
    fig, ax = plt.subplots(ncols=k)
    for i in range(k):
        centroids = np.asarray(kmeans.centroids[i].cpu()).reshape(-1,1)
        con_mat = centroids * np.transpose(centroids)
        ax[i].imshow(con_mat)
        #original = scaler.inverse_transform(centroids)
        #con_mat = original * np.transpose(original)
        #con_mats.append([original, diag, con_mat])
    plt.show()
    
#save_file(con_mats, path, "con_mat")



#%%
x_np = scaler.fit_transform(np.concatenate((x_dmt,x_ec,x_eo)))
colors = len(x_dmt)*[0] + len(x_ec)*[1] + len(x_eo)*[2]

pca = PCA(n_components=2)
pca_data = pca.fit_transform(x_np)
dmt_pca = pca.transform(scaler.transform(x_dmt))
ec_pca = pca.transform(scaler.transform(x_ec))
eo_pca = pca.transform(scaler.transform(x_eo))

fig, ax = plt.subplots()
#ax = fig.add_subplot()#projection='3d')
ax.scatter(pca_data[:,0], pca_data[:,1], c=labels.cpu())#, c=colors)#labels.cpu())
#ax.scatter(dmt_pca[:,0], dmt_pca[:,1])#, c=colors)#labels.cpu())
#ax.scatter(ec_pca[:,0], ec_pca[:,1])#, c=colors)#labels.cpu())
#ax.scatter(eo_pca[:,0], eo_pca[:,1])#, c=colors)#labels.cpu())
ax.set_aspect('equal')
plt.show()

#%%

# fig, ax = plt.subplots()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(dmt_pca[:,0], dmt_pca[:,1], dmt_pca[:,2], alpha=0.1)#, c=colors)#labels.cpu())
# ax.scatter(ec_pca[:,0], ec_pca[:,1], ec_pca[:,2], alpha=0.1)#, c=colors)#labels.cpu())
# ax.scatter(eo_pca[:,0], eo_pca[:,1], eo_pca[:,2], alpha=0.1)#, c=colors)#labels.cpu())
# ax.set_aspect('equal')
# plt.show()


#%%

from scipy.stats import norm
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fig, ax = plt.subplots()
ax.hist(dmt_pca[:,1], alpha=0.5, bins=30, density=True)
ax.hist(ec_pca[:,1], alpha=0.5, bins=30, density=True)
ax.hist(eo_pca[:,1], alpha=0.5, bins=30, density=True)

xmin, xmax = ax.get_xlim()
x = np.linspace(xmin, xmax, 100)

mu1, std1 = norm.fit(dmt_pca[:,1])
mu2, std2 = norm.fit(ec_pca[:,1])
mu3, std3 = norm.fit(eo_pca[:,1])

p1 = norm.pdf(x, mu1, std1)
p2 = norm.pdf(x, mu2, std2)
p3 = norm.pdf(x, mu3, std3)

ax.plot(x, p1, colors[0], linewidth=2)
ax.plot(x, p2, colors[1], linewidth=2)
ax.plot(x, p3, colors[2], linewidth=2)

ax.axvline(x=mu1, linestyle="--", linewidth=2, color=colors[0])
ax.axvline(x=mu2, linestyle="--", linewidth=2, color=colors[1])
ax.axvline(x=mu3, linestyle="--", linewidth=2, color=colors[2])

plt.show

#%%