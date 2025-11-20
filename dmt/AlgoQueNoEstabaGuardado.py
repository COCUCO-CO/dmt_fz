
import pickle

def save_file(data, folder, file):
    with open(folder+file+".pkl", 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_file(file):
    with open(file, 'rb') as handle:
        return pickle.load(handle)

#%%

import os

path = path = "E:\\FedeZ\\DMT\\fwd-inv-stc\\"

files_list = []
for cond in ["EO\\", "EC\\", "DMT\\"]:
    files_names = sorted([x for x in os.listdir(path+cond) if x.startswith('syncro')])
    files_list += [path+cond+x for x in files_names]


#%%
import numpy as np

from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD as SVD
# from sklearn.manifold import Isomap
# from sklearn.manifold import MDS
# from sklearn.manifold import TSNE

from sklearn.metrics import pairwise_distances
#from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

import optuna
from optuna.samplers import GridSampler


def objective(trial):
    n_comps = trial.suggest_int("n_comps", 2, 25)
    k_clusters = trial.suggest_int("k_clusters", 2, 25)
    decomposer = PCA(n_components=n_comps)
    data = decomposer.fit_transform(eigen_stc_array)
    kmedoids = KMedoids(n_clusters=k_clusters).fit(data) #, n_init="auto", random_state=0
    cluster_labels = kmedoids.labels_
    simi = pairwise_distances(data, metric="euclidean")
    silhouette = silhouette_score(simi, cluster_labels)
    return silhouette

#value_range = list(range(2,9))+[25,50,75]
search_space = {'n_comps': list(range(2,11))+[15,20,25],
                'k_clusters': list(range(2,21))}

#study = optuna.create_study(sampler=GridSampler(search_space))  # Create a new study.
#study.optimize(objective, n_jobs=-1)#, n_trials=1000)  # Invoke optimization of the objective function.


#%%
from tqdm import tqdm

band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

eigen_eeg_list = []
eigen_stc_list = []

for splits in [0]:
    result_dict = {}
    for band in band_list:
        for file in tqdm(files_list):
            subject_data = load_file(file)

            #eigen_eeg = subject_data["eigen_eeg"][band][splits]
            eigen_stc = subject_data["eigen_stc"][band][splits]
            
            #eigen_eeg_list.append(eigen_eeg)
            eigen_stc_list.append(eigen_stc)
        
        all_epochs = []
        for epoch_list in eigen_stc_list:
            for epoch in epoch_list:
                all_epochs.append(epoch)

        eigen_stc_array = np.asarray(all_epochs)
        study = optuna.create_study(study_name="this_study", direction="maximize", sampler=GridSampler(search_space))  # Create a new study.
        study.optimize(objective, n_jobs=20)#, n_trials=1000)  # Invoke optimization of the objective function.
        df = study.trials_dataframe()
        result_dict[band] = df[["params_k_clusters","params_n_comps","value"]].sort_values("value", ascending=False)
        del study
    save_file(result_dict, path, "clusters_splits"+str(splits))
        

#%%

# from tqdm import tqdm

# band_list = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

# result_dict = {}
# for band in band_list:
#     for file in tqdm(files_list):
#         subject_data = load_file(file)

#         #amplitudes_eeg = subject_data["amplitudes_eeg"][band]
#         amplitudes_stc = subject_data["amplitudes_stc"][band]
        
#         num_epochs = amplitudes_stc.shape[0]
#         #num_channels = amplitudes_stc.shape[1]
#         #num_samples = amplitudes_stc.shape[2]
        
#         for epoch in range(num_epochs):
#             signal = amplitudes_stc[epoch,:,:]
#             median = np.median(signal, axis=1)
#             thresholded = np.where(signal/median[:,None]>=1, 1, 0)
            
        
# save_file(result_dict, path, "clusters_splits"+str(splits))


#%%
# eyes_open_files = sorted([x for x in os.listdir(path+"EO\\") if x.startswith('syncro')])
# eyes_open_files_list = [path+"EO\\"+x for x in eyes_open_files]

# eyes_closed_files = sorted([x for x in os.listdir(path+"EC\\") if x.startswith('syncro')])
# eyes_closed_files_list = [path+"EC\\"+x for x in eyes_closed_files]

# dmt_files = sorted([x for x in os.listdir(path+"DMT\\") if x.startswith('syncro')])
# dmt_files_list = [path+"DMT\\"+x for x in dmt_files]

# files_list = eyes_open_files_list + eyes_closed_files_list + dmt_files_list


#%%

# from sklearn.model_selection import GridSearchCV
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import make_scorer

# class KMedoidsWrapper(KMedoids):
#     def predict(self, X,):
#         return self.labels_

# # The scorer function
# def silhouette_scorer(cluster_labels, data): #y_pred, y_true
#     simi = pairwise_distances(data, metric="euclidean")
#     silhouette = silhouette_score(simi, cluster_labels)
#     return silhouette

# pca = PCA()
# kmedoids = KMedoidsWrapper()

# pipe = Pipeline(steps=[("decomposer", pca), ("clustering", kmedoids)])
# param_grid = {"decomposer__n_components": np.arange(2,9),
#               "clustering__n_clusters": np.arange(2,9)}

# search = GridSearchCV(pipe, param_grid, scorer=make_scorer(silhouette_scorer), n_jobs=-1)

#%%

# def explore_grid(eigen_array, n_comps_range, k_clusters_range):
#     heatmap = np.zeros((len(n_comps_range),len(k_clusters_range)))
#     for i, n_comps in enumerate(n_comps_range):
#         for j, k_clusters in enumerate(k_clusters_range):
#             decomposer = PCA(n_components=n_comps)
#             data = decomposer.fit_transform(eigen_array)
#             kmedoids = KMedoids(n_clusters=k_clusters) #, n_init="auto", random_state=0
#             kmedoids.fit(data) 
#             cluster_labels = kmedoids.labels_
#             simi = pairwise_distances(data, metric="euclidean")
#             silhouette = silhouette_score(simi, cluster_labels)
#             print(n_comps,k_clusters,silhouette)
#             heatmap[i, j] = silhouette
#     return heatmap

# value_range = list(range(2,9))+[25,50,75]
# heatmap = explore_grid(eigen_stc_array, value_range, value_range)

#%%

# import optuna
# from optuna.terminator import TerminatorCallback
# from optuna.terminator import report_cross_validation_scores as report_score

# decomps = ["pca", "svd", "isomap", "mds", "tsne"]

# def objective(trial):
#     n_comps = trial.suggest_int("n_comps", 2, 25)
#     k_clusters = trial.suggest_int("k_clusters", 2, 25)
    
#     #decomp = trial.suggest_categorical('decomp', decomps)
#     #if decomp=="pca":
#     decomposer = PCA(n_components=n_comps)
#     #if decomp=="svd": decomposer = SVD(n_components=n_comps)
#     #if decomp=="isomap": decomposer = Isomap(n_components=n_comps) #n_neighbors=5, radius=None, 
#     #if decomp=="mds": decomposer = MDS(n_components=n_comps)
#     #if decomp=="tsne": decomposer = TSNE(n_components=n_comps)
#     data = decomposer.fit_transform(eigen_stc_array)
    
#     kmedoids = KMedoids(n_clusters=k_clusters).fit(data) #, n_init="auto", random_state=0
#     cluster_labels = kmedoids.labels_
#     simi = pairwise_distances(data, metric="euclidean")
#     silhouette = silhouette_score(simi, cluster_labels)
#     #report_score(trial, [silhouette])
    
#     return silhouette