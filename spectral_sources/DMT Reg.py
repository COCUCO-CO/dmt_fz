from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from scipy.stats import pearsonr

import pandas as pd
import numpy as np

from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt

from tqdm import tqdm

#%%
folder = "E:\\FedeZ\\DMT\\spectral_sources\\"

bands = {
    "Alpha": pd.read_csv(folder+'alpha.csv', header=None),
    "Beta": pd.read_csv(folder+'beta.csv', header=None),
    "Delta":  pd.read_csv(folder+'delta.csv', header=None),
    "Gamma 1": pd.read_csv(folder+'gamma1.csv', header=None),
    "Gamma 2": pd.read_csv(folder+'gamma2.csv', header=None),
    "Theta": pd.read_csv(folder+'theta.csv', header=None),
    "DMT Alpha": pd.read_csv(folder+'DMT_alpha.csv', header=None),
    "DMT Beta": pd.read_csv(folder+'DMT_beta.csv', header=None),
    "DMT Delta": pd.read_csv(folder+'DMT_delta.csv', header=None),
    "DMT Gamma 1": pd.read_csv(folder+'DMT_gamma1.csv', header=None),
    "DMT Gamma 2": pd.read_csv(folder+'DMT_gamma2.csv', header=None),
    "DMT Theta": pd.read_csv(folder+'DMT_theta.csv', header=None)
}

labels = list(pd.read_csv(folder+"target_labels.txt", header=None)[0])
targets = pd.read_csv(folder+'target.csv', header=None)
targets.columns = labels

#%%

networks = {
    "FPN": np.array([7, 8, 11, 12, 13, 14, 61, 62, 65, 66])-1,
    "DMN 1": np.array([23, 24, 31, 32, 35, 36, 65, 66, 67, 68])-1,
    "DMN 2": np.array([59,60,61,62,85,86])-1,
    "DMN A": np.array([29,30,31,32,87,88])-1,
    "DMN V": np.array([35,36,37,38,39,40, 55, 56, 65, 66, 67, 68])-1,
    "Sensorimotor": np.array([1,2,7,8,19,20,57,58,63,64,69,70])-1,
    "Visual": np.array([43,44,45,46,47,48,49,50,51,52,54])-1,
    "Frontal": np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32])-1,
    "Parietal": np.array([33,34,35,36,57,58,59,60,61,62,63,64,65,66,67,68,69,70])-1,
    "Temporal": np.array([37,38,39,40,79,80,81,82,83,84,85,86,87,88,89,90])-1,
    "Subcortical": np.array([41,42,71,72,73,74,75,76,77,78])-1,
    "Occipital": np.array([43,44,45,46,47,48,49,50,51,52,53,54,55,56])-1,
    }

#%%

reg = RandomForestRegressor(n_estimators=200)
reg_s = RandomForestRegressor(n_estimators=200)

scaler = StandardScaler()

history = []

for target in range(targets.shape[1]):
  for network_name, network in networks.items():
    for band_name, band in bands.items():
      r2_list = []
      rpearson_list = []
      y_pred_list = []
      y_true_list = []

      r2_list_s = []
      rpearson_list_s = []
      y_pred_list_s = []
      y_true_list_s = []

      X = np.array(band)
      X = X[:, network]
      y = np.array(targets.iloc[:,target])
      target_name = targets.iloc[:,target].name
      
      print(target_name,"-",network_name,"-",band_name)

      for i in tqdm(range(200)): 
        X_train, X_test, y_train, y_test, y_train_s, y_test_s = train_test_split(X, y, shuffle(y), test_size=0.2)

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        y_train = scaler.fit_transform(y_train.reshape(-1,1))
        y_test = scaler.transform(y_test.reshape(-1,1))
        y_train_s = scaler.fit_transform(y_train_s.reshape(-1,1))
        y_test_s = scaler.transform(y_test_s.reshape(-1,1))

        reg.fit(X_train, np.ravel(y_train))

        y_pred = reg.predict(X_test)
        y_true = np.ravel(y_test)
        r2 = r2_score(y_true, y_pred)
        r, pvalue = pearsonr(y_true, y_pred)
        r2_list.append(r2)
        rpearson_list.append(r)
        y_pred_list.append(y_pred)
        y_true_list.append(y_true)

        reg_s.fit(X_train, np.ravel(y_train_s))

        y_pred_s = reg_s.predict(X_test)
        y_true_s = np.ravel(y_test_s)
        r2_s = r2_score(y_true_s, y_pred_s)
        r_s, pvalue_s = pearsonr(y_true_s, y_pred_s)
        r2_list_s.append(r2_s)
        rpearson_list_s.append(r_s)
        y_pred_list_s.append(y_pred_s)
        y_true_list_s.append(y_true_s)

      dictio = {
          "Target": target_name,
          "Network": network_name,
          "Band": band_name,
          "R2": r2_list,
          "r Pearson": rpearson_list,
          "y_pred": y_pred_list,
          "y_true": y_true_list,
          "R2 Shuffle": r2_list_s,
          "r Pearson Shuffle": rpearson_list_s,
          "y_pred Shuffle": y_pred_list_s,
          "y_true Shuffle": y_true_list_s
          }
      history.append(dictio)
      r_mean = sum(rpearson_list)/len(rpearson_list)
      print("r mean:", r_mean,"\n")
      
#%%

import pickle

with open('history_explore.pickle', 'wb') as handle:
    pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    