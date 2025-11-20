from skimage import color
from matplotlib import cm
from matplotlib.colors import to_rgb, to_rgba #,rgb2hex
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from itertools import product

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection

cmap = plt.get_cmap('plasma')

folder = "E:\\FedeZ\\DMT\\spectral_sources\\"

aal90_labels = pd.read_csv(folder+"AAL90.csv", delimiter=";",index_col=0)["Label"].to_list()
labels = pd.read_csv(folder+'target_labels.txt', header=None)[0].to_list()
target = pd.read_csv(folder+'target.csv', names=labels)
pre_theta = pd.read_csv(folder+'theta.csv', names=aal90_labels)


def to_gray(x):
  # RGBA to RGB
  rgba = cm.twilight_shifted(x)
  rgb = to_rgb(rgba)
  # RGB to Gray
  gray = color.rgb2gray(np.asarray(rgb))
  return cm.gray(gray)


def PlotCorr(band, fdr=True, alpha=0.05, pvalue=0.05, absolute=True):
  survived_dict = {x:[] for x in range(20)}
  rpearson_matrix = np.zeros((90,20))
  pvalues_matrix = np.ones((90,20))
  corr_matrix = np.zeros((4,90,20))

  for roi, score in product(range(90), range(20)):
    array1 = band.iloc[:,roi].to_numpy()
    array2 = target.iloc[:,score].to_numpy()
    r, p = pearsonr(array1, array2, alternative="less") #alternative="two-sided")
    rpearson_matrix[roi,score] = (abs(r) if absolute==True else r)
    pvalues_matrix[roi,score] = p
    corr_matrix[:, roi,score] = to_rgba(cm.plasma(abs(r)))

  max_value = np.max(rpearson_matrix)
  min_value = np.min(rpearson_matrix)

  #rpearson_matrix = (rpearson_matrix - min_value) / (max_value - min_value)

  for roi, score in product(range(90), range(20)):
   r = rpearson_matrix[roi,score]
   corr_matrix[:, roi,score] = to_rgba(cm.plasma(abs(r)))

  if fdr==False:
    for roi, score in product(range(90), range(20)):
      if pvalues_matrix[roi,score] > pvalue:
        corr_matrix[:, roi,score] = to_gray(abs(rpearson_matrix[roi,score]))
        #rpearson_matrix[roi,score] = 0
      else:
        survived_dict[score].append(roi)

  if fdr==True:
    p_corrected = fdrcorrection(pvalues_matrix.flatten(),alpha=alpha)[0].reshape(90,20) #method="negcorr"
    for roi, score in product(range(90), range(20)):
      if p_corrected[roi,score] != True:
        corr_matrix[:,roi,score] = to_gray(abs(rpearson_matrix[roi,score]))
        #rpearson_matrix[roi,score] = 0
      else:
        survived_dict[score].append(roi)

  fig, ax = plt.subplots(figsize=(60,36))
  ax.imshow(np.transpose(corr_matrix), origin='lower')

  ax.set_xticks(np.arange(90))
  ax.set_yticks(np.arange(20))

  ax.set_xticklabels(aal90_labels, fontsize=10, rotation='vertical')
  #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
  ax.set_yticklabels(target.columns[:-3], fontsize=10)

  #norm = plt.Normalize(radii[0], radii[-1])
  m = plt.cm.ScalarMappable(cmap=cmap) #norm
  m.set_array([min_value, max_value])
  divider = make_axes_locatable(ax)
  cax = divider.append_axes("right", size="1%", pad=0.07)
  plt.colorbar(m, cax=cax).set_label('r Pearson', size=15)

  for i in range(20):
    for j in range(90):
      ax.text(j, i, str(rpearson_matrix[j, i])[:5],
              ha="center", va="center", color="k", fontsize=6)

  plt.show()
  return survived_dict

#%%

survived_dict = PlotCorr(pre_theta, fdr=True, alpha=0.05, absolute=False)

#%%

from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

scores_dict = {}
for score in survived_dict:
  X = pre_theta.iloc[:,survived_dict[score]]
  y = target.iloc[:,score]
  if X.shape[1] > 5:
    quest = labels[score]
    print(quest, X.shape)
    scores_dict[quest] = []
    for i in tqdm(range(1000)):
      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
      regr = RandomForestRegressor(n_estimators=1000)
      regr.fit(X_train, y_train)
      y_pred = regr.predict(X_test)
      r, p = pearsonr(y_test, y_pred)
      scores_dict[quest].append(r)