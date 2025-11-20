
import pandas as pd
import numpy as np

folder = "E:\\FedeZ\\DMT\\spectral_sources\\"
aal90_labels = pd.read_csv(folder+"AAL90.csv", delimiter=";", index_col=0)["Label"].to_list()
pre_theta = pd.read_csv(folder+'theta.csv', header=None, names=aal90_labels)

labels = list(pd.read_csv(folder+"target_labels.txt", header=None)[0])
targets = pd.read_csv(folder+'target.csv', header=None, names=labels)


#%%

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from tqdm import tqdm

model = RandomForestRegressor(n_estimators=1000, n_jobs=-1)
r_pearson = []
r2_coeffdet = []

for i in tqdm(range(1000)):
  X_train, X_test, y_train, y_true = train_test_split(pre_theta, targets.iloc[:,0], test_size=0.5, random_state=i)
  
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  r2score = r2_score(y_true.to_numpy(), y_pred)
  rpearson, pvalue = pearsonr(y_true.to_numpy(), y_pred)
  r2_coeffdet.append(r2score)
  r_pearson.append(rpearson)

#%%

print(sum(r_pearson)/len(r_pearson))
print(sum(r2_coeffdet)/len(r2_coeffdet))