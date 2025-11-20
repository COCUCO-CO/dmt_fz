

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import pearsonr

import pandas as pd
import numpy as np

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

for band, X_data in bands.items():
    for target, y_data in targets.iteritems():
        
        train = np.concatenate([X_train, y_train], axis=1)
        test = np.concatenate([X_test, y_test], axis=1)
        
        X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
        
        predictor = TabularPredictor(label=target, eval_metric=metric)
        predictor.fit(train, ds_args={"validation_procedure":"cv", "n_folds":5})
        
        best_model = predictor._trainer.load_model("LightGBMXT").model
        pred_proba = best_model.predict(test.iloc[:,:-1])

#%%

from sklearn.model_selection import ShuffleSplit

ShuffleSplit(n_splits=1, test_size=0.2, random_state=0).get_n_splits(X)
5
>>> print(rs)
ShuffleSplit(n_splits=5, random_state=0, test_size=0.25, train_size=None)
>>> for i, (train_index, test_index) in enumerate(rs.split(X)):

for target in labels:
    
    dictio["model_"+metric] = {}
    
    