from autogluon.tabular import TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split

path="C:\\Users\\xochipilli\\Desktop\\"
dataset = pd.read_excel(path+"ABC_proc_speed.xlsx").iloc[:,1:]
target = pd.read_excel(path+"NPS_calc_proc_speed.xlsx")["imp_Proc_speed"]
dataset["target"] = target

train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=32, stratify=dataset["target"])

predictor = TabularPredictor(label="target", eval_metric="f1_weighted", verbosity=0, sample_weight='balance_weight')
predictor.fit(train_data = train_data) #, presets="best_quality")
predictor.leaderboard(test_data)

#%%

import shap
best_model = predictor._trainer.load_model("LightGBM").model
used_features = [int(x) for x in best_model.feature_name()]
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(test_data.loc[:,used_features])
shap.summary_plot(shap_values,test_data.loc[:,used_features])



