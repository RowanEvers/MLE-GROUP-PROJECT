# Gradient Boosted regression model with tuned parameters from the training code 
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd 

final_regression_model = GradientBoostingRegressor(learning_rate=np.float64(0.055144535042022436),
                          max_depth=2, max_features='sqrt', min_samples_leaf=6,
                          n_estimators=121, random_state=42,
                          subsample=np.float64(0.8483662071069908))   # parameters taken from training file from the randomised 


df_train = pd.read_excel('TrainDataset2025.xls')
exists = df_train.isin([999]).any().any()
print(exists)
# since 999 are Na values:
df_train.replace(999, np.nan, inplace=True)
# drop any na rows 
X = df_train.drop(['ID','pCR (outcome)','RelapseFreeSurvival (outcome)'], axis=1)
y = df_train['RelapseFreeSurvival (outcome)']

# import the test data 
df_test_final = pd.read_excel('FinalTestDataset2025.xls')

X_test_final = df_test_final.drop(['ID'], axis=1)

# create the pipeline so we can impute and scale:
preprocess = Pipeline(steps=[('scaler', StandardScaler()),
                       ('imputer',KNNImputer()) ]) 

# fit using the whole train dataset
gb_pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('model', final_regression_model)
])

X = df_train.drop(['ID','pCR (outcome)','RelapseFreeSurvival (outcome)'], axis=1)
y = df_train['RelapseFreeSurvival (outcome)']
# test with the test dataset 
gb_pipeline.fit(X,y)
y_pred = final_regression_model.predict(X_test_final)
print(y_pred)
pd.DataFrame(y_pred, columns=['RelapseFreeSurvival (prediction)']).to_csv('RFSPrediction.csv', index=False)