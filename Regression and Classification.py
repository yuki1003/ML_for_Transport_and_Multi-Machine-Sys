#!/usr/bin/env python
# coding: utf-8

# ### To do:
# - delete month + distance variables
# - validate model
# - change weekend to friday 20:00 in data preparation
# -

# In[98]:


"""
Authors:
Group:
Description: this file will be used for clustering
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
import pyarrow.parquet as pq
import seaborn as sb
from sklearn.cluster import KMeans 
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgbm
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
#from sklearn import cluster
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from lightgbm import LGBMClassifier
from matplotlib import pyplot


# In[32]:


from sklearn import metrics
from sklearn.ensemble import GradientBoostingRegressor


# In[48]:


import seaborn as sns


# In[2]:


#import data
pq_taxi = pq.read_table("yellow_tripdata_processed.parquet")
df_taxi = pq_taxi.to_pandas()


# Predict:
#     - fare amount
#     - trip duration
#    
#    
#   - Attributes need to be numerical 
#   - We need the year, month, date, day of week, hour etc. 
#   - dropoff_longitude & dropoff_latitude (or code the zone's?) --> better to work with latitude and longitude. Coding the zone's will be very confusing as there are many zone's. 
#   
#   
#   Columns:
#   - pickup_longitude
#   - pickup_latitude
#   - dropoff_longitude
#   - dropoff_latitude
#   - passenger count
#   - year
#   - month
#   - date
#   - day of week
#   - hour
#   - distance
#   
#   (['fare_amount', 'pickup_longitude', 'pickup_latitude',
#        'dropoff_longitude', 'dropoff_latitude', 'passenger_count',
#        'H_Distance', 'Year', 'Month', 'Date', 'Day of Week', 'Hour'],

# In[4]:


df_taxi.head()


# In[5]:


df_taxi.columns


# In[6]:


df_taxi.dtypes


# In[7]:


len(df_taxi)


# In[8]:


#first try for smaller dataset
df_taxi_sampled = df_taxi.sample(n=10000)
len(df_taxi_sampled)


# ## Correlation

# In[52]:


col = ['trip_distance', 'passenger_count' ,'fare_amount', 'tip_amount', 'total_amount', 'trip_duration_minutes', 'pickup_day_no', 'pickup_month', 'pickup_hour', 'total_amount']
#label = ['Distance', 'Fare', 'Tip', 'Total$', 'Duration', '', '']

# plot correlation plot
fig, ax = plt.subplots(figsize=(16, 5))
corr_matrix = df_taxi.corr()
sns.heatmap(corr_matrix, annot=True, ax=ax)
ax.set_title("Correlation Between Continuous Attributes")
plt.show()


# ## Regression models

# In[16]:


#select only the necessary columns as X
#Datetime not included, due to dtype
X = df_taxi_sampled[["PULocationID", "DOLocationID", "trip_distance", "passenger_count","fare_amount","tip_amount","trip_duration_minutes","pickup_day_no","pickup_month","pickup_hour",]]


# In[10]:


X.dtypes


# In[11]:


y = df_taxi_sampled["total_amount"]


# In[12]:


y.dtype


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)


# In[18]:


X_train.columns


# ### Random forest prediction

# In[19]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)


# In[20]:


print(rf_predict)


# In[22]:


rf.score(X_test, y_test)


# ### Gradiënt boosting with LightGBM

# In[24]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': -1,
        'verbose': 0,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.6,
        'reg_aplha': 1,
        'reg_lambda': 0.001,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1     
    }


# In[25]:


pred_test_y = np.zeros(X_test.shape[0])
pred_test_y.shape


# In[26]:


train_set = lgbm.Dataset(X_train, y_train, silent=True)
train_set


# In[27]:


#reset to 300 boosting rounds
model = lgbm.train(params, train_set = train_set, num_boost_round=50)


# In[28]:


pred_test_y = model.predict(X_test, num_iteration = model.best_iteration)


# In[29]:


print(pred_test_y)


# In[42]:


# ALTERNATIVE MODEL?

# fit the model
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)
# predict and evaluate

gbr_train_pred = reg.predict(X_train)
gbr_test_pred = reg.predict(X_test)
train_rmse = mean_squared_error(y_train, gbr_train_pred, squared=False)
test_rmse = mean_squared_error(y_test, gbr_test_pred, squared=False)
train_r2 = r2_score(y_train, gbr_train_pred)
test_r2 = r2_score(y_test, gbr_test_pred)

print("Gradient Boost Regression")
print("Train RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
print("Train R2", train_r2)
print("Test R2", test_r2)


# In[43]:


reg.feature_importances_


# ### Gradiënt boosting with XGBoost

# In[34]:


#!pip install xgboost


# In[36]:


dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
#dtest = xgb.DMatrix(test_df)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_params = {
    'min_child_weight': 1, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.7, 
    'max_depth': 10,
    'subsample': 0.7,
    'n_estimators': 50, #set back to 5000
    'n_jobs': -1, 
    'booster' : 'gbtree', 
    'silent': 1,
    'eval_metric': 'rmse'}

model = xgb.train(xgb_params, dtrain, 700, watchlist, early_stopping_rounds=100, maximize=False, verbose_eval=50)


# In[37]:


y_train_pred = model.predict(dtrain)
y_pred = model.predict(dvalid)
print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')


# In[ ]:





# ## Classification models

# In[53]:


df_taxi.head()


# In[54]:


print(df_taxi["total_amount"].max())
print(df_taxi["total_amount"].min())
print(df_taxi["total_amount"].mean())


# In[55]:


import seaborn as sns
sns.displot(df_taxi, x="total_amount")


# In[56]:


df_taxi.sort_values('total_amount')


# In[72]:


#creating categories
#spread evenly over categories?
df_taxi['bins'] = pd.cut(x=df_taxi['total_amount'], bins=[1, 25, 50, 75, 100, 200])
df_taxi.head()


# In[73]:


#renaming categories
df_taxi['bins'] = df_taxi['bins'].cat.rename_categories(["Low", "Medium","High","Very High","Extremely High"])
df_taxi.head()


# In[74]:


df_taxi['bins'].dtype


# In[75]:


df_taxi_sampled_2 = df_taxi.sample(n=10000)


# In[76]:


X2 = df_taxi_sampled_2[["PULocationID", "DOLocationID", "trip_distance", "passenger_count","tip_amount","trip_duration_minutes","pickup_day_no","pickup_month","pickup_hour",]]


# In[ ]:


#lightgbm.LGBMClassifier


# In[77]:


y2 = df_taxi_sampled_2["bins"]


# In[83]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=25)


# In[85]:


# evaluate lightgbm algorithm for classification
# define the model
model = LGBMClassifier()
# evaluate the model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
n_scores = cross_val_score(model, X_train2, y_train2, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(n_scores), std(n_scores)))


# #### Exploring the number of trees

# In[96]:


# get a list of models to evaluate
def get_models():
	models = dict()
	trees = [3, 5, 7, 9, 11, 13]
	for n in trees:
		models[str(n)] = LGBMClassifier(n_estimators=n)
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train2, y_train2, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores


# In[99]:


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# #### Exploring the tree depth

# In[100]:


# get a list of models to evaluate
def get_models():
	models = dict()
	for i in range(1,11):
		models[str(i)] = LGBMClassifier(max_depth=i, num_leaves=2**i)
	return models
 
# evaluate a give model using cross-validation
def evaluate_model(model):
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X_train2, y_train2, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores
 
# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# #### Exploring Boosting type

# In[101]:


# get a list of models to evaluate
def get_models():
	models = dict()
	types = ['gbdt', 'dart', 'goss']
	for t in types:
		models[t] = LGBMClassifier(boosting_type=t)
	return models


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# #### Final model predictions

# In[103]:


#FINAL MODEL, MAKE PREDICTIONS
# make predictions using lightgbm for classification
model = LGBMClassifier()
# fit the model on the whole dataset
model.fit(X_test2, y_test2)
# make a single prediction
yhat = model.predict(X_test2)


# In[104]:


yhat


# In[105]:


X_test['predicted_class'] = yhat


# In[106]:


X_test


# In[125]:


data = X2.columns.to_list()
df = pd.DataFrame(data, columns=['Features'])
df['Importance'] = model.feature_importances_
df


# In[128]:


def plotImp(model, X , num = 20, fig_size = (40, 20)):
    feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})
    plt.figure(figsize=fig_size)
    sns.set(font_scale = 5)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", 
                                                        ascending=False)[0:num])
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances-01.png')
    plt.show()


# In[129]:


plotImp(model, X_test2)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ## Predicting trip_distance

# In[151]:


#select only the necessary columns as X
X = df_taxi_sampled[['passenger_count', 'PULocationID', 'DOLocationID',
       'fare_amount', 'tip_amount', 'total_amount', 'tip_percent',
       'trip_duration_minutes', 'pickup_hour',
       'pickup_month', 'pickup_day_no']]


# In[152]:


X.dtypes


# In[153]:


y = df_taxi_sampled["trip_distance"]


# In[154]:


y.dtype


# In[155]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)


# #### Random forest prediction

# In[156]:


rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_predict = rf.predict(X_test)


# In[157]:


X_test['predicted_distance'] = rf_predict
X_test


# In[158]:


print(rf_predict)


# In[160]:


rf.score(X_test, y_test)


# In[ ]:




