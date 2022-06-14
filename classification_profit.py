# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 15:51:00 2022

@author: Yuki1
"""

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import seaborn as sb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from copy import deepcopy

#import data
pq_taxi = pq.read_table("data/yellow_tripdata_processed.parquet")
df_taxi = pq_taxi.to_pandas()

X = df_taxi['passenger_count',
            'trip_distance',
            'PULocationID',
            'DOLocationID',
            'pickup_time',
            'day_of_week']

y = ...

# Split All Data to train data & test data
# X_train,X_test,y_train,y_test=train_test_split(X,
                                               # df_taxi['has_tipped'],
                                               # test_size=0.25,
                                               # random_state=0)

# clf = LogisticRegression().fit(X_train, y_train)
# clf.predict()