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
#from sklearn import cluster

#import data
pq_taxi = pq.read_table("data/yellow_tripdata_processed.parquet")
df_taxi = pq_taxi.to_pandas()

#%% Looking at time periods of the day with most pickups

def get_sec(time_str):
    #converts a string in format %H:%M:%S into number of seconds since midnight
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

pickup_hour_min_sec = [date.strftime('%H:%M:%S') for date in df_taxi['tpep_pickup_datetime']]
pickup_seconds = [[get_sec(time_str)] for time_str in pickup_hour_min_sec]

# performing K-means clustering
kmeans_clusters = KMeans(n_clusters=20, random_state=5)
kmeans_clusters.fit(pickup_seconds)
pickup_seconds_clusters_df = pd.DataFrame(pickup_seconds, columns=["pickup_seconds"])
pickup_seconds_clusters_df['cluster'] = kmeans_clusters.labels_

# getting top 3 clusters of time periods of the day with most pick-ups
top3seconds_clusters = pickup_seconds_clusters_df['cluster'].value_counts().index[0:3]

# printing the top 3 time periods of the day with most pick-ups
labels = ['1st', '2nd', '3rd']
for i in range(0,3):
    print(labels[i] + ' best time period')
    cluster_df = pickup_seconds_clusters_df[pickup_seconds_clusters_df.cluster == top3seconds_clusters[i]]
    min_seconds = min(cluster_df.pickup_seconds)
    min_minute, min_second = divmod(min_seconds, 60)
    min_hour, min_minute = divmod(min_minute, 60)
    max_seconds = max(cluster_df.pickup_seconds)
    max_minute, max_second = divmod(max_seconds, 60)
    max_hour, max_minute = divmod(max_minute, 60)
    print("\t%d:%02d:%02d to %d:%02d:%02d" % (min_hour, min_minute, min_second, max_hour, max_minute, max_second))
