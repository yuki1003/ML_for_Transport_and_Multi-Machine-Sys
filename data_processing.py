"""
Authors:
Group:
Description: This file will only be used for data processing
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

pq_trips = pq.read_table("yellow_tripdata_2022-03.parquet")
df_trips = trips.to_pandas()

print(df_trips.head())