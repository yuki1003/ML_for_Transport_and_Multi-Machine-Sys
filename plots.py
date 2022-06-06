"""
Authors:
Group:
Description: this file will be used for data analysis 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime as datetime
import pyarrow.parquet as pq
import seaborn as sb

from data_processing import df_trips

#%% Plotting heat maps
#sb.distplot(df_trips[(df_trips['fare_amount'] >= 2.5)&(df_trips['fare_amount'] <= 150)]['fare_amount'])


sb.distplot(df_trips[(df_trips['tip_amount']>= 0.0 & df_trips['tip_amount'] <= 50)])