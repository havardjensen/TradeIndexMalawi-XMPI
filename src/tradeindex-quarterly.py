# <div style="text-align: center;">
# <img src="NSA-logo.svg" alt="alt text" title="Title" height="50%"/>
# </div>
#
# # Import and export price index - XMPI
#
# **Quarterly production**

# This program runs all the needed programs for creating the index for a quarter. Most of the other programs are included and executed from here. The parameteres will change for each period

# ## Import the necessary Python packages
# If the packages are not installed, use `pip install` from a terminal window to install, for instance `pip install pyarrow` and `pip install jupytext --upgrade`

# +
# #!pip install pyarrow itables
# #!pip install dash
# -

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from itables import init_notebook_mode, show
import matplotlib.pyplot as plt
import seaborn as sns
import itables
#pd.set_option('display.float_format',  '{:18,.0}'.format)
pd.set_option('display.float_format', lambda x: f'{x:15,.0f}' if abs(x)>1e5 else f'{x:15.2f}')
# Initialize interactive display mode
itables.init_notebook_mode(all_interactive=True)

# ## Tradedata for a quarter

year = 2022
quarter  = 1
flow = 'import'
exec(open("T010 Read trade quarter.py").read())

# ## Outlier control

# ### Set treshold for what is acceptable dispersion for prices at transactionlevel for each HS.
#
# **Standard deviations from the mean:**
#
# This method assesses how far each data point is from the mean (average) value of the dataset. It uses the concept of standard deviation, which measures the spread or variability of the data around the mean.
#
# **Absolute Deviation from Median (MAD):**
#
# The Median Absolute Deviation (MAD) measures the distance between each data point and the median of the dataset. It’s robust against outliers and skewed data because the median is less affected by extreme values than the mean.
#
# **Acceptable price change from base price**
#
# The limits are set at own discretion
#
# ## Choose which outliers to be removed. 
# #### outlier_sd = 1 iteration of standard deviation from mean, 
# #### outlier_sd2 = 2 iteration of standard deviation from mean, 
# #### outlier_MAD = 1 iteration of absolute deviation from median
#
# ## outlier_sd is recommended and should be the same as in yearly program

selected_outlier = 'outlier_sd'

# +
# Deviation from median
outlier_dev_median = 5

# Deviation from mean
outlier_sd = 2

#Treshold price change
price_limit_low = 0.5
price_limit_high = 1.5
# -

exec(open("T015 Outlier control quarter.py").read())

# ### Price control quarter
# Check the data for extreme prices

exec(open("T40M Price_control.py").read())

# ## Impute prices quarter

exec(open("T50M Impute_prices.py").read())

# ## Calculate unchained index quarter

exec(open("T60M Index_unchained.py").read())

# ## Calculate chained index 
# This will start when all quarters of the first index year have been processed. There is ine syntax for chaining the first year and one for further chaining.

# ### Attention! only change first_index_year, when updating reference year (not the same as base year)

first_index_year = 2022 # This should not be changed unless we start again with a new reference year
if (year == first_index_year) & (quarter == 4):
    exec(open("T71M Chain_first_year.py").read())
elif year > first_index_year:
    exec(open("T72M Chain_next_years.py").read())
else:
    print("Not executed because we are executing one of the first 3 quarters of the first year. The chained index starts when all quarters of the first index year are executed.")

# ## Visualization of index

exec(open("T80M_Graph.py").read())


