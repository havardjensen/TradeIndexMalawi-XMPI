# <div style="text-align: center;">
# <img src="NSO.png" alt="alt text" title="Title" height="50%"/>
# </div>    
#
# # Import and export price index - XMPI
#     
# **Yearly (1. quarter) - Selection of HS, Calculation of weights and base price**
#
# This program runs all the programs that shall be executed yearly. Most of the other programs are included and executed from here. 
#
# Important steps:
#
# - Select limits for outliers
# - Select parameteres for sample selection - What is acceptable price variation for each commodity. Stricter parameter --> less heterogeniety
# - Check coverage - How much of the population does the sample cover.
# - Check imputation of base prices

# ## Import the necessary Python packages
# If the packages are not installed, use `pip install` from a terminal window to install, for instance `pip install pyarrow` and `pip install jupytext --upgrade`
#
# for ipywidgets and widgets in jupyterlab you might have to install spesific to the "user": `pip install ipywidgets --upgrade --user` and `pip install jupyterlab_widgets --user`
#
# Most packages can be installed by activating cell below:

# +
# #!pip install pyarrow itables
# #!pip install upsetplot
# #!pip install matplotlib-venn
# #!pip install jupytext --upgrade
# #!pip install ipywidgets
# #!pip install ipywidgets --upgrade
# #!pip install plotly
# -

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import itables
import matplotlib.pyplot as plt
from itables import init_notebook_mode, show
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display
#pd.set_option('display.float_format',  '{:18,.0}'.format)
pd.set_option('display.float_format', lambda x: f'{x:15,.0f}' if abs(x)>1e5 else f'{x:15.2f}')
# Initialize interactive display mode
itables.init_notebook_mode(all_interactive=True)

# ## Select year and flow for selection of sample and calculation of base price

year = 2024
flow = 'import'

# ## Import chapter and section correspondance

# + active=""
# correspondance_file = f'../cat/chapter_section.parquet'
# if os.path.exists(correspondance_file):
#     answer = input(f"File {correspondance_file} already exists, do you want to import it again (y/n)", )
#     if answer.lower() == 'y':
#         exec(open("005 Import_chapter_catalog.py").read())
#     else:
#         print(f"File {correspondance_file} not replaced")
# else:
#     exec(open("005 Import_chapter_catalog.py").read())   
# -

# ## Import commodity and sitc correspondance

# + active=""
# correspondance_file = f'../cat/commodity_sitc.parquet'
# if os.path.exists(correspondance_file):
#     answer = input(f"File {correspondance_file} already exists, do you want to import it again (y/n)", )
#     if answer.lower() == 'y':
#         exec(open("007 Import_commodities_catalog.py", encoding="utf-8").read())
#     else:
#         print(f"File {correspondance_file} not replaced")
# else:
#     exec(open("007 Import_commodities_catalog.py").read())           
# -

# ## Import files for selected year

quarter = 1
exec(open("T010 Read trade quarter.py").read())
quarter = 2
exec(open("T010 Read trade quarter.py").read())
quarter = 3
exec(open("T010 Read trade quarter.py").read())
quarter = 4
exec(open("T010 Read trade quarter.py").read())

# ## Outlier control

#
# **Standard deviations from the mean:**
#
# This method assesses how far each data point is from the mean (average) value of the dataset. It uses the concept of standard deviation, which measures the spread or variability of the data around the mean.
#
# **Absolute Deviation from Median (MAD):**
#
# The Median Absolute Deviation (MAD) measures the distance between each data point and the median of the dataset. It’s robust against outliers and skewed data because the median is less affected by extreme values than the mean.

# ## Choose which outliers to be removed. 
# #### outlier_sd = 1 iteration of standard deviation from mean 
# #### outlier_sd2 = 2 iteration of standard deviation from mean 
# #### outlier_MAD = 1 iteration of absolute deviation from median
#
# ## outlier_sd is recommended

selected_outlier = 'outlier_sd'

# #### Set treshold for what is acceptable dispersion for prices at transactionlevel for each HS.

# +
# Accepted treshold for standard deviation from mean (default=2)
outlier_sd = 2

# Deviation from median (default=3)
outlier_dev_median = 3

# +
quarter = 1
exec(open("T015 Outlier control quarter.py").read())

quarter = 2
exec(open("T015 Outlier control quarter.py").read())

quarter = 3
exec(open("T015 Outlier control quarter.py").read())

quarter = 4
exec(open("T015 Outlier control quarter.py").read())
# -

# ## Create weight base data and delete outliers

exec(open("A10M CreateWeightBasePopulation.py").read())

# ## Create weight base population
# This syntax will select the commodities to use for the index for the next year. It will be the base for that index. We set the parameters for selecting the commodities here

share_total=0.001
no_of_months=2
no_of_months_seasons=2
section_seasons='II'
price_cv=0.5
max_by_min=10
max_by_median=5
median_by_min=5
share_small=0.0001
n_transactions_year = 2
exec(open("A20M CreateWeightBase.py").read())
basedata

# ## Calculate the base prices

exec(open("A30M Base_price.py").read())
baseprice


