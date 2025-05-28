# # Read csv file from external trade and add some new columns

# ## Read csv file
# We use the pandas read_csv to import the file to a Python pandas dataframe. With the dtype parameter we decide the column types.

# + active=""
# import pandas as pd
# import numpy as np
# import json
# from pathlib import Path
# from itables import init_notebook_mode, show
# import matplotlib.pyplot as plt
# import seaborn as sns
# #pd.set_option('display.float_format',  '{:18,.0}'.format)
# pd.set_option('display.float_format', lambda x: f'{x:15,.0f}' if abs(x)>1e5 else f'{x:15.2f}')
# import itables
# # Initialize interactive display mode
# itables.init_notebook_mode(all_interactive=True)
#
#
# year = 2020
# quarter = 1
# flow = 'export'

# +
dtyp = {
    "flow": "object",
    "year": "int32",
    "month": "object",
    "comno": "object",
    "HS_description": "object",
    "ItemID": "object",
    "Partner": "object",
    #"valUSD": "float64",
    "value": "float64",
    "weight": "float64",
    "quantity": "float64",
    "unit": "object"
}



xmpi_names = ['flow', 
              'year',
              'month',
              'comno',
              'HS_description',
              'ItemID',
              'Partner',
              'valUSD',
              'value',
              'weight',
              'quantity',
              'unit',
              'quarter'
]


# -

def read_trade_data(path, prefix, flow, year, quarter, suffix):
    trade_data = pd.read_csv(f'{path}{prefix}{flow}_{year}Q{quarter}.{suffix}',
                             sep=';', 
                             decimal=',',
                             #dtype=dtyp,
                             names=xmpi_names,
                             skiprows=1,
                             header=None,
                             na_values={'.', ' .', 'missing', 'N/A'}
                             )
    #trade_data['flow'] = np.where(trade_data['CPC'] == '4000', '1', '2')
    #trade_data['country'] = np.where(trade_data['flow'] == '2', trade_data['Corigin'], trade_data['Partner'])
    #trade_data['quarter'] = (((trade_data['month'].astype(int) - 1) // 3) + 1).astype('str')
    print(f'{trade_data.shape[0]} rows read from csv file {path}{prefix}{flow}_{year}Q{quarter}.{suffix}.\n')
    return trade_data


# +
tradedata = read_trade_data(path='../data/',
            prefix='',
            flow=flow,
            year=year,
            quarter=quarter,
            suffix='txt')

# After reading the data into trade_data:
for col, col_type in dtyp.items():
    tradedata[col] = tradedata[col].astype(col_type)

tradedata['valUSD'] = pd.to_numeric(tradedata['valUSD'], errors='coerce')

# Convert entire 'comno' column to string explicitly first
tradedata['comno'] = tradedata['comno'].astype(str)

# Now add leading zero to those with length 7
tradedata.loc[tradedata['comno'].str.len() == 7, 'comno'] = '0' + tradedata.loc[tradedata['comno'].str.len() == 7, 'comno']


tradedata
# -

# ## Read parquet files
# Parquet files with correspondances to sitc and section

sitccat = pd.read_parquet('../cat/commodity_sitc.parquet')
print(f'{sitccat.shape[0]} rows read from parquet file ../cat/commodity_sitc.parquet\n')
sectioncat = pd.read_parquet('../cat/chapter_section.parquet')
print(f'{sectioncat.shape[0]} rows read from parquet file ../cat/sectioncat.parquet\n')

# ## Merge trade data with sitc catalog
# We add sitc and sitc2 from the correspondance table

# +
# Perform the merge
t_sitc = pd.merge(tradedata, sitccat, on='comno', how='left', indicator=True)

# Display the result of the merge
print(f'Result of merge with SITC catalog for {flow}, for {year}q{quarter}:')
display(pd.crosstab(t_sitc['_merge'], columns='Frequency', margins=True))

# Check if there are any "left_only" entries
left_only_data = t_sitc.loc[t_sitc['_merge'] == 'left_only']
if not left_only_data.empty:
    print(f'List of commodity numbers that do not have SITC codes for {flow}, for {year}q{quarter}:')
    
    # Crosstab for the 'left_only' entries
    display(pd.crosstab(left_only_data['comno'], columns='Frequency'))
else:
    print(f"No missing SITC codes for {flow}, for {year}q{quarter}.")

# Clean up by dropping the '_merge' column
t_sitc.drop(columns='_merge', inplace=True)

# -

# ## Merge trade data with chapter catalog
# We add section from the correspondance table

# + active=""
# t_sitc['chapter'] = t_sitc['comno'].str[0:2]
# t_section = pd.merge(t_sitc, sectioncat, on='chapter', how='left', indicator=True)
# print(f'Result of merge with chapter catalog for {flow}, for {year}q{quarter}:')
# display(pd.crosstab(t_section['_merge'], columns='Frequency', margins=True))
# if len(t_section.loc[t_section['_merge'] == 'left_only']) > 0:
#     print(f'List of chapters that do not have section code for {flow}, for {year}q{quarter}:')
#     display(pd.crosstab(t_section.loc[t_section['_merge'] == 'left_only', 'chapter'], columns='Frequency', margins=True))
# t_section.drop(columns='_merge', inplace=True)

# +
# Convert 'comno' to string first, then extract first two characters
t_sitc['chapter'] = t_sitc['comno'].astype(str).str[0:2]

# Perform the merge with section catalog
t_section = pd.merge(t_sitc, sectioncat, on='chapter', how='left', indicator=True)

# Display the merge result
print(f'Result of merge with chapter catalog for {flow}, for {year}q{quarter}:')
display(pd.crosstab(t_section['_merge'], columns='Frequency', margins=True))

# Check for unmatched chapters
left_only_chapters = t_section.loc[t_section['_merge'] == 'left_only']
if not left_only_chapters.empty:
    print(f'List of chapters that do not have section code for {flow}, for {year}q{quarter}:')
    display(pd.crosstab(left_only_chapters['chapter'], columns='Frequency'))
else:
    print(f"No missing section codes for {flow}, for {year}q{quarter}.")

# Drop the helper column
t_section.drop(columns='_merge', inplace=True)

# -

# ## Check if month is missing

print('rows with NaN in month column: ',t_section['month'].isna().sum())  # This will show how many NaN values are present


# ## print number of rows that have 0 in weight or value

# +
rows_with_zero = t_section[(t_section['weight'] == 0) | (t_section['value'] == 0)]
print("Number of rows with 0 in weight or value:", len(rows_with_zero))

#t_section['weight'] = np.where(t_section['weight'] == 0, 1, t_section['weight'])
# -

# ## Choose whether to use weight or quantity for the UV-weight

use_quantity = pd.read_excel('../cat/use_quantity.xlsx', dtype=str)
use_quantity_list = use_quantity['use_quantity'].tolist()

t_section['weight'] = np.where(t_section['comno'].isin(use_quantity_list), t_section['quantity'], t_section['weight'])

# ## Save as parquet
# The quarter file is save as a parquet file

t_section.to_parquet(f'../data/{flow}_{year}q{quarter}.parquet')
print(f'\nNOTE: Parquet file ../data/{flow}_{year}q{quarter}.parquet written with {t_section.shape[0]} rows and {t_section.shape[1]} columns\n')


