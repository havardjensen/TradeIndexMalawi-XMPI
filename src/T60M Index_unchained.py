# # Create unchained index

# ## Read parquet file with unchained indexes for previous quarter(s)
# This is done from quarter 2 every year

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
#
# import itables
# # Initialize interactive display mode
# itables.init_notebook_mode(all_interactive=True)
# init_notebook_mode(all_interactive=True)
#
# year = 2024
# quarter = 1
# flow = 'import'
#
# -

if quarter > 1:
    indexunchainedfile = f'../data/index_unchained_{flow}_{year}.parquet'
    index_unchained_previous = pd.read_parquet(indexunchainedfile)
    print(f'{index_unchained_previous.shape[0]} rows read from parquet file {indexunchainedfile}\n')

# Calculate base year as the previous year

year_base = year - 1

year_base

# ## Read parquet files

# +
imputefile = f'../data/price_impute_{flow}_{year}q{quarter}.parquet'
price_impute = pd.read_parquet(imputefile)
print(f'{price_impute.shape[0]} rows read from parquet file {imputefile}\n')

basepricefile = f'../data/base_price{flow}_{year_base}.parquet'
baseprice = pd.read_parquet(basepricefile)
print(f'{baseprice.shape[0]} rows read from parquet file {basepricefile}\n')
baseprice.drop(columns='year', inplace=True)
# -

baseprice

price_impute

# ## Merge prices with base prices

prices = pd.merge(price_impute, baseprice, on=['flow', 'comno'], how='left', indicator=True)
display(pd.crosstab(prices['_merge'], columns='Frequency', margins=True))
prices.drop(columns='_merge', inplace=True)

prices

# ### Extra check of price change in chaining period

check = (prices['price_rel'] > 1.40) | (prices['price_rel'] < 0.6)
selected_columns = ['comno', 'sitc1', 'HS_sum', 'price_rel']
outliers = prices[check][selected_columns]


if quarter == 1:
    print('Extra check before establishing chaining for new short indices')
    print('Do an extra check for these commodities before they are accepted as good unit values. Compare data between baseperiod and current period.')
    print('Focus on important commodities')
    display(outliers)
else:
    print()


# ## Calculate unchained index and index weight for commodities

prices['index_unchained'] = prices['price'] / prices['base_price'] * 100
prices['index_weight'] = prices['index_unchained'] * prices['Weight_HS']
prices['level'] = 'Commodity'
prices['series'] = prices['comno']

prices

# ### Mark comnos within special series

special_series1 = pd.read_excel('../cat/special_series1.xlsx', dtype=str)
prices_special1 = pd.merge(prices, special_series1, on=['comno'], how='right', indicator=True)
display(pd.crosstab(prices_special1['_merge'], columns='Frequency', margins=True))
prices_special1.drop(columns='_merge', inplace=True)

prices_special1

# ### Totals without

special_series2_total_without = pd.read_excel('../cat/special_series2_total_without.xlsx', dtype=str)
prices_special2 = pd.merge(prices, special_series2_total_without, on=['comno'], how='left', indicator=True)
display(pd.crosstab(prices_special2['_merge'], columns='Frequency', margins=True))
prices_special2.drop(columns='_merge', inplace=True)

total_without_diamonds = prices_special2[~(prices_special2['special_serie2'] == 'Total_without_diamonds')].copy()
total_without_fish = prices_special2[~(prices_special2['special_serie2'] == 'Total_without_fish')].copy()
total_without_fuel = prices_special2[~(prices_special2['special_serie2'] == 'Total_without_fuel')].copy()


# ## Function for creating unchained index for a series

def unchained(dataframe, groupvars:list, level, series):
    index_agg = dataframe.groupby(groupvars, as_index=False).agg(
        Weight_HS=('Weight_HS', 'sum'),
    index_weight=('index_weight', 'sum')
    )
    index_agg['index_unchained'] = index_agg['index_weight'] / index_agg['Weight_HS']
    index_agg['level'] = level
    index_agg['series'] = index_agg[series]
    return index_agg


# ## Unchained index for section

# +
#index_agg = prices.groupby(['year', 'quarter', 'flow', 'section'], as_index=False).agg(
#    Weight_HS=('Weight_HS', 'sum'),
#    index_weight=('index_weight', 'sum')
#)
#index_agg['index_unchained'] = index_agg['index_weight'] / index_agg['Weight_HS']
#index_agg['level'] = 'Section'
#index_agg['series'] = 'section'

index_section = unchained(prices, groupvars=['year', 'quarter', 'flow', 'section'], level='Section', series='section')
# -

path = f'../data/index_section_ex.csv'
index_section.to_csv(path)

# ## Unchained index for total

index_total = unchained(prices, groupvars=['year', 'quarter', 'flow'], level='Total', series='flow')

# +
total = index_total.copy()  # Create a copy of index_total

# Assign the values of Weight_HS to Weight_total
total['Weight_total'] = total['Weight_HS']

# Keep only the Weight_total column
total = total[['Weight_total']]
# -

# ## Unchained index for sitc1 and sitc2

index_sitc1 = unchained(prices, groupvars=['year', 'quarter', 'flow', 'sitc1'], level='Sitc1', series='sitc1')
index_sitc2 = unchained(prices, groupvars=['year', 'quarter', 'flow', 'sitc2'], level='Sitc2', series='sitc2')

# ### Special series (combination of different HS)

index_special_series = unchained(prices_special1, groupvars=['year', 'quarter', 'flow', 'special_serie'], level='special_serie', series='special_serie')

# ### Special series (Total without combination of different HS)

total_without_diamonds_index = unchained(total_without_diamonds, groupvars=['year', 'quarter', 'flow'], level='special_serie', series='flow')
total_without_diamonds_index['series'] = 'Total_without_diamonds'
total_without_diamonds_index['special_series2'] = 'Total_without_diamonds'
total_without_fish_index = unchained(total_without_fish, groupvars=['year', 'quarter', 'flow'], level='special_serie', series='flow')
total_without_fish_index['series'] = 'Total_without_fish'
total_without_diamonds_index['special_series2'] = 'Total_without_fish'
total_without_fuel_index = unchained(total_without_fuel, groupvars=['year', 'quarter', 'flow'], level='special_serie', series='flow')
total_without_fuel_index['series'] = 'total_without_fuel'
total_without_fuel_index['special_series2'] = 'total_without_fuel'


# ## Add index files together

indexes = [prices, index_section, index_sitc1, index_sitc2, index_total , index_special_series, total_without_diamonds_index , total_without_fish_index, total_without_fuel_index]
index_unchained = pd.concat(indexes)
varlist = ['flow', 'year', 'quarter', 'Weight_HS', 'index_unchained', 'level', 'series']
index_unchained = index_unchained[varlist]

# +
# Step 1: Get the Weight_HS where level is 'Total'
total_weights = index_unchained[index_unchained['level'] == 'Total']['Weight_HS']

# Step 2: Divide Weight_HS by the corresponding total weight
index_unchained['share_total'] = (index_unchained['Weight_HS'] / total_weights.values[0])*100
# -

columns_to_drop = ['Weight_HS']  # Specify the column names you want to drop
index_unchained.drop(columns=columns_to_drop, inplace=True)

# ## Add indexes to previous quarters
# We will also delete previous version if a quarter is executed again

if quarter > 1:
    index_unchained = pd.concat([index_unchained, index_unchained_previous])
    groupvars = ['flow', 'level', 'series', 'year', 'quarter']
    index_unchained = index_unchained[index_unchained.duplicated(groupvars, keep='last') == False].sort_values(groupvars)

# Create a crosstab with the percentage change displayed
display(pd.crosstab(
    [index_unchained['level'], index_unchained['series']], 
    columns=[index_unchained['year'], index_unchained['quarter']],
    values=index_unchained['index_unchained'],
    aggfunc='mean'
))

# +
index_unchained1 = index_unchained.copy()

# Calculate lagged values
index_unchained1['lag_index_unchained'] = np.where(index_unchained1['quarter'] > 1,
    index_unchained1['index_unchained'].shift(1),
    100  # Set to 100 if quarter <= 1
)

# Compute percentage_change based on the conditions
index_unchained1['percentage_change'] = np.where(
    index_unchained1['quarter'] > 1,
    (index_unchained1['index_unchained'] / index_unchained1['lag_index_unchained'] - 1) * 100,
    ((index_unchained1['index_unchained'] / 100) - 1) * 100
)

# Round the change to 1 decimal place
index_unchained1['percentage_change'] = index_unchained1['percentage_change'].round(1)

# Optionally, drop the lag column if not needed
index_unchained1.drop(columns=['lag_index_unchained'], inplace=True)

# Ensure year and quarter are integers
index_unchained1['year'] = index_unchained1['year'].astype(int)
index_unchained1['quarter'] = index_unchained1['quarter'].astype(int)

# Create crosstab for index_unchained1
crosstab_index = pd.crosstab(
    [index_unchained1['level'], index_unchained1['series']], 
    columns=[index_unchained1['year'], index_unchained1['quarter']],
    values=index_unchained1['index_unchained'],
    aggfunc='mean'
)

# Create crosstab for percentage_change
crosstab_index2 = pd.crosstab(
    [index_unchained1['level'], index_unchained1['series']], 
    columns=[index_unchained1['year'], index_unchained1['quarter']],
    values=index_unchained1['percentage_change'],
    aggfunc='mean'
)

# Create crosstab for share_total
crosstab_index3 = pd.crosstab(
    [index_unchained1['level'], index_unchained1['series']], 
    columns=[index_unchained1['year'], (index_unchained1['quarter'] == 0).astype(int)],
    values=index_unchained1['share_total'],
    aggfunc='mean'
).round(2)

# Convert the percentage_change values to formatted strings with a percentage sign
crosstab_index2 = crosstab_index2.astype(str) + '%'

# Convert the percentage_change values to formatted strings with a percentage sign
crosstab_index3 = crosstab_index3.astype(str) + '%'

# Merge both crosstabs
final_result = pd.concat([crosstab_index, crosstab_index2, crosstab_index3], axis=1, keys=['Index Unchained', 'Percentage Change', 'Share_total'])

# Display the final result
print(f"{flow.capitalize()}. Crosstab of Index Unchained and Percentage Change:")
display(final_result)

# -

# ## Save result as parquet file

indexunchainedfile = f'../data/index_unchained_{flow}_{year}.parquet'
index_unchained.to_parquet(indexunchainedfile)
print(f'\nNOTE: Parquet file {indexunchainedfile} written with {index_unchained.shape[0]} rows and {index_unchained.shape[1]} columns\n')




