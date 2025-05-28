# # Base prices

# ## Add parquet files for the whole year together
# This will be all the data, including the outliers

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
# selected_outlier = 'outlier_sd'
# year = 2023
# flow = 'import'
# quarter = 4
# -

data_dir = Path('../data')
trade = pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob(f'{flow}_{year}_q*.parquet')
)
print(f'{trade.shape[0]} rows read from parquet files for {year}\n')

# ## Open the weight base for previous year and aggregate to section

weight_base = pd.read_parquet(f'../data/weight_base_{flow}_{year}.parquet')
print(f'{weight_base.shape[0]} rows read from parquet file ../data/weight_base_{flow}_{year}.parquet\n')

# ## Extract weight at commodity level

commodity_weights = (weight_base[weight_base.duplicated(['year', 'flow', 'comno'], keep='first') == False]
            .sort_values(['year', 'flow', 'comno'])                 
)
commodity_weights = commodity_weights[['year', 'flow', 'comno', 'Weight_HS']]

# ## Match the weight to the tradedata
# We use inner join because we only want to include those who are in the weight base

trade_with_weights = pd.merge(trade, commodity_weights, on=['year', 'flow', 'comno'], how='inner')

# Crosstab of frequencies outlier
crosstab = pd.crosstab(trade_with_weights[selected_outlier], columns='Frequency', margins=True)
# Calculate relative percentages
crosstab['Percentage (%)'] = ((crosstab['Frequency'] / crosstab.loc['All', 'Frequency']) * 100).map('{:.1f}'.format)
# Keep only 'Frequency' and 'Percentage (%)' columns
crosstab = crosstab[['Frequency', 'Percentage (%)']]
print(f'{flow.capitalize()}, {year}. Frequencies of transactions tagged as outlier')
display(crosstab)

trade_with_weights_no_outliers = trade_with_weights.copy()

# +
# Filter the DataFrame to keep only transactions where all specified conditions are False
trade_with_weights_no_outliers = trade_with_weights_no_outliers.loc[
    (trade_with_weights_no_outliers[selected_outlier] == False)

].copy()
# -

# #### Aggregate up to month per HS

aggvars = ['year', 'flow', 'comno', 'quarter', 'month', 'section', 'chapter', 
           'sitc1', 'sitc2', 'Weight_HS']
tradedata_month_base = trade_with_weights_no_outliers.groupby(aggvars, as_index=False).agg(
    weight=('weight', 'sum'),
    value=('value', 'sum'),
    n_transactions = ('n_transactions', 'mean')
)


print(f'{flow.capitalize()}, {year}. Aggregated to month.Tradedata')
show(tradedata_month_base, maxBytes=0)

# ## Calculate the price per month per HS and other measures
#

# +
tradedata_month_base['price'] = tradedata_month_base['value'] / tradedata_month_base['weight']

tradedata_month_base['price_median'] = tradedata_month_base.groupby(['flow', 'comno'])['price'].transform('median')
tradedata_month_base['price_sd'] = tradedata_month_base.groupby(['flow', 'comno'])['price'].transform('std')
tradedata_month_base['price_mean'] = tradedata_month_base.groupby(['flow', 'comno'])['price'].transform('mean')
# extra step: Calculate the standard deviation to mean ratio for each transaction
tradedata_month_base['price_mean_ratio'] = tradedata_month_base['price']/tradedata_month_base['price_mean']
tradedata_month_base['price_median_ratio'] = tradedata_month_base['price']/tradedata_month_base['price_median']
tradedata_month_base['price_sd_ratio'] = tradedata_month_base['price']/tradedata_month_base['price_sd']
tradedata_month_base['price_cv'] = tradedata_month_base['price_sd']/tradedata_month_base['price_mean']
# -

# ## Extreme price difference from median price per HS per month - Tag outliers

tradedata_month_base['outlier_time'] = np.select([tradedata_month_base['price'] / tradedata_month_base['price_median'] < 0.5,
                                           tradedata_month_base['price'] / tradedata_month_base['price_median'] > 2,
                                           ],
                                          [1, 2],
                                          default=0
                                         )
display(pd.crosstab(tradedata_month_base['outlier_time'], columns=tradedata_month_base['flow'], margins=True))

tradedata_month_base['outlier_time'] = tradedata_month_base['outlier_time'].apply(lambda x: True if x in [1, 2] else False)


# +
count_true_outliers = tradedata_month_base.groupby('comno')[['outlier_time']].sum()

print(f'{flow.capitalize()}, {year}. Outlier time per comno')
display(count_true_outliers.loc[count_true_outliers['outlier_time'] == True])
# -

# ## Extract those who are not outliers

# ### Used in further calculation:

# +
# Filter the DataFrame to keep only transactions where all specified conditions are False
trade_without_outliers = tradedata_month_base.loc[
    (tradedata_month_base['outlier_time'] == False)

].copy()

# +
# Grouping trade_without_outliers by specified variables and aggregating value and weight
trade_without_outliers_r = trade_without_outliers.groupby(
    ['year', 'quarter', 'flow', 'comno', 'section', 'Weight_HS'], 
    as_index=False
).agg(
    value_quarter=('value', 'sum'),     # Sum the value for each group
    weight_quarter=('weight', 'sum')     # Sum the weight for each group
)

trade_without_outliers_r['price'] = trade_without_outliers_r['value_quarter'] / trade_without_outliers_r['weight_quarter']
# -

# ## Transpose the data so it will be easier to impute missing prices

trade_without_outliers_r = trade_without_outliers_r.pivot(index=['year', 'flow', 'comno', 'section', 'Weight_HS'], 
                                                        columns='quarter', 
                                                        values= ['price']
                                                       )
trade_without_outliers_r.columns = trade_without_outliers_r.columns.get_level_values(0) + '_' + trade_without_outliers_r.columns.get_level_values(1).astype('str')
trade_without_outliers_r.reset_index(inplace=True)
print(f'{flow.capitalize()}, {year}. Comno with a missing quarterly price in base year')
display(trade_without_outliers_r.loc[trade_without_outliers_r.isna().any(axis=1)])

# ### Impute by section

# ## Compute values for price relatives and product

trade_without_outliers_r['price_rel_1'] = trade_without_outliers_r['price_4'] / trade_without_outliers_r['price_3']
trade_without_outliers_r['price_rel_2'] = trade_without_outliers_r['price_4'] / trade_without_outliers_r['price_2']
trade_without_outliers_r['product_1'] = trade_without_outliers_r['price_rel_1'] * trade_without_outliers_r['Weight_HS']
trade_without_outliers_r['product_2'] = trade_without_outliers_r['price_rel_2'] * trade_without_outliers_r['Weight_HS']

# ## Compute products aggregated to section

trade_without_outliers_r['prod_sum_1'] = trade_without_outliers_r.groupby(['year', 'flow', 'section'])['product_1'].transform('sum')
trade_without_outliers_r['prod_sum_2'] = trade_without_outliers_r.groupby(['year', 'flow', 'section'])['product_2'].transform('sum')

# ## Create weights by section for those who have value for product_1

trade_without_outliers_r['weight_section'] = trade_without_outliers_r['Weight_HS'] * trade_without_outliers_r['product_1'].notna()
trade_without_outliers_r['weight_section'] = trade_without_outliers_r.groupby(['year', 'flow', 'section'])['weight_section'].transform('sum')

# ## Impute prices for those who are missing
# Not all will necessarily get prices in this round

trade_without_outliers_r['impute_base'] = trade_without_outliers_r['price_4'].isna().astype('int') 
trade_without_outliers_r['price_4'] = np.where((trade_without_outliers_r['price_3'].notna()) &
                                               (trade_without_outliers_r['impute_base'] == 1),
                                               trade_without_outliers_r['price_3']*
                                               trade_without_outliers_r['prod_sum_1']/
                                               trade_without_outliers_r['weight_section'],
                                               trade_without_outliers_r['price_4']
                                              )
trade_without_outliers_r['price_4'] = np.where((trade_without_outliers_r['price_3'].isna()) &
                                               (trade_without_outliers_r['impute_base'] == 1),
                                                trade_without_outliers_r['price_2']*
                                                trade_without_outliers_r['prod_sum_2']/
                                                trade_without_outliers_r['weight_section'],
                                                trade_without_outliers_r['price_4']
                                              )
display(pd.crosstab(trade_without_outliers_r['impute_base'], columns='Frequency', margins=True))
display(trade_without_outliers_r.loc[trade_without_outliers_r.isna().any(axis=1)])

# ### Impute by Total

# ## Compute products aggregated flow

trade_without_outliers_r['prod_sum_1'] = trade_without_outliers_r.groupby(['year', 'flow'])['product_1'].transform('sum')
trade_without_outliers_r['prod_sum_2'] = trade_without_outliers_r.groupby(['year', 'flow'])['product_2'].transform('sum')

# ## Create weights by flow for those who have value for product_1

trade_without_outliers_r['weight_flow'] = trade_without_outliers_r['Weight_HS'] * trade_without_outliers_r['product_1'].notna()
trade_without_outliers_r['weight_flow'] = trade_without_outliers_r.groupby(['year', 'flow'])['weight_flow'].transform('sum')

# ## Impute prices for those who are still missing
# All rows should have prices after these operations. It is checked with a table.

trade_without_outliers_r['impute_base'] = np.where(trade_without_outliers_r['impute_base'] == 1,
                                                  trade_without_outliers_r['price_4'].isin([np.nan, np.inf]).astype('int') + 1,
                                                  trade_without_outliers_r['impute_base'])
trade_without_outliers_r['price_4'] = np.where((trade_without_outliers_r['price_3'].notna()) &
                                               (trade_without_outliers_r['impute_base'] == 2),
                                               trade_without_outliers_r['price_3']*
                                               trade_without_outliers_r['prod_sum_1']/
                                               trade_without_outliers_r['weight_flow'],
                                               trade_without_outliers_r['price_4']
                                              )
trade_without_outliers_r['price_4'] = np.where((trade_without_outliers_r['price_3'].isna()) &
                                               (trade_without_outliers_r['impute_base'] == 2),
                                               trade_without_outliers_r['price_2']*
                                               trade_without_outliers_r['prod_sum_2']/
                                               trade_without_outliers_r['weight_flow'],
                                               trade_without_outliers_r['price_4']
                                              )
display(pd.crosstab(trade_without_outliers_r['impute_base'], columns='Frequency', margins=True))
trade_without_outliers_r.rename(columns = {'price_4': 'base_price'}, inplace = True)
# Check if all have prices
if len(trade_without_outliers_r.loc[trade_without_outliers_r['base_price'].isna() | np.isinf(trade_without_outliers_r['base_price'])]) > 0:
    # Display the frequency of NaN and inf values in 'base_price'
    problematic_prices = trade_without_outliers_r.loc[trade_without_outliers_r['base_price'].isna() | np.isinf(trade_without_outliers_r['base_price']), 'base_price']
    display(pd.crosstab(problematic_prices, columns='Frequency'))
else:
    print(f'\n{flow.capitalize()}, {year}. No missing base prices.\n')

# # save prices for whole baseyear

trade_without_outliers_r.to_parquet(f'../data/prices_baseyear_{flow}_{year}.parquet')
print(f'\nNOTE: Parquet file ../data/prices_baseyear_{flow}_{year}.parquet written with {trade_without_outliers_r.shape[0]} rows and {trade_without_outliers_r.shape[1]} columns\n')

# ## Save as parquet file

baseprice = trade_without_outliers_r[['year', 'flow', 'comno', 'base_price', 'impute_base']]
baseprice.to_parquet(f'../data/base_price{flow}_{year}.parquet')
print(f'\nNOTE: Parquet file ../data/base_price{flow}_{year}.parquet written with {baseprice.shape[0]} rows and {baseprice.shape[1]} columns\n')

# ### Visualization of transactions per hs and monthly prices and outlier control time

# +
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# Ensure month is sorted numerically
def preprocess_dataset(dataset):
    dataset = dataset.sort_values(by='month')  # Sort by month (numeric)
    return dataset

# Preprocess datasets
trade_with_weights_no_outliers = preprocess_dataset(trade_with_weights_no_outliers)
tradedata_month_base = preprocess_dataset(tradedata_month_base)

# Function to filter data for a specific comno and plot it on given axes
def plot_transactions_for_comno(ax, dataset, comno_value, x_var, y_var, hue):
    filtered_data = dataset[dataset['comno'] == comno_value]
    sns.scatterplot(data=filtered_data, x=x_var, y=y_var, 
                    hue=hue, ax=ax, palette='muted', legend='full')
    ax.set_title(f'{flow.capitalize()}, {year}. Transactions for comno {comno_value} - Detection_method: {hue}')
    ax.set_xlabel(x_var)
    ax.set_ylabel(y_var)

# Function to update plots based on selected comno, dataset, and axes
def update_plot(comno_value, dataset_name, x_var, y_var):
    dataset = {
        'trade_with_weights': trade_with_weights,
        'trade_with_weights_no_outliers': trade_with_weights_no_outliers
    }[dataset_name]
    
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))  # Only two plots now


    # Second plot
    plot_transactions_for_comno(axs[0], dataset, comno_value, x_var, y_var, selected_outlier)
    axs[0].set_title(f'{flow.capitalize()}, {year}. Transaction data - prices for comno {comno_value}')
    
    # Third plot
    plot_transactions_for_comno(axs[1], tradedata_month_base, comno_value, x_var, y_var, 'outlier_time')
    axs[1].set_title(f'{flow.capitalize()}, {year}. Monthly data - Unit values per month for comno {comno_value}')
    
    plt.tight_layout()
    plt.show()

# Specify the comno values and sort them
comno_values = sorted(trade_with_weights_no_outliers['comno'].unique().tolist())

# Define list of axis variables
axis_variables = ['price', 'trans_n', 'value', 'weight', 'month', 'quarter']

# Create widgets
comno_dropdown = widgets.Combobox(
    options=comno_values,
    value=str(comno_values[0]),
    description='Search comno:',
    layout=widgets.Layout(width='300px'),
    placeholder='Type or select a comno'
)

dataset_dropdown = widgets.Dropdown(
    options=['trade_with_weights', 'trade_with_weights_no_outliers'],
    value='trade_with_weights',
    description='Select dataset:',
    layout=widgets.Layout(width='300px')
)

x_dropdown = widgets.Dropdown(
    options=axis_variables,
    value='value',
    description='Select X:',
    layout=widgets.Layout(width='300px')
)

y_dropdown = widgets.Dropdown(
    options=axis_variables,
    value='price',
    description='Select Y:',
    layout=widgets.Layout(width='300px')
)

output = widgets.interactive_output(update_plot, {
    'comno_value': comno_dropdown,
    'dataset_name': dataset_dropdown,
    'x_var': x_dropdown,
    'y_var': y_dropdown
})

display(dataset_dropdown, comno_dropdown, x_dropdown, y_dropdown, output)

# -




