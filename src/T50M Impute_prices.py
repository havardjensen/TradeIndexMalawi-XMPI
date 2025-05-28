# # Impute missing prices

# Calculate base year and previous quarter

# year= 2019
# quarter=1
# flow='export'

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
# year = 2023
# quarter = 1
# flow = 'import'
# price_limit_low = 0.3
# price_limit_high = 2.5
#
# -

year_base = year - 1
quarter_1 = 4 if quarter == 1 else quarter - 1
#year_1 = year - 1

# ## Read parquet files

# +
if quarter == 1:
    basepricefile = f'../data/base_price{flow}_{year_base}.parquet'
else:
    basepricefile = f'../data/price_impute_{flow}_{year}Q{quarter_1}.parquet'

baseprice = pd.read_parquet(basepricefile)
print(f'{baseprice.shape[0]} rows read from parquet file {basepricefile}\n')

if quarter == 1:
    baseprice.rename(columns={'base_price': 'price'}, inplace=True)
    baseprice.drop(columns='impute_base', inplace=True)

baseprice['qrt'] = 0    

weightbasefile = f'../data/weight_base_{flow}_{year_base}.parquet'
weight_base = pd.read_parquet(weightbasefile)
print(f'{weight_base.shape[0]} rows read from parquet file {weightbasefile}\n')

tradedatafile = f'../data/tradedata_no_outlier_{flow}_{year}_q{quarter}.parquet'
tradedata_no_outlier = pd.read_parquet(tradedatafile)
print(f'{tradedata_no_outlier.shape[0]} rows read from parquet file {tradedatafile}\n')
# -

# ## Aggregate

tradedata_no_outlier['qrt'] = 1
tradedata_qrt = tradedata_no_outlier.groupby(['year', 'flow', 'comno', 'qrt'], as_index=False).agg(
    value=('value', 'sum'),
    weight=('weight', 'sum')
)

# ## Add files together

tradedata_qrt['price'] = tradedata_qrt['value'] / tradedata_qrt['weight']
tradedata_qrts = pd.concat([baseprice, tradedata_qrt])
display(pd.crosstab(tradedata_qrts['qrt'], columns='Frequency', margins=True))

# ## Restructure file

qrt_r = tradedata_qrts.pivot(index=['flow', 'comno'], columns='qrt', values= ['price'])
qrt_r.columns = [f'{x}_{y}' for x, y in qrt_r.columns]
qrt_r = qrt_r.reset_index()

# ## Merge with weight_base

# +
prices = pd.merge(weight_base, qrt_r, on=['flow', 'comno'], how='left')

prices.drop(columns=['year', 'quarter'], inplace=True)


# Calculate the relative price ratio 'price_rel' by dividing 'price_1' by 'price_0'
# This represents the change in price between the two time periods represented by price_1 and price_0
prices['price_rel'] = prices['price_1'] / prices['price_0']

# Calculate the 'product' column, which is a weighted value of the price ratio ('price_rel')
# This is done by multiplying 'price_rel' by 'Weight_HS' to reflect the influence of each product
prices['product'] = prices['price_rel'] * prices['Weight_HS']

# Calculate the sum of 'product' for each group of 'flow' and 'section' using 'transform'
# The result is stored in 'prod_sum' and repeated for each row within the group
prices['prod_sum'] = prices.groupby(['flow', 'section'])['product'].transform('sum')

# Calculate 'Weight_section' by setting it to 'Weight_HS' where 'product' is not NaN
# This effectively considers only rows with valid 'product' values when computing section weights
prices['Weight_section'] = prices['Weight_HS'] * (prices['product'].notna())

# Aggregate 'Weight_section' by summing it within each 'flow' and 'section' group using 'transform'
# The result gives a total weight per section and flow, repeated for each row in the group
prices['Weight_section'] = prices.groupby(['flow', 'section'])['Weight_section'].transform('sum')
# -

# ## Impute missing values on section level

# +
prices['impute'] = prices['price_1'].isna().astype(int)
prices['price_1'] =  np.where(prices['impute'] == 1,
                              prices['price_0'] * prices['prod_sum'] /  prices['Weight_section'],
                              prices['price_1']
                             )

prices['price_rel'] =  np.where(prices['impute'] == 1,
                                prices['price_1'] / prices['price_0'],
                                prices['price_rel']
                               )  
# -

# ## Impute missing values on flow level

# +
prices['prod_sum'] = prices.groupby(['flow'])['product'].transform('sum')

prices['Weight_flow'] = prices['Weight_HS'] * prices['product'].notna()
prices['Weight_flow'] = prices.groupby(['flow'])['Weight_flow'].transform('sum')

prices['impute'] = np.where(prices['impute'] == 1,
                            prices['price_1'].isna().astype(int) + 1,
                            prices['impute']
                           )

prices['price_1'] =  np.where(prices['impute'] == 2,
                              prices['price_0'] * prices['prod_sum'] /  prices['Weight_flow'],
                              prices['price_1']
                             )

prices['price_rel'] =  np.where(prices['impute'] == 2,
                                prices['price_1'] / prices['price_0'],
                                prices['price_rel']
                               )

display(pd.crosstab(prices['impute'], columns='Frequency', margins=True))
display(prices['price_1'].describe())
# -

# ## Save result as parquet file

# +
prices.drop(columns=['Weight_flow', 'price_0'], inplace=True)
prices.rename(columns={'price_1': 'price'}, inplace=True)
prices['year'] = year
prices['quarter'] = quarter

imputefile = f'../data/price_impute_{flow}_{year}q{quarter}.parquet'
prices.to_parquet(imputefile)
print(f'\nNOTE: Parquet file {imputefile} written with {prices.shape[0]} rows and {prices.shape[1]} columns\n')
# -




