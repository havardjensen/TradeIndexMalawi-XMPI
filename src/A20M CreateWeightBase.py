# # Create base population

# ## Open parquet file

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
# year = 2021
# quarter = 1
# flow = 'import'
#
# import itables
#
# # Initialize interactive display mode
# itables.init_notebook_mode(all_interactive=True)
#
#
# share_total=0.05
# no_of_months=5
# no_of_months_seasons=3
# section_seasons='II'
# price_cv=0.5
# max_by_min=10
# max_by_median=5
# median_by_min=5
# share_small=0.0001
# n_transactions_year = 30
# -

basedata = pd.read_parquet(f'../data/{flow}_{year}.parquet')
print(f'{basedata.shape[0]} rows read from parquet file ../data/{flow}_{year}.parquet\n')

basedata

# ## Compute share of total

basedata['share_total'] = basedata['HS_sum'] / basedata['T_sum'] 
smaller = basedata.loc[basedata['share_total'] <= share_total]
valsum_ex_big = smaller.groupby('flow', as_index=False).agg(T_sum_small = ('value', 'sum'))

# ## Table of big commodities

print(f'Coomodities with share of total more than {share_total} for {flow} {year}')
big = basedata.loc[basedata['share_total'] > share_total]
display(pd.crosstab(big['comno'], 
                    columns='Sum',
                    values=big['value'],
                    aggfunc='sum', margins=True))

# ## Add sum of small commodities to the trade data

basedata = pd.merge(basedata, valsum_ex_big, on='flow', how='left')

# ## Drop all the rows but the first within each comno to get data for the whole year

# +
basedata = (basedata[basedata.duplicated(['flow', 'comno'], keep='first') == False]
            .sort_values(['flow', 'comno'])
)

basedata.drop(columns=[#'price', 
                       'month' 
                       #'weight', 
                       #'value'
                      ], inplace=True)
# -

basedata

# ## Calculate new columns
# the share the small have of the total within the commodities and more

basedata['share_small'] = basedata['HS_sum'] / basedata['T_sum_small']
basedata['max_by_min'] = basedata['price_max'] / basedata['price_min']
basedata['max_by_median'] = basedata['price_max'] / basedata['price_median']
basedata['median_by_min'] = basedata['price_median'] / basedata['price_min']

# ## Select the sample for the index
# We start with an overview of the original number of commodities

print(f'{flow.capitalize()} {year}. Number of commodities before selection')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))

basedata1 = basedata.copy()

print(f'{flow.capitalize()} {year}. basedata before selection')
show(basedata, maxBytes=0)

# ## Extract total value for population

Tpop_sum = basedata['T_sum'][0]
print(f'\n{flow.capitalize()}, {year}. Total value={Tpop_sum:,.0f}.\n')

# ### Extract those with enough months and Seasonal commodities

basedata = basedata.loc[(basedata['no_of_months'] >= no_of_months) | ((basedata['no_of_months'] >= no_of_months_seasons) &
                        (basedata['section'] == section_seasons))]
print(f'{flow.capitalize()} {year}. Number of commodities after selection of at least {no_of_months} months or seasonal commodities in section {section_seasons} with {no_of_months_seasons} or more months')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))



# ### Extract those with enough transactions for the year

basedata = basedata.loc[(basedata['n_transactions_year'] >= n_transactions_year)]
print(f'{flow.capitalize()} {year}. Number of commodities after selection of at least {n_transactions_year} transactions in the year')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))

# ### price_cv

basedata = basedata.loc[(basedata['price_cv'] < price_cv)]
print(f'{flow.capitalize()} {year}. Number of commodities after selection of those with price co-variance less than {price_cv}')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))

# ### max by min

basedata = basedata.loc[(basedata['max_by_min'] < max_by_min)]
print(f'{flow.capitalize()} {year}. Number of commodities after selection of those with maximum by minimum less than {max_by_min}')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))

# ### max by median

basedata = basedata.loc[(basedata['max_by_median'] < max_by_median)]
print(f'{flow.capitalize()} {year}. Number of commodities after selection of those with maximum by median less than {max_by_median}')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))

# ### median by min

basedata = basedata.loc[(basedata['median_by_min'] < median_by_min)]
print(f'{flow.capitalize()} {year}. Number of commodities after selection of those with median by minimum less than {median_by_min}')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))

# ### share small

basedata = basedata.loc[(basedata['share_small'] > share_small)]
print(f'{flow.capitalize()} {year}. Number of commodities after selection of those with share of small {share_small} or more')
display(pd.crosstab(basedata['flow'], columns='Frequency', margins=True))

# ## Import labels from json file

with open('../cat/labels.json') as json_file:
    labels = json.load(json_file)
labels


# ## Function to calculate the coverage
# As we calculate the coverage for different aggregation levels, we make a function for it

def coverage(df: pd.DataFrame, groupcol, aggcol) -> pd.DataFrame:
    result = df.groupby(['year', 'flow', groupcol], as_index=False).agg(
        Ssample_sum=('HS_sum', 'sum'),
        spop_sum=(aggcol, 'mean'),
        Sno_of_comm=(aggcol, 'size')
        )
    result['Tsample_sum'] = result.groupby(['year', 'flow'])['Ssample_sum'].transform('sum')
    result['Tpop_sum'] = Tpop_sum
    result['Tno_of_comm'] = result.groupby(['year', 'flow'])['Sno_of_comm'].transform('sum')
    result['Scoverage'] = result['Ssample_sum'] * 100 / result['spop_sum']
    result['Tcoverage'] = result['Tsample_sum'] * 100 / result['Tpop_sum']
    result['share_total'] = result['spop_sum'] * 100 / result['Tpop_sum']
    result = result.replace(labels)
    return result


# ## Coverage on different aggregation levels

print(f'{flow.capitalize()} {year}. Coverage of section')
display(coverage(basedata, 'section', 'S_sum'))
print(f'{flow.capitalize()} {year}. Coverage of sitc 1')
display(coverage(basedata, 'sitc1', 'S1_sum'))
print(f'{flow.capitalize()} {year}. Coverage of sitc 2')
display(coverage(basedata, 'sitc2', 'S2_sum'))

# ## Keep only columns that will be used later

keepcol = ['flow',
           'year',
           'comno',
           'sitc1',
           'sitc2',
           'chapter',
           'section',
           'quarter',
           'T_sum',
           'HS_sum',
           'S_sum',
           'C_sum',
           'S1_sum',
           'S2_sum'
           ]
basedata = basedata[keepcol]


# ## Function to calculate weights

def calculate_weights(df: pd.DataFrame, level, aggcol, mult1, mult2, weight) -> pd.DataFrame:
    # Filter out duplicate rows based on 'year', 'flow', and the specified 'level' column
    # Keep only the first occurrence of each duplicate and sort by 'year', 'flow', and 'level'
    result = (df[df.duplicated(['year', 'flow', level], keep='first') == False]
              .sort_values(['year', 'flow', level])
              )

    # If the aggregation level is 'section', calculate a column where 'aggcol' is the sum of 'mult2'
    # for each combination of 'year' and 'flow'
    if level == 'section':
        result[aggcol] = result.groupby(['year', 'flow'])[mult2].transform('sum')
    else:
        # Otherwise, calculate 'aggcol' as the sum of 'mult2' grouped by 'year', 'flow', and 'section'
        result[aggcol] = result.groupby(['year', 'flow', 'section'])[mult2].transform('sum')

    # Calculate the 'weight' by multiplying 'mult1' with 'mult2' and dividing by 'aggcol'
    result[weight] = result[mult1] * result[mult2] / result[aggcol]

    # Select only the relevant columns for the final output
    result = result[['year', 'flow', level, aggcol, weight]]

    # Return the resulting DataFrame with calculated weights
    return result


# ## Calculate weights for section

section_weights = calculate_weights(
    basedata,          # The DataFrame containing your data.
    level='section',   # The level of aggregation; 'section' will be used to determine how data is grouped.
    aggcol='Tsample_sum', # The name of the new column to store the sum of 'S_sum' within each group.
    mult1='T_sum',         # The first multiplier column used in calculating weights.
    mult2='S_sum',         # The second multiplier column, also used to calculate the weights.
    weight='Weight_S'      # The name of the new column to store the calculated weights.
)

# ## Add section weights to weight data

basedata = pd.merge(basedata, section_weights, on=(['year', 'flow', 'section']), how='left')

# ## Calculate weights for chapter

chapter_weights = calculate_weights(basedata, 
                                    level='chapter', 
                                    aggcol='Ssample_sum', 
                                    mult1='Weight_S', 
                                    mult2='C_sum',
                                    weight='Weight_C'
                                   )                                 

# ## Add chapter weights to weight data

basedata = pd.merge(basedata, chapter_weights, on=(['year', 'flow', 'chapter']), how='left')

# ## Calculate weight_HS
# This is the weight for commodities

basedata['Csample_sum'] = basedata.groupby(['year', 'flow', 'chapter'])['HS_sum'].transform('sum')
basedata['Weight_HS'] = basedata['Weight_C'] * basedata['HS_sum'] / basedata['Csample_sum']

# ## Save as parquet file

# +

basedata.to_parquet(f'../data/weight_base_{flow}_{year}.parquet')
print(f'\nNOTE: Parquet file ../data/weight_base_data_{flow}_{year}.parquet written with {basedata.shape[0]} rows and {basedata.shape[1]} columns\n')
