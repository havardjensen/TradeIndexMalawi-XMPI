# # Create chained index for next years

# Calculate base year as the previous year

import pandas as pd
import numpy as np
#year = 2024
#flow = 'Export'
year_base = year - 1

# ## Read parquet files

# +
indexchainedfile = f'../data/index_{flow}_chained.parquet'
index_chained = pd.read_parquet(indexchainedfile)
print(f'{index_chained.shape[0]} rows read from parquet file {indexchainedfile}\n')

indexunchainedfile = f'../data/index_unchained_{flow}_{year}.parquet'
index_unchained = pd.read_parquet(indexunchainedfile)
print(f'{index_unchained.shape[0]} rows read from parquet file {indexunchainedfile}\n')
# -

# ## Select last quarter previous year

index_chained_q4 = index_chained.loc[(index_chained['year'] == year_base) & (index_chained['quarter'] == 4)]
varlist = ['flow', 'level', 'series', 'index_chained']
index_chained_q4 = index_chained_q4[varlist]
index_chained_q4.rename(columns={'index_chained': 'index_chained_base'}, inplace=True)

# ## Match last quarter chained with index current year

index_chained_detailed = pd.merge(index_unchained, index_chained_q4, on=['flow', 'level', 'series'], how='left')

# ## Select total level for the first quarter

index_chained_total = index_chained_detailed.loc[(index_chained_detailed['level'] == 'Total') & (index_chained_detailed['quarter'] == 1)].copy()
index_chained_total.rename(columns={'index_chained_base': 'index_chained_total'}, inplace=True)
index_chained_total = index_chained_total[['flow', 'index_chained_total']]

# ## Match chained base to every row

index_chained_quarter = pd.merge(index_chained_detailed, index_chained_total, on=['flow'], how='left')

# ## Compute chain factor and chained index
# If base index is missing, we will use the index for the total

index_chained_quarter['index_chained_base'] = np.where(index_chained_quarter['index_chained_base'].isna(),
                                                       index_chained_quarter['index_chained_total'],
                                                       index_chained_quarter['index_chained_base']
)
index_chained_quarter['chain_factor'] = index_chained_quarter['index_chained_base'] / 100
index_chained_quarter['index_chained'] = index_chained_quarter['chain_factor'] * index_chained_quarter['index_unchained']
index_chained_quarter.drop(columns=['index_chained_total', 'index_chained_base', 'chain_factor'], inplace=True)
index_chained_quarter['year'] = index_chained_quarter['year'].astype(int)
index_chained_quarter['quarter'] = index_chained_quarter['quarter'].astype(int)

# ## Add chained index this quarter to chained index file

index_chained = pd.concat([index_chained, index_chained_quarter])
groupvars = ['flow', 'level', 'series', 'year', 'quarter']
index_chained = index_chained[index_chained.duplicated(groupvars, keep='last') == False].sort_values(groupvars)

# ## Table for last 4 years

last4 = index_chained.loc[index_chained['year'] >= year - 3]
display(pd.crosstab([last4['level'], last4['series']], 
                    columns=[last4['year'], last4['quarter']], 
                    values=last4['index_chained'], 
                    aggfunc='mean'))

# ## Save result as parquet file

index_chained.to_parquet(indexchainedfile)
print(f'\nNOTE: Parquet file {indexchainedfile} written with {index_chained.shape[0]} rows and {index_chained.shape[1]} columns\n')


