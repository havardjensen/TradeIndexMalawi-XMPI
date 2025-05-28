# # Create chained index for first year

# Calculate base year as the previous year

year_base = year - 1

# ## Read parquet files

indexunchainedfile = f'../data/index_unchained_{flow}_{year}.parquet'
index_unchained = pd.read_parquet(indexunchainedfile)
print(f'{index_unchained.shape[0]} rows read from parquet file {indexunchainedfile}\n')

# ## Aggregate and calculate chained indexes

index_unchained['index_mean'] = index_unchained.groupby(['year', 'flow', 'series', 'level'])['index_unchained'].transform('mean')
index_chained = index_unchained.copy()
index_chained['index_chained'] = index_chained['index_unchained'] * 100 / index_chained['index_mean']
#index_chained['period'] = index_chained['year'].astype(str) + 'Q' + index_chained['quarter'].astype(str)
index_chained.drop(columns='index_mean', inplace=True)
# Ensure year and quarter are integers
index_chained['year'] = index_chained['year'].astype(int)
index_chained['quarter'] = index_chained['quarter'].astype(int)
display(pd.crosstab([index_chained['level'], index_chained['series']], 
                    columns=[index_chained['year'], index_chained['quarter']], 
                    values=index_chained['index_chained'], 
                    aggfunc='mean'))

# ## Save result as parquet file

indexfile = f'../data/index_{flow}_chained.parquet'
index_chained.to_parquet(indexfile)
print(f'\nNOTE: Parquet file {indexfile} written with {index_chained.shape[0]} rows and {index_chained.shape[1]} columns\n')
