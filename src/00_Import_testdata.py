# ---
# jupyter:
#   jupytext:
#     formats: py:hydrogen
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Trade examples

# %%
import pandas as pd
import numpy as np

# %%
fileprefix = 'Trade Data 2020-2025 - new.parquet'
input_file = f'../data/{fileprefix}'

month_dict = {
    'January': 1,
    'February': 2,
    'March': 3,
    'April': 4,
    'May': 5,
    'June': 6,
    'July': 7,
    'August': 8,
    'September': 9,
    'October': 10,
    'November': 11,
    'December': 12
}


df = pd.read_parquet(input_file)

df.columns = df.columns.str.lower()

# Replace month names with numbers
df['month'] = df['month'].replace(month_dict)

df['supplementary unit'] = df['supplementary unit'].replace('missing', None)

# Create a new column 'Quarter' based on the month
df['quarter'] = ((df['month'].astype(int) - 1) // 3) + 1

# Add leading zero to HS codes with length 7
df.loc[df['hs code'].str.len() == 7, 'hs code'] = '0' + df.loc[df['hs code'].str.len() == 7, 'hs code']

# Group by 'Year' and 'Quarter'
grouped = df.groupby(['year', 'quarter', 'flow'])

for (year, quarter, flow), group in grouped:
    # Define the output file name using the year and quarter
    output_file = f'../data/{flow}_{year}Q{quarter}.txt'

    # Drop the 'Quarter' column before saving
    #group = group.drop(columns=['quarter', 'flow'])
 
    # Get the number of rows in the group
    num_rows = len(group)
    
    # Save the group to a CSV file with semi-colon as the delimiter
    group.to_csv(output_file, sep=';', index=False)
    print(f"File ../data/{flow}_{year}Q{quarter}.txt successfully created with {num_rows} rows.")

# %%

# %%
df.dtypes

# %%
df.dtypes

# %%
grouped_size_df = df.groupby(['year', 'quarter']).size().reset_index(name='Count')
print(grouped_size_df)
