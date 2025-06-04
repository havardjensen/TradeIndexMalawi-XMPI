# # Create weight base population

# ## Add parquet files for the whole year together

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
# import ipywidgets as widgets
# from IPython.display import display
#
#
# year = 2024
# quarter = 1
# flow = 'export'
# selected_outlier= 'outlier_sd'
#
# import itables
#
# # Initialize interactive display mode
# itables.init_notebook_mode(all_interactive=True)
# -

sitc1_label = pd.read_parquet('../cat/SITC_label.parquet')
sitc1_label = sitc1_label.loc[sitc1_label['level'] == 1]

data_dir = Path('../data')
tradedata = pd.concat(
    pd.read_parquet(parquet_file)
    for parquet_file in data_dir.glob(f'{flow}_{year}_q*.parquet')
)
#tradedata.to_parquet(f'../data/{flow}_{year}.parquet')
print(f'{tradedata.shape[0]} rows read from parquet files for {year}\n')

tradedata['price'] = tradedata['value'] / tradedata['weight']

# ### List rows where price is set to Infinity

if len(tradedata.loc[np.isinf(tradedata['price'])]) > 0:
    print(f'{flow.capitalize()} {year}. Rows where price could not be calculated.\n')
    display(tradedata.loc[np.isinf(tradedata['price'])])

# ## Population weights - Add sums for aggregated levels

tradedata['T_sum'] = tradedata.groupby(['flow'])['value'].transform('sum')
tradedata['HS_sum'] = tradedata.groupby(['flow', 'comno'])['value'].transform('sum')
tradedata['S_sum'] = tradedata.groupby(['flow', 'section'])['value'].transform('sum')
tradedata['C_sum'] = tradedata.groupby(['flow', 'chapter'])['value'].transform('sum')
tradedata['S1_sum'] = tradedata.groupby(['flow', 'sitc1'])['value'].transform('sum')
tradedata['S2_sum'] = tradedata.groupby(['flow', 'sitc2'])['value'].transform('sum')

tradedata.groupby(['flow', 'section'])['value'].agg('sum')

pd.crosstab(tradedata['section'], columns='Freq', margins=True)

tradedata['comno'].loc[tradedata['section'].isna()].value_counts()

tradedata.to_parquet(f'../data/tradedata_{flow}_{year}.parquet')
print(f'\nNOTE: Parquet file ../data/tradedata_{flow}_{year}.parquet written with {tradedata.shape[0]} rows and {tradedata.shape[1]} columns\n')

# #### Create datasets for analysis of removal of outliers 

# Filter the DataFrame to keep only transactions where all specified conditions are False
tradedata_no_MAD = tradedata.loc[
    (tradedata['outlier_MAD'] == False)
].copy()

# Filter the DataFrame to keep only transactions where all specified conditions are False
tradedata_no_sd = tradedata.loc[
    (tradedata['outlier_sd'] == False)
].copy()

# Filter the DataFrame to keep only transactions where all specified conditions are False
tradedata_no_sd2 = tradedata.loc[
    (tradedata['outlier_sd'] == False) &
    (tradedata['outlier_sd2'] == False)
].copy()

# Create dataframe with all outliers
tradedata_with_outlier = tradedata.copy()

# ## Delete outliers
# The limit is set before we run this syntax. We use axis=0 to avoid a lot of messages

# + active=""
# # Crosstab for sum of values
# crosstab = pd.crosstab(tradedata[selected_outlier], 
#                        columns='Sum', 
#                        values=tradedata['value'], 
#                        margins=True, 
#                        aggfunc='sum')
#
# # Add percentage column
# crosstab['Percentage (%)'] = (crosstab['Sum'] / crosstab.loc['All', 'Sum'] * 100).map('{:.1f}'.format)
#
# # Ensure 'Sum' and 'Percentage (%)' columns are numeric
# crosstab['Sum'] = pd.to_numeric(crosstab['Sum'], errors='coerce')
# crosstab['Percentage (%)'] = pd.to_numeric(crosstab['Percentage (%)'], errors='coerce')
#
# # Display with formatted sum and percentage
# print(f'Value of price outliers for {flow} in {year}')
# display(crosstab.style.format({'Sum': '{:.0f}','ALL': '{:.0f}', 'Percentage (%)': '{:.1f}%'}))

# +
print('')
print('')
print(f'Discriptiv statistics for {flow} in {year} grouped by outlier')
print('')
print('')

display(tradedata.groupby(selected_outlier).agg(
    value_count=('value', 'count'),
    value_mean=('value', 'mean'),
    value_sum=('value', 'sum'),
    value_std=('value', 'std')
    )
)        

print('')
print('')
print(f'List of price outliers for {flow} in {year}')
print('')
print('')

# Filter for outliers, sort by 'value' in descending order, and display the top 100
top_outliers = tradedata.loc[tradedata[selected_outlier] == 1].sort_values(by='value', ascending=False).head(100)

display(top_outliers)

# -
# Remove outliers
tradedata = tradedata.loc[tradedata[selected_outlier] == 0].copy()

# #### Count transactions within each comno after removal of outliers

tradedata['n_transactions'] = tradedata.groupby(['flow', 'comno', 'quarter', 'month'])['value'].transform('count')

# ## Aggregate to months as there are often more transactions for the same commodity within the same month

aggvars = ['year', 'flow', 'comno', 'quarter', 'month', 'section', 'chapter', 
           'sitc1', 'sitc2', 'T_sum', 'S_sum', 'C_sum', 'S1_sum', 'S2_sum', 'HS_sum']
tradedata_month = tradedata.groupby(aggvars, as_index=False).agg(
    weight=('weight', 'sum'),
    value=('value', 'sum'),
    n_transactions_month = ('n_transactions', 'mean')
)
tradedata_month['price'] = tradedata_month['value'] / tradedata_month['weight']

# ## Add columns for to check for homogenity in the data
# These columns will be checked against the edge values that we choose

tradedata_month['no_of_months'] = tradedata_month.groupby(['flow', 'comno'])['month'].transform('count')
for stat in ['max', 'min', 'median', 'mean']:
    tradedata_month[f'price_{stat}'] = tradedata_month.groupby(['flow', 'comno'])['price'].transform(stat)
tradedata_month['price_sd'] = tradedata_month.groupby(['flow', 'comno'])['price'].transform('std')
tradedata_month['n_transactions_year'] = tradedata_month.groupby(['flow', 'comno'])['n_transactions_month'].transform('sum')
tradedata_month['price_cv'] = tradedata_month['price_sd'] / tradedata_month['price_mean']

# ## Save as parquet file

tradedata_month.to_parquet(f'../data/{flow}_{year}.parquet')
print(f'\nNOTE: Parquet file ../data/{flow}_{year}.parquet written with {tradedata_month.shape[0]} rows and {tradedata_month.shape[1]} columns\n')

# #### Visualizing data

# + active=""
# markdown_text = """
# ### Price Coefficient of Variation (price_cv)
#
# The Price Coefficient of Variation (price_cv)** is a statistical measure that shows the degree of variability in price relative to the mean price across different transactions. It is calculated as:
#
#     price_cv = (Standard Deviation of Prices) / (Mean Price)
#
# Interpretation:
# - Low price_cv: Indicates that prices are relatively stable, meaning they are close to the mean price. This suggests minimal variability in the price of a product across different transactions.
#   
# - High price_cv: Indicates a wide range of prices, meaning prices are spread out significantly from the mean. This suggests high volatility or inconsistency in the price of the product.
#
# In summary, the lower the price_cv, the more consistent the pricing; the higher the price_cv, the more unpredictable the pricing.
# """
#
# # Displaying the markdown text
# print(f'\n{flow.capitalize()} {year}.\n{markdown_text}')

# +
print('')
print('')
print(f'Scatter plot for price cv and number of transactions for {flow} in {year}')
print('')
print('')



# Helper function to calculate price_mean, price_sd, and price_cv within each dataset
def calculate_price_cv(df):
    df['price_mean'] = df.groupby(['flow', 'comno'])['price'].transform('mean')
    df['price_sd'] = df.groupby(['flow', 'comno'])['price'].transform('std')
    df['price_cv'] = (df['price_sd'] / df['price_mean']).fillna(0)  # Handle NaNs by filling with 0
    return df

# Function to aggregate data after calculating price_cv
def aggregate_and_calculate_price(df):
    #df['qrt'] = 1  # Assign a constant quarter value for simplicity
    
    aggregated_df = df.groupby(['year', 'flow', 'comno'], as_index=False).agg(
        value=('value', 'sum'),
        weight=('weight', 'sum'),
        n_transactions=('n_transactions', 'first')  # Assuming n_transactions does not vary
    )
    
    aggregated_df['price'] = aggregated_df['value'] / aggregated_df['weight']
    return aggregated_df[['comno', 'price', 'n_transactions']]

# List of datasets to process
datasets = [tradedata_with_outlier, tradedata_no_MAD, tradedata_no_sd, tradedata_no_sd2]
dataset_names = ['With outliers', 'outlier_MAD removed', 'outlier_sd removed', 'outlier_sd2 removed']

consolidated_data = pd.DataFrame()
for i, dataset in enumerate(datasets):
    dataset = calculate_price_cv(dataset)
    aggregated_data = aggregate_and_calculate_price(dataset)
    
    # Merge the aggregated data with the calculated price_cv from the original dataset
    aggregated_data = aggregated_data.merge(
        dataset[['comno', 'price_cv']].drop_duplicates(), on='comno', how='left'
    )
    
    aggregated_data['Dataset'] = dataset_names[i]
    consolidated_data = pd.concat([consolidated_data, aggregated_data], ignore_index=True)

# Pivot the consolidated data
price_comparison = consolidated_data.pivot_table(
    index='comno', 
    columns='Dataset', 
    values=['price', 'price_cv'],
    aggfunc='first'  # Ensures we only keep one instance of the value
)

# Flatten the MultiIndex columns for clarity
price_comparison.columns = [f'{col[0]}_{col[1]}' for col in price_comparison.columns]

# Aggregate base_price and n_transactions separately, keeping just one instance of each
base_data = consolidated_data[['comno', 'n_transactions']].drop_duplicates()

# Merge base data to include base_price and n_transactions in the comparison
price_comparison = price_comparison.merge(base_data, on='comno', how='left')

# Create a function to plot price_cv
def plot_price_cv(dataset_name):
    # Filter for the selected dataset
    data_to_plot = consolidated_data[consolidated_data['Dataset'] == dataset_name]
    
    # Count number of comnos with price_cv > 0.5
    number_of_prices = data_to_plot['price'].dropna().count()  # Count of non-NaN prices
    count_high_cv = (data_to_plot['price_cv'] < 0.5).sum()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=data_to_plot, x='n_transactions', y='price_cv', color='blue', label='Price CV')
    
    # Set plot title and include the count of comnos with price_cv > 0.5
    plt.title(f'{flow.capitalize()} {year}. Price Coefficient of Variation (Count of comnos with CV < 0.5: {count_high_cv} of {number_of_prices})')
    plt.xlabel('n_transactions')
    plt.ylabel('Price CV')
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()

# Create a dropdown for dataset selection
dataset_dropdown = widgets.Dropdown(
    options=dataset_names,
    value=dataset_names[0],  # Default dataset
    description='Select dataset:',
    layout=widgets.Layout(width='300px')
)

# Create an interactive output for the plot
output = widgets.interactive_output(plot_price_cv, {'dataset_name': dataset_dropdown})

# Display the dropdown and output
display(dataset_dropdown, output)

# +
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display, clear_output

# --- Setup ---
comno_counts = tradedata_no_sd['comno'].value_counts()
valid_comno_values = sorted(comno_counts[comno_counts > 1].index.tolist())
unique_quarters = sorted(tradedata_no_sd['quarter'].dropna().unique().tolist())

# --- Widgets ---
comno_combobox = widgets.Combobox(
    placeholder='Type or select HS code',
    options=valid_comno_values,
    value=valid_comno_values[0],
    description='Select HS:',
    ensure_option=True,
    layout=widgets.Layout(width='300px')
)

dataset_dropdown = widgets.Dropdown(
    options=[('With outliers', 'tradedata_with_outlier'), ('Without outliers', 'tradedata')],
    value='tradedata_with_outlier',
    description='Dataset:',
    layout=widgets.Layout(width='300px')
)

# Only first quarter selected
quarter_checkboxes = [
    widgets.Checkbox(value=(i == 0), description=str(q), indent=False)
    for i, q in enumerate(unique_quarters)
]
quarter_box = widgets.VBox(quarter_checkboxes)

def get_selected_quarters():
    return [int(cb.description) for cb in quarter_checkboxes if cb.value]

# --- Output area ---
plot_output = widgets.Output()

# --- Plotting ---
def plot_price_distribution(ax, dataset, comno_value, hue, selected_quarters):
    filtered_data = dataset[
        (dataset['comno'] == comno_value) &
        (dataset['quarter'].isin(selected_quarters))
    ]

    if filtered_data.empty:
        ax.text(0.5, 0.5, 'No data for selected filters', ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return

    if hue in filtered_data.columns:
        sns.histplot(data=filtered_data, x='price', hue=hue, kde=True, palette='muted', bins=30, alpha=0.7, ax=ax)
    else:
        sns.histplot(data=filtered_data, x='price', kde=True, bins=30, alpha=0.7, ax=ax)

    ax.set_title(f'HS {comno_value} | Quarters: {", ".join(map(str, selected_quarters))}')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')

# --- Update function (triggered by button) ---
def update_price_distribution(_):
    with plot_output:
        clear_output(wait=True)

        dataset_name = dataset_dropdown.value
        comno_value = comno_combobox.value
        selected_quarters = get_selected_quarters()

        if not comno_value or not selected_quarters:
            print("Please select both a HS code and at least one quarter.")
            return

        dataset = {
            'tradedata_with_outlier': tradedata_with_outlier,
            'tradedata': tradedata
        }[dataset_name]

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_price_distribution(ax, dataset, comno_value, selected_outlier, selected_quarters)
        plt.tight_layout()
        plt.show()

# --- Button ---
update_button = widgets.Button(description="Update Plot", button_style='primary', icon='refresh')
update_button.on_click(update_price_distribution)

# --- Layout ---
display(
    widgets.HBox([comno_combobox, dataset_dropdown]),
    widgets.Label("Select Quarters:"),
    quarter_box,
    update_button,
    plot_output
)


# +
print('')
print('')
print(f'Piechart of weightbase for {flow} in {year}')
print('')
print('')


# Load SITC1 label data
sitc1_label = pd.read_parquet("../cat/SITC_label.parquet")
sitc1_label = sitc1_label[sitc1_label["level"] == 1][["code", "name"]]  # Keep relevant columns

# Remove duplicates for each SITC1
df_pie = tradedata_month.drop_duplicates(subset="sitc1").sort_values("sitc1")

# Ensure `sitc1` and `code` are the same type
df_pie["sitc1"] = df_pie["sitc1"].astype(str)
sitc1_label["code"] = sitc1_label["code"].astype(str)

# Merge with labels
df_pie = df_pie.merge(sitc1_label, left_on="sitc1", right_on="code", how="left")

# Create a combined label with both code and name
df_pie["label"] = df_pie["sitc1"] + " - " + df_pie["name"]

# Define colors
colors = plt.cm.Paired.colors  

# Create figure
fig, ax = plt.subplots(figsize=(8, 8))

# Generate pie chart
wedges, texts, autotexts = ax.pie(
    df_pie["S1_sum"], 
    labels=None,  # Remove default labels to avoid stacking
    autopct="%1.1f%%", 
    startangle=140, 
    colors=colors, 
    pctdistance=0.85  
)

# Adjust percentage label styling
for autotext in autotexts:
    autotext.set_color("black")
    autotext.set_fontweight("bold")

# Add labels outside the pie with fixed offsets
for wedge, label in zip(wedges, df_pie["label"]):
    angle = (wedge.theta2 + wedge.theta1) / 2  # Get middle angle of each slice
    x = 1.2 * wedge.r * np.cos(np.radians(angle))  # Fixed position offset
    y = 1.2 * wedge.r * np.sin(np.radians(angle))  # Fixed position offset
    ax.annotate(
        label, 
        xy=(wedge.r * np.cos(np.radians(angle)), wedge.r * np.sin(np.radians(angle))),
        xytext=(x, y),  # Label position outside pie
        ha="center", fontsize=10, 
        bbox=dict(boxstyle="round,pad=0.3", edgecolor="gray", facecolor="white"),
        arrowprops=dict(arrowstyle="-", connectionstyle="arc3,rad=0.2")
    )

# Title
plt.title(f"Pie chart of weightbase for SITC1 for {flow} in {year}", fontsize=14)

plt.show()
# -


