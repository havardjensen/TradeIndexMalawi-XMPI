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
# flow = 'import'
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

tradedata_month

# ## Save as parquet file

tradedata_month.to_parquet(f'../data/{flow}_{year}.parquet')
print(f'\nNOTE: Parquet file ../data/{flow}_{year}.parquet written with {tradedata_month.shape[0]} rows and {tradedata_month.shape[1]} columns\n')

# #### Visualizing data

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import display, clear_output


def combined_price_ui(tradedata_with_outlier, tradedata, tradedata_month):
    # Shared HS code selector based on intersection of all datasets' comno
    comno_sets = [
        set(tradedata_with_outlier['comno'].dropna().unique()),
        set(tradedata['comno'].dropna().unique()),
        set(tradedata_month['comno'].dropna().unique())
    ]
    comno_values = sorted(set.intersection(*comno_sets))
    
    # Widgets
    comno_combobox = widgets.Combobox(
        placeholder='Type or select HS code',
        options=comno_values,
        value=comno_values[0] if comno_values else None,
        description='HS code:',
        ensure_option=True,
        layout=widgets.Layout(width='300px')
    )
    
    dataset_dropdown = widgets.Dropdown(
        options=[('With outliers', 'tradedata_with_outlier'), ('Without outliers', 'tradedata')],
        value='tradedata_with_outlier',
        description='Dataset:',
        layout=widgets.Layout(width='200px')
    )
    
    unique_quarters = sorted(tradedata['quarter'].dropna().unique().tolist())
    quarter_checkboxes = [widgets.Checkbox(value=(i == 0), description=str(q), indent=False) for i, q in enumerate(unique_quarters)]
    quarter_box = widgets.VBox(quarter_checkboxes)
    
    right_axis_dropdown = widgets.Dropdown(
        options=[('None', 'none'), ('Value', 'value'), ('Weight', 'weight'), ('Both', 'both')],
        value='none',
        description='Show:',
        layout=widgets.Layout(width='200px')
    )
    
    plot_output = widgets.Output()
    
    def get_selected_quarters():
        return [int(cb.description) for cb in quarter_checkboxes if cb.value]
    
    def plot_price_distribution(ax, dataset, comno_value, selected_quarters):
        filtered_data = dataset[
            (dataset['comno'] == comno_value) &
            (dataset['quarter'].isin(selected_quarters))
        ]
        if filtered_data.empty:
            ax.text(0.5, 0.5, 'No data for selected filters', ha='center', va='center', fontsize=14)
            ax.set_axis_off()
            return

        if 'outlier_sd' in filtered_data.columns:
            filtered_data = filtered_data.copy()
            filtered_data['outlier_label'] = filtered_data['outlier_sd'].apply(
                lambda x: 'Outlier' if x else 'No Outlier'
            )
            counts = filtered_data['outlier_label'].value_counts()
            enable_kde = counts.min() >= 2  # KDE only if all groups have at least 2 points

            sns.histplot(
                data=filtered_data,
                x='price',
                hue='outlier_label',
                kde=enable_kde,
                bins=30,
                alpha=0.7,
                ax=ax,
                multiple='stack'
            )

            # Manually create legend handles with fixed colors for clarity
            hue_order = ['No Outlier', 'Outlier']
            colors = ['tab:blue', 'tab:orange']
            handles = [mpatches.Patch(color=colors[i], label=hue_order[i]) for i in range(len(hue_order))]
            ax.legend(title='Outlier Status', handles=handles)

        else:
            sns.histplot(
                data=filtered_data,
                x='price',
                kde=True,
                bins=30,
                alpha=0.7,
                ax=ax
            )

        ax.set_title(f'HS {comno_value} | Quarters: {", ".join(map(str, selected_quarters))}')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
    
    def plot_line(ax_left, dataset, comno, right_selection):
        filtered = dataset[dataset['comno'] == comno]
        if filtered.empty:
            ax_left.text(0.5, 0.5, 'No data for selected HS code', ha='center', va='center')
            ax_left.set_axis_off()
            return
        filtered = filtered.sort_values('date')

        sns.lineplot(data=filtered, x='date', y='price', color='tab:blue', marker='o', ax=ax_left, label='Price')
        ax_left.set_ylabel('Price', color='tab:blue')
        ax_left.tick_params(axis='y', labelcolor='tab:blue')
        ax_left.set_xlabel('Date')
        ax_left.set_title(f'HS {comno} — Price vs. Value & Weight')
        ax_left.grid(True)

        legend_handles = [mlines.Line2D([], [], color='tab:blue', marker='o', label='Price')]

        if right_selection != 'none':
            ax_right = ax_left.twinx()
            if right_selection in ['value', 'both']:
                sns.lineplot(data=filtered, x='date', y='value',
                             color='gray', linestyle='--', marker='s', ax=ax_right)
                legend_handles.append(mlines.Line2D([], [], color='gray', linestyle='--', marker='s', label='Value'))

            if right_selection in ['weight', 'both']:
                sns.lineplot(data=filtered, x='date', y='weight',
                             color='darkgray', linestyle=':', marker='^', ax=ax_right)
                legend_handles.append(mlines.Line2D([], [], color='darkgray', linestyle=':', marker='^', label='Weight'))

            ax_right.set_ylabel('Value / Weight', color='gray')
            ax_right.tick_params(axis='y', labelcolor='gray')

        # Stats box
        latest = filtered.iloc[-1]
        stats_text = (
            f"Months: {latest.get('no_of_months', '')}\n"
            f"Max: {latest.get('price_max', 0):,.2f}\n"
            f"Min: {latest.get('price_min', 0):,.2f}\n"
            f"Median: {latest.get('price_median', 0):,.2f}\n"
            f"Mean: {latest.get('price_mean', 0):,.2f}\n"
            f"SD: {latest.get('price_sd', 0):,.2f}\n"
            f"Transactions: {latest.get('n_transactions_year', '')}\n"
            f"CV: {latest.get('price_cv', 0):,.2f}"
        )
        ax_left.text(
            0.98, 0.95, stats_text,
            transform=ax_left.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3')
        )
        ax_left.legend(handles=legend_handles, loc='upper left')
    
    def update_plot(_):
        with plot_output:
            clear_output(wait=True)
            comno = comno_combobox.value
            dataset_name = dataset_dropdown.value
            selected_quarters = get_selected_quarters()
            right_selection = right_axis_dropdown.value
            
            if not comno:
                print("Please select a valid HS code.")
                return
            if not selected_quarters:
                print("Please select at least one quarter.")
                return
            
            dataset_dist = {
                'tradedata_with_outlier': tradedata_with_outlier,
                'tradedata': tradedata
            }[dataset_name]
            
            monthly_data = tradedata_month.copy()
            for col in ['month', 'year', 'value', 'weight', 'price']:
                monthly_data[col] = pd.to_numeric(monthly_data[col], errors='coerce')
            monthly_data['date'] = pd.to_datetime(monthly_data[['year', 'month']].assign(day=1), errors='coerce')
            monthly_data = monthly_data.dropna(subset=['date'])
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 12))
            
            plot_price_distribution(axes[0], dataset_dist, comno, selected_quarters)
            plot_line(axes[1], monthly_data, comno, right_selection)
            
            plt.tight_layout()
            plt.show()
    
    update_button = widgets.Button(description="Update Plot", button_style='primary', icon='refresh')
    update_button.on_click(update_plot)
    
    display(
        widgets.HBox([comno_combobox, dataset_dropdown, right_axis_dropdown]),
        widgets.Label("Select Quarters:"),
        quarter_box,
        update_button,
        plot_output
    )



# +
import pandas as pd
import plotly.express as px
import ipywidgets as widgets
from IPython.display import display

def price_cv_ui(datasets, dataset_names, flow, year):
    """
    Interactive visualization of Price Coefficient of Variation (CV)
    with selectable X-axis (transactions or value).
    """

    def aggregate_to_monthly(df):
        df_month = df.groupby(['year', 'flow', 'comno', 'quarter', 'month'], as_index=False).agg(
            weight=('weight', 'sum'),
            value=('value', 'sum'),
            n_transactions_month=('comno', 'size')
        )
        df_month['price'] = df_month['value'] / df_month['weight']
        return df_month

    def calculate_price_cv_monthly(df_month):
        df_month['price_mean'] = df_month.groupby(['flow', 'comno'])['price'].transform('mean')
        df_month['price_sd'] = df_month.groupby(['flow', 'comno'])['price'].transform('std')
        df_month['price_cv'] = (df_month['price_sd'] / df_month['price_mean']).fillna(0)
        df_month['n_transactions_year'] = df_month.groupby(['flow', 'comno'])['n_transactions_month'].transform('sum')
        return df_month

    def aggregate_final(df_month):
        return df_month.groupby(['flow', 'year', 'comno'], as_index=False).agg(
            price=('price', 'mean'),
            value=('value', 'sum'),
            n_transactions_year=('n_transactions_year', 'first'),
            price_cv=('price_cv', 'first')
        )[['comno', 'price', 'value', 'n_transactions_year', 'price_cv']]

    consolidated = pd.DataFrame()
    for i, dataset in enumerate(datasets):
        df_month = aggregate_to_monthly(dataset)
        df_month = calculate_price_cv_monthly(df_month)
        final_df = aggregate_final(df_month)
        final_df['Dataset'] = dataset_names[i]
        consolidated = pd.concat([consolidated, final_df], ignore_index=True)

    def plot_dynamic_xaxis(dataset_name, x_variable):
        data = consolidated[consolidated['Dataset'] == dataset_name]
        total = data['price'].notna().sum()
        below_cv = (data['price_cv'] < 0.5).sum()

        x_labels = {
            'n_transactions_year': 'Number of Transactions (Year)',
            'value': 'Total Value'
        }

        title = f"{flow.capitalize()} {year} — Price CV (CV < 0.5: {below_cv} of {total})"

        fig = px.scatter(
            data,
            x=x_variable,
            y='price_cv',
            hover_data={'comno': True, 'price_cv': ':.3f', x_variable: ':.0f'},
            labels={
                x_variable: x_labels[x_variable],
                'price_cv': 'Price CV'
            },
            title=title
        )
        fig.update_traces(marker=dict(size=7, color='blue', opacity=0.7, line=dict(width=0.5, color='black')))
        fig.update_layout(height=600, title_font_size=18, template='plotly_white')
        fig.show()

    # UI widgets
    dataset_dropdown = widgets.Dropdown(
        options=dataset_names,
        value=dataset_names[0],
        description='Dataset:',
        layout=widgets.Layout(width='300px')
    )

    x_axis_dropdown = widgets.Dropdown(
        options=[
            ('Number of Transactions', 'n_transactions_year'),
            ('Total Value', 'value')
        ],
        value='n_transactions_year',
        description='X-axis:',
        layout=widgets.Layout(width='300px')
    )

    output = widgets.interactive_output(
        plot_dynamic_xaxis,
        {'dataset_name': dataset_dropdown, 'x_variable': x_axis_dropdown}
    )

    display(widgets.HBox([dataset_dropdown, x_axis_dropdown]), output)



# +

# 1. Histogram Plot UI
#price_distribution_ui(tradedata_no_sd, tradedata_with_outlier)

# 2. Lineplot UI
#price_lineplot_ui(tradedata_month)

combined_price_ui(tradedata_with_outlier, tradedata, tradedata_month)

# 3. Price CV Plot UI
#flow = 'import'
#year = 2024
datasets = [tradedata_with_outlier, tradedata_no_sd, tradedata_no_sd2]  # as example
dataset_names = ['With outliers', 'outlier_sd removed', 'outlier_sd2 removed']
price_cv_ui(datasets, dataset_names, flow, year)


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




