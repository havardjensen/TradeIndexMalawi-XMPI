#
#
#

# ### T015

# #### Q-Q plot of distribution z-score for each HS

# + active=""
# import matplotlib.pyplot as plt
# import ipywidgets as widgets
# from IPython.display import display
# import scipy.stats as stats
#
# # Function to plot Q-Q plot for the selected comno
# def plot_qq_plot_for_comno(comno_value):
#     filtered_data = t_section[t_section['comno'] == comno_value]
#
#     # Create a figure for the Q-Q plot
#     fig, ax = plt.subplots(figsize=(6, 6))
#
#     # Ensure z_score exists and plot Q-Q plot
#     if 'z_score' in filtered_data.columns:
#         stats.probplot(filtered_data['z_score'], dist="norm", plot=ax)
#         ax.set_title(f'{flow.capitalize()}. {year} quarter {quarter}: Q-Q Plot of Z-scores for comno {comno_value}')
#     else:
#         print(f"{flow.capitalize()} {year}, q {quarter}. z_score column not found in t_section dataset for comno {comno_value}.")
#
#     # Show plot
#     plt.tight_layout()
#     plt.show()
#
# # Specify the comno values you want to visualize and sort them
# comno_values = sorted(t_section['comno'].unique().tolist())  # Sort unique comno values
#
# # Create a dropdown menu for comno selection
# comno_dropdown = widgets.Select(
#     options=comno_values,
#     value=comno_values[0],  # Default value
#     description='Select comno:',
#     layout=widgets.Layout(width='300px', height='200px'),
# )
#
# # Create an interactive output for the Q-Q plot
# output = widgets.interactive_output(plot_qq_plot_for_comno, {
#     'comno_value': comno_dropdown
# })
#
# # Display the dropdown and output
# display(comno_dropdown, output)
#

# + active=""
# import matplotlib.pyplot as plt
# from matplotlib_venn import venn3
#
# # Step 1: Define the categorization function with the new thresholds
# def categorize_transactions(count):
#     if count < 10:
#         return 'Less than 10 transactions'
#     elif 10 <= count <= 30:
#         return 'Between 11-30 transactions'
#     else:
#         return 'Above 30 transactions'
#
# # Apply the function to categorize transaction counts
# t_section['transaction_category'] = t_section['n_transactions'].apply(categorize_transactions)
#
# # Step 2: Loop through each category and create separate Venn diagrams
# categories = t_section['transaction_category'].unique()  # Get unique categories
#
# for category in categories:
#     # Filter the DataFrame for the current category
#     category_data = t_section[t_section['transaction_category'] == category]
#     
#     # Create a Venn diagram for the current category
#     plt.figure(figsize=(8, 8))
#     venn3(subsets=(
#         sum(category_data['outlier_MAD'] & ~category_data['outlier_sd2'] & ~category_data['outlier_sd']),  # Only MAD
#         sum(~category_data['outlier_MAD'] & category_data['outlier_sd2'] & ~category_data['outlier_sd']),  # Only SD2
#         sum(category_data['outlier_MAD'] & category_data['outlier_sd2'] & ~category_data['outlier_sd']),   # MAD and SD2
#         sum(~category_data['outlier_MAD'] & ~category_data['outlier_sd2'] & category_data['outlier_sd']),  # Only SD
#         sum(category_data['outlier_MAD'] & ~category_data['outlier_sd2'] & category_data['outlier_sd']),   # MAD and SD
#         sum(~category_data['outlier_MAD'] & category_data['outlier_sd2'] & category_data['outlier_sd']),   # SD2 and SD
#         sum(category_data['outlier_MAD'] & category_data['outlier_sd2'] & category_data['outlier_sd']),    # All three
#     ), set_labels=('MAD', 'SD2', 'SD'))
#     
#     # Customize the plot title with relevant details
#     plt.title(f'{flow.capitalize()}, {year} quarter {quarter}: Outliers Detected by Different Methods\nHS with: {category}')
#     plt.show()


# + active=""
# from matplotlib_venn import venn3, venn2
# import matplotlib.pyplot as plt
#
# # comparing three methods: outlier_MAD, outlier_HB, and outlier_sd
# plt.figure(figsize=(8, 8))
# venn3(subsets=(
#     sum(t_section['outlier_MAD'] & ~t_section['outlier_sd'] & ~t_section['outlier_sd2']),
#     sum(~t_section['outlier_MAD'] & t_section['outlier_sd'] & ~t_section['outlier_sd2']),
#     sum(t_section['outlier_MAD'] & t_section['outlier_sd'] & ~t_section['outlier_sd2']),
#     sum(~t_section['outlier_MAD'] & ~t_section['outlier_sd'] & t_section['outlier_sd2']),
#     sum(t_section['outlier_MAD'] & ~t_section['outlier_sd'] & t_section['outlier_sd2']),
#     sum(~t_section['outlier_MAD'] & t_section['outlier_sd'] & t_section['outlier_sd2']),
#     sum(t_section['outlier_MAD'] & t_section['outlier_sd'] & t_section['outlier_sd2']),
# ), set_labels=('MAD', 'SD', 'SD2'))
# plt.title(f'Outliers Detected by different Methods in quarter {quarter}, {year} for {flow}')
# plt.show()
# -

# ### Visualization of transaction and outliers - T015

# + active=""
# import matplotlib.pyplot as plt
# import seaborn as sns
# import ipywidgets as widgets
# from IPython.display import display
#
# # Function to filter data for a specific comno and plot it on given axes
# def plot_transactions_for_comno(ax, dataset, comno_value, x_var, y_var, hue):
#     # Filter data for the given comno
#     filtered_data = dataset[dataset['comno'] == comno_value]
#
#     # Plot transactions, color-coding based on the hue variable
#     sns.scatterplot(data=filtered_data, x=x_var, y=y_var, 
#                     hue=hue, ax=ax, palette='muted', legend='full')
#
#     # Add plot labels and title
#     ax.set_title(f'{flow.capitalize()}. {year} quarter {quarter}: Transactions for comno {comno_value} - Detection_method: {hue}')
#     ax.set_xlabel(x_var)
#     ax.set_ylabel(y_var)
#
# # Function to update plots based on selected comno, dataset, and axes
# def update_plot(comno_value, dataset_name, x_var, y_var):
#     # Select the appropriate dataset based on the dropdown
#     dataset = {
#         't_section': t_section
#     }[dataset_name]
#
#     # Create a grid of subplots
#     num_hues = len(hue_variables)
#     fig, axs = plt.subplots(nrows=(num_hues + 1) // 2, ncols=2, figsize=(14, 6 * ((num_hues + 1) // 2)))
#     axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing
#
#     # Loop through each hue variable and create a plot in the corresponding subplot
#     for i, hue in enumerate(hue_variables):
#         plot_transactions_for_comno(axs[i], dataset, comno_value, x_var, y_var, hue)
#
#     # If there are any empty subplots, remove them
#     for j in range(i + 1, len(axs)):
#         fig.delaxes(axs[j])
#
#     # Adjust layout
#     plt.tight_layout()
#     plt.show()
#
# # Specify the comno values you want to visualize and sort them
# comno_values = sorted(t_section['comno'].unique().tolist())  # Corrected dataset reference
#
# # Define the hues you want to visualize
# hue_variables = ['outlier_sd', 'outlier_sd2', 'outlier_MAD']  # Example hue variables
#
# # Define the list of variables for the x- and y-axes
# axis_variables = ['price', 'value', 'weight', 'country', 'z_score2', 'month', 'quarter']
#
# # Create a dropdown menu for comno selection
# comno_dropdown = widgets.Select(
#     options=comno_values,
#     value=comno_values[0],  # Default value
#     description='Select comno:',
#     layout=widgets.Layout(width='300px', height='200px'),
# )
#
# # Create a dropdown menu for dataset selection
# dataset_dropdown = widgets.Dropdown(
#     options=['t_section'],
#     value='t_section',  # Default dataset
#     description='Select dataset:',
#     layout=widgets.Layout(width='300px')
# )
#
# # Create dropdowns for selecting x- and y-axis variables
# x_dropdown = widgets.Dropdown(
#     options=axis_variables,
#     value='value',  # Default x-axis
#     description='Select X:',
#     layout=widgets.Layout(width='300px')
# )
#
# y_dropdown = widgets.Dropdown(
#     options=axis_variables,
#     value='price',  # Default y-axis
#     description='Select Y:',
#     layout=widgets.Layout(width='300px')
# )
#
# # Create an interactive output for the plot
# output = widgets.interactive_output(update_plot, {
#     'comno_value': comno_dropdown,
#     'dataset_name': dataset_dropdown,
#     'x_var': x_dropdown,
#     'y_var': y_dropdown
# })
#
# # Display the dropdowns and output
# display(dataset_dropdown, comno_dropdown, x_dropdown, y_dropdown, output)


# -


