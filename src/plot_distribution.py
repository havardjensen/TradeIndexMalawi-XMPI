# +
import matplotlib.pyplot as plt
import seaborn as sns
import ipywidgets as widgets
from IPython.display import display

# Function to filter data for a specific comno and plot the distribution
def plot_price_distribution(ax, dataset, comno_value, hue):
    # Filter data for the given comno
    filtered_data = dataset[dataset['comno'] == comno_value]

    # Plot the distribution of prices, color-coding based on the hue variable
    sns.histplot(data=filtered_data, x='price', hue=hue, ax=ax, 
                 kde=True, palette='muted', bins=30, alpha=0.7)

    # Add plot labels and title
    ax.set_title(f'Price Distribution for comno {comno_value} - Hue: {hue}')
    ax.set_xlabel('Price')
    ax.set_ylabel('Frequency')

# Function to update plots based on selected comno, dataset, and hue variables
def update_price_distribution(comno_value, dataset_name):
    # Select the appropriate dataset based on the dropdown
    dataset = {
        'trade_with_weights': trade_with_weights,
        'trade_with_weights_no_outliers': trade_with_weights_no_outliers
    }[dataset_name]

    # Create a grid of subplots
    num_hues = len(hue_variables)
    fig, axs = plt.subplots(nrows=(num_hues + 1) // 2, ncols=2, figsize=(14, 6 * ((num_hues + 1) // 2)))
    axs = axs.flatten()  # Flatten the 2D array of axes for easier indexing

    # Loop through each hue variable and create a plot in the corresponding subplot
    for i, hue in enumerate(hue_variables):
        plot_price_distribution(axs[i], dataset, comno_value, hue)

    # If there are any empty subplots, remove them
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Specify the comno values you want to visualize and sort them
comno_values = sorted(trade_without_outliers['comno'].unique().tolist())

# Define the hues you want to visualize
hue_variables = ['outlier_sd']  # Example hue variables

# Create dropdowns for selecting options
comno_dropdown = widgets.Select(
    options=comno_values,
    value=comno_values[0],  # Default value
    description='Select comno:',
    layout=widgets.Layout(width='300px', height='200px'),
)

dataset_dropdown = widgets.Dropdown(
    options=['trade_with_weights', 'trade_with_weights_no_outliers'],
    value='trade_with_weights',  # Default dataset
    description='Select dataset:',
    layout=widgets.Layout(width='300px')
)

# Create an interactive output for the plot
output = widgets.interactive_output(update_price_distribution, {
    'comno_value': comno_dropdown,
    'dataset_name': dataset_dropdown,
})

# Display the dropdowns and output
display(dataset_dropdown, comno_dropdown, output)

