index_chained = pd.read_parquet(f'../data/index_{flow}_chained.parquet')

index_chained['periode'] = index_chained['year'].astype(str) + '0' + index_chained['quarter'].astype(str)

# Sort data by series, level, and period (to ensure the order is correct)
index_chained = index_chained.sort_values(by=['level', 'series', 'year', 'quarter'])

# Calculate the percentage change
index_chained['pct_change_1Q'] = index_chained.groupby(['level', 'series'])['index_chained'].pct_change(fill_method=None) * 100
index_chained['pct_change_4Q'] = index_chained.groupby(['level', 'series'])['index_chained'].pct_change(periods=4, fill_method=None) * 100
index_chained

import dash
from dash import dcc, html
import pandas as pd
import plotly.express as px

# Data
df = index_chained.copy()

# Extract unique levels for dropdown
unique_levels = df['level'].unique()
dropdown_options = [{'label': str(level), 'value': str(level)} for level in unique_levels]

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Dropdown for selecting level
    dcc.Dropdown(
        id='level-dropdown',
        options=dropdown_options,
        placeholder="Select a Level",
        multi=False
    ),
    
    html.Br(),
    
    # Checklist (multi-select series)
    dcc.Checklist(
        id='series-checklist',
        inline=True,  # Makes it horizontal
        style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '10px',
            'fontSize': '12px',
            'alignItems': 'center',
            'lineHeight': '1.2'
        },
        value=[],  # Starts with none selected
    ),

    html.Br(),

    # Graph for the line plot
    dcc.Graph(id='line-plot'),

    html.Br(),

    # Output
    html.Div(id='output-container'),

    dcc.Graph(figure={}, id='line-plot_change'),

    dcc.Graph(figure={}, id='line-plot_change4')
])

# Callback to update checklist options based on selected level
@app.callback(
    dash.Output('series-checklist', 'options'),
    dash.Output('series-checklist', 'value'),
    dash.Input('level-dropdown', 'value')
)
def update_series_options(selected_level):
    if selected_level:
        # Get unique series for the selected level
        filtered_series = df[df['level'] == selected_level]['series'].unique()
        options = [{'label': s, 'value': s} for s in filtered_series]
        return options, []  # Start with none selected
    return [], []

# Callback to update the line plot for selected level & series
@app.callback(
    dash.Output('line-plot', 'figure'),
    [dash.Input('level-dropdown', 'value'),
     dash.Input('series-checklist', 'value')]
)
def update_line_plot(selected_level, selected_series):
    if not selected_level or not selected_series:
        return px.line(title="No data selected")

    # Filter the dataset to the selected level and series
    filtered_data = df[(df['level'] == selected_level) & (df['series'].isin(selected_series))].copy()

    if filtered_data.empty:
        return px.line(title="No data available for selection")

    # Convert 'periode' to numeric for proper sorting
   # filtered_data['periode'] = filtered_data['periode'].astype(int)

    # Sort values by 'year', 'quarter', and 'periode'
    filtered_data = filtered_data.sort_values(by=['year', 'quarter', 'periode'])

    # Create the line plot using Plotly Express
    fig = px.line(
        filtered_data,
        x='periode',
        y='index_chained',
        color='series',
        title=f'Chained index for {selected_series} - {flow}',
        markers=True
    )

    return fig


@app.callback(
    dash.Output('line-plot_change', 'figure'),
    [dash.Input('level-dropdown', 'value'),
     dash.Input('series-checklist', 'value')]
)
def update_line_plot_change(selected_level, selected_series):
    if not selected_level or not selected_series:
        return px.line(title="No data selected")

    # Filter the dataset to the selected level and series
    filtered_data = df[(df['level'] == selected_level) & (df['series'].isin(selected_series))].copy()

    if filtered_data.empty:
        return px.line(title="No data available for selection")

    # Convert 'periode' to numeric for proper sorting
   # filtered_data['periode'] = filtered_data['periode'].astype(int)

    # Sort values by 'year', 'quarter', and 'periode'
    filtered_data = filtered_data.sort_values(by=['year', 'quarter', 'periode'])

    # Create the line plot using Plotly Express
    fig = px.line(
        filtered_data,
        x='periode',
        y='pct_change_1Q',
        color='series',
        title=f'Percentage change from previous period, for {selected_series} - {flow}',
        markers=True
    )
    return fig

@app.callback(
    dash.Output('line-plot_change4', 'figure'),
    [dash.Input('level-dropdown', 'value'),
     dash.Input('series-checklist', 'value')]
)
def update_line_plot_change4(selected_level, selected_series):
    if not selected_level or not selected_series:
        return px.line(title="No data selected")

    # Filter the dataset to the selected level and series
    filtered_data = df[(df['level'] == selected_level) & (df['series'].isin(selected_series))].copy()

    if filtered_data.empty:
        return px.line(title="No data available for selection")

    # Convert 'periode' to numeric for proper sorting
   # filtered_data['periode'] = filtered_data['periode'].astype(int)

    # Sort values by 'year', 'quarter', and 'periode'
    filtered_data = filtered_data.sort_values(by=['year', 'quarter', 'periode'])

    # Create the line plot using Plotly Express
    fig = px.line(
        filtered_data,
        x='periode',
        y='pct_change_4Q',
        color='series',
        title=f'Percentage change from same period previous year, for {selected_series} - {flow}',
        markers=True
    )
    return fig




# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=1814)
