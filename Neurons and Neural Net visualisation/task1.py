import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np

import io

# Initialize the Dash app
app = dash.Dash(__name__)

# Function to generate new data for the first scatter plot

def generate_data(mean_x,mean_y,std_dev_x,std_dev_y,sample_size):
    x = np.random.normal(loc=mean_x, scale=std_dev_x, size=sample_size)
    y = np.random.normal(loc=mean_y, scale=std_dev_y, size=sample_size)
    return x, y

# Function to generate new data for the second scatter plot

# Initialize data for both plots
n = 100
mean_x1 = np.random.uniform(-1,1)
mean_x2 = np.random.uniform(-1,1)
mean_y1 = np.random.uniform(-1,1)
mean_y2 = np.random.uniform(-1,1)
std_x1 = 1
std_y1 = 1
std_x2 = 1
std_y2 = 1

x1, y1 = generate_data(mean_x1,mean_y1,std_x1,std_y1,n)
x2, y2 = generate_data(mean_x2,mean_y2,std_x2,std_y2,n)


# Combine data into a DataFrame
df = pd.DataFrame({
    'X': np.concatenate((x1, x2)),
    'Y': np.concatenate((y1, y2)),
    'Dataset': ['Scatter Plot 1'] * len(x1) + ['Scatter Plot 2'] * len(x2)
})

# Create initial figure using Plotly Express
fig = px.scatter(df, x='X', y='Y', color='Dataset',
                 labels={'X': 'X-axis', 'Y': 'Y-axis'}, color_discrete_sequence=['yellow', 'magenta'])

# Update layout of the figure
fig.update_layout(
    plot_bgcolor='teal',    # Change the plot area background color to teal
    paper_bgcolor='teal',   # Change the entire figure background color to teal
    font=dict(color='white') # Change font color to white
)

# Layout of the app
app.layout = html.Div(
    style={'backgroundColor': 'teal'},  # Set the background color to teal
    children=[
        html.H1("2 Dataset generator", style={'margin-left': '150px', 'color': 'white','font-family': 'Arial'}),
        html.Div(
            style={'display': 'flex', 'justify-content': 'center'},
            children = [
                dcc.Graph(
                    id='combined-scatter-plot',
                    style={'height': '350px', 'width': '60%'},
                    figure=fig  # Set the initial figure with two scatter plots
                ),
                html.Div(
                    children = [
                        html.Div(
                            style={'display': 'flex', 'justify-content': 'center', 'margin-top': '20px'},
                            children=[
                                html.Div(
                                    children = [
                                        html.Label('Standard Deviation for x1:', style={'color': 'white', 'font-family': 'Arial', 'font-size': '12px'}),
                                        dcc.Slider(
                                            id='std-x1-slider',  # Slider ID
                                            min=0,             # Minimum value
                                            max=2,               # Maximum value
                                            step=0.05,            # Step size
                                            value=std_x1,        # Default value
                                            marks={i: str(i) for i in range(0, 3)}  # Marks for each integer in the range
                                        )
                                    ],
                                    style={'width': '40%','margin-right':'20px'}
                                ),
                                html.Div(
                                    children = [
                                        html.Label('Standard Deviation for y1:', style={'color': 'white', 'font-family': 'Arial', 'font-size': '12px'}),
                                        dcc.Slider(
                                            id='std-y1-slider',  # Slider ID
                                            min=0,             # Minimum value
                                            max=2,               # Maximum value
                                            step=0.05,            # Step size
                                            value=std_y1,        # Default value
                                            marks={i: str(i) for i in range(0, 3)}  # Marks for each integer in the range
                                        )
                                    ],
                                    style={'width': '40%','margin-left':'20px'}
                                )
                            ]),

        html.Div(
            style={'display': 'flex', 'justify-content': 'center', 'margin-top': '20px'},
            children=[
                html.Div(
                    children = [
                        html.Label('Standard Deviation for x2:', style={'color': 'white', 'font-family': 'Arial', 'font-size': '12px'}),
                        dcc.Slider(
                            id='std-x2-slider',  # Slider ID
                            min=0,             # Minimum value
                            max=2,               # Maximum value
                            step=0.05,            # Step size
                            value=std_x2,        # Default value
                            marks={i: str(i) for i in range(0, 3)}  # Marks for each integer in the range
                        )
                    ],
                    style={'width': '40%','margin-right':'20px'}
                ),
        html.Div(
            children = [
                html.Label('Standard Deviation for y2:', style={'color': 'white', 'font-family': 'Arial', 'font-size': '12px'}),
                dcc.Slider(
                    id='std-y2-slider',  # Slider ID
                    min=0,             # Minimum value
                    max=2,               # Maximum value
                    step=0.05,            # Step size
                    value=std_y2,        # Default value
                    marks={i: str(i) for i in range(0, 3)}  # Marks for each integer in the range
            )
            ],
            style={'width': '40%','margin-left':'20px'}
        )
    ]
        ),
        html.Div(
            children = [
                html.Label('Define the sizes of datasets:', style={'color': 'white', 'font-family': 'Arial', 'font-size': '12px'}),
                dcc.Slider(
                    id='size-slider',  # Slider ID
                    min=0,             # Minimum value
                    max=200,               # Maximum value
                    step=1,            # Step size
                    value=n,        # Default value
                    marks={i: str(i) for i in range(0, 201,20)}  # Marks for each integer in the range
            )
            ],
            style = {'width': '90%', 'margin-top':'50px'}

        )
                    ]
                )
            ]

        ),
        html.Div(
            style={'display': 'flex', 'justify-content': 'center', 'margin-top': '20px'},
            children = [
                html.Button("Generate new means", id='regenerate-button', n_clicks=0),
                html.Button("Save to CSV", id='save-button', n_clicks=0, style={'margin-left': '10px'}),
                dcc.Download(id="download-dataframe-csv"),
                dcc.Store(id='data-store')
            ]
        )
])

# Callback to update the scatter plots
@app.callback(
    Output('combined-scatter-plot', 'figure'),
    Output('data-store', 'data'),
    [Input('regenerate-button', 'n_clicks'),
     Input('std-x1-slider', 'value'),
     Input('std-y1-slider', 'value'),
     Input('std-x2-slider', 'value'),
     Input('std-y2-slider', 'value'),
     Input('size-slider', 'value')]  # Add slider value as input
)
def update_scatter_plot(n_clicks, std_x1,std_y1,std_x2,std_y2,n):  # Update function signature
    # Generate new data each time the button is clicked
    mean_x1 = np.random.uniform(-1, 1)
    mean_x2 = np.random.uniform(-1, 1)
    mean_y1 = np.random.uniform(-1, 1)
    mean_y2 = np.random.uniform(-1, 1)

    x1, y1 = generate_data(mean_x1, mean_y1, std_x1, std_y1, n)
    x2, y2 = generate_data(mean_x2, mean_y2, std_x2, std_y2, n)

    # Combine new data into a DataFrame
    df = pd.DataFrame({
        'X': np.concatenate((x1, x2)),
        'Y': np.concatenate((y1, y2)),
        'Dataset': ['Class 1'] * len(x1) + ['Class 2'] * len(x2)
    })

    # Create new figure using Plotly Express
    fig = px.scatter(df, x='X', y='Y', color='Dataset', title='Combined Scatter Plots',
                     labels={'X': 'X-axis', 'Y': 'Y-axis'}, color_discrete_sequence=['yellow', 'magenta'])

    # Update layout of the figure
    fig.update_layout(
        plot_bgcolor='teal',
        paper_bgcolor='teal',
        font=dict(color='white')
    )

    return fig, {'x1': x1.tolist(), 'y1': y1.tolist(), 'x2': x2.tolist(), 'y2': y2.tolist()}

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("save-button", "n_clicks"),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def save_to_csv(n_clicks, stored_data):
    if stored_data is None:
        return
    df = pd.DataFrame({
        'x1': stored_data['x1'], 'y1': stored_data['y1'],
        'x2': stored_data['x2'], 'y2': stored_data['y2']
    })
    return dcc.send_data_frame(df.to_csv, "scatter_data.csv")


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
