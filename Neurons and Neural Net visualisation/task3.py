import numpy as np
import pandas as pd
import random
import dash
from dash import dcc, html
import plotly.express as px
import plotly.graph_objects as go
import csv
from torch import nn
from neuralnet import *

#for 3: heaviside, logistic
#for 4: sin,tanh
#for 5: sign, ReLU, Leaky ReLu



df = pd.read_csv("scatter_data.csv")
df = df.drop(df.columns[0], axis=1)

data_list = []
for i in range(len(df)):
    data_list.append((df['x1'][i],df['y1'][i],0))
    data_list.append((df['x2'][i],df['y2'][i],1))

split = int(len(data_list)*0.8)
data_train = random.sample(data_list, split)
data_eval = [item for item in data_list if item not in data_train]


#Creating the neural_network
neuralnet1 = NeuralNet(data_train,(4,3,2),'sigmoid')
neuralnet1.train()

# app2 = dash.Dash(__name__)
# df2 = pd.DataFrame({
#     'X': np.concatenate((np.array(df['x1']), np.array(df['x2']))),
#     'Y': np.concatenate((np.array(df['y1']), np.array(df['y2']))),
#     'Dataset': ['Class 1'] * len(df) + ['Class 2'] * len(df)
# })
#
# def create_scatter_line(df,neuron):
#     # Scatter plot
#     fig = px.scatter(df, x='X', y='Y', color='Dataset',
#                  labels={'X': 'X-axis', 'Y': 'Y-axis'}, color_discrete_sequence=['navy', 'magenta'])
#
#
#     slope = -neuron.w[0]/neuron.w[1]
#     intercept = -neuron.b/neuron.w[1]
#     x_range = np.linspace(-2,2, 100)
#     y_range = slope * x_range + intercept
#     fig.add_trace(go.Scatter(x=x_range, y=y_range, mode='lines', name='decision boundary',line=dict(color='rgb(60, 26, 2)')))
#     fig.update_layout(
#         plot_bgcolor='rgb(237, 237, 213)',
#         paper_bgcolor='rgb(228, 228, 176)',
#         font=dict(color='rgb(60, 26, 2)'),
#         yaxis=dict(range=[-2, 2]),  # Set y-axis limits
#         title = dict(text=f'Decision boundary for {neuron.activation_function.upper()}' , font=dict(family='Bodoni MT', size=24, color='rgb(60, 26, 2)')))
#     return fig
#
#
#
# graph_style = {'width': '75%', 'display': 'block'}
# # Layout of the app
# app2.layout = html.Div(
#     style={'backgroundColor': 'rgb(228, 228, 176)'},  # Set the background color to teal
#     children=[
#         html.Div(
#             style={'position': 'fixed', 'top': '0','left':'0','height':'120px','width':'100%','backgroundColor': 'rgb(228, 228, 176)','border-top': '3px solid rgb(60, 26, 2)',  'border-bottom': '2px solid rgb(60, 26, 2)','zIndex': '1000' },
#             children=[html.H1("Decision boundaries for different activation functions", style={'margin-left': '50px', 'color': 'rgb(60, 26, 2)','font-family': 'Bodoni MT Condensed','fontSize': '60px'}),]
#         ),
#         html.Div(
#             children = [
#         html.Div(style={'height':'120px'}),
#         html.Div(children=[dcc.Graph(id='graph1', figure=create_scatter_line(df2, n1))], style=graph_style),
# ])])
#
# if __name__ == '__main__':
#     app2.run_server(debug=True)
#


