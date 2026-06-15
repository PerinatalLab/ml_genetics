
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table, callback, Patch, no_update
from dash.exceptions import PreventUpdate
from base64 import b64encode
import io
import importlib
import sys
import os
hostname = os.uname().nodename
if hostname == 'BlackBeast':
    path = '/home/hedvigs/snake_book/econ'
    site = 'home'
elif hostname == 'hedvig-hp-elitedesk-800-g5-twr':
    path = '/home/hedvigs/PycharmProjects/homewrs/plab_workflow'
    site = 'work'
elif hostname == 'work-computer':
    path = '/mnt/work/workbench/hedvigs/snake_book/econ'
    site = 'server'
elif hostname == 'wl-241113-007':
    path = '/home/hedvigs/wslGit/snake_book/econ'
    site = 'silverFlex'

sys.path.append(path)
from workflow.scripts.tables import table_functions as tf
from workflow.scripts.figures import figure_functions as ff
from workflow.scripts.data_management.setup_data import read_config
#app = Dash(__name__)

csc_file = path + '/results/report/sum_file_26.csv'
dfc = pd.read_csv(csc_file, sep='\t')

dfc = tf.rename_subsets(dfc)
renamed_dfc = tf.rename_cols(dfc)
cat_namesc, val_namesc, rownamesc  = tf.get_rownames(renamed_dfc)

cm_cols = [col for col in val_namesc if '(cM' in col]
auc_cols = [col for col in val_namesc if 'AUC' in col]
nonc_cols = [col for col in val_namesc if col not in cm_cols]
nonp_cols = [col for col in val_namesc if '(' not in col]

auc_rem = ['AUC(perm)','AUC(prob)','AUC(binary)', 'AUC(cMedian)']
vvar = [var for var in auc_cols if var not in auc_rem]
cols = renamed_dfc.columns.to_list()
idv = [var for var in cols if var not in vvar]

melted_df = pd.melt(
    renamed_dfc, 
    id_vars=idv, #['Model', 'SNPs', 'MaternalFetal', 'Fold'], 
    value_vars=vvar, 
    var_name='Type', 
    value_name='AUC')

melted_df.sort_values(by="MaternalFetal", inplace=True)

df = melted_df

categorical_cols, value_cols, rownames  = tf.get_rownames(df)
categorical_cols.append('Fold')
# settings for app 
colorscales = px.colors.named_colorscales()
op = [{'label': i, 'value': i} for i in df.columns]


fig = px.box(df, x=categorical_cols[1], y=value_cols[-1], color=categorical_cols[2], notched=True)
# yaxis_dict = dict(
#                 autorange=True,
#                 showgrid=True,
#                 zeroline=True,
#                 dtick=5,
#                 gridcolor='rgb(255, 255, 255)',
#                 gridwidth=0.1,
#                 zerolinecolor='rgb(255, 255, 255)',
#                 zerolinewidth=20,
#                 )
margin_dict = dict(
                l=40,
                r=10,
                b=80,
                t=140,
                )
paper_color = 'rgb(243, 243, 243)'
plot_color = 'rgba(215, 234, 256, 0.5)'
my_discrete=['#C55A11','#398D63','#4472C4', '#E49E2C','#B6487C']

app = Dash(__name__)

app.layout = html.Div(
    children=[
        html.Div(
            children=[
            html.H2("Metrics by configuration", 
                    style={'textAlign': 'center'}),
            html.P("Filter by:"),
            dcc.Checklist(
                id='filter-by', 
                options = categorical_cols,
                value=[], 
                inline=True,
            ),
    #         html.Button("Add Filter", id="add-filter-btn", n_clicks=0),
    #         html.Div(id="dropdown-container-div", children=[]),
    #         html.Div(id="dropdown-container-output-div"),
    # #        html.Div(id="dropdown-container-div", children=[]),
            dcc.Dropdown(
                id='filter-options', 
                options=[], 
                multi=True),
            html.P("x-axis:"),
            dcc.Dropdown(
                id='x-axis', 
                options=op,
                value=categorical_cols[0], 
            ),
            html.P("y-axis (metric):"),
            dcc.Dropdown(
                id='y-axis', 
                options=value_cols, 
                value=value_cols[0], 
            ),
            html.P("color:"),
            dcc.Dropdown(
                id='color', 
                options=categorical_cols, 
                value=categorical_cols[1], 
            ),
            html.H4("Type of plot:", 
                    style={'textAlign': 'center'}),
            dcc.RadioItems(
                id='plot-choice', 
                options=['Scatter', 'Box'], 
                value='Box'), 
            ], 
            style={'padding': 10,'width':'20%'}),
        html.Div(
            children=[
                dcc.Graph(
                    id="graph", 
                    figure=fig),
                dcc.Tooltip(
                    id="graph-tooltip"),
                dash_table.DataTable(
                    data=df.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in df.columns],
                    id='table',
                    sort_action='native',
                    filter_action="native",
                    filter_options={"placeholder_text": "Filter column..."},
                    fixed_rows={'headers': True}, 
                    style_table={'height': '300px', 'overflowY': 'auto',},
                    style_as_list_view=True,
                    style_cell={'padding': '20px'},
                    style_data_conditional=[
                        {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(220, 220, 220)',}
                        ],
                    style_header={'backgroundColor': 'white', 'fontWeight': 'bold'},
                    ),
                ],
            style={'width': '70%'}
            ),
        ],
    style={'display': 'flex', 'flexDirection': 'row'})

@app.callback(
    Output("filter-options", "options"),
    Input("filter-by", "value"),
    prevent_initial_call=True,
)
def update_filter_options(filter_by):    
    if filter_by:
        filter_options=np.unique(df[filter_by])
#        return filter_options
        return [{'label': option, 'value': option} for option in filter_options]
    else:
        return []

@app.callback(
    Output("table", "data"), 
    Output("graph", "figure", allow_duplicate=True), 
    Input("x-axis", "value"), 
    Input("y-axis", "value"),
    Input("color", "value"),
    Input("filter-by", "value"),
    Input("filter-options", "value"),
    Input("plot-choice", "value"),
    prevent_initial_call=True,
)
def update_output(x, y, color, filter_by, filter_values, plot_choice):
    filtered_title = '<br>'
    if filter_by and filter_values:
        filtered_df = df.copy()
        patched_fig = Patch()
        #filtered_title = '<br>'
        for filter_col, filter_val in zip(filter_by, filter_values):
            filtered_df = filtered_df[filtered_df[filter_col] == filter_val]
            filtered_title= filtered_title + f'{filter_col}:{filter_val} <br>'
        #filtered_df = df
        #filter_value = str(filter_value)
        #for filteropt in filter_by:
        #    filtered_df = filtered_df[filtered_df[filteropt]==filter_value]
        if plot_choice == 'Scatter':
            patched_fig = px.scatter(data_frame=filtered_df, x=x, y=y, color=color, color_discrete_sequence=px.colors.qualitative.Vivid)
        else:
            patched_fig = px.box(data_frame=filtered_df, x=x, y=y, color=color, color_discrete_sequence=px.colors.qualitative.Vivid)
        fdf = filtered_df
        pfig = patched_fig
    else:
        if plot_choice == 'Scatter':
            fig = px.scatter(
                data_frame=df, 
                x=x, 
                y=y, 
                color=color,
                color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(
                scattermode='group', 
                scattergap=.95)
        else:
            fig = px.box(data_frame=df, x=x, y=y, color=color,color_discrete_sequence=px.colors.qualitative.Vivid)
        fdf=df
        pfig = fig
    if 'lr' in y:
        pfig.add_shape(
            type='line', 
            line=dict(dash='dash'),
            x0=-1, 
            x1=9, 
            y0=1, 
            y1=1)
    elif 'auc' in y.lower():
        pfig.add_shape(
            type='line', 
            line=dict(
                dash='dash', 
                color=px.colors.qualitative.Vivid[4], 
                width=3),
            name=f'Average {y} for all {color}', 
            showlegend=True, 
            opacity=0.5, 
            xref="paper",
            x0=0, 
            x1=1, 
            y0=fdf[y].mean(),
            y1=fdf[y].mean() )
        for i, cname in enumerate(fdf[color].unique()):
            cname_df = fdf[fdf[color]==cname]
            mean_cname = cname_df[y].mean()
            pfig.add_shape(
                type='line', 
                line=dict(dash='dash', color=px.colors.qualitative.Vivid[i]), 
                name=f'Average {y} for {cname}',
                xref="paper",
                x0=0, 
                x1=1, 
                y0=mean_cname,
                y1=mean_cname, 
                showlegend=True, 
                opacity=0.7 )

    pfig.update_layout(
        title=f'Filters {filtered_title}',# {y} by {x}',
        margin=margin_dict,
        paper_bgcolor=paper_color,
        plot_bgcolor=plot_color,
        showlegend=True)
    
    return fdf.to_dict('records'), pfig

if __name__ == '__main__':
    app.run_server(debug=True, port=4030)
