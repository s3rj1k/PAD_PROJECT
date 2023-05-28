import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
from dash import dcc, html, dash_table
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.nonparametric.smoothers_lowess import lowess


def layout_app():
    layout = html.Div([
        html.Div([
            html.H1('Apartment 42 electricity usage and weather data for 2016 year.',
                    style={'margin-right': '50px', 'flex-grow': '1'}),
            html.Label('Scale (10min, H, 4H, D, 7D...):',
                       style={'font-size': '1.75em', 'margin-left': '20px', 'margin-right': '10px'}),
            dcc.Input(
                id='input-interval',
                type='text',
                value='12H',
                style={'font-size': '1.75em', 'margin-right': '10px', 'width': '200px', 'textAlign': 'center'}
            ),
            html.Button('Update', id='update-button', n_clicks=0, style={'font-size': '1.75em'})
        ], style={'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center'}),
        dcc.Tabs(id="tabs", value='Data',
                 children=[
                     dcc.Tab(label='Data', value='Data'),
                     dcc.Tab(label='Usage', value='Usage'),
                     dcc.Tab(label='Regression Plot', value='Regression'),
                     dcc.Tab(label='OLS Summary', value='OLS_Summary'),
                     dcc.Tab(label='Heatmap', value='Heatmap')
                 ], style={'margin-bottom': '20px'}),
        html.Div(id='tabs-content', style={'width': '80%', 'margin': '0 auto'})
    ])
    return layout


def render_content_data(df):
    return html.Div([
        html.Div(
            dash_table.DataTable(
                id='datatable',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                sort_action="native",
                filter_action="native",
                page_action="native",
                page_current=0,
                page_size=25,
                style_cell={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'minWidth': '100px', 'width': '100px', 'maxWidth': '100px',
                    'overflow': 'hidden'
                },
                style_header={
                    'textAlign': 'center'
                },
            ),
            style={'margin-bottom': '20px', 'height': '80vh', 'overflow': 'auto'}
        ),
    ])


def render_content_usage(df):
    return html.Div([
        dcc.Graph(
            id='graph',
            figure=px.line(df, x='Date', y='Total Usage [kW]', markers=True, height=800),
        )
    ], style={'height': '80vh'})


def render_content_regression(df):
    regression_dropdowns = html.Div([
        dcc.Dropdown(
            id='regression-type',
            options=[
                {'label': 'Linear Regression', 'value': 'linear'},
                {'label': 'LOWESS Regression', 'value': 'lowess'},
                {'label': 'Nonlinear Regression', 'value': 'nonlinear'}
            ],
            value='linear',
            style={'width': '300px', 'margin-right': '20px'},
            clearable=False,
            searchable=False
        ),
        dcc.Dropdown(
            id='independent-variable',
            options=[{'label': col, 'value': col} for col in df.columns if
                     col not in ['Date', 'Total Usage [kW]']],
            value=df.columns[2],
            style={'width': '300px'},
            clearable=False,
            searchable=False
        )
    ], style={'display': 'flex', 'justify-content': 'center', 'padding-bottom': '20px'})

    regression_plot = dcc.Graph(id='regression-plot', style={'height': '800px'})

    return html.Div([regression_dropdowns, regression_plot])


def update_regression_plot_linear(df, independent_variable):
    x = df[independent_variable]
    y = df['Total Usage [kW]']

    # Perform linear regression
    coefficients = np.polyfit(x, y, deg=1)
    line = coefficients[0] * x + coefficients[1]

    # Create scatter plot and regression line
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=x, y=line, mode='lines', name='Linear Regression'))

    # Set plot layout
    fig.update_layout(title=f'Linear Regression: {independent_variable} vs Total Usage',
                      xaxis_title=independent_variable,
                      yaxis_title='Total Usage [kW]')

    return fig


def update_regression_plot_lowess(df, independent_variable):
    x = df[independent_variable]
    y = df['Total Usage [kW]']

    # Perform LOWESS regression
    lowess_data = lowess(y, x, frac=0.5)

    # Create scatter plot and LOWESS curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=lowess_data[:, 0], y=lowess_data[:, 1], mode='lines', name='LOWESS Regression'))

    # Set plot layout
    fig.update_layout(title=f'LOWESS Regression: {independent_variable} vs Total Usage',
                      xaxis_title=independent_variable,
                      yaxis_title='Total Usage [kW]')

    return fig


def update_regression_plot_nonlinear(df, independent_variable):
    x = df[independent_variable].values.reshape(-1, 1)
    y = df['Total Usage [kW]']

    # Perform polynomial regression
    model = make_pipeline(PolynomialFeatures(2), LinearRegression())
    model.fit(x, y)
    x_range = np.linspace(df[independent_variable].min(), df[independent_variable].max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    # Create scatter plot and regression line
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[independent_variable], y=df['Total Usage [kW]'], mode='markers', name='Data'))
    fig.add_trace(go.Scatter(x=x_range.squeeze(), y=y_range, mode='lines', name='Polynomial Regression'))

    # Set plot layout
    fig.update_layout(title=f'Polynomial Regression: {independent_variable} vs Total Usage',
                      xaxis_title=independent_variable,
                      yaxis_title='Total Usage [kW]')

    return fig


def render_content_heatmap(df):
    # Compute correlations
    correlation_matrix = df.corr()

    # Create heatmap
    heatmap = go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        hoverongaps=False)

    annotations = []
    for i, row in enumerate(correlation_matrix.values):
        for j, value in enumerate(row):
            annotations.append(
                dict(
                    showarrow=False,
                    text=round(value, 2),  # rounding to 2 decimal places
                    xref='x',
                    yref='y',
                    x=correlation_matrix.columns[j],
                    y=correlation_matrix.columns[i]
                )
            )

    layout = go.Layout(
        title='Correlation Heatmap',
        autosize=False,
        width=800,
        height=800,
        annotations=annotations,
        margin=dict(l=50, r=50, b=100, t=100, pad=4)
    )

    fig = go.Figure(data=heatmap, layout=layout)

    return html.Div(
        dcc.Graph(figure=fig),
        style={'display': 'flex', 'justify-content': 'center'}
    )


def render_ols_summary(df):
    regression_dropdowns = html.Div([
        dcc.Dropdown(
            id='independent-variable-summary',
            options=[{'label': 'ALL', 'value': 'ALL'}] + [{'label': col, 'value': col} for col in df.columns if
                                                          col not in ['Date', 'Total Usage [kW]']],
            value=df.columns[2],
            style={'width': '300px', 'margin-right': '20px'},
            clearable=False,
            searchable=False
        ),
    ], style={'display': 'flex', 'justify-content': 'center', 'padding-bottom': '20px'})

    summary_content = html.Div(id='summary-content')

    return html.Div([regression_dropdowns, summary_content])


def update_regression_summary(df, independent_variable):
    if independent_variable == 'ALL':
        # If 'ALL' is selected, include all variables in the regression model
        variables = [col for col in df.columns if col not in ['Date', 'Total Usage [kW]']]
    else:
        variables = [independent_variable]

    return html.Div(perform_regression(df, variables))


def perform_regression(df, variables):
    # Perform linear regression and return OLS summary table
    X = df[variables]
    y = df['Total Usage [kW]']
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract all tables from summary
    tables = results.summary().tables

    # Convert tables from SimpleTable to DataFrame
    df_summary = [pd.read_html(table.as_html(), header=0, index_col=0)[0] for table in tables]

    # Create a separate DataTable for each table in the summary
    summary_divs = []
    for i, df_table in enumerate(df_summary):
        summary_divs.append(html.Div([
            dash_table.DataTable(
                data=df_table.reset_index().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df_table.reset_index().columns],
                style_cell={'whiteSpace': 'normal', 'height': 'auto'},
            )
        ]))

    return summary_divs
