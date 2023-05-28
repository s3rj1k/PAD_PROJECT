import dash
import pandas as pd
from dash import html
from dash.dependencies import Input, Output, State

from app import update_regression_plot_linear, update_regression_plot_lowess, render_content_regression, \
    render_content_usage, render_content_data, layout_app, update_regression_plot_nonlinear, render_content_heatmap, \
    render_ols_summary, update_regression_summary
from data import get_significant_data

# Initiate the app (`suppress_callback_exceptions` due to dynamic content generation)
dashboard_app = dash.Dash(__name__, suppress_callback_exceptions=True)
dashboard_app.layout = layout_app()

# Define a hidden Div inside your app layout to store the intermediate dataframe
dashboard_app.layout = html.Div([dashboard_app.layout, html.Div(id='intermediate-data', style={'display': 'none'})])


@dashboard_app.callback(Output('intermediate-data', 'children'),
                        Input('update-button', 'n_clicks'),
                        State('input-interval', 'value'))
def update_dataframe(n_clicks, interval):
    if n_clicks is None:
        # avoid updating when the app is loaded for the first time
        return dash.no_update
    df = get_significant_data(interval)
    return df.to_json(date_format='iso', orient='split')


@dashboard_app.callback(Output('tabs-content', 'children'),
                        [Input('tabs', 'value'),
                         Input('intermediate-data', 'children')])
def render_content(tab, json_df):
    df = pd.read_json(json_df, orient='split')
    if tab == 'Data':
        return render_content_data(df)
    elif tab == 'Usage':
        return render_content_usage(df)
    elif tab == 'Regression':
        return render_content_regression(df)
    elif tab == 'Heatmap':
        return render_content_heatmap(df)
    elif tab == 'OLS_Summary':
        return render_ols_summary(df)


@dashboard_app.callback(Output('regression-plot', 'figure'),
                        [Input('regression-type', 'value'),
                         Input('independent-variable', 'value'),
                         Input('intermediate-data', 'children')])
def update_regression_plot(regression_type, independent_variable, json_df):
    df = pd.read_json(json_df, orient='split')
    if regression_type == 'linear':
        return update_regression_plot_linear(df, independent_variable)
    elif regression_type == 'lowess':
        return update_regression_plot_lowess(df, independent_variable)
    elif regression_type == 'nonlinear':
        return update_regression_plot_nonlinear(df, independent_variable)


@dashboard_app.callback(Output('summary-content', 'children'),
                        [Input('independent-variable-summary', 'value'),
                         Input('intermediate-data', 'children')])
def render_summary(independent_variable, json_df):
    df = pd.read_json(json_df, orient='split')
    return update_regression_summary(df, independent_variable)


# Run server app
dashboard_app.run_server(debug=True)
