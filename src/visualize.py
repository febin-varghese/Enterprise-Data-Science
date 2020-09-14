import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from src.simulation import get_optimum_beta_gamma, confirmed_cumulative


class Visualize:
    def __init__(self, data_path: str):
        """
        Create dashboard to display COVID-19 data
        :param data_path: path to the .csv file
        """
        self.input_data = pd.read_csv(data_path, sep=';')
        self.fig = go.Figure()
        self.app = dash.Dash()
        self.app.layout = self.define_layout()

        @self.app.callback(Output('main_window_slope', 'figure'), [Input('country_drop_down', 'value'),
                                                                   Input('doubling_time', 'value')])
        def update_figure(country_list, show_doubling):
            if '_DR' in show_doubling:
                my_yaxis = {'type': "log",
                            'title': 'Approximated doubling rate over 3 days (larger numbers are better)'
                            }
            else:
                my_yaxis = {'type': "log",
                            'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                            }

            traces = []
            for each in country_list:

                df_plot = self.input_data[self.input_data['country'] == each]

                if show_doubling == 'doubling_rate_filtered':
                    df_plot = df_plot[
                        ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                         'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
                else:
                    df_plot = df_plot[
                        ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                         'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
                traces.append(dict(x=df_plot.date, y=df_plot[show_doubling], mode='markers+lines', opacity=0.9,
                                   name=each, marker=dict(size=4)
                                   )
                              )
            return {
                'data': traces,
                'layout': dict(width=1280, height=720, xaxis={'title': 'Timeline', 'tickangle': -45, 'nticks': 20,
                                                              'tickfont': dict(size=14, color="#7f7f7f"),
                                                              }, yaxis=my_yaxis,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               plot_bgcolor='#EEEEEE', paper_bgcolor='#F5F5F5', font={'color': '#FA5252'}
                               )
            }

    def define_layout(self):
        """
        Define the layout of Dash dashboard
        :return: None
        """
        layout = html.Div([

            dcc.Markdown('''#  Enterprise Data Science on COVID-19 (Corona virus disease 2019) data

            Goal of the project is to learn data science by applying a cross industry standard process;
            it covers the full walkthrough of: automated data gathering, data transformations, filtering and 
            machine learning to approximating the doubling time, and (static) deployment of responsive dashboard.

            '''),

            dcc.Markdown(''' ## Multi-Select Country for visualization'''),

            dcc.Dropdown(id='country_drop_down',
                         options=[{'label': each, 'value': each} for each in self.input_data['country'].unique()],
                         value=['Germany', 'India'], multi=True
                         ),

            dcc.Markdown('''## Select Timeline of confirmed COVID-19 cases or the approximated doubling time'''),

            dcc.Dropdown(id='doubling_time',
                         options=[
                             {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
                             {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
                             {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
                             {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
                         ], value='confirmed', clearable=False, multi=False
                         ),

            dcc.Graph(figure=self.fig, id='main_window_slope')
        ])
        return layout


class VisualizeSIR:
    def __init__(self):
        """
        Create dashboard to display COVID-19 simulated curves
        """
        self.df_confirmed, self.country_list = confirmed_cumulative()
        self.fig = go.Figure()
        self.app = dash.Dash()
        self.app.layout = self.define_layout()

        @self.app.callback(Output(component_id='SIR', component_property='figure'),
                           [Input(component_id='country_drop_down', component_property='value')]
                           )
        def update_figure(country_list):
            traces = []
            for pos, each in enumerate(country_list):
                traces.append(dict(x=self.df_confirmed.date, y=self.df_confirmed[each], mode='lines', opacity=0.9,
                                   name=each
                                   )
                              )
                fit_line, idx = get_optimum_beta_gamma(self.df_confirmed, each, susceptible_percentage=5)
                traces.append(dict(x=self.df_confirmed.date[idx:], y=fit_line, mode='markers+lines', opacity=0.9,
                                   name=each + '_simulated'
                                   )
                              )

            return {'data': traces,
                    'layout': dict(width=1280, height=720, xaxis={'title': 'Timeline', 'tickangle': -25, 'nticks': 20,
                                                                  'tickfont': 18, 'titlefont': 20
                                                                  },
                                   yaxis={'type': 'log', 'title': 'Number of infected people (log scale)',
                                          'tickfont': 18, 'titlefont': 20
                                          },
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                   plot_bgcolor='#EEEEEE', paper_bgcolor='#F5F5F5', font={'color': '#FA5252'}
                                   )
                    }

    def define_layout(self):
        """
        Define the layout of Dash dashboard
        :return: None
        """
        layout = html.Div([
            dcc.Markdown('''# COVID-19 SIR (Susceptible, Infectious, or Recovered) Model
            
            This dashboard displays the updated status of COVID-19 in selected countries as well as the 
            simulated curves for the respective countries. The simulated curves are generated using SIR modelling. 
            The fitted curve can be used to predict the trend of COVID-19
            '''),
            html.Div([
                html.Div([
                    dcc.Markdown(''' ## Multi-Select Country for visualization'''),
                    dcc.Dropdown(id='country_drop_down',
                                 options=[{'label': each, 'value': each} for each in self.country_list],
                                 value=['Germany', "India", "France"], multi=True)
                ]),

            ]),
            dcc.Markdown(''' ''', style={'text-align': 'center', 'padding': 10, }),
            dcc.Graph(figure=self.fig, id='SIR', style={'color': '#000099', 'background-color': '#F7F9F9'})
        ])
        return layout


class MainDashboard:
    def __init__(self, data_path: str):
        self.input_data = pd.read_csv(data_path, sep=';')
        self.fig = go.Figure()
        self.df_confirmed, self.country_list = confirmed_cumulative()
        self.app = dash.Dash()
        self.tab_names = ["Data", "Simulation"]
        self.tabs_styles = {'height': '44px'}
        self.tab_style = {'borderBottom': '1px solid #d6d6d6', 'padding': '6px', 'fontWeight': 'bold'}
        self.tab_selected_style = {'borderTop': '1px solid #d6d6d6', 'borderBottom': '1px solid #d6d6d6',
                                   'backgroundColor': '#119DFF', 'color': 'white', 'padding': '6px'
                                   }
        self.app.layout = self.define_layout()
        self.app.config.suppress_callback_exceptions = True

        @self.app.callback(Output('tabs-content', 'children'), [Input('tabs1', 'value')])
        def render_content(tab):
            if tab == 'tab-1':
                return self.data_layout()
            elif tab == 'tab-2':
                return self.simulation_layout()

        @self.app.callback(Output('main_window_slope', 'figure'), [Input('country_drop_down', 'value'),
                                                                   Input('doubling_time', 'value')])
        def update_figure(country_list, show_doubling):
            if '_DR' in show_doubling:
                my_yaxis = {'type': "log",
                            'title': 'Approximated doubling rate over 3 days (larger numbers are better)'
                            }
            else:
                my_yaxis = {'type': "log",
                            'title': 'Confirmed infected people (source johns hopkins csse, log-scale)'
                            }

            traces = []
            for each in country_list:

                df_plot = self.input_data[self.input_data['country'] == each]

                if show_doubling == 'doubling_rate_filtered':
                    df_plot = df_plot[
                        ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                         'date']].groupby(['country', 'date']).agg(np.mean).reset_index()
                else:
                    df_plot = df_plot[
                        ['state', 'country', 'confirmed', 'confirmed_filtered', 'confirmed_DR', 'confirmed_filtered_DR',
                         'date']].groupby(['country', 'date']).agg(np.sum).reset_index()
                traces.append(dict(x=df_plot.date, y=df_plot[show_doubling], mode='markers+lines', opacity=0.9,
                                   name=each, marker=dict(size=4)
                                   )
                              )
            return {
                'data': traces,
                'layout': dict(width=1280, height=720, xaxis={'title': 'Timeline', 'tickangle': -45, 'nticks': 20,
                                                              'tickfont': dict(size=14, color="#7f7f7f"),
                                                              }, yaxis=my_yaxis,
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                               plot_bgcolor='#EEEEEE', paper_bgcolor='#F5F5F5', font={'color': '#FA5252'}
                               )
            }

        @self.app.callback(Output(component_id='SIR', component_property='figure'),
                           [Input(component_id='country_drop_down', component_property='value')]
                           )
        def update_figure(country_list):
            traces = []
            for pos, each in enumerate(country_list):
                traces.append(dict(x=self.df_confirmed.date, y=self.df_confirmed[each], mode='lines', opacity=0.9,
                                   name=each
                                   )
                              )
                fit_line, idx = get_optimum_beta_gamma(self.df_confirmed, each, susceptible_percentage=5)
                traces.append(dict(x=self.df_confirmed.date[idx:], y=fit_line, mode='markers+lines', opacity=0.9,
                                   name=each + '_simulated'
                                   )
                              )

            return {'data': traces,
                    'layout': dict(width=1280, height=720, xaxis={'title': 'Timeline', 'tickangle': -25, 'nticks': 20,
                                                                  'tickfont': 18, 'titlefont': 20
                                                                  },
                                   yaxis={'type': 'log', 'title': 'Number of infected people (log scale)',
                                          'tickfont': 18, 'titlefont': 20
                                          },
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                                   plot_bgcolor='#EEEEEE', paper_bgcolor='#F5F5F5', font={'color': '#FA5252'}
                                   )
                    }

    def define_layout(self):
        layout = html.Div([dcc.Tabs(id='tabs1', value='tab-1',
                                    children=[dcc.Tab(label=self.tab_names[0], value='tab-1', style=self.tab_style,
                                                      selected_style=self.tab_selected_style),
                                              dcc.Tab(label=self.tab_names[1], value='tab-2', style=self.tab_style,
                                                      selected_style=self.tab_selected_style),
                                              ]),
                           html.Div(id='tabs-content')
                           ])
        return layout

    def data_layout(self):
        """
        Define the layout of Dash dashboard
        :return: None
        """
        layout = html.Div([

            dcc.Markdown('''#  Enterprise Data Science on COVID-19 (Corona virus disease 2019) data

            Goal of the project is to learn data science by applying a cross industry standard process;
            it covers the full walkthrough of: automated data gathering, data transformations, filtering and 
            machine learning to approximating the doubling time, and (static) deployment of responsive dashboard.

            '''),

            dcc.Markdown(''' ## Multi-Select Country for visualization'''),

            dcc.Dropdown(id='country_drop_down',
                         options=[{'label': each, 'value': each} for each in self.input_data['country'].unique()],
                         value=['Germany', 'India'], multi=True
                         ),

            dcc.Markdown('''## Select Timeline of confirmed COVID-19 cases or the approximated doubling time'''),

            dcc.Dropdown(id='doubling_time',
                         options=[
                             {'label': 'Timeline Confirmed ', 'value': 'confirmed'},
                             {'label': 'Timeline Confirmed Filtered', 'value': 'confirmed_filtered'},
                             {'label': 'Timeline Doubling Rate', 'value': 'confirmed_DR'},
                             {'label': 'Timeline Doubling Rate Filtered', 'value': 'confirmed_filtered_DR'},
                         ], value='confirmed', clearable=False, multi=False
                         ),

            dcc.Graph(figure=self.fig, id='main_window_slope')
        ])
        return layout

    def simulation_layout(self):
        """
        Define the layout of Dash dashboard
        :return: None
        """
        layout = html.Div([
            dcc.Markdown('''# COVID-19 SIR (Susceptible, Infectious, or Recovered) Model

            This dashboard displays the updated status of COVID-19 in selected countries as well as the 
            simulated curves for the respective countries. The simulated curves are generated using SIR modelling. 
            The fitted curve can be used to predict the trend of COVID-19
            '''),
            html.Div([
                html.Div([
                    dcc.Markdown(''' ## Multi-Select Country for visualization'''),
                    dcc.Dropdown(id='country_drop_down',
                                 options=[{'label': each, 'value': each} for each in self.country_list],
                                 value=['Germany', "India", "France"], multi=True)
                ]),

            ]),
            dcc.Markdown(''' ''', style={'text-align': 'center', 'padding': 10, }),
            dcc.Graph(figure=self.fig, id='SIR', style={'color': '#000099', 'background-color': '#F7F9F9'})
        ])
        return layout
