import pandas as pd
import numpy as np
import git
from scipy import signal
from sklearn import linear_model


class Data:
    def __init__(self, update_data: bool = False):
        """
        COVID-19 data
        :param update_data: Flag to pull data from the John Hopkins repository
        """
        self.repository_path = 'data/raw/COVID-19'
        if update_data:
            self.pull_data()
        self.data_path = r"data/raw/COVID-19/csse_covid_19_data/csse_covid_19_time_series/" \
                         r"time_series_covid19_confirmed_global.csv"
        self.relational_data_path = r'data/processed/COVID_relational_confirmed.csv'
        self.final_data_path = r'data/processed/COVID_final_set.csv'
        self.reg = linear_model.LinearRegression(fit_intercept=True)
        self.degree = 1

        self.final_result = pd.DataFrame()

    def pull_data(self):
        """
        Pull the data from the John Hopkins github repository.
        :return: None
        """
        repo = git.Repo(self.repository_path)
        o = repo.remotes.origin
        print("Pulling data...")
        o.pull()
        print("Successfully pulled data.")

    def store_relational_data(self):
        """
        Process the raw data and create relational data. Save the results in ".csv" format
        :return: None
        """
        pd_raw = pd.read_csv(self.data_path)

        pd_data_base = pd_raw.rename(columns={'Country/Region': 'country', 'Province/State': 'state'})

        pd_data_base['state'] = pd_data_base['state'].fillna('no')

        pd_data_base = pd_data_base.drop(['Lat', 'Long'], axis=1)

        pd_relational_model = pd_data_base.set_index(['state', 'country']).T.stack(level=[0, 1]).reset_index() \
            .rename(columns={'level_0': 'date', 0: 'confirmed'}, )

        pd_relational_model['date'] = pd_relational_model.date.astype('datetime64[ns]')

        pd_relational_model.to_csv(self.relational_data_path, sep=';', index=False)
        print(' Number of rows stored: ' + str(pd_relational_model.shape[0]))
        print(' Latest date is: ' + str(max(pd_relational_model.date)))

    def get_doubling_time_via_regression(self, in_array):
        """

        :param in_array:
        :return:
        """
        y = np.array(in_array)
        X = np.arange(-1, 2).reshape(-1, 1)

        assert len(in_array) == 3
        self.reg.fit(X, y)
        intercept = self.reg.intercept_
        slope = self.reg.coef_

        return intercept / slope

    def savgol_filter(self, df_input, column='confirmed', window=5):
        """

        :param df_input:
        :param column:
        :param window:
        :return:
        """
        df_result = df_input

        filter_in = df_input[column].fillna(0)  # attention with the neutral element here

        result = signal.savgol_filter(np.array(filter_in),
                                      window,  # window size used for filtering
                                      self.degree)
        df_result[str(column + '_filtered')] = result
        return df_result

    def rolling_reg(self, df_input, col='confirmed'):
        """

        :param df_input:
        :param col:
        :return:
        """
        days_back = 3
        result = df_input[col].rolling(
            window=days_back,
            min_periods=days_back).apply(self.get_doubling_time_via_regression, raw=False)
        return result

    def calc_filtered_data(self, filter_on='confirmed'):
        """

        :param filter_on:
        :return:
        """
        pd_JH_data = pd.read_csv(self.relational_data_path, sep=';', parse_dates=[0])
        df_input = pd_JH_data.sort_values('date', ascending=True).copy()
        must_contain = set(['state', 'country', filter_on])
        assert must_contain.issubset(set(df_input.columns)), \
            ' Error in calc_filtered_data not all columns in data frame'

        df_output = df_input.copy()  # we need a copy here otherwise the filter_on column will be overwritten

        pd_filtered_result = df_output[['state', 'country', filter_on]].groupby(['state', 'country']).apply(
            self.savgol_filter)  # .reset_index()
        df_output = pd.merge(df_output, pd_filtered_result[[str(filter_on + '_filtered')]], left_index=True,
                             right_index=True, how='left')
        self.final_result = df_output.copy()

    def calc_doubling_rate(self, filter_on='confirmed'):
        """

        :param filter_on:
        :return:
        """
        df_input = self.final_result
        must_contain = set(['state', 'country', filter_on])
        assert must_contain.issubset(
            set(df_input.columns)), ' Error in calc_filtered_data not all columns in data frame'

        pd_DR_result = df_input.groupby(['state', 'country']).apply(self.rolling_reg, filter_on).reset_index()

        pd_DR_result = pd_DR_result.rename(columns={filter_on: filter_on + '_DR', 'level_2': 'index'})

        # we do the merge on the index of our big table and on the index column after groupby
        df_output = pd.merge(df_input, pd_DR_result[['index', str(filter_on + '_DR')]], left_index=True,
                             right_on=['index'], how='left')
        df_output = df_output.drop(columns=['index'])

        self.final_result = df_output

    def save_final_results(self):
        """
        Save results to a csv file
        :return: None
        """
        mask = self.final_result['confirmed'] > 100
        self.final_result['confirmed_filtered_DR'] = self.final_result['confirmed_filtered_DR'].where(mask,
                                                                                                      other=np.NaN)
        self.final_result.to_csv(self.final_data_path, sep=';', index=False)
        print(self.final_result[self.final_result['country'] == 'Germany'].tail())


