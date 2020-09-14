import pandas as pd
import numpy as np
from datetime import datetime
from scipy import optimize
from scipy.integrate import odeint
from pathlib import Path


def confirmed_cumulative():
    """
    Get COVID-19 confirmed case for all countries
    :return: pandas dataframe
    """
    # get large data frame
    if Path.cwd().name == 'src':
        data_path = Path.cwd().parent / 'data/processed/COVID_final_set.csv'
    else:
        data_path = Path.cwd() / 'data/processed/COVID_final_set.csv'
    full_data_frame = pd.read_csv(data_path, sep=';')
    full_data_frame.reset_index(drop=True)

    countries = full_data_frame.country.unique()

    # convert date to datetime format
    t_idx = [datetime.strptime(date, "%Y-%m-%d") for date in full_data_frame.date]
    full_data_frame['date'] = t_idx

    # fetch confirmed cases of all countries
    df = full_data_frame.drop(['state'], axis=1).groupby(['country', 'date'])['confirmed'].sum()
    confirmed = pd.DataFrame()
    confirmed['date'] = df['Canada'].index
    for each in countries:
        confirmed[each] = df[each].values
    return confirmed, countries


class SIRModel:
    def __init__(self, df, country, population, percentage=5):
        """
        Creates the simulated curves of COVID-19 using SIR (Susceptible, Infectious, or Recovered) model
        :param df: panda dataframe
        :param country: Name of country
        :param population: Population of the country
        :param percentage: Susceptible percentage
        """
        self.df = df
        self.country = country
        self.population = population
        self.percentage = percentage
        self.N0 = (self.percentage / 100) * self.population
        self.pop_t = None
        self.p_cov = None
        self.p_err = None
        self.fitted = None

        self._get_SIR_initials()

    def _get_index(self, percentage):
        """
        Initial day of infected population
        :param percentage: percentage of susceptible
        :return:
        """
        self.idx_I0 = np.where(self.df[self.country] > self.N0 * (percentage / 100))[0][0]

    def _initial_infected(self, percentage=0.05):
        """
        Initially infected population based on percentage.
        :param percentage:
        :return:
        """
        self._get_index(percentage)
        self.ydata = np.array(self.df[self.country][self.idx_I0:])

    def _set_time(self):
        """
        Set time period based on initially infected index
        :return:
        """
        self._initial_infected()
        self.t = np.arange(len(self.ydata))

    def _get_SIR_initials(self, R0=0):
        """
        Set up initial values for SIR model.
        :param R0: Recovery index
        :return: None
        """
        self._set_time()
        self.I0 = self.ydata[0]
        self.S0 = self.N0 - self.I0
        self.R0 = R0

        self.SIR = np.array([self.S0, self.I0, self.R0])

    def calculate_SIR(self, SIR, t, beta, gamma):
        """
        :param SIR: list:   S: susceptible population
                            I: infected people
                            R: recovered people
        :param t:
        :param beta: infection rate
        :param gamma: recovery rate
        :return:
        """
        S, I, R = SIR
        dS_dt = -beta * S * I / self.N0
        dI_dt = beta * S * I / self.N0 - gamma * I
        dR_dt = gamma * I

        return dS_dt, dI_dt, dR_dt

    def fit_ode(self, x, beta, gamma):
        """
        Helper function for the integration
        :param x:
        :param beta: infection rate
        :param gamma: recovery rate
        :return:
        """
        self._get_SIR_initials()
        return odeint(self.calculate_SIR, (self.S0, self.I0, self.R0), self.t, args=(beta, gamma))[:, 1]

    def fitted_curve(self):
        """
        Fit the curve using optimize.curve_fit from Scipy library.
        :return:
        """
        self.pop_t, self.p_cov = optimize.curve_fit(self.fit_ode, self.t, self.ydata)
        self.p_err = np.sqrt(np.diag(self.p_cov))
        self.fitted = self.fit_ode(self.t, *self.pop_t)
        # return the final fitted curve
        return self.fitted


def get_optimum_beta_gamma(df, country, susceptible_percentage=5):
    # get world population. World population data is stored in a csv file.
    if Path.cwd().name == 'src':
        pop_path = Path.cwd().parent / 'data/processed/population_data.csv'
    else:
        pop_path = Path.cwd() / 'data/processed/population_data.csv'
    df_population = pd.read_csv(pop_path, sep=';', index_col=0)
    population = df_population.T[country].values[0]
    limit = len(df) - 1
    periods = [[39, 70]]
    p = 70
    duration = [10, 20, 25, 30, 30, 30, 30]
    for d in duration:
        periods.append([p, p + d])
        p += d
    periods.append([p, limit])
    # fit curve
    fit_line = np.array([])
    dynamic_beta = []
    dynamic_gamma = []
    dynamic_R0 = []
    for n, element in enumerate(periods):
        try:
            OBJ_SIR = SIRModel(df[element[0]:element[1]], country=country, population=population,
                               percentage=susceptible_percentage)
            fit_line = np.concatenate([fit_line, OBJ_SIR.fitted_curve()])
            dynamic_beta.append(OBJ_SIR.pop_t[0])
            dynamic_gamma.append(OBJ_SIR.pop_t[1])
            dynamic_R0.append(OBJ_SIR.pop_t[0] / OBJ_SIR.pop_t[1])
        except:
            periods = periods[n + 1:]
            dynamic_beta.append(np.nan)
            dynamic_gamma.append(np.nan)
            dynamic_R0.append(np.nan)

    # get starting point
    idx = SIRModel(df, country=country, population=population).idx_I0

    return fit_line, idx
