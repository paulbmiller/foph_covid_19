from get_data import download_data
import pandas as pd
import numpy as np
from zipfile import ZipFile
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

plt.style.use('ggplot')
# Ignore sarimax warnings
warnings.filterwarnings('ignore')

ZIP_FILE_NAME = 'sources_csv.zip'
CASES_GEOREGION = 'data/COVID19Cases_geoRegion.csv'
DEATHS_GEOREGION = 'data/COVID19Death_geoRegion.csv'
VACCDOSESADMIN = 'data/COVID19VaccDosesAdministered.csv'
VACCDOSESDELIV = 'data/COVID19VaccDosesDelivered.csv'
FULLYVACC = 'data/COVID19FullyVaccPersons.csv'


def preprocess(filename):
    """Function which reads in the given CSV file, converts the datum column to datetime and removes FL rows."""
    zf = ZipFile(ZIP_FILE_NAME)
    df = pd.read_csv(zf.open(filename))

    # Change the column name datum to date for consistency
    if 'datum' in df.columns:
        df.rename(columns={'datum': 'date'}, inplace=True)

    df['date'] = pd.to_datetime(df['date'])
    df.set_index(['geoRegion', 'date'], inplace=True)

    # Remove Liechtenstein data
    df.drop('FL', axis=0, level='geoRegion', inplace=True)
    df.drop('CHFL', axis=0, level='geoRegion', inplace=True)

    return df


def check_sum_cases():
    # Function to check if the sum of cases in cantons is equal to the CH values
    df = preprocess(CASES_GEOREGION)

    dates = df.loc['CH', :]._get_label_or_level_values('date')
    cantons = df.drop('CH', axis=0, level='geoRegion')
    cantons_sum = cantons.sum(axis=0, level='date')

    consistent = True

    for date in dates:
        if cantons_sum.loc[date].entries != df.loc['CH', date].entries:
            print('Sum of cases by geographical region are inconsistent.')
            consistent = False

    if consistent:
        print('Sum of cases by geographical region are consistent.')


def check_sum_fullyvacc():
    # Function to check if the sum of canton values for the sumTotal column are equal to the CH column values
    df = preprocess(FULLYVACC)

    dates = df.loc['CH', :]._get_label_or_level_values('date')
    cantons = df.drop('CH', axis=0, level='geoRegion')['sumTotal']
    cantons_sum = cantons.sum(axis=0, level='date')

    consistent = True

    for date in dates:
        if cantons_sum[date] != df.loc['CH', date]['sumTotal']:
            print('Sum of cantons is not the same as the CH total for fully vaccinated')
            consistent = False

    if consistent:
        print('Sum of cases for fully vaccinated are consistent.')


def visualize(fn, column, plot_title, plot_herd_immunities=False, agg_cantons=False, sarima_forecast=0, ax=None,
              column_label=None, colors=['cadetblue', 'coral', 'lightpink', 'limegreen']):
    """
    Function to draw plots for a column.

    :param fn: Name of the file which contains the column we want to plot
    :param column: Name of the column to plot
    :param plot_title: Title of the plot or subplot
    :param plot_herd_immunities: Whether we want to plot the lines for herd immunities for vaccines
    :param agg_cantons: Whether we want to use the aggregation of cantons (True) or the values at the CH index (False)
    :param sarima_forecast: Number of days to forecast using a SARIMA model
    :param ax: The axis to which matplotlib will draw the plot
    :param column_label: The label we want to use for drawing the column
    :param colors: Tuple of colors used
    :return: None
    """
    df = preprocess(fn)

    if not column_label:
        column_label = column

    if plot_herd_immunities:
        CH_pop_total = df.loc['CH', :]['pop'].values[0]

    if agg_cantons:
        df = df.drop('CH', axis=0, level='geoRegion')
        df = df[column].sum(axis=0, level='date')
        df = df.to_frame()
    else:
        df = df.loc['CH', :]

    df = df[df[column] > 0]

    # result = seasonal_decompose(df[column], model='add')
    # result.plot()

    if sarima_forecast > 0:
        stepwise_fit = auto_arima(df[column], start_p=0, start_q=0, max_p=5, max_q=5, m=7, trace=True)
        print(stepwise_fit.summary())

        model = SARIMAX(df[column], order=stepwise_fit.order, seasonal_order=stepwise_fit.seasonal_order)
        results = model.fit(disp=False)
        # print(results.summary())

        # Predict ´sarima_forecast´ days
        predictions = results.predict(start=len(df[column]), end=len(df[column]) + sarima_forecast,
                                      typ='levels').rename('SARIMA forecast')
        df = pd.concat([df, predictions], axis=1)

    if not ax:
        ax = plt.gca()

    print(df)

    ax.plot(df[column], label=column_label, color=colors[0])
    if sarima_forecast > 0:
        ax.plot(df['SARIMA forecast'], label='SARIMA forecast', color=colors[1])
    ax.set_title(plot_title)

    ax.yaxis.get_major_formatter().set_scientific(False)
    if plot_herd_immunities:
        ax.axhline(y=CH_pop_total * 0.8, color=colors[2], label='polio herd immunity')
        ax.axhline(y=CH_pop_total * 0.95, color=colors[3], label='measles herd immunity')
    ax.legend()

    if not ax:
        plt.show()


if __name__ == '__main__':
    download_data()
    check_sum_cases()
    check_sum_fullyvacc()
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    visualize(CASES_GEOREGION, 'entries', 'Switzerland Covid-19 confirmed cases', ax=ax1, column_label='confirmed',
              colors=['cadetblue'])
    visualize(DEATHS_GEOREGION, 'entries', 'Switzerland Covid-19 deaths', ax=ax2, column_label='deaths',
              colors=['orangered'])
    visualize(FULLYVACC, 'sumTotal', 'Forecasting of fully vaccinated people using a SARIMA model', agg_cantons=True,
              plot_herd_immunities=True, sarima_forecast=30, ax=ax3)
    visualize(VACCDOSESADMIN, 'sumTotal', 'Forecasting of administered vaccines using a SARIMA model',
              sarima_forecast=30, agg_cantons=True, ax=ax4)
    plt.show()
