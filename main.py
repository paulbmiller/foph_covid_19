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

    dates = df.loc['CH', :].index.get_level_values('date')
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

    dates = df.loc['CH', :].index.get_level_values('date')
    cantons = df.drop('CH', axis=0, level='geoRegion')['sumTotal']
    cantons_sum = cantons.sum(axis=0, level='date')

    consistent = True

    for date in dates:
        if cantons_sum[date] != df.loc['CH', date]['sumTotal']:
            print('Sum of cantons is not the same as the CH total for fully vaccinated')
            consistent = False

    if consistent:
        print('Sum of cases for fully vaccinated are consistent.')


def visualize(fn, column, plot_title, plot_herd_immunities=False, agg_cantons=False, sarima_forecast=None, ax=None,
              column_label=None, colors=None, remove_cumul=False, avg_7days=False):
    """
    Function to draw plots for a column.

    :param fn: Name of the file which contains the column we want to plot
    :param column: Name of the column to plot
    :param plot_title: Title of the plot or subplot
    :param plot_herd_immunities: Whether we want to plot the lines for herd immunities for vaccines
    :param agg_cantons: Whether we want to use the aggregation of cantons (True) or the values at the CH index (False)
    :param sarima_forecast: Tuple of the column to forecast and the number of days
    :param ax: The axis to which matplotlib will draw the plot
    :param column_label: The label we want to use for drawing the column
    :param colors: Tuple of colors used
    :param remove_cumul: Whether we want to diff the column before plotting
    :param avg_7days: Plot the 7 days average of the given column
    :return: None
    """
    df = preprocess(fn)

    if colors is None:
        colors = ['cadetblue', 'coral', 'limegreen', 'lightpink', 'orange']
    curr_color = 0

    if column_label is None:
        column_label = column

    if plot_herd_immunities:
        ch_pop_total = df.loc['CH', :]['pop'].values[0]

    if agg_cantons:
        df = df.drop('CH', axis=0, level='geoRegion')
        df = df[column].sum(axis=0, level='date')
        df = df.to_frame()
    else:
        df = df.loc['CH', :]

    df = df[df[column] > 0]

    # result = seasonal_decompose(df[column], model='add')
    # result.plot()

    if ax is None:
        ax = plt.gca()

    print(df)

    if remove_cumul:
        first_val = df[column].iloc[0]
        df[column] = df[column].diff()
        df[column].iloc[0] = first_val

    ax.plot(df[column], label=column_label, color=colors[curr_color])
    curr_color += 1

    if avg_7days:
        df['avg_7days'] = df[column].rolling(7).mean()
        ax.plot(df['avg_7days'], label='7d average', color=colors[curr_color])
        curr_color += 1

    if sarima_forecast is not None:
        column, nb_days = sarima_forecast
        stepwise_fit = auto_arima(df[column].dropna(), start_p=0, start_q=0, max_p=5, max_q=5, m=7, trace=True)
        print(stepwise_fit.summary())

        model = SARIMAX(df[column].dropna(), order=stepwise_fit.order, seasonal_order=stepwise_fit.seasonal_order)
        results = model.fit(disp=False)
        # print(results.summary())

        # Predict ´sarima_forecast´ days
        predictions = results.predict(start=len(df[column].dropna()), end=len(df[column].dropna()) + nb_days,
                                      typ='levels').rename('SARIMA forecast')
        last_date = df.index[-1]
        df = pd.concat([df, predictions], axis=1)

        # In order to have a line from the last value
        df['SARIMA forecast'].loc[last_date] = df[column].loc[last_date]

        ax.plot(df['SARIMA forecast'], label='SARIMA forecast', color=colors[curr_color])
        curr_color += 1

    ax.set_title(plot_title)

    ax.yaxis.get_major_formatter().set_scientific(False)
    if plot_herd_immunities:
        ax.axhline(y=ch_pop_total * 0.8, color=colors[curr_color], label='polio herd immunity')
        curr_color += 1
        ax.axhline(y=ch_pop_total * 0.95, color=colors[curr_color], label='measles herd immunity')
        curr_color += 1
    ax.legend()

    if ax is None:
        plt.show()


if __name__ == '__main__':
    download_data()
    check_sum_cases()
    check_sum_fullyvacc()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    visualize(VACCDOSESADMIN, 'sumTotal', 'New administered vaccines in Switzerland', agg_cantons=True,
              remove_cumul=True, avg_7days=True, column_label='daily doses administered',
              sarima_forecast=('sumTotal', 30), ax=ax1)
    visualize(VACCDOSESDELIV, 'sumTotal', 'Total vaccines available in Switzerland', agg_cantons=True,
              column_label='sum of delivered doses', ax=ax2)
    visualize(VACCDOSESADMIN, 'sumTotal', 'Total vaccines doses available and administered in Switzerland',
              agg_cantons=True, column_label='sum of administered doses', ax=ax2, colors=['orange'])
    visualize(FULLYVACC, 'sumTotal', 'New fully vaccinated people in Switzerland', agg_cantons=True,
              remove_cumul=True, avg_7days=True, sarima_forecast=('sumTotal', 30), ax=ax3, column_label='daily fully vacc')
    visualize(FULLYVACC, 'sumTotal', 'Total fully vaccinated people in Switzerland', agg_cantons=True,
              sarima_forecast=('sumTotal', 30), ax=ax4, column_label='sum of fully vaccinated')

    plt.show()
