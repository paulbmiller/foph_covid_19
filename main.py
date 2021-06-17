from get_data import download_data
import pandas as pd
from zipfile import ZipFile
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import os
import shutil

plt.style.use('ggplot')
# Ignore sarimax warnings
warnings.filterwarnings('ignore')

ZIP_FILE_NAME = 'sources_csv.zip'
CASES_GEOREGION = 'data/COVID19Cases_geoRegion.csv'
DEATHS_GEOREGION = 'data/COVID19Death_geoRegion.csv'
VACCDOSESADMIN = 'data/COVID19VaccDosesAdministered.csv'
VACCDOSESDELIV = 'data/COVID19VaccDosesDelivered.csv'
FULLYVACC = 'data/COVID19FullyVaccPersons.csv'
VACCPERSONS = 'data/COVID19VaccPersons.csv'


def create_fully_vacc():
    zf = ZipFile(ZIP_FILE_NAME, 'r')
    if FULLYVACC not in zf.namelist():
        df = pd.read_csv(zf.open(VACCPERSONS))
        zf.close()
        df = df[df['type'] == 'COVID19FullyVaccPersons']
        df.to_csv('COVID19VaccPersons.csv')
        zf = ZipFile(ZIP_FILE_NAME, 'a')
        zf.write('COVID19VaccPersons.csv', FULLYVACC)
        os.remove('COVID19VaccPersons.csv')
    zf.close()


def preprocess(filename):
    """Function which reads in the given CSV file, converts the datum column to datetime and removes FL rows."""
    zf = ZipFile(ZIP_FILE_NAME)
    df = pd.read_csv(zf.open(filename))
    zf.close()

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


def aggregate_cantons(df, column):
    df = df.drop('CH', axis=0, level='geoRegion')
    df = df[column].sum(axis=0, level='date')
    df = df.to_frame()
    return df


def remove_cumul(df, column):
    first_val = df[column].iloc[0]
    df[column] = df[column].diff()
    df[column].iloc[0] = first_val
    return df


def sarima_forecast(df, column, nb_days):
    """Add a column which contains the Sarima forecast ´nb_days´ days in the future."""
    stepwise_fit = auto_arima(df[column].dropna(), start_p=0, start_q=0, max_p=5, max_q=5, m=7, trace=True)
    # print(stepwise_fit.summary())

    model = SARIMAX(df[column].dropna(), order=stepwise_fit.order, seasonal_order=stepwise_fit.seasonal_order)
    results = model.fit(disp=False)
    # print(results.summary())

    # Predict ´sarima_forecast´ days
    predictions = results.predict(start=len(df[column].dropna()), end=len(df[column].dropna()) + nb_days,
                                  typ='levels').rename('SARIMA_forecast')
    last_date = df.index[-1]
    df = pd.concat([df, predictions], axis=1)

    # In order to have a line from the last value
    df['SARIMA_forecast'].loc[last_date] = df[column].loc[last_date]

    return df


def avg_7days(df, column):
    df['avg_7days'] = df[column].rolling(7).mean()
    return df


def plot_col(df_column, column_label, color, ax=None):
    if ax is None:
        ax = plt.gca()
    else:
        ax.plot(df_column, label=column_label, color=color)


if __name__ == '__main__':
    download_data()
    create_fully_vacc()
    check_sum_cases()
    check_sum_fullyvacc()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    subplots = [
        {'filename': VACCDOSESADMIN, 'column': 'sumTotal', 'plot_title': 'New administered vaccines in Switzerland',
         'plot_herd_immunities': False, 'agg_cantons': True, 'sarima_nb_days': 30, 'ax': ax1,
         'column_label': 'daily doses administered', 'colors': None, 'remove_cumul': True, 'avg_7days': True},
        {'filename': VACCDOSESDELIV, 'column': 'sumTotal', 'plot_title': 'Total vaccines available in Switzerland',
         'plot_herd_immunities': False, 'agg_cantons': True, 'sarima_nb_days': None, 'ax': ax2,
         'column_label': 'sum of delivered doses', 'colors': None, 'remove_cumul': False, 'avg_7days': False},
        {'filename': VACCDOSESADMIN, 'column': 'sumTotal',
         'plot_title': 'Total vaccines doses available and administered in Switzerland', 'plot_herd_immunities': False,
         'agg_cantons': True, 'sarima_nb_days': None, 'ax': ax2, 'column_label': 'sum of administered doses',
         'colors': ['orange'], 'remove_cumul': False, 'avg_7days': False},
        {'filename': FULLYVACC, 'column': 'sumTotal', 'plot_title': 'New fully vaccinated people in Switzerland',
         'plot_herd_immunities': False, 'agg_cantons': True, 'sarima_nb_days': 30, 'ax': ax3,
         'column_label': 'daily fully vacc', 'colors': None, 'remove_cumul': True, 'avg_7days': True},
        {'filename': FULLYVACC, 'column': 'sumTotal', 'plot_title': 'Total fully vaccinated people in Switzerland',
         'plot_herd_immunities': False, 'agg_cantons': True, 'sarima_nb_days': 30, 'ax': ax4,
         'column_label': 'sum of fully vaccinated', 'colors': None, 'remove_cumul': False, 'avg_7days': False}]

    for subplot in subplots:
        df = preprocess(subplot['filename'])
        column = subplot['column']

        if subplot['colors'] is None:
            colors = ['cadetblue', 'coral', 'limegreen', 'lightpink', 'orange']
        else:
            colors = subplot['colors']
        curr_color = 0

        if subplot['plot_herd_immunities']:
            ch_pop_total = df.loc['CH', :]['pop'].values[0]

        if subplot['agg_cantons']:
            df = aggregate_cantons(df, column)
        else:
            df = df.loc['CH', :]

        df = df[df[column] > 0]

        if subplot['remove_cumul']:
            df = remove_cumul(df, column)

        plot_col(df[column], subplot['column_label'], colors[curr_color], subplot['ax'])
        curr_color += 1

        if subplot['avg_7days']:
            df = avg_7days(df, column)
            plot_col(df['avg_7days'], '7d average', colors[curr_color], subplot['ax'])
            curr_color += 1

        if subplot['sarima_nb_days'] is not None:
            df = sarima_forecast(df, column, subplot['sarima_nb_days'])
            plot_col(df['SARIMA_forecast'], 'SARIMA forecast', colors[curr_color])
            curr_color += 1

        subplot['ax'].set_title(subplot['plot_title'])

        subplot['ax'].yaxis.get_major_formatter().set_scientific(False)
        if subplot['plot_herd_immunities']:
            subplot['ax'].axhline(y=ch_pop_total * 0.8, color=colors[curr_color], label='polio herd immunity')
            curr_color += 1
            subplot['ax'].axhline(y=ch_pop_total * 0.95, color=colors[curr_color], label='measles herd immunity')
            curr_color += 1
        subplot['ax'].legend()

    plt.show()
