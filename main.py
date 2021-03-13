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


def visualize(fn, column, plot_title, days=30, plot_herd_immunities=False):
    df = preprocess(fn)
    CH_pop_total = df.loc['CH', :]['pop'].values[0]

    df_cantons = df.drop('CH', axis=0, level='geoRegion')
    df_cantons_sum = df_cantons[column].sum(axis=0, level='date')

    CH_vacc_df = df_cantons_sum.to_frame()

    CH_vacc_df = CH_vacc_df[CH_vacc_df[column] > 0]

    CH_vacc_df['percentTotal'] = CH_vacc_df[column] / CH_pop_total * 100

    # CH_vacc_df.plot(y='sumTotal', legend=True, title='Sum of fully vaccinated over total CH population')
    # CH_vacc_df.plot(y='percentTotal', legend=True, title='Percentage of fully vaccinated over total CH population')
    # CH_vacc_df.plot(y='logTotal', legend=True, title='Natural logarithm of the sum of fully vaccinated in CH')

    # result = seasonal_decompose(CH_vacc_df['sumTotal'], model='add')
    # result.plot()

    stepwise_fit = auto_arima(CH_vacc_df[column], start_p=0, start_q=0, max_p=5, max_q=5, m=7, trace=True)
    print(stepwise_fit.summary())

    model = SARIMAX(CH_vacc_df[column], order=stepwise_fit.order, seasonal_order=stepwise_fit.seasonal_order)
    results = model.fit(disp=False)
    # print(results.summary())

    # Predict 30 days
    predictions = results.predict(start=len(CH_vacc_df[column]), end=len(CH_vacc_df[column]) + days,
                                  typ='levels').rename('SARIMA forecast')

    CH_vacc_df = pd.concat([CH_vacc_df, predictions], axis=1)

    CH_vacc_df[[column, 'SARIMA forecast']].plot(
        title=plot_title)
    plt.gcf().axes[0].yaxis.get_major_formatter().set_scientific(False)
    if plot_herd_immunities:
        plt.axhline(y=CH_pop_total * 0.8, color='g', label='polio herd immunity')
        plt.axhline(y=CH_pop_total * 0.95, color='c', label='measles herd immunity')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    download_data()
    check_sum_cases()
    check_sum_fullyvacc()
    # visualize_fully_vacc()
    visualize(FULLYVACC, 'sumTotal', 'Forecasting of fully vaccinated people using a SARIMA model',
              plot_herd_immunities=True)
    visualize(VACCDOSESADMIN, 'sumTotal', 'Forecasting of administered vaccines using a SARIMA model')
