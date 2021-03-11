import pickle
import datetime
from urllib.request import urlopen
import json

def download_data():
    new_available, version = new_data_available()

    if new_available:
        with urlopen('https://www.covid19.admin.ch/api/data/20210310-kc5zw1wj/downloads/sources-csv.zip') as zip_file:
            with open('sources_csv.zip', 'wb') as out_file:
                out_file.write(zip_file.read())
                pickle.dump(version, open('data_version.p', 'wb'))
                print('Data successfully downloaded from the FOPH')
    else:
        print('We already have the latest version of the data')


def new_data_available():
    with urlopen('https://www.covid19.admin.ch/api/data/context') as f:
        data = json.load(f)
    latest_version = data['dataVersion']

    try:
        my_version = pickle.load(open('data_version.p', 'rb'))
    except FileNotFoundError:
        return (True, latest_version)

    if my_version != latest_version:
        return (True, latest_version)
    else:
        return (False, latest_version)

