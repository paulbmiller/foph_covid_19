import pickle
from urllib.request import urlopen
import json


def download_data():
    new_available, version, latest_url = new_data_available()

    if new_available:
        with urlopen(latest_url) as zip_file:
            with open('sources_csv.zip', 'wb') as out_file:
                out_file.write(zip_file.read())
                with open('data_version.p', 'wb') as version_file:
                    pickle.dump(version, version_file)
                print('Data successfully downloaded from the FOPH')
    else:
        print('We already have the latest version of the data')


def new_data_available():
    with urlopen('https://www.covid19.admin.ch/api/data/context') as f:
        data = json.load(f)
    latest_version = data['dataVersion']
    latest_url = data['sources']['zip']['csv']

    try:
        my_version = pickle.load(open('data_version.p', 'rb'))
    except FileNotFoundError:
        return True, latest_version, latest_url

    if my_version != latest_version:
        return True, latest_version, latest_url
    else:
        return False, latest_version, latest_url


if __name__ == '__main__':
    download_data()
