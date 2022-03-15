from datetime import datetime
import os
import requests
from typing import List

import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import strip_accents_unicode

class AnalysisDataset:
    
    DATA_PROJECTS_FILE_CSV = "data/projects.csv"
    DATA_PROJECTS_FILE_H5 = "data/projects.h5"

    def __init__(self, path):

        # DataFrame
        self.data = None
        
        # Define regex parse for date
        self.dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        # Train set
        self.x_train = None
        self.y_train = None

        # Test set
        self.x_test = None
        self.y_test = None

        # Labels
        self.labels = None

        # Load the corpora
        self.__load(path)
    
    def __load(self, path, cache=False):
        # Reload previously processed dataset, if available
        if (os.path.isfile(path)):
            print('> Load compressed HD5 file...')
            self.data = pd.read_hdf(path, 'data', mode='r')
            return

        # Download CSV file, if necessary
        if not os.path.isfile(AnalysisDataset.DATA_PROJECTS_FILE_CSV):
            self.__download_data(AnalysisDataset.DATA_PROJECTS_FILE_CSV)

        # Read original CSV file
        print("> Loading from raw CSV file...")
        self.data = pd.read_csv(AnalysisDataset.DATA_PROJECTS_FILE_CSV, encoding='Windows-1252', na_values=[None], parse_dates=['start_date','end_date'], date_parser=self.dateparse)

        # Preprocess original dataset
        print("> Preprocessing...")
        self.__preprocess()
        
        # Save the DataFrame in HDF5 for faster reload.
        print('> Save compressed and processed DataFrame...')
        self.data.to_hdf(path, key='data', complevel=9)

    def __download_data(self, path:str):
        print("> Downloading raw CSV file...")
        url = 'https://filesender.renater.fr/?s=download&token=2f5ed948-6e35-4bf1-88ed-7ab5dd0411b9'
        r = requests.get(url, allow_redirects=True)
        open(path, 'wb').write(r.content)

    def print_statistics(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.float_format', '{:.2f}'.format)

        print(self.data.info())
        print()
        print(self.data.describe().transpose())
        print()
        print(self.data.select_dtypes(object).describe().transpose())

        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')
        pd.reset_option('display.float_format')

    def __preprocess(self):
        print('> Calculating elapsed days...')
        self.data['elapsed_days'] = self.data.apply(lambda row: (row.end_date - row.start_date).days, axis=1)

        print('> Convert textual columns to \'str\'...')
        labels = [
            'id',
        ]
        for label in labels:
            self.data[label] = self.data.loc[self.data[label].notna(), label].astype(str)

        print()
        print('> Normalizing string columns...')
        for label, content in self.data.select_dtypes(object).iteritems():
            self.data[label] = self.data[label].apply(lambda x: strip_accents_unicode(str(x).upper().strip()) if not str(x) in (None, 'nan') else None)
            self.data.loc[(self.data[label] == ''), label] = None
            self.data[label].replace('\s+', ' ', regex=True, inplace=True)