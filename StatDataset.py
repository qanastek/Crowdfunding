from typing import List

import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import strip_accents_unicode

class StatDataset:

    def __init__(self, path):

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
        print('> Load from raw dataset...')
        # df:DataFrame = pd.read_csv(path, encoding='utf-8', na_values=[None,])
        df:DataFrame = pd.read_csv(path, encoding='cp1252', na_values=[None])

        print()
        print(df.info())
        print()
        print(df.describe().transpose())
        print()
        print(df.select_dtypes(object).describe().transpose())

        self.__preprocess(df)

        print()
        print(df.info())
        print()
        print(df.describe().transpose())
        print()
        print(df.select_dtypes(object).describe().transpose())

    def __preprocess(self, df:DataFrame):
            print()
            print('> Normalizing string columns...')
            for label, content in df.select_dtypes(object).iteritems():
                df[label] = df[label].apply(lambda x: strip_accents_unicode(str(x).upper().strip()) if not str(x) in (None, 'nan') else None)
                df.loc[(df[label] == ''), label] = None
                df[label].replace('\s+', ' ', regex=True, inplace=True)