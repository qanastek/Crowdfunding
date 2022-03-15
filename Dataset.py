from typing import List

import pandas as pd
from pandas import DataFrame

class Dataset:

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
