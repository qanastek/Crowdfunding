from typing import List
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from os.path import exists
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_extraction.text import strip_accents_unicode
from sklearn.model_selection import train_test_split, GridSearchCV

class Dataset:

    def __init__(self, path, save_gzip_path=None, clean_gzip=False):

        # Ratios
        self.train_ratio = 0.90

        # Train set
        self.x_train = None
        self.y_train = None

        # Test set
        self.x_test = None
        self.y_test = None

        # Labels
        self.labels = None
        self.labels_path = "data/labels.npy"
        self.label_encoder = preprocessing.LabelEncoder()

        # Define regex parse for date
        self.dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        self.numeric_features = ["age", "goal"]
        self.categorical_features = ["category", "subcategory", "country", "sex", "currency"]

        # Load the corpora
        self.__load(path, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip)
    
    def __transform(self, sub_df: pd.DataFrame, mode="train"):

        # Get labels
        if mode == "train":
            print(">>> TRAIN <<<")
            Y = self.label_encoder.fit_transform(sub_df.state.to_list())
            np.save(self.labels_path, self.label_encoder.classes_)
            self.labels = list(self.label_encoder.classes_)
            print("Labels : ", self.labels)
        else:
            print(">>> TEST <<<")
            Y = self.label_encoder.transform(sub_df.state.to_list())
        print("> label_encoder.fit_transform - DONE!")

        # Get elapsed time in days
        sub_df['elapsed_days'] = sub_df.apply(lambda row: (row.end_date - row.start_date).days, axis=1)
        print("> Add elapsed_days - DONE!")

        # Drop useless columns
        sub_df = sub_df.drop(['id', 'name', 'pledged', 'backers', 'state', 'start_date', 'end_date'], axis=1)
        print("> Drop columns - DONE!")

        # Transform to categorial
        for c in self.categorical_features:
            sub_df[c] = sub_df[c].astype('category')
        sub_df =  pd.get_dummies(sub_df, columns=self.categorical_features, prefix=self.categorical_features)
        print("> To categorial - DONE!")

        # Transform to numpy array
        X = sub_df.to_numpy()
        print("> Converted to NumPy array - DONE!\n")

        return X, Y

    def __load(self, path, save_gzip_path=None, clean_gzip=False):

        # Search for compressed & preprocessed data files
        if save_gzip_path != None and exists(save_gzip_path+'.npz') and not clean_gzip :

            loaded = np.load(save_gzip_path+'.npz')
            self.x_train, self.y_train = loaded['x_train'], loaded['y_train']
            self.x_test, self.y_test   = loaded['x_test'], loaded['y_test']
            print("> Data loaded - DONE!")

        else : # Read original CSV file

            df: pd.DataFrame = pd.read_csv(path, encoding='Windows-1252', na_values=[None], parse_dates=['start_date','end_date'], date_parser=self.dateparse)
            print("> DataFrame read - DONE!")

            # Get train index
            train_idx = int(len(df)*self.train_ratio)

            # Split into train and test
            df_train = df.iloc[:train_idx, :]
            df_test  = df.iloc[train_idx:, :]

            # Transform sub-dataframes
            self.x_train, self.y_train = self.__transform(df_train, mode="train")
            self.x_test, self.y_test   = self.__transform(df_test, mode="test")

            # Save sub-dataframes
            if save_gzip_path != None:
                np.savez_compressed(save_gzip_path, x_train=self.x_train, y_train=self.y_train, x_test=self.x_test , y_test=self.x_test )