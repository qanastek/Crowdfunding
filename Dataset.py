from typing import List
from os.path import exists
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Dataset:

    def __init__(self, path, shuffle=True, seed=0, verbose=True, save_gzip_path=None, clean_gzip=False, train_ratio=0.99, normalizer="StandardScaler"):
        """
        Constructor for the dataset
        """

        # Shuffle
        self.shuffle = shuffle
        self.seed = seed

        # Normalizer
        self.normalizer = normalizer

        # Verbose
        self.verbose = verbose

        # Ratios
        self.train_ratio = train_ratio

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

        self.numeric_features = ["age", "goal", "elapsed_days"]
        self.categorical_features = ["category", "subcategory", "country", "sex", "currency"]

        # Load the corpora
        self.__load(path, save_gzip_path=save_gzip_path, clean_gzip=clean_gzip)
    
    def __transform(self, sub_df: pd.DataFrame, mode="train"):
        """
        Apply a transformation of the input and references data to be compatible with Scikit-Learn
        """

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

        # Drop useless columns
        sub_df = sub_df.drop(['id', 'name', 'pledged', 'backers', 'state', 'start_date', 'end_date'], axis=1)
        print("> Drop columns - DONE!")

        if self.normalizer == "MinMaxScaler":
            scaler = MinMaxScaler()
        elif self.normalizer == "StandardScaler":
            scaler = StandardScaler()

        if self.normalizer != None and scaler != None:
            sub_df[self.numeric_features] = scaler.fit_transform(sub_df[self.numeric_features])

        # Transform to numpy array
        X = sub_df.to_numpy()
        print("> Converted to NumPy array - DONE!\n")

        return X, Y

    def __load(self, path, save_gzip_path=None, clean_gzip=False):
        """
        Load CSV files and apply pre-processing
        """

        # Search for compressed & preprocessed data files
        if save_gzip_path != None and exists(save_gzip_path+'.npz') and not clean_gzip:

            loaded = np.load(save_gzip_path + '.npz')
            self.x_train, self.y_train = loaded['x_train'], loaded['y_train']
            self.x_test, self.y_test   = loaded['x_test'], loaded['y_test']
            print("> Data loaded - DONE!")

        else:

            # Read original CSV file
            df: pd.DataFrame = pd.read_csv(path, encoding='Windows-1252', na_values=[None], parse_dates=['start_date','end_date'], date_parser=self.dateparse)
            print("> DataFrame read - DONE!")

            # Get elapsed time in days
            df['elapsed_days'] = df.apply(lambda row: (row.end_date - row.start_date).days, axis=1)
            print("> Add elapsed_days - DONE!")

            # Transform to categorial
            for c in self.categorical_features:
                df[c] = df[c].astype('category')
            df = pd.get_dummies(df, columns=self.categorical_features, prefix=self.categorical_features)

            # Shuffle the data with reproducible results
            if self.shuffle:
                df = shuffle(df, random_state=self.seed)
                print("> Shuffle - DONE!")

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
                np.savez_compressed(save_gzip_path, x_train=self.x_train, y_train=self.y_train, x_test=self.x_test, y_test=self.y_test)
