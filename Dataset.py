from typing import List
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV

class Dataset:

    def __init__(self, path):

        # Ratios
        self.train_ratio = 0.90

        # Train set
        self.x_train = None
        self.y_train = None

        # Test set
        self.x_test = None
        self.y_test = None

        # Labels
        self.labels = "data/labels.npy"
        self.label_encoder = preprocessing.LabelEncoder()

        # Define regex parse for date
        self.dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        self.numeric_features = ["age", "goal"]
        self.categorical_features = ["category", "subcategory", "country", "sex", "currency"]

        # Load the corpora
        self.__load(path)
    
    def __transform(self, sub_df: pd.DataFrame, mode="train"):

        # Get labels
        if mode == "train":
            print(">>> TRAIN <<<")
            Y = self.label_encoder.fit_transform(sub_df.state.to_list())
            np.save(self.labels, self.label_encoder.classes_)
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
        sub_df = pd.get_dummies(sub_df, columns=self.categorical_features, prefix=self.categorical_features).head()
        print("> To categorial - DONE!")

        # Transform to numpy array
        X = sub_df.to_numpy()
        print("> Converted to NumPy array - DONE!\n")

        return X, Y

    def __load(self, path, cache=False):

        # Read original CSV file
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
