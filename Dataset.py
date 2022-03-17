from typing import List
from os.path import exists
from datetime import datetime

import numpy as np
import pandas as pd

from currency_converter import CurrencyConverter

from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class Dataset:

    def __init__(
        self,
        path,
        shuffle = True,
        seed = 0,
        verbose = True,
        save_gzip_path = None,
        clean_gzip = False,
        train_ratio = 0.60,
        dev_ratio = 0.20,
        test_ratio = 0.20,
        normalizer = "StandardScaler",
        normalize_currency = True,
        num_strategy = None,
        cat_strategy = "NaN_Token"
    ):
        """
        Constructor for the dataset
        """

        # Shuffle
        self.shuffle = shuffle
        self.seed = seed

        # Normalizer
        self.normalizer = normalizer
        self.normalize_currency = normalize_currency
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

        # Verbose
        self.verbose = verbose

        # Ratios
        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

        # Train set
        self.x_train = None
        self.y_train = None
        
        # Dev set
        self.x_dev = None
        self.y_dev = None

        # Test set
        self.x_test = None
        self.y_test = None

        # Labels
        self.labels = None
        self.labels_path = "data/labels.npy"
        self.label_encoder = preprocessing.LabelEncoder()

        # GZIP Caching
        self.save_gzip_path = save_gzip_path
        self.clean_gzip = clean_gzip

        # Define regex parse for date
        self.dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        self.numeric_features = ["age", "goal", "elapsed_days"]
        self.categorical_features = ["category", "subcategory", "country", "sex"]

        # Currency converter
        self.cc = CurrencyConverter()

        # Load the corpora
        self.__load(path)
    
    def __transform(self, sub_df: pd.DataFrame, mode="train"):
        """
        Apply a transformation of the input and references data to be compatible with Scikit-Learn
        """

        # Get labels
        print(">>> " + mode.upper() + " <<<")
        if mode == "train":
            Y = self.label_encoder.fit_transform(sub_df.state.to_list())
            np.save(self.labels_path, self.label_encoder.classes_)
            self.labels = list(self.label_encoder.classes_)
            print("Labels : ", self.labels)
        else:
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

    def __load(self, path):
        """
        Load CSV files and apply pre-processing
        """

        # Search for compressed & preprocessed data files
        if self.save_gzip_path != None and exists(self.save_gzip_path + ".npz") and not self.clean_gzip:

            loaded = np.load(self.save_gzip_path + '.npz')
            self.x_train, self.y_train = loaded['x_train'], loaded['y_train']
            self.x_dev, self.y_dev   = loaded['x_dev'], loaded['y_dev']
            self.x_test, self.y_test   = loaded['x_test'], loaded['y_test']
            print("> Data loaded from cache - DONE!")

        else:

            # Read original CSV file
            df: pd.DataFrame = pd.read_csv(path, encoding='Windows-1252', na_values=[None], parse_dates=['start_date','end_date'], date_parser=self.dateparse)
            print("> DataFrame read - DONE!")

            # Remove useless states and merge others
            df = df.drop(df[df.state.str.upper().isin(["LIVE", "SUSPENDED", "UNDEFINED"])].index)
            df.state = df.state.replace("canceled", "failed")
            print("> Remove useless states and merge others - DONE!")

            # Downsampling the data
            min_occurences = min(df.groupby(['state']).size().reset_index(drop=True))
            df = pd.concat([
                df[df.state.isin([col])].sample(min_occurences) for col in ["failed", "successful"]
            ])
            print("> Downsampling the data based on the labels - DONE!")
            
            print("> Labels distribution :")
            for row in df.groupby(['state']).size():
                print(" >", row)

            # Get elapsed time in days
            df['elapsed_days'] = df.apply(lambda row: (row.end_date - row.start_date).days, axis=1)
            print("> Get elapsed time in days - DONE!")

            # Remove elements with more than 365 days
            df = df[df['elapsed_days'] <= 365]
            print("> Remove extrema for the elapsed time in days - DONE!")

            # Normalize Currency
            if self.normalize_currency:
                # Transform to USD
                df['goal'] = df.apply(lambda row: self.cc.convert(row.goal, row.currency, 'USD'), axis=1)
                df = df.drop(['currency'], axis=1)
                print("> Currencies normalized - DONE!")
            else:
                self.categorical_features.append('currency')

            # Fill missing numerical values
            if self.num_strategy != None:

                if self.num_strategy == "mean":
                    df[self.numeric_features] = df[self.numeric_features].fillna(df.mean().round(1))
                    print("> Replacing missing numerical by mean value - DONE!")

                elif self.num_strategy == "median":
                    df[self.numeric_features] = df[self.numeric_features].fillna(df.median().round(1))
                    print("> Replacing missing numerical by median value - DONE!")

                elif type(self.num_strategy) in [int,float]:
                    df[self.numeric_features] = df[self.numeric_features].fillna(value=float(self.num_strategy))
                    print("> Replacing missing numerical by a default value - DONE!")

            # Fill missing categorical values
            if self.cat_strategy != None:

                for c in self.categorical_features:

                    if self.cat_strategy == "most_frequent":
                        value = df[c].mode()[0]
                        print("> Replacing missing categorical by the most frequent value - DONE!")

                    elif type(self.cat_strategy) in [str]:
                        value = self.cat_strategy
                        print("> Replacing missing categorical by a default value - DONE!")

                    df[c] = df[c].fillna(value=value)

            # Transform to categorial
            for c in self.categorical_features:
                df[c] = df[c].astype('category')
            df = pd.get_dummies(df, columns=self.categorical_features, prefix=self.categorical_features)

            # Shuffle the data with reproducible results
            if self.shuffle:
                df = shuffle(df, random_state=self.seed)
                print("> Shuffle - DONE!")

            # Split into train and test
            length = len(df)

            # Index and split for train
            train_start_idx, train_end_idx = 0, int(length*self.train_ratio)
            df_train = df.iloc[train_start_idx:train_end_idx, :]

            # Index and split for dev
            dev_start_idx, dev_end_idx = train_end_idx, train_end_idx + int(length*self.dev_ratio)
            df_dev  = df.iloc[dev_start_idx:dev_end_idx, :]

            # Index and split for test
            test_start_idx, test_end_idx = dev_end_idx, dev_end_idx + int(length*self.test_ratio)
            df_test  = df.iloc[test_start_idx:test_end_idx, :]

            # Transform sub-dataframes
            self.x_train, self.y_train = self.__transform(df_train, mode="train")
            self.x_dev, self.y_dev     = self.__transform(df_dev, mode="dev")
            self.x_test, self.y_test   = self.__transform(df_test, mode="test")
            
            # Save sub-dataframes
            if self.save_gzip_path != None:

                np.savez_compressed(

                    self.save_gzip_path,

                    x_train=self.x_train,
                    y_train=self.y_train,

                    x_dev=self.x_dev,
                    y_dev=self.y_dev,

                    x_test=self.x_test,
                    y_test=self.y_test
                )
