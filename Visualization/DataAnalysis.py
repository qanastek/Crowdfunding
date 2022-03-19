from datetime import datetime
import os
import requests
import seaborn as sns
from typing import List
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from currency_converter import CurrencyConverter
from sklearn.feature_extraction.text import strip_accents_unicode

class DataAnalysis:
    
    DATA_PROJECTS_FILE_CSV = "data/projects.csv"
    DATA_PROJECTS_FILE_H5 = "data/projects.h5"
    DATA_PROJECTS_URL = 'https://filesender.renater.fr/?s=download&token=2f5ed948-6e35-4bf1-88ed-7ab5dd0411b9'

    def __init__(self, path):

        sns.set_theme(style="whitegrid")
        sns.color_palette("crest", as_cmap=True)

        # DataFrame
        self.data = None
        
        # Define regex parse for date
        self.dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        # Currency converter
        self.cc = CurrencyConverter()

        # Load the corpora
        self.__load(path)

        # Create the output directory
        os.makedirs("./plots", exist_ok=True)

    def __load(self, path, cache=False):
        # Reload previously processed dataset, if available
        if (os.path.isfile(path)):
            print('> Load compressed HD5 file...')
            self.data = pd.read_hdf(path, 'data', mode='r')
            return

        # Download CSV file, if necessary
        if not os.path.isfile(DataAnalysis.DATA_PROJECTS_FILE_CSV):
            self.__download_data(DataAnalysis.DATA_PROJECTS_FILE_CSV)

        # Read original CSV file
        print("> Loading from raw CSV file...")
        self.data = pd.read_csv(DataAnalysis.DATA_PROJECTS_FILE_CSV, encoding='Windows-1252', na_values=[None], parse_dates=['start_date','end_date'], date_parser=self.dateparse)

        # Preprocess original dataset
        print("> Preprocessing...")
        self.__preprocess()
        
        # Save the DataFrame in HDF5 for faster reload.
        print('> Save compressed and processed DataFrame...')
        self.data.to_hdf(path, key='data', complevel=9)

    def __download_data(self, path:str):
        print("> Downloading raw CSV file...")
        r = requests.get(DataAnalysis.DATA_PROJECTS_URL, allow_redirects=True)
        open(path, 'wb').write(r.content)

    def get_country(self, currency:str):
        if currency == 'USD':
            return 'US'
        elif currency == 'CAD':
            return 'CA'
        elif currency == 'GBP':
            return 'GB'
        elif currency == 'AUD':
            return 'AU'
        elif currency == 'DKK':
            return 'DK'
        elif currency == 'SEK':
            return 'SE'
        elif currency == 'NOK':
            return 'NO'
        elif currency == 'NZD':
            return 'NZ'
        elif currency == 'CHF':
            return 'CH'
        else:
            return None

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
        print('> Convert textual columns to \'str\'...')
        labels = [
            'id',
        ]
        for label in labels:
            self.data[label] = self.data.loc[self.data[label].notna(), label].astype(str)

        print('> Normalizing string columns...')
        for label, content in self.data.select_dtypes(object).iteritems():
            self.data[label] = self.data[label].apply(lambda x: strip_accents_unicode(str(x).upper().strip()) if not str(x) in (None, 'nan') else None)
            self.data.loc[(self.data[label] == ''), label] = None
            self.data[label].replace('\s+', ' ', regex=True, inplace=True)

    def __build_numerical_boxplots(self):
        vars = [
            'age',
            'goal',
            'usd_goal',
            'pledged',
            'usd_pledged',
            'backers',
            'elapsed_days',
        ]
        for var in vars:
            plt.subplots()
            # sns.boxplot(data=self.data, x=var, palette="tab10")
            sns.boxenplot(data=self.data, x=var, palette="tab10")
            plt.savefig('plots/num_boxenplot_{}.png'.format(var), bbox_inches='tight')
            plt.close()

    def __build_numerical_pairplot(self):
        vars = [
            'age',
            'log_goal',
            'log_pledged',
            'log_backers',
            'elapsed_days',
        ]

        g = sns.PairGrid(self.data, vars=vars, hue="category", palette="viridis", dropna=True)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.scatterplot, size=self.data["state"])
        g.add_legend(title="", adjust_subtitles=True)
        g.savefig('plots/num_pairgrid.png')
        return

    def build_plots_numerical(self):
        corr=self.data.corr()
        plt.subplots(figsize=(16,9))
        heatmap=sns.heatmap(corr,
                            xticklabels=corr.columns,
                            yticklabels=corr.columns)
        plt.tight_layout()
        plt.savefig('plots/correlation-matrix.png', bbox_inches='tight')

        self.__build_numerical_boxplots()
        self.__build_numerical_pairplot()
        return



    def build_plots_categorial(self):

        # Before removing useless classes
        vars = [
            'category',
            'country',
            'sex',
            'currency',
            'state',
        ]
        for var in vars:
            plt.subplots()
            sns.countplot(data=self.data, y=var, palette="crest")
            plt.savefig('plots/before_cat_catplot_{}.png'.format(var), bbox_inches='tight')
            plt.close()

        # Remove useless states and merge others
        self.data = self.data.drop(self.data[self.data.state.str.upper().isin(["LIVE", "SUSPENDED", "UNDEFINED"])].index)
        self.data.state = self.data.state.replace("CANCELED", "FAILED")
        print("> Remove useless states and merge others - DONE!")

        # After removing useless classes
        vars = [
            'category',
            'country',
            'sex',
            'currency',
            'state',
        ]
        for var in vars:
            plt.subplots()
            sns.countplot(data=self.data, y=var, palette="crest")
            plt.savefig('plots/after_cleaning_cat_catplot_{}.png'.format(var), bbox_inches='tight')
            plt.close()
        
        # Downsampling the data
        min_occurences = min(self.data.groupby(['state']).size().reset_index(drop=True))
        print(f"min_occurences = {min_occurences}")
        self.data = pd.concat([
            self.data[self.data.state.isin([col])].sample(min_occurences) for col in ["FAILED", "SUCCESSFUL"]
        ])
        
        # After downsampling
        vars = [
            'category',
            'country',
            'sex',
            'currency',
            'state',
        ]
        for var in vars:
            plt.subplots()
            sns.countplot(data=self.data, y=var, palette="crest")
            plt.savefig('plots/after_downsampling_cat_catplot_{}.png'.format(var), bbox_inches='tight')
            plt.close()





        num_vars = [
            'age',
            'goal',
            'usd_goal',
            'usd_pledged',
            'elapsed_days',
        ]
        num_vars_log = [
            'age',
            'log_goal',
            'log_goal',
            'log_pledged',
            'elapsed_days',
        ]

        self.data.country.fillna(self.data.currency.apply(lambda x: self.get_country(x)), inplace=True)
        self.data.country.fillna('UNK', inplace=True)

        sns.catplot(y='country',
            hue=None, col="currency",
            data=self.data, kind="count",
            height=4, aspect=.7, col_wrap=4, sharex=False)
        plt.savefig('plots/grid_countplot_{}-{}.png'.format('country', 'currency'), bbox_inches='tight')
        plt.close()
        
        sns.catplot(y='currency',
            hue=None, col="country",
            data=self.data, kind="count",
            height=4, aspect=.7, col_wrap=4, sharex=False)
        plt.savefig('plots/grid_countplot_{}-{}.png'.format('currency', 'country'), bbox_inches='tight')
        plt.close()

        return