from datetime import datetime
import os
import requests
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List

import pandas as pd
import numpy as np
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import strip_accents_unicode
from currency_converter import CurrencyConverter

class AnalysisDataset:
    
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

        # # Transform currencies into USD
        # self.data['usd_goal'] = self.data.apply(lambda row: self.cc.convert(row.goal, row.currency, 'USD'), axis=1)
        # self.data['usd_pledged'] = self.data.apply(lambda row: self.cc.convert(row.pledged, row.currency, 'USD'), axis=1)

        # # Remove outliers
        # self.data = self.data.drop(self.data[(self.data.start_date == datetime.fromtimestamp(0)) | (self.data.end_date == datetime.fromtimestamp(0))].index)

        # print('> Calculating elapsed days...')
        # self.data['elapsed_days'] = self.data.apply(lambda row: (row.end_date - row.start_date).days, axis=1)

        # # Apply log transformation for right/left skewed data 
        # # self.data['log_goal'] = np.log10(self.data['goal'].replace(0, 0.1))
        # self.data['log_goal'] = np.log10(self.data['usd_goal'].replace(0, 0.1))
        # # self.data['log_pledged'] = np.log10(self.data['pledged'].replace(0, 0.1))
        # self.data['log_pledged'] = np.log10(self.data['usd_pledged'].replace(0, 0.1))
        # self.data['log_backers'] = np.log10(self.data['backers'].replace(0, 0.1))
        # self.data['log_elapsed_days'] = np.log10(self.data['elapsed_days'].replace(0, 0.1))
        # # df = df.drop(columns = ['goal', 'pledged', 'backers', 'elapsed_days'])
        
        # # Remove useless states and merge others
        # self.data = self.data.drop(self.data[self.data.state.str.upper().isin(["LIVE", "SUSPENDED", "UNDEFINED"])].index)
        # self.data.state = self.data.state.replace("CANCELED", "FAILED")
    

    
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
        r = requests.get(AnalysisDataset.DATA_PROJECTS_URL, allow_redirects=True)
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
        # if row.country.isna():
        #     if row.currency == 'USD':
        #         return 'US'
        #     else:
        #         return None
        # else:
        #     return row.country


    def print_statistics(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.float_format', '{:.2f}'.format)

        print(self.data.info())
        print()
        print(self.data.describe().transpose())
        print()
        print(self.data.select_dtypes(object).describe().transpose())
        # print()
        # self.data = self.data.apply(lambda row : get_country(row), axis=1)
        # print(self.data[])

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
        # filtered = self.data.loc[self.data["state"].isin(["SUCCESSFUL", "FAILED"])]

        g = sns.PairGrid(self.data, vars=vars, hue="category", palette="viridis", dropna=True)
        # g = sns.pairplot(filtered, vars=vars, hue="category", palette="viridis", plot_kws={'alpha': 0.5, 's': 100, 'edgecolor': 'k'}, dropna=True)
        # g.map_diag(sns.kdeplot, hue=None, color="0.3")
        g.map_diag(sns.kdeplot)
        # g.map_offdiag(sns.scatterplot, size=self.data["sex"])
        g.map_offdiag(sns.scatterplot, size=self.data["state"])
        # g._legend.remove() # Remove built-in legend from pairplot
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
        vars = [
            'category',
            'country',
            'sex',
            'currency',
            'state',
        ]
        for var in vars:
            plt.subplots()
            # sns.catplot(ax=ax, data=self.data, y=var, kind="count", palette="crest")
            sns.countplot(data=self.data, y=var, palette="crest")
            plt.savefig('plots/cat_catplot_{}.png'.format(var), bbox_inches='tight')
            plt.close()
        
        # g = sns.FacetGrid(attend, col="subject", col_wrap=4, height=2, ylim=(0, 10))
        # g.map(sns.pointplot, "solutions", "score", order=[1, 2, 3], color=".3", ci=None)

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
        # sns.catplot(x=var,
        #     hue="state", col="category",
        #     data=self.data, kind="count",
        #     height=4, aspect=.7, col_wrap=4)
        # sns.catplot(y='country', x='age',
        #     hue="state", col="category",
        #     data=self.data.sample(1000),
        #     height=4, aspect=.7, col_wrap=4)
        # plt.savefig('plots/grid_scatter_cagtegory-{}.png'.format('test'), bbox_inches='tight')
        # plt.close()
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
        # for var in num_vars_log:
        #     # plt.subplots()
        #     # g = sns.FacetGrid(self.data, col='category', col_wrap=4, sharex=False,)
        #     # g.map(sns.boxplot, var)
        #     sns.catplot(x=var,
        #         hue="state", col="category",
        #         data=self.data, kind="count",
        #         height=4, aspect=.7, col_wrap=4)
        #     # g = sns.catplot(x="state", y=var,
        #     #     hue="sex", col="category",
        #     #     data=self.data, kind="swarm",
        #     #     height=4, aspect=.7,)
        #     plt.savefig('plots/grid_scatter_cagtegory-{}.png'.format(var), bbox_inches='tight')
        #     plt.close()

        # plt.subplots()
        # # sns.catplot(ax=ax, data=self.data, y=var, kind="count", palette="crest")
        # sns.countplot(data=self.data, y='category', hue="state", palette="crest")
        # plt.savefig('plots/cat_catplot_cagtegory-state.png'.format(var), bbox_inches='tight')
        # plt.close()
        return