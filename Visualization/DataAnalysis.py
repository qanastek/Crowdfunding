import os
import sys
import matplotlib
from matplotlib.axis import Axis
import requests
import seaborn as sns
from seaborn import FacetGrid
from typing import List
from datetime import datetime
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

    def __init__(self, path, output_dir="./plots/"):

        self.output_dir = output_dir

        sns.set_theme(style="whitegrid")
        sns.color_palette("crest", as_cmap=True)

        # DataFrame
        self.data = None
        
        # Define regex parse for date
        self.dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

        # Currency converter
        self.cc = CurrencyConverter()

        # Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Redirect STDOUT
        sys.stdout = open(self.output_dir + "log.txt", 'w')

        # Load the corpora
        self.__load(path)
        
        sys.stdout = sys.__stdout__

    def __load(self, path, cache=False):
        """
        Load, Transform and Save the data
        """
        
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
        """
        Download the dataset from a remote source
        """
        print("> Downloading raw CSV file...")
        r = requests.get(DataAnalysis.DATA_PROJECTS_URL, allow_redirects=True)
        open(path, 'wb').write(r.content)


    def print_statistics(self):
        """
        Display basic statistics
        """

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_colwidth', None)
        pd.set_option('display.float_format', '{:.2f}'.format)

        print(self.data.info())
        print()
        print(self.data.describe().transpose())
        print()
        print(self.data.select_dtypes(object).describe().transpose())
        if ('duration_days' in self.data.columns):
            print()
            print(self.data.groupby(by=['country'])['usd_goal'].sum().sort_values(ascending=False))
            print()
            print(self.data.groupby(by=['country'])['usd_goal'].describe().unstack(1).transpose())
            print()
            print(self.data[self.data.state.isin(['SUCCESSFUL'])].groupby(by=['country'])['id'].count())

        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')
        pd.reset_option('display.float_format')

    def clean_data(self):
        self.data = self.data.drop(self.data[self.data.state.str.upper().isin(["LIVE", "SUSPENDED", "UNDEFINED"])].index)
        self.data.state = self.data.state.replace("CANCELED", "FAILED")

        self.data['usd_goal'] = self.data.apply(lambda row: self.cc.convert(row.goal, row.currency, 'USD'), axis=1)

        self.data['log_goal'] = np.log10(self.data['usd_goal'].replace(0, 0.1))
        
        print(self.data.loc[self.data.country.isin([None]), 'country'].size)
        self.data.country.fillna(self.data.currency.apply(lambda c: c[:2] if c != 'EUR' else None), inplace=True)
        print(self.data.loc[self.data.country.isin([None]), 'country'].size)
        self.data.country.fillna('UNK', inplace=True)
        self.data.sex.fillna('UNK', inplace=True)
        # self.data.dropna(subset=['sex'], inplace=True) # Could try simply dropping missing values (<2% of instances)

        self.data = self.data.drop(self.data[(self.data.start_date == datetime.fromtimestamp(0)) | (self.data.end_date == datetime.fromtimestamp(0))].index)
        self.data['duration_days'] = self.data.apply(lambda row: (row.end_date - row.start_date).days, axis=1)
        return

    def sample_data(self):
        min_occurences = min(self.data.groupby(['state']).size().reset_index(drop=True))
        print(f"min_occurences = {min_occurences}")
        self.data = pd.concat([
            self.data[self.data.state.isin([col])].sample(min_occurences) for col in ["FAILED", "SUCCESSFUL"]
        ])
        return

    def __preprocess(self):
        """
        Normalize types
        """

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

    def build_plots_numerical(self, data_state:str, with_pair_grid=False):
        """
        Export correlation matrix
        """

        self.__build_numerical_correlation_matrix(data_state)
        self.__build_numerical_boxplots(data_state)
        if with_pair_grid:
            self.__build_numerical_pairplot(data_state)
        return

    def __build_numerical_correlation_matrix(self, data_state:str):
        # Create any missing directories in the output path
        final_output_dir = '{}/{}/numerical'.format(self.output_dir, data_state)
        if not os.path.isdir(final_output_dir):
            os.makedirs(final_output_dir)

        corr=self.data.corr()
        sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
        plt.tight_layout()
        plt.savefig('{}/{}_correlation-matrix.png'.format(final_output_dir, data_state), bbox_inches='tight')
        plt.close()

    def __build_numerical_boxplots(self, data_state:str):
        """
        Export boxplots
        """

        # Create any missing directories in the output path
        final_output_dir = '{}/{}/numerical'.format(self.output_dir, data_state)
        if not os.path.isdir(final_output_dir):
            os.makedirs(final_output_dir)

        vars = [
            'age'
        ]
        bins = [
            np.arange(10, 80, 2)
        ]

        if ('duration_days' in self.data.columns):
            vars.append('duration_days')
            bins.append(np.arange(0, 100, 1))

        if ('log_goal' in self.data.columns):
            vars.append('log_goal')
            bins.append(np.arange(-4, 10, 0.25))

        for var, b in zip(vars, bins):
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
            sns.boxplot(data=self.data, x=var, ax=ax_box)
            sns.histplot(data=self.data, x=var, ax=ax_hist, bins=b, edgecolor="black")
            ax_box.set(xlabel='')
            plt.savefig('{}/{}_jointplot_{}.png'.format(final_output_dir, data_state, var), bbox_inches='tight')
            plt.close()

        # var = 'age'

        # # creating a figure composed of two matplotlib.Axes objects (ax_box and ax_hist)
        # f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

        # bins = np.arange(10, 80, 2)
        # sns.boxplot(self.data[var], x=var, ax=ax_box)
        # sns.histplot(data=self.data, x=var, ax=ax_hist, bins=bins, edgecolor="black")
        # ax_box.set(xlabel='')
        # plt.savefig('{}/{}_jointplot_{}.png'.format(final_output_dir, data_state, var), bbox_inches='tight')
        # plt.close()

    def __build_numerical_pairplot(self, data_state:str):
        """
        Export pairplots
        """

        # Create any missing directories in the output path
        final_output_dir = '{}/{}/multivariate'.format(self.output_dir, data_state)
        if not os.path.isdir(final_output_dir):
            os.makedirs(final_output_dir)

        # vars = self.data.select_dtypes('number')
        vars = [
            'age',
        ]

        if ('duration_days' in self.data.columns):
            vars.append('duration_days')

        if ('log_goal' in self.data.columns):
            vars.append('log_goal')

        g:sns.PairGrid = sns.PairGrid(self.data, vars=vars, hue="state", palette="tab10", diag_sharey=False)
        g.map_diag(sns.kdeplot)
        # g.map_offdiag(sns.scatterplot, size=self.data["state"])
        g.map_offdiag(sns.kdeplot, alpha=0.6)

        # Prevents a bug when only a single variable is present...
        if (len(vars) > 1):
            g.add_legend(title="", adjust_subtitles=True)

        g.savefig('{}/{}_pairgrid.png'.format(final_output_dir, data_state))
        plt.close()
        return

    def build_plots_categorial(self, data_state:str):
        """
        Export categorical plots
        """

        self.__build_categorial_countplots(data_state)
        self.__build_categorial_catplots(data_state)

        print('start_date: ', self.data.loc[(self.data.start_date == datetime.fromtimestamp(0)), 'start_date'].size)
        print('  end_date: ', self.data.loc[:, 'start_date'].min())
        print('  end_date: ', self.data.loc[:, 'start_date'].max())
        print('  end_date: ', self.data.loc[(self.data.end_date == datetime.fromtimestamp(0)), 'end_date'].size)
        print('  end_date: ', self.data.loc[:, 'end_date'].min())
        print('  end_date: ', self.data.loc[:, 'end_date'].max())
        return

    def __build_categorial_countplots(self, data_state:str):
        """
        Export categorical plots
        """

        # Create any missing directories in the output path
        final_output_dir = '{}/{}/categorical'.format(self.output_dir, data_state)
        if not os.path.isdir(final_output_dir):
            os.makedirs(final_output_dir)

        # Before removing useless classes
        vars = [
            'category',
            'subcategory',
            'country',
            'sex',
            'currency',
            'state',
        ]
        for var in vars:
            plt.subplots()
            ax = sns.countplot(data=self.data, y=var, palette="crest", order=self.data[var].value_counts()[:20].index)
            for container in ax.containers:
                ax.bar_label(container)
            plt.savefig('{}/{}_countplot_{}.png'.format(final_output_dir, data_state, var), bbox_inches='tight')
            plt.close()
        return
        
    def __build_categorial_catplots(self, data_state:str):
        """
        Export pairplots
        """

        # Create any missing directories in the output path
        final_output_dir = '{}/{}/multivariate'.format(self.output_dir, data_state)
        if not os.path.isdir(final_output_dir):
            os.makedirs(final_output_dir)

        col_vars = [
            'state',
            'sex',
            'country',
        ]
        y_vars = [
            'name',
            'category',
            'subcategory',
            'country',
            'sex',
            'state',
        ]
        for col in col_vars:
            for y in y_vars:
                if (y == col):
                    continue
                
                print('[{}-{}]'.format(col, y))
                g:FacetGrid = sns.catplot(
                    y=y, hue=None, col=col, data=self.data,
                    kind="count",
                    order=self.data[y].value_counts(ascending=False, sort=True).iloc[:20].index,
                    # order=self.data[var].value_counts().index,
                    col_wrap=3)

                for ax in g.axes:
                    ax.bar_label(ax.containers[0])
                g.tight_layout()
                g.savefig('{}/{}_catplots-{}-{}.png'.format(final_output_dir, data_state, col, y))
                plt.close()

        x = 'age'
        for y in y_vars:
            ax = sns.boxplot(
                x=x, y=y, data=self.data,
                order=self.data[y].value_counts(ascending=False, sort=True).iloc[:20].index,
                orient="h")
            plt.tight_layout()
            plt.savefig('{}/{}_catplots-{}-{}.png'.format(final_output_dir, data_state, x, y))
            plt.close()

        x = 'duration_days'
        if (x in self.data.columns):
            for y in y_vars:
                ax = sns.boxplot(
                    x=x, y=y, data=self.data,
                    order=self.data[y].value_counts(ascending=False, sort=True).iloc[:20].index,
                    orient="h")
                plt.tight_layout()
                plt.savefig('{}/{}_catplots-{}-{}.png'.format(final_output_dir, data_state, x, y))
                plt.close()

        x = 'log_goal'
        if (x in self.data.columns):
            for y in y_vars:
                ax = sns.boxplot(
                    x=x, y=y, data=self.data,
                    order=self.data[y].value_counts(ascending=False, sort=True).iloc[:20].index,
                    orient="h")

                ax.set_xscale('linear')
                plt.tight_layout()
                plt.savefig('{}/{}_catplots-{}-{}.png'.format(final_output_dir, data_state, x, y))
                plt.close()
        return