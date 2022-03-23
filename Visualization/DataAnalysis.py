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

        print()
        state = 'SUCCESSFUL'
        successes = self.data[self.data.state.isin([state])]
        # self.data['success_rate'] = self.data[self.data.state.isin([state])].sum(axis=1) / df['population'] * 1000
        # self.data['success_rate'] = self.data.apply(lambda x: x)[self.data.state.isin([state])]
        # y = 'category'
        # all_by_y = self.data.groupby([y, 'state']).size().reset_index(name='count')
        # all_by_y['rate'] = all_by_y.apply(lambda row : row['count'] / len(self.data[self.data.category.isin([row.category])]) * 100, axis=1)
        # print(all_by_y.head(20))

        # if ('duration_days' in self.data.columns):
        #     print()
        #     print(self.data.groupby(by=['country'])['usd_goal'].sum().sort_values(ascending=False))
        #     print()
        #     print(self.data.groupby(by=['country'])['usd_goal'].describe().unstack(1).transpose())
        #     print()
        #     print(self.data[self.data.state.isin(['SUCCESSFUL'])].groupby(by=['country'])['id'].count())

        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_colwidth')
        pd.reset_option('display.float_format')

    def clean_data(self):
        self.data = self.data.drop(self.data[self.data.state.str.upper().isin(["LIVE", "SUSPENDED", "UNDEFINED"])].index)
        self.data.state = self.data.state.replace("CANCELED", "FAILED")


        print(self.data.loc[self.data.country.isin([None]), 'country'].size)
        self.data.country.fillna(self.data.currency.apply(lambda c: c[:2] if c != 'EUR' else None), inplace=True)
        print(self.data.loc[self.data.country.isin([None]), 'country'].size)
        self.data.country.fillna('UNK', inplace=True)
        self.data.sex.fillna('UNK', inplace=True)
        # self.data.dropna(subset=['sex'], inplace=True) # Could try simply dropping missing values (<2% of instances)

        self.data = self.data.drop(self.data[(self.data.start_date == datetime.fromtimestamp(0)) | (self.data.end_date == datetime.fromtimestamp(0))].index)
        self.data['duration_days'] = self.data.apply(lambda row: (row.end_date - row.start_date).days, axis=1)
        
        self.data['usd_goal'] = self.data.apply(lambda row: self.cc.convert(row.goal, row.currency, 'USD'), axis=1)
        self.data['log_goal'] = np.log10(self.data['usd_goal'].replace(0, 0.1))
        self.data['log_duration'] = np.log10(self.data['duration_days'].replace(0, 0.1))
        self.data['log_age'] = np.log10(self.data['age'].replace(0, 0.1))
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

        if (not 'log_goal' in self.data.columns):
            return

        vars = [
            'log_age',
            'log_goal',
            'log_duration',
        ]

        methods = [
            'pearson',
            'kendall',
            'spearman',
        ]
        for method in methods:
            corr = self.data[vars].corr(method=method)
            sns.heatmap(corr, vmin=-1, vmax=1, annot=True)
            plt.tight_layout()
            plt.savefig('{}/{}_correlation-matrix_{}.png'.format(final_output_dir, data_state, method), bbox_inches='tight')
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

        if ('log_goal' in self.data.columns):
            vars.extend([
                'log_goal',
                'log_duration',
                'duration_days',
            ])
            bins.extend([
                np.arange(-4, 10, 0.25), # log_goal
                # np.arange(0, 100, 1), # duration_days
                np.arange(0, 3, 0.1), # log_duration
            ])

        for var, b in zip(vars, bins):
            f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
            sns.boxplot(data=self.data, x=var, ax=ax_box)
            sns.histplot(data=self.data, x=var, ax=ax_hist, bins=b, edgecolor="black")
            ax_box.set(xlabel='')
            plt.savefig('{}/{}_jointplot_{}.png'.format(final_output_dir, data_state, var), bbox_inches='tight')
            plt.close()

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

        if ('log_goal' in self.data.columns):
            vars.extend([
                # 'log_age',
                'log_goal',
                'duration_days',
                # 'log_duration',
            ])

        g:sns.PairGrid = sns.PairGrid(
            self.data.sample(150000), vars=vars, 
            hue="state", hue_order=['FAILED', 'SUCCESSFUL'],
            palette="tab10", 
            diag_sharey=False)
        g.map_diag(sns.kdeplot)
        g.map_offdiag(sns.kdeplot, alpha=0.8)

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

        y_vars = [
            'name',
            'category',
            'subcategory',
            'country',
            'sex',
            'state',
        ]

        col = 'state'

        y='category'
        all_by_y = self.data.groupby([y, col]).size().reset_index(name='count')
        all_by_y['rate'] = all_by_y.apply(lambda row : row['count'] / len(self.data[self.data[y].isin([row[y]])]) * 100, axis=1)
        state = 'SUCCESSFUL'
        successes = all_by_y[all_by_y.state.isin([state])]
        ax = sns.barplot(
            data=successes, y=y, x='rate',
            palette="tab10", 
            order=successes.sort_values('rate', ascending=False)[y][:20])
        ax.set(title='state = {}'.format(state))
        for container in ax.containers:
            ax.bar_label(container)
        plt.tight_layout()
        plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        plt.close()
        state = 'FAILED'
        failures = all_by_y[all_by_y.state.isin([state])]
        ax = sns.barplot(
            data=failures, y=y, x='rate',
            palette="tab10", 
            order=failures.sort_values('rate', ascending=False)[y][:20])
        ax.set(title='state = {}'.format(state))
        for container in ax.containers:
            ax.bar_label(container)
        plt.tight_layout()
        plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        plt.close()
        y = 'subcategory'
        all_by_y = self.data.groupby([y, col]).size().reset_index(name='count')
        all_by_y['rate'] = all_by_y.apply(lambda row : row['count'] / len(self.data[self.data[y].isin([row[y]])]) * 100, axis=1)
        state = 'SUCCESSFUL'
        successes = all_by_y[all_by_y.state.isin([state])]
        ax = sns.barplot(
            data=successes, y=y, x='rate',
            palette="tab10", 
            order=successes.sort_values('rate', ascending=False)[y][:20])
        ax.set(title='state = {}'.format(state))
        for container in ax.containers:
            ax.bar_label(container)
        plt.tight_layout()
        plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        plt.close()
        state = 'FAILED'
        failures = all_by_y[all_by_y.state.isin([state])]
        ax = sns.barplot(
            data=failures, y=y, x='rate',
            palette="tab10", 
            order=failures.sort_values('rate', ascending=False)[y][:20])
        ax.set(title='state = {}'.format(state))
        for container in ax.containers:
            ax.bar_label(container)
        plt.tight_layout()
        plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        plt.close()
        y = 'country'
        all_by_y = self.data.groupby([y, col]).size().reset_index(name='count')
        all_by_y['rate'] = all_by_y.apply(lambda row : row['count'] / len(self.data[self.data[y].isin([row[y]])]) * 100, axis=1)
        state = 'SUCCESSFUL'
        successes = all_by_y[all_by_y.state.isin([state])]
        ax = sns.barplot(
            data=successes, y=y, x='rate',
            palette="tab10", 
            order=successes.sort_values('rate', ascending=False)[y][:20])
        ax.set(title='state = {}'.format(state))
        for container in ax.containers:
            ax.bar_label(container)
        plt.tight_layout()
        plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        plt.close()
        state = 'FAILED'
        failures = all_by_y[all_by_y.state.isin([state])]
        ax = sns.barplot(
            data=failures, y=y, x='rate',
            palette="tab10", 
            order=failures.sort_values('rate', ascending=False)[y][:20])
        ax.set(title='state = {}'.format(state))
        for container in ax.containers:
            ax.bar_label(container)
        plt.tight_layout()
        plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        plt.close()

        # state = 'FAILED'
        # failures = self.data[self.data.state.isin([state])]
        # y = 'category'
        # ax = sns.countplot(
        #     data=failures, y=y,
        #     palette="tab10", 
        #     order=failures[y].value_counts(ascending=False, sort=True)[:20].index)
        # ax.set(title='state = {}'.format(state))
        # for container in ax.containers:
        #     ax.bar_label(container)
        # plt.tight_layout()
        # plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        # plt.close()
        # y = 'subcategory'
        # ax = sns.countplot(
        #     data=failures, y=y,
        #     palette="tab10", 
        #     order=failures[y].value_counts(ascending=False, sort=True)[:20].index)
        # ax.set(title='state = {}'.format(state))
        # for container in ax.containers:
        #     ax.bar_label(container)
        # plt.tight_layout()
        # plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        # plt.close()
        # y = 'country'
        # ax = sns.countplot(
        #     data=failures, y=y,
        #     palette="tab10", 
        #     order=failures[y].value_counts(ascending=False, sort=True)[:20].index)
        # ax.set(title='state = {}'.format(state))
        # for container in ax.containers:
        #     ax.bar_label(container)
        # plt.tight_layout()
        # plt.savefig('{}/{}_countplot_{}-{}-{}.png'.format(final_output_dir, data_state, col, y, state), bbox_inches='tight')
        # plt.close()

        x = 'age'
        for y in y_vars:
            ax = sns.boxplot(
                x=x, y=y, data=self.data,
                order=self.data[y].value_counts(ascending=False, sort=True).iloc[:20].index,
                orient="h")
            plt.tight_layout()
            plt.savefig('{}/{}_boxplot-{}-{}.png'.format(final_output_dir, data_state, x, y))
            plt.close()

        x = 'duration_days'
        if (x in self.data.columns):
            for y in y_vars:
                ax = sns.boxplot(
                    x=x, y=y, data=self.data,
                    order=self.data[y].value_counts(ascending=False, sort=True).iloc[:20].index,
                    orient="h")
                plt.tight_layout()
                plt.savefig('{}/{}_boxplot-{}-{}.png'.format(final_output_dir, data_state, x, y))
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
                plt.savefig('{}/{}_boxplot-{}-{}.png'.format(final_output_dir, data_state, x, y))
                plt.close()
        return