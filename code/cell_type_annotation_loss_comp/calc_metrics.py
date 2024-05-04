import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

class metrics():

    def __init__(self):

        self.metrics = None

    def calc_metrics(self, path: str):
        """
        Simply reads the results and concatenates them with previously read results.

        Parameters:
        path (str): Path to the input .csv file of results.

        Returns:
        None
        """

        results = pd.read_csv(f'{path}.csv', index_col=0)
            
        if self.metrics is None:
            self.metrics = results
        else:
            temp = results
            self.metrics = pd.concat([self.metrics, temp], ignore_index=True)

    def read_results(self, path: str):
        """
        Reads a CSV file and updates the performance metrics dataframe.

        Parameters
        ----------
        path : str, optional
            The file path and name of the CSV file to read.

        Returns
        -------
        None

        Notes
        -----
        This method reads a CSV file containing performance metrics and updates the metrics dataframe.
        """
        if self.metrics is not None:
            metrics = pd.read_csv(f'{path}.csv', index_col=0)
            self.metrics = pd.concat([self.metrics, metrics], axis="rows")
        else:
            self.metrics = pd.read_csv(f'{path}.csv', index_col=0)

    def download_results(self,path: str):
        """
        Saves the performance metrics dataframe as a CSV file.

        Parameters
        ----------
        name : str, optional
            The file path and name for the CSV file (default is 'benchmarks/results/Benchmark_results' (file name will then be Benchmark_results.csv)).

        Returns
        -------
        None

        Notes
        -----
        This method exports the performance metrics dataframe to a CSV file.
        """
        self.metrics.to_csv(f'{path}.csv', index=True, header=True)
        self.metrics = None