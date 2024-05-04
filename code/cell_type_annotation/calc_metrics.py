import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score

class metrics():

    def __init__(self):

        self.metrics = None

    def calc_metrics(self, path: str, method_name: str, dataset_name: str):
        """
        Calculate evaluation metrics for a given method and dataset.
        This method calculates accuracy, balanced accuracy, and F1 score for each fold of the cross-testing
        results stored in the CSV file specified by `path`. It then stores these metrics along with the method name,
        dataset name, and fold index in a DataFrame.

        Parameters:
        path (str): Path to the CSV file containing results.
        method_name (str): Name of the method/model being evaluated.
        dataset_name (str): Name of the dataset.

        Returns:
        None
        """

        results = pd.read_csv(f'{path}.csv', index_col=0)

        unique_folds = np.unique(results.fold)

        for fold_idx in unique_folds:

            results_temp = results.loc[results['fold'] == fold_idx, :]

            # Extract the unique labels
            unique_labels1 = np.unique(results_temp.true_label)
            unique_labels2 = np.unique(results_temp.pred)
            unique_labels = np.unique(np.concatenate([unique_labels1,unique_labels2]))

            # Convert string labels to numerical labels
            label_encoder_temp = LabelEncoder()
            label_encoder_temp.fit(unique_labels)
            y_true = label_encoder_temp.transform(results_temp.true_label)
            y_pred = label_encoder_temp.transform(results_temp.pred)

            # Calculate accuracy
            print(f"Method {method_name} | Fold {fold_idx}")
            accuracy = accuracy_score(y_true, y_pred)
            print("Accuracy:", accuracy)

            # Calculate balanced accuracy
            balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
            print("Balanced Accuracy:", balanced_accuracy)

            # Calculate F1 score
            f1 = f1_score(y_true, y_pred, average='weighted')
            print("F1 Score:", f1)
            
            if self.metrics is None:
                self.metrics = pd.DataFrame({"method": method_name, 
                                            "accuracy": accuracy, 
                                            "balanced_accuracy": balanced_accuracy,
                                            "f1_score": f1,
                                            "dataset": dataset_name,
                                            "fold": fold_idx}, index=[0])
            else:
                temp = pd.DataFrame({"method": method_name, 
                                    "accuracy": accuracy, 
                                    "balanced_accuracy": balanced_accuracy,
                                    "f1_score": f1,
                                    "dataset": dataset_name,
                                    "fold": fold_idx}, index=[0])
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