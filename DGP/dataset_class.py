#######################################################################################
# Author: Jenna Fields, edits by Franny Dean
# Script: dataset_class.py
# Function: Dataset wrappers for IV generation
# Date: 02/06/2026
#######################################################################################
from seed_utils import set_seed

set_seed(42)

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from models.treatment_effect_estimators.simple_estimators import OLS_splits, TSLS_splits, diff_in_means, TSLS_df, OLS_df

#######################################################################################

class ParentDataset():
    def __init__(self, df, 
                 x_cols, 
                 z_cols,
                 train_indices, 
                 val_indices,
                 test_indices,
                 true_c_cols = None, 
                 u_cols = None):
        """
        Parent class for datasets used in the DGP framework.
        Args:
            df (pd.DataFrame): DataFrame containing the data.
            x_cols (list): List of all input feature names in df.
            z_cols (list): List of instrumental variable names in df.
            train_indices (list): Indices for the training set.
            val_indices (list): Indices for the validation set.
            test_indices (list): Indices for the test set.
            true_c_cols (list, optional): List of true observed confounder names in df.
            u_cols (list, optional): List of unobserved confounder names in df
        """
        self.n = len(df)
        self.x = df[x_cols].values
        if true_c_cols is not None:
            self.true_c = df[true_c_cols].values
        self.true_c_cols = true_c_cols
        self.z = df[z_cols].values
        self.z_cols = z_cols
        if u_cols is not None:
            self.u = df[u_cols].values
        self.u_cols = u_cols
        self.t = df[['t']].values
        self.y = df[['y']].values
        self.ite = df[['ite']].values
        self.x_cols = x_cols

        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        # Assign in child class
        self.df = None

        self.is_tensor = False

    def __len__(self):
        return self.n
    
    def to_tensor(self):
        """
        Convert the dataset attributes to PyTorch tensors.
        """
        if self.is_tensor:
            return
        self.is_tensor = True
        self.x = torch.from_numpy(self.x).float()
        if hasattr(self, 'true_c'):
            self.true_c = torch.from_numpy(self.true_c).float()
        if hasattr(self, 'u'):
            self.u = torch.from_numpy(self.u).float()
        self.z = torch.from_numpy(self.z).float()
        self.t = torch.from_numpy(self.t).float()
        self.y = torch.from_numpy(self.y).float()
        self.ite = torch.from_numpy(self.ite).float()
    
    def to_numpy(self):
        """
        Convert the dataset attributes to NumPy arrays.
        """
        if not self.is_tensor:
            return
        self.is_tensor = False
        self.x = self.x.detach().numpy()
        if hasattr(self, 'true_c'):
            self.true_c = self.true_c.detach().numpy()
        if hasattr(self, 'u'):
            self.u = self.u.detach().numpy()
        self.z = self.z.detach().numpy()
        self.t = self.t.detach().numpy()
        self.y = self.y.detach().numpy()
        self.ite = self.ite.detach().numpy()

    def get_split(self, split_type='train', data_type='np'):
        """
        Get a specific data split (train, validation, or test).
        
        Args:
            split_type (str): Type of split ('train', 'val', or 'test'). Defaults to 'train'.
            data_type (str): Return type ('np' for numpy or 'torch' for tensors). Defaults to 'np'.
            
        Returns:
            SplitDataset: Dataset object for the specified split.
        """
        if split_type == 'train':
            use_indices = self.train_indices
        elif split_type == 'val':
            use_indices = self.val_indices
        elif split_type == 'test':
            use_indices = self.test_indices
        else:
            raise ValueError("split_type must be either 'train', 'val', or 'test'")
        
        if data_type == 'np':
            return SplitDataset(self.df.iloc[use_indices], 
                                self.x_cols, 
                                self.z_cols, 
                                self.true_c_cols, 
                                self.u_cols)
        elif data_type == 'torch':
            dataset =  SplitDataset(self.df.iloc[use_indices], 
                                self.x_cols,
                                self.z_cols,  
                                self.true_c_cols, 
                                self.u_cols)
            dataset.to_tensor()
            return dataset
        else:
            raise ValueError("data_type must be either 'np' or 'torch'")
        
    def train(self, data_type = 'np'):
        """
        Get the training split.
        
        Args:
            data_type (str): Return type ('np' or 'torch'). Defaults to 'np'.
            
        Returns:
            SplitDataset: Training dataset.
        """
        return self.get_split('train', data_type)

    def val(self, data_type = 'np'):
        """
        Get the validation split.
        
        Args:
            data_type (str): Return type ('np' or 'torch'). Defaults to 'np'.
            
        Returns:
            SplitDataset: Validation dataset.
        """
        return self.get_split('val', data_type)
    
    def test(self, data_type = 'np'):
        """
        Get the test split.
        
        Args:
            data_type (str): Return type ('np' or 'torch'). Defaults to 'np'.
            
        Returns:
            SplitDataset: Test dataset.
        """
        return self.get_split('test', data_type)
    
    def correlation_matrix(self):
        """
        Compute the correlation matrix for all variables in the dataset.
        
        Includes input features (excluding instruments), instruments, unobserved
        confounders (if present), treatment, and outcome.
        
        Returns:
            pd.DataFrame: Correlation matrix as a pandas DataFrame.
        """
        x_labels = [i for i in self.x_cols if i not in self.z_cols]
        if hasattr(self, 'u'):
            correlation_matrix = pd.DataFrame(np.concatenate([self.df[x_labels].values, self.z, self.u, self.t, self.y], 1), 
                                                                    columns=x_labels + 
                                                                            [f'z{i + 1}' for i in range(len(self.z_cols))] + 
                                                                            self.u_cols +
                                                                            ['t', 'y']).corr()
        else:
            correlation_matrix = pd.DataFrame(np.concatenate([self.df[x_labels].values, self.z, self.t, self.y], 1), 
                                                                    columns=x_labels + 
                                                                            [f'z{i + 1}' for i in range(len(self.z_cols))] + 
                                                                            ['t', 'y']).corr()
        return correlation_matrix
    
    def plot_correlation_matrix(self):
        """
        Plot a heatmap of the correlation matrix.
        
        Creates a seaborn heatmap visualization of correlations between all
        variables in the dataset.
        
        Returns:
            None: Displays the plot.
        """
        correlation_matrix = self.correlation_matrix()
        plt.figure(figsize=(12, 12))
        sns.heatmap(correlation_matrix, annot=True, cmap='twilight_shifted', fmt='.2f', square=True, linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        plt.show()
    
    def run_evaluations(self, plot_corr=True, verbose=True):
        """
        Run standard evaluations on the dataset.
        
        Performs various analyses including correlation visualization, treatment
        percentage, CATE estimation, difference-in-means, TSLS, and OLS.
        
        Args:
            plot_corr (bool): Whether to plot correlation matrix. Defaults to True.
            verbose (bool): Print detailed evaluation results. Defaults to True.
            
        Returns:
            tuple: (mean_diff, tsls_result, ols_result) - Estimates from different methods.
        """
        if plot_corr:
            self.plot_correlation_matrix()
        treated_percentage = np.mean(self.t)
        if verbose:
            print(f"Treated Percentage: {treated_percentage:.2f}")
            print()
        
        true_cate = self.ite.mean()
        if verbose:
            print(f"True CATE: {true_cate:.2f}")
            print()

        mean_diff = diff_in_means(self.df, verbose=verbose)
        mean_diff_error = np.abs(mean_diff - true_cate)
        if verbose:
            print(f"Mean Difference Error: {mean_diff_error:.2f}")
            print()

        x_cols = [i for i in self.x_cols if i not in self.z_cols]
        z_cols = [i for i in self.z_cols if i in self.x_cols or i == 'group_instr']
        if verbose:
            print(f"X Columns: {x_cols}")
            print(f"Z Columns: {z_cols}")

        if verbose:
            print("Running TSLS on entire dataset...")
        tsls_result = TSLS_df(self.df, x_cols, z_cols, verbose=verbose)
        tsls_error = np.abs(tsls_result - true_cate)
        if verbose:
            print(f"TSLS Error: {tsls_error:.2f}")
            print()
        
        if verbose:
            print("Running TSLS by split...")
        train_cate, val_cate, test_cate = TSLS_splits(self.df[self.df['split'] == 'train'], self.df[self.df['split'] == 'val'], self.df[self.df['split'] == 'test'], x_cols, z_cols, verbose=verbose)
        if verbose:
            print()

        if verbose:
            print("Running OLS on entire dataset...")
        ols_result = OLS_df(self.df, x_cols, z_cols, verbose=verbose)
        ols_error = np.abs(ols_result - true_cate)
        if verbose:
            print(f"OLS Error: {ols_error:.2f}")
            print()
        
        if verbose:
            print("Running OLS by split...")
        train_cate, val_cate, test_cate = OLS_splits(self.df[self.df['split'] == 'train'], self.df[self.df['split'] == 'val'], self.df[self.df['split'] == 'test'], x_cols, z_cols, verbose=verbose)
        if verbose:
            print()

        return mean_diff, tsls_result, ols_result

        
    def generate_df(self):
        pass 

    def save_csv(self, filename):
        """
        Save the dataset to a CSV file.
        """
        if self.df is None:
            raise ValueError("DataFrame is not generated. Call generate_df() first.")
        self.df.to_csv(filename, index=False)

class SplitDataset():
    def __init__(self, df, 
                 x_cols, 
                 z_cols,
                 true_c_cols = None, 
                 u_cols = None):
        
        """
        Initialize a dataset for a specific split (train, val, or test).
        
        Args:
            df (pd.DataFrame): DataFrame containing the split data.
            x_cols (list): List of input feature column names.
            z_cols (list): List of instrumental variable column names.
            true_c_cols (list, optional): List of true confounder column names. Defaults to None.
            u_cols (list, optional): List of unobserved confounder column names. Defaults to None.
        """
        self.n = len(df)
        self.x = df[x_cols].values
        if true_c_cols is not None:
            self.true_c = df[true_c_cols].values
        self.z = df[z_cols].values
        if u_cols is not None:
            self.u = df[u_cols].values
        self.t = df[['t']].values
        self.y = df[['y']].values
        self.ite = df[['ite']].values

        self.is_tensor = False
    
    def to_tensor(self):
        """
        Convert the dataset attributes to PyTorch arrays.
        """
        if self.is_tensor:
            return
        self.is_tensor = True
        self.x = torch.from_numpy(self.x).float()
        if hasattr(self, 'true_c'):
            self.true_c = torch.from_numpy(self.true_c).float()
        if hasattr(self, 'u'):
            self.u = torch.from_numpy(self.u).float()
        self.z = torch.from_numpy(self.z).float()

        self.t = torch.from_numpy(self.t).float()
        self.y = torch.from_numpy(self.y).float()
        self.ite = torch.from_numpy(self.ite).float()
    
    def to_numpy(self):
        """
        Convert the dataset attributes to NumPy arrays.
        """
        if not self.is_tensor:
            return
        self.is_tensor = False
        self.x = self.x.detach().numpy()
        if hasattr(self, 'true_c'):
            self.true_c = self.true_c.detach().numpy()
        self.z = self.z.detach().numpy()
        if hasattr(self, 'u'):
            self.u = self.u.detach().numpy()

        self.t = self.t.detach().numpy()
        self.y = self.y.detach().numpy()
        self.ite = self.ite.detach().numpy()

class ECG_DGPDataset(ParentDataset): # For the ZNet generated output of ECG synthetic data
    def __init__(self, df, 
                 x_cols,
                 c_cols, 
                 z_cols, 
                 u_cols=None, 
                 treatment_effect=None, 
                 train_size=0.6, 
                 valid_size=0.2): 
        """
        Dataset class for ECG-based synthetic data with ZNet-generated outputs.
        
        Args:
            df (pd.DataFrame): DataFrame containing ECG outcome, treatment, and covariate data. (ECGs in separate folder.)
            x_cols (list): List of input feature (ECG) column names.
            c_cols (list): List of confounder column names.
            z_cols (list): List of instrumental variable column names.
            u_cols (list, optional): List of unobserved confounder column names. Defaults to None.
            treatment_effect (array-like, optional): Treatment effect values. Uses 'ite' from df if None.
            train_size (float): Proportion of data for training. Defaults to 0.6.
            valid_size (float): Proportion of data for validation. Defaults to 0.2.
        """
        self.full_data = df

        if treatment_effect is None:
            treatment_effect = df['ite'].values
        X = df[x_cols].values
        shuffled_indices = np.random.RandomState(seed=42).permutation(np.arange(X.shape[0]))

        train_index = int(len(df) * train_size)
        val_index = int(len(df) * (train_size + valid_size))
        train_indices = shuffled_indices[:train_index]
        val_indices = shuffled_indices[train_index:val_index]
        test_indices = shuffled_indices[val_index:]

        super().__init__(df, x_cols, z_cols, train_indices, val_indices, test_indices, c_cols, u_cols)

        self.n = len(df)

        self.x = df[x_cols].values
        self.true_c = df[c_cols].values
        self.z = df[z_cols].values

        self.u = df[u_cols].values
        self.t = df[['t']].values
        self.y = df[['y']].values

        self.t_cf = df[['t_cf']].values
        self.y_cf = df[['y_cf']].values

        self.ite = treatment_effect

        self.df = self.generate_df()

        self.is_tensor = False
        self.x_cols = x_cols
        self.c_cols = c_cols
        self.z_cols = z_cols
        self.u_cols = u_cols
    
    def to_tensor(self):
        if self.is_tensor:
            return
        self.is_tensor = True
        super().to_tensor()
        self.t_cf = torch.from_numpy(self.t_cf).float()
        self.y_cf = torch.from_numpy(self.y_cf).float()
        self.ite = torch.from_numpy(self.ite).float()
    
    def to_numpy(self):
        if not self.is_tensor:
            return
        self.is_tensor = False
        super().to_numpy()
        self.t_cf = self.t_cf.detach().numpy()
        self.y_cf = self.y_cf.detach().numpy()
        self.ite = self.ite.detach().numpy()

    
    def generate_df(self):
        """
        Generates a DataFrame from the dataset.
        """
        # Here we just add any columns that would be missing from X
        z_not_in_x_cols = [f'z{i[1:]}' if i.startswith('x') else i for i in self.z_cols if i not in self.x_cols]
        z_not_in_x = self.z[:, [self.z_cols.index(i) for i in z_not_in_x_cols]]
        data = np.concatenate([self.x, z_not_in_x, self.u, self.t, self.y, self.ite.reshape(-1, 1)], axis=1)
        columns = self.x_cols + z_not_in_x_cols + self.u_cols + ['t', 'y', 'ite']
        df = pd.DataFrame(data, columns=columns)
        df['split'] = np.where(df.index.isin(self.train_indices), 'train',
                               np.where(df.index.isin(self.val_indices), 'val', 'test'))
        return df
               
    

class GeneratedECGIVDataset(ParentDataset):
    def __init__(self, original_df, x, z, dataset_type, train_indices, val_indices, test_indices):
        """
        Args:
            original_df (dataframe): The original dataset pandas dataframe.
            x (np.ndarray): The features.
            z (np.ndarray): The instrumental variables.
            dataset_type (str): The type of dataset ('znet').
        """


        data = np.concatenate([x, z, original_df['t'].to_numpy().reshape(-1, 1), original_df['y'].to_numpy().reshape(-1, 1), original_df['ite'].to_numpy().reshape(-1, 1)], axis=1)
        columns = [f'{dataset_type}_x{i + 1}' for i in range(x.shape[1])] + \
                 [f'{dataset_type}_z{i + 1}' for i in range(z.shape[1])]+ \
                ['t', 'y', 'ite']
        df = pd.DataFrame(data, columns=columns)

        super().__init__(df,
                    [f'{dataset_type}_x{i + 1}' for i in range(x.shape[1])], 
                    [f'{dataset_type}_z{i + 1}' for i in range(z.shape[1])],
                    train_indices, 
                    val_indices,
                    test_indices)

        self.original_dataset = original_df

        self.df = self.generate_df()

    def generate_df(self):
        """
        Generates a DataFrame from the dataset.
        """
        data = np.concatenate([self.x, self.z, self.t, self.y, self.ite.reshape(-1, 1)], axis=1)
        columns = self.x_cols + \
                self.z_cols+ \
                ['t', 'y', 'ite']
        df = pd.DataFrame(data, columns=columns)
        df['split'] = np.where(df.index.isin(self.train_indices), 'train',
                               np.where(df.index.isin(self.val_indices), 'val', 'test'))
        return df
    
    def get_combined_dataset(self):
        orig_df = self.original_dataset.generate_df()
        znet_df = self.generate_df()
        combined_df = pd.concat([orig_df, znet_df], axis=1)
        combined_df = combined_df.loc[:,~combined_df.columns.duplicated()]
        return combined_df
    
    def evaluate_u_z(self, verbose=True, split='train'):
        if verbose:
            print(f"Covariance of U on Z for {split}:")
        covariances = []
        correlations = []
        assert split in ['train', 'val', 'test'], "split must be either 'train', 'val', or 'test'"
        indices = self.train_indices if split == 'train' else self.val_indices if split == 'val' else self.test_indices
        u_column_names = [c for c in self.original_dataset.columns if c.startswith("u")]
        for i in range(self.z.shape[1]):
            for col in u_column_names:
                z_column = self.z[indices, i]
                u_column = self.original_dataset.loc[indices, col].to_numpy()

                # Compute covariance and correlation
                covariance = np.cov(u_column, z_column)[0, 1]
                correlation = covariance / (np.std(u_column) * np.std(z_column))

                correlations.append(abs(correlation))
                covariances.append(abs(covariance))
            
        average_correlation = np.mean(correlations)
        average_covariance = np.mean(covariances)
        if verbose:
            print("Average |correlation| of U with each column of Z:")
            print(average_correlation)
            print("Average |covariance| of U with each column of Z:")
            print(average_covariance)

        return covariances, correlations

class ZNetECGDataset(GeneratedECGIVDataset):
    def __init__(self, original_df, x, z,  
                 train_indices, val_indices, test_indices):
        """
        Args:
            original_df (dataframe): The original dataset pandas dataframe.
            x (np.ndarray): The features.
            z (np.ndarray): The instrumental variables.
        """

        super().__init__(original_df, x, z, 'znet', train_indices, val_indices, test_indices)

class DGPDataset(ParentDataset):
    def __init__(self, df, 
                 x_cols, 
                 true_c_cols, 
                 true_z_cols, 
                 u_cols, 
                 treatment_effect=None, 
                 train_size=0.6, 
                 valid_size=0.2): 
        """
        Args:
            df (pd.DataFrame): Dataframe containing the data.
            x_cols (list): List of feature names which we should input into new dataset. Instrumental variable candidates (non-group) are included
            true_c_cols (list): List of feature names which are true confounders.
            true_z_cols (list): List of feature names which are true instrumental variables.
            u_cols (list): List of feature names which are unobserved confounders.
            treatment_effect (np.ndarray): Treatment effect values. If None, will use the 'ite' column from df.
            train_size (float): Proportion of data to use for training.
            valid_size (float): Proportion of data to use for validation.
        """

        if treatment_effect is None:
            treatment_effect = df['ite'].values
        X = df[x_cols].values
        shuffled_indices = np.random.RandomState(seed=42).permutation(np.arange(X.shape[0]))

        train_index = int(len(df) * train_size)
        val_index = int(len(df) * (train_size + valid_size))
        train_indices = shuffled_indices[:train_index]
        val_indices = shuffled_indices[train_index:val_index]
        test_indices = shuffled_indices[val_index:]

        super().__init__(df, x_cols, true_z_cols, train_indices, val_indices, test_indices, true_c_cols, u_cols)

        self.t_cf = df[['t_cf']].values
        self.y_cf = df[['y_cf']].values

        self.ite = treatment_effect

        self.df = self.generate_df()

        self.is_tensor = False
    
    def to_tensor(self):
        if self.is_tensor:
            return
        self.is_tensor = True
        super().to_tensor()
        self.t_cf = torch.from_numpy(self.t_cf).float()
        self.y_cf = torch.from_numpy(self.y_cf).float()
        self.ite = torch.from_numpy(self.ite).float()
    
    def to_numpy(self):
        if not self.is_tensor:
            return
        self.is_tensor = False
        super().to_numpy()
        self.t_cf = self.t_cf.detach().numpy()
        self.y_cf = self.y_cf.detach().numpy()
        self.ite = self.ite.detach().numpy()

    
    def generate_df(self):
        """
        Generates a DataFrame from the dataset.
        """
        # Here we just add any columns that would be missing from X
        z_not_in_x_cols = [f'z{i[1:]}' if i.startswith('x') else i for i in self.z_cols if i not in self.x_cols]
        z_not_in_x = self.z[:, [self.z_cols.index(i) for i in z_not_in_x_cols]]
        data = np.concatenate([self.x, z_not_in_x, self.u, self.t, self.y, self.ite.reshape(-1, 1)], axis=1)
        columns = self.x_cols + z_not_in_x_cols + self.u_cols + ['t', 'y', 'ite']
        df = pd.DataFrame(data, columns=columns)
        df['split'] = np.where(df.index.isin(self.train_indices), 'train',
                               np.where(df.index.isin(self.val_indices), 'val', 'test'))
        return df


class GeneratedIVDataset(ParentDataset):
    def __init__(self, dgp_dataset : DGPDataset, x, z, dataset_type):
        """
        Dataset wrapper for IV representations generated by methods like ZNet or AutoIV.
        
        Args:
            dgp_dataset (DGPDataset): The original DGP dataset.
            x (np.ndarray): Generated confounder representation of shape (n_samples, x_dim).
            z (np.ndarray): Generated instrumental variable representation of shape (n_samples, z_dim).
            dataset_type (str): Name of the generation method ('znet', 'autoiv', 'giv', etc.).
        """


        data = np.concatenate([x, z, dgp_dataset.t, dgp_dataset.y, dgp_dataset.ite.reshape(-1, 1)], axis=1)
        columns = [f'{dataset_type}_x{i + 1}' for i in range(x.shape[1])] + \
                 [f'{dataset_type}_z{i + 1}' for i in range(z.shape[1])]+ \
                ['t', 'y', 'ite']
        df = pd.DataFrame(data, columns=columns)

        super().__init__(df,
                    [f'{dataset_type}_x{i + 1}' for i in range(x.shape[1])], 
                    [f'{dataset_type}_z{i + 1}' for i in range(z.shape[1])],
                    dgp_dataset.train_indices, 
                    dgp_dataset.val_indices,
                    dgp_dataset.test_indices)

        self.original_dataset = dgp_dataset

        self.df = self.generate_df()

    def generate_df(self):
        """
        Generates a DataFrame from the dataset.
        """
        data = np.concatenate([self.x, self.z, self.t, self.y, self.ite.reshape(-1, 1)], axis=1)
        columns = self.x_cols + \
                self.z_cols+ \
                ['t', 'y', 'ite']
        df = pd.DataFrame(data, columns=columns)
        df['split'] = np.where(df.index.isin(self.train_indices), 'train',
                               np.where(df.index.isin(self.val_indices), 'val', 'test'))
        return df
    
    def get_combined_dataset(self):
        orig_df = self.original_dataset.generate_df()
        znet_df = self.generate_df()
        combined_df = pd.concat([orig_df, znet_df], axis=1)
        combined_df = combined_df.loc[:,~combined_df.columns.duplicated()]
        return combined_df
    
    def evaluate_u_z(self, verbose=True, split='train'):
        """
        Evaluate independence between unobserved confounders U and generated instruments Z.
        
        Computes covariances and correlations between all U columns and Z columns
        to verify the exogeneity assumption.
        
        Args:
            verbose (bool): Print evaluation results. Defaults to True.
            split (str): Data split to evaluate ('train', 'val', or 'test'). Defaults to 'train'.
            
        Returns:
            tuple: (covariances, correlations) - Lists of absolute covariances and correlations.
        """
        if verbose:
            print(f"Covariance of U on Z for {split}:")
        covariances = []
        correlations = []
        assert split in ['train', 'val', 'test'], "split must be either 'train', 'val', or 'test'"
        indices = self.train_indices if split == 'train' else self.val_indices if split == 'val' else self.test_indices
        for i in range(self.z.shape[1]):
            for j in range(self.original_dataset.u.shape[1]):
                z_column = self.z[indices, i]
                u_column = self.original_dataset.u[indices, j]

                # Compute covariance and correlation
                covariance = np.cov(u_column, z_column)[0, 1]
                correlation = covariance / (np.std(u_column) * np.std(z_column))

                correlations.append(abs(correlation))
                covariances.append(abs(covariance))
            
        average_correlation = np.mean(correlations)
        average_covariance = np.mean(covariances)
        if verbose:
            print("Average |correlation| of U with each column of Z:")
            print(average_correlation)
            print("Average |covariance| of U with each column of Z:")
            print(average_covariance)

        return covariances, correlations

class ZNetDataset(GeneratedIVDataset):
    def __init__(self, dgp_dataset : DGPDataset, x, z):
        """
        Args:
            dgp_dataset (DGPDataset): The original dataset.
            x (np.ndarray): The features.
            z (np.ndarray): The instrumental variables.
        """

        super().__init__(dgp_dataset, x, z, 'znet')

class AutoIVDataset(GeneratedIVDataset):
    def __init__(self, dgp_dataset : DGPDataset, x, z):
        """
        Args:
            dgp_dataset (DGPDataset): The original dataset.
            x (np.ndarray): The features.
            z (np.ndarray): The instrumental variables.
        """

        super().__init__(dgp_dataset, x, z, 'autoiv')

class GIVDataset(GeneratedIVDataset):
    def __init__(self, dgp_dataset : DGPDataset, x, z):
        """
        Args:
            dgp_dataset (DGPDataset): The original dataset.
            x (np.ndarray): The features.
            z (np.ndarray): The instrumental variables.
        """

        super().__init__(dgp_dataset, x, z, 'giv')

class VIVDataset(GeneratedIVDataset):
    def __init__(self, dgp_dataset : DGPDataset, x, z):
        """
        Args:
            dgp_dataset (DGPDataset): The original dataset.
            x (np.ndarray): The features.
            z (np.ndarray): The instrumental variables.
        """

        super().__init__(dgp_dataset, x, z, 'viv')

class TrueIVDataset(GeneratedIVDataset):
    def __init__(self, dgp_dataset : DGPDataset):
        """
        Args:
            dgp_dataset (DGPDataset): The original dataset.
            x (np.ndarray): The features.
            z (np.ndarray): The instrumental variables.
        """
        df = dgp_dataset.generate_df()

        x_cols = [c for c in dgp_dataset.x_cols if c not in dgp_dataset.z_cols]
        self.x = df[x_cols].values
        self.z = df[dgp_dataset.z_cols].values

        super().__init__(dgp_dataset, self.x, self.z, 'trueiv')
