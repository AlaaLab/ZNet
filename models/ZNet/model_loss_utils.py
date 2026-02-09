#######################################################################################
# Author: Jenna Fields, Franny Dean
# Script: model_loss_utils.py
# Function: Define loss functions for ZNet.
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed

set_seed(42)

from tqdm import tqdm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn.functional as F
from torch import nn
import torch
import pandas as pd
from sklearn.manifold import TSNE
import itertools
import statsmodels.api as sm  
import sys

#######################################################################################
# Losses

class PearsonLoss(nn.Module):
    def __init__(self, is_matrix=False):
        """
        Compute Pearson correlation loss between predictions and targets.
        
        Args:
            is_matrix (bool): If True, computes correlation matrix between all feature pairs.
                            If False, computes element-wise correlation. Defaults to False.
        """
        super(PearsonLoss, self).__init__()
        self.matrix_loss = is_matrix
    
    def pearson_correlation_matrix(self, x, y):
        """
        Compute the Pearson correlation matrix between two matrices x and y.
        
        x: Tensor of shape (m, n)
        y: Tensor of shape (m, l)
        
        Returns: 
            A tensor of shape (n, l) representing the Pearson correlation between each feature of x and each feature of y.
        """
        if x.shape[0] == 1:
            return torch.tensor(0.0, requires_grad=True)
        if y.shape[0] == 1: 
            return torch.tensor(0.0, requires_grad=True)
        # Step 1: Normalize x and y (subtract mean and divide by std)
        x_mean = x.mean(dim=0, keepdim=True)
        x_std = x.std(dim=0, keepdim=True)
        x_normalized = (x - x_mean) / (x_std + 1e-10)  # Avoid division by zero

        y_mean = y.mean(dim=0, keepdim=True)
        y_std = y.std(dim=0, keepdim=True)
        y_normalized = (y - y_mean) / (y_std + 1e-10)  # Avoid division by zero
        
        # Step 2: Compute Pearson correlation between each feature of x and each feature of y
        correlation_matrix = torch.mm(x_normalized.t(), y_normalized) / x.shape[0]
        correlation_matrix = torch.nan_to_num(correlation_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.mean(correlation_matrix)
    
    def pearson_correlation_loss(self, x, y):
        """
        Compute and return the average Pearson correlation between covariates x and outcome y.

        X is a matrix of shape (n, m) and Y is a matrix of shape (n, 1) where n is the number of samples.
        """

        if x.ndimension() == 1:
            x = x.unsqueeze(1)
        if y.ndimension() == 1:
            y = y.unsqueeze(1)
        if x.shape[0] == 1:
            return torch.tensor(0.0, requires_grad=True)
        if y.shape[0] == 1: 
            return torch.tensor(0.0, requires_grad=True)
        # Normalize each feature in x
        x = x - x.mean(dim=0, keepdim=True)
        
        # Normalize y
        y = y - y.mean(dim=0, keepdim=True)

        num = torch.sum(x * y, dim=0)  # Compute covariance
        denom = torch.sqrt(torch.sum(x ** 2, dim=0)) * torch.sqrt(torch.sum(y ** 2, dim=0))
        
        # Pearson correlation is the element-wise ratio of covariance and norms
        corr = num / (denom + 1e-10) # Avoid dividing by 0
        corr = torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.nanmean(corr) # Handle NaN values and take average

    def forward(self, predictions, targets):
        """
        Compute Pearson correlation loss.
        
        Args:
            predictions (torch.Tensor): Predicted values.
            targets (torch.Tensor): Target values.
            
        Returns:
            torch.Tensor: Pearson correlation (matrix or element-wise based on is_matrix).
        """
        if self.matrix_loss:
            return self.pearson_correlation_matrix(predictions, targets)
        else:
            return self.pearson_correlation_loss(predictions, targets)

class KDEMutualInformation(nn.Module):
    """
    KDE-based Mutual Information Loss.
    
    Estimates mutual information between two variables using kernel density estimation.
    Can compute MI for individual variable pairs or between all feature pairs in a matrix.
    """
    def __init__(self, sigma=0.5, is_matrix=False):
        """
        Initialize KDE mutual information estimator.
        
        Args:
            sigma (float): Bandwidth parameter for Gaussian kernel. Controls smoothness
                         of the KDE. Defaults to 0.5.
            is_matrix (bool): If True, computes MI matrix between all feature pairs.
                            If False, treats inputs as single variables. Defaults to False.
        """
        super(KDEMutualInformation, self).__init__()
        self.sigma = sigma # Bandwidth - controls the smoothness of the KDE by specifying the "width" of the Gaussian kernels ie how much should nearby points influence?
        self.matrix_loss = is_matrix
    
    def _kde_entropy(self, data):
        """
        Estimate entropy of data using Kernel Density Estimation
        
        Args:
            data: Tensor of shape (n_samples, n_features)
        Returns:
            Entropy estimate
        """
        n_samples, n_features = data.shape
        
        if n_samples < 2:
            return torch.tensor(0.0, requires_grad=True, device=data.device)

        # Pairwise distances between all samples
        dist_matrix = torch.cdist(data, data, p=2)
        
        # Gaussian kernels: K(xi, xj) = exp(-||xi - xj||² / (2σ²))
        K = torch.exp(-dist_matrix ** 2 / (2 * self.sigma ** 2))
        
        # Remove diagonal (self-distances) and normalize
        K = K - torch.eye(n_samples, device=data.device)
        
        # Density estimate at each point (exclude self)
        # p(xi) = (1/(n-1)) * Σ K(xi, xj) / (2πσ²)^(d/2)
        normalization = (2 * torch.pi * self.sigma ** 2) ** (n_features / 2)
        density = torch.sum(K, dim=1) / ((n_samples - 1) * normalization)
        
        # Handle edge cases
        density = torch.clamp(density, min=1e-10)
        
        # Entropy: H = -E[log p(x)]
        entropy = -torch.mean(torch.log(density))
        
        return entropy
    
    def kde_mutual_information_matrix(self, x, y):
        """
        Compute KDE-based MI between each feature of x and each feature of y
        
        Args:
            x: Tensor of shape (n_samples, n_features_x)
            y: Tensor of shape (n_samples, n_features_y)
        Returns:
            Average MI across all feature pairs
        """
        n_samples = x.shape[0]
        n_features_x = x.shape[1]
        n_features_y = y.shape[1]
        
        total_mi = 0.0
        count = 0
        
        for i in range(n_features_x):
            for j in range(n_features_y):
                # Extract individual features
                x_feat = x[:, i:i+1]  # Keep 2D
                y_feat = y[:, j:j+1]  # Keep 2D
                
                # Individual entropies
                H_x = self._kde_entropy(x_feat)
                H_y = self._kde_entropy(y_feat)
                
                # Joint entropy
                xy_joint = torch.cat([x_feat, y_feat], dim=1)
                H_xy = self._kde_entropy(xy_joint)
                
                # MI = H(X) + H(Y) - H(X,Y)
                mi = H_x + H_y - H_xy
                total_mi += mi
                count += 1
        
        return total_mi / count if count > 0 else torch.tensor(0.0, requires_grad=True, device=x.device)
    
    def kde_mutual_information_simple(self, x, y):
        """
        Compute KDE-based MI treating x and y as single variables
        
        Args:
            x: Tensor (will be flattened)
            y: Tensor (will be flattened)
        Returns:
            MI estimate
        """
        # Ensure 2D shape
        if x.ndim == 1:
            x = x.unsqueeze(1)
        if y.ndim == 1:
            y = y.unsqueeze(1)
            
        # # If multi-dimensional, flatten to 1D
        # if x.shape[1] > 1:
        #     x = x.view(-1, 1)
        # if y.shape[1] > 1:
        #     y = y.view(-1, 1)
        
        # Individual entropies
        H_x = self._kde_entropy(x)
        H_y = self._kde_entropy(y)
        
        # Joint entropy
        xy_joint = torch.cat([x, y], dim=1)
        H_xy = self._kde_entropy(xy_joint)
        
        # MI = H(X) + H(Y) - H(X,Y)
        mi = H_x + H_y - H_xy
        
        return mi
    
    def forward(self, predictions, targets):
        """
        Forward pass - computes mutual information estimate.
        
        Args:
            predictions (torch.Tensor): Predicted values/features.
            targets (torch.Tensor): Target values/features.
            
        Returns:
            torch.Tensor: MI estimate (higher = more mutual information).
        """
        if self.matrix_loss:
            return self.kde_mutual_information_matrix(predictions, targets)
        else:
            return self.kde_mutual_information_simple(predictions, targets)


        
class FeatureGenFormLoss(nn.Module):
    def __init__(self, kl_loss_coeff, feature_corr_loss_coeff):
        """
        Loss to enforce distributional and independence constraints on generated features.
        
        Combines KL divergence loss (to encourage standard normal distribution) and
        feature correlation loss (to encourage independence between feature dimensions).
        
        Args:
            kl_loss_coeff (float): Coefficient for KL divergence loss.
            feature_corr_loss_coeff (float): Coefficient for feature correlation loss.
        """
        super(FeatureGenFormLoss, self).__init__()
        self.kl_loss_coeff = kl_loss_coeff  
        self.feature_corr_loss_coeff = feature_corr_loss_coeff

    def forward(self, x):
        """
        Compute combined KL divergence and feature correlation losses.
        
        Args:
            x (torch.Tensor): Feature tensor of shape (n_samples, n_features).
            
        Returns:
            tuple: (kl_loss, pearson_cols) - KL divergence and correlation losses.
        """
        if x.shape[0] > 1:
            mean_x = torch.mean(x, dim=0)
            std_x = torch.std(x, dim=0)
            std_x = torch.clamp(std_x, min=1e-6)  # Prevent zero variance
            var_x = std_x ** 2

            log_var = torch.log(var_x)
            log_var = torch.clamp(log_var, min=-10, max=10)  # Prevent extreme values

            kl_loss = torch.mean((mean_x**2 + var_x - 1 - log_var) / 2)
            kl_loss = torch.nan_to_num(kl_loss, nan=0.0)
        else:
            kl_loss = torch.tensor(0.0, requires_grad=True, device=x.device)

        if x.shape[0] > 1 and x.shape[-1] > 1:
            corrcoef = torch.corrcoef(x.T)
            corrcoef = torch.nan_to_num(corrcoef, nan=0.0)  # Handle NaNs from corrcoef
            pearson_cols = torch.abs(corrcoef).triu(1)
            pearson_cols = pearson_cols[pearson_cols != 0]
            if pearson_cols.numel() > 0:
                pearson_cols = torch.mean(pearson_cols)
            else:
                pearson_cols = torch.tensor(0.0, requires_grad=True, device=x.device)
        else:
            pearson_cols = torch.tensor(0.0, requires_grad=True, device=x.device)

        return kl_loss, pearson_cols


class ZNetLoss(nn.Module):
    def __init__(self, 
                 use_mi_corr_loss=False, 
                 use_mi_matrix_loss=False, 
                 kl_loss_coeff=0, 
                 feature_corr_loss_coeff=0,
                 c_pearson_loss_alpha = 1, 
                 c_mse_loss_alpha = 1, 
                 z_pearson_loss_alpha = 1, 
                 z_t_loss_alpha = 0, 
                 pearson_matrix_alpha = 0, 
                 t_hat_alpha = 1, 
                 train_xt_net=False,
                 binary_outcome=False):
        
        """
        Initialize the ZNetLoss class - main loss function for ZNet.
        
        Combines multiple objectives: C-Y correlation, Z-residual correlation,
        C-Z independence, treatment prediction, and distributional constraints.
        
        Args:
            use_mi_corr_loss (bool): Use mutual information instead of Pearson for correlation. Defaults to False.
            use_mi_matrix_loss (bool): Use MI for matrix independence loss. Defaults to False.
            kl_loss_coeff (float): Coefficient for KL divergence loss. Defaults to 0.
            feature_corr_loss_coeff (float): Coefficient for feature correlation loss. Defaults to 0.
            c_pearson_loss_alpha (float): Weight for C-Y correlation. Defaults to 1.
            c_mse_loss_alpha (float): Weight for C-Y MSE loss. Defaults to 1.
            z_pearson_loss_alpha (float): Weight for Z-residual correlation. Defaults to 1.
            z_t_loss_alpha (float): Weight for Z-T correlation. Defaults to 0.
            pearson_matrix_alpha (float): Weight for C-Z independence. Defaults to 0.
            t_hat_alpha (float): Weight for treatment prediction loss. Defaults to 1.
            train_xt_net (bool): Whether to train X,T→Y network. Defaults to False.
            binary_outcome (bool): Whether outcome is binary. Defaults to False.
        """
        super(ZNetLoss, self).__init__()
        if binary_outcome:
            self.reg_loss = torch.nn.functional.binary_cross_entropy_with_logits
        else:
            self.reg_loss = torch.nn.functional.mse_loss 
        self.binary_func = lambda x: torch.sigmoid(x.clone()) if binary_outcome else x.clone()
        self.class_loss = torch.nn.functional.binary_cross_entropy_with_logits
        if use_mi_corr_loss:
            self.corr_loss = KDEMutualInformation()
        else:
            self.corr_loss = PearsonLoss()
        if use_mi_matrix_loss:
            self.matrix_loss = KDEMutualInformation(is_matrix=True)
        else:
            self.matrix_loss = PearsonLoss(is_matrix=True)

        self.c_pearson_loss_alpha = c_pearson_loss_alpha
        self.c_mse_loss_alpha = c_mse_loss_alpha
        self.z_pearson_loss_alpha = z_pearson_loss_alpha
        self.z_t_loss_alpha = z_t_loss_alpha
        self.pearson_matrix_alpha = pearson_matrix_alpha
        self.t_hat_alpha = t_hat_alpha
        self.train_xt_net = train_xt_net

        self.feature_loss = FeatureGenFormLoss(kl_loss_coeff, feature_corr_loss_coeff)
 
    def forward(self, c, z, t_hat, c_y, x_t_y, x, y, t) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all loss components for ZNet.
        
        Args:
            c (torch.Tensor): Confounder representation.
            z (torch.Tensor): Instrumental variable representation.
            t_hat (torch.Tensor): Predicted treatment from Z.
            c_y (torch.Tensor): Predicted outcome from C and t.
            x_t_y (torch.Tensor): Predicted outcome from X and t.
            x (torch.Tensor): Input features.
            y (torch.Tensor): True outcomes.
            t (torch.Tensor): True treatments.
            
        Returns:
            list: [pearson_c, pearson_z, pearson_matrix, class_t_hat, kl_c,
                  feature_loss_c, kl_z, feature_loss_z, mse_c, pearson_z_t, mse_xt]
        """

        if isinstance(self.corr_loss, PearsonLoss):
            pearson_c = 1 - self.corr_loss(c, y) ** 2 # We want to maximize the pearson correlation of C..
            pearson_z = self.corr_loss(z, y - self.binary_func(x_t_y)) ** 2  # And minimize it for Z and the errors on y
            pearson_z_t = 1 - self.corr_loss(z, t) ** 2 # We want to maximize the pearson correlation of z and t
        elif isinstance(self.corr_loss, KDEMutualInformation):
            pearson_c = -self.corr_loss(c, y) # Maximize so negate
            pearson_z = self.corr_loss(z, y - self.binary_func(x_t_y)) # Minimize
            pearson_z_t = - self.corr_loss(z, t) # Maximize so negate
        else:
            print(f"[DEBUG] Unexpected corr_loss type: {type(self.corr_loss)}")
            pearson_c = None
            pearson_z = None
            pearson_z_t = None

        if isinstance(self.matrix_loss, PearsonLoss):
            pearson_matrix = self.matrix_loss(c, z) ** 2
        elif isinstance(self.matrix_loss, KDEMutualInformation):
            pearson_matrix = self.matrix_loss(c, z)
        else:
            print(f"[DEBUG] Unexpected matrix_loss type: {type(self.matrix_loss)}")
            pearson_matrix = None

        class_t_hat = self.class_loss(t_hat, t)

        kl_c, feature_loss_c = self.feature_loss(c)
        kl_z, feature_loss_z = self.feature_loss(z)

        mse_c = self.reg_loss(c_y, y) # We want to maximize prediction of y from x hat
        if self.train_xt_net:
            mse_xt = self.reg_loss(x_t_y, y) # We want to maximize prediction of y from x and t

        # if any(np.isnan(v) for v in [pearson_c.item(), pearson_z.item(), pearson_matrix.item(), class_t_hat.item(), kl_c.item(), feature_loss_c.item(), kl_z.item(), feature_loss_z.item(), mse_c.item(), pearson_z_t.item()]):
        #     raise ValueError("NaN detected in loss components")
        return [pearson_c,
                pearson_z,
                pearson_matrix,
                class_t_hat,
                kl_c,
                feature_loss_c,
                kl_z,
                feature_loss_z,
                mse_c,
                pearson_z_t,
                mse_xt if self.train_xt_net else None]


#######################################################################################
# ZNet Layers
    
class ZNetModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 c_dim, 
                 z_dim, 
                 output_dim, 
                 hidden_dim_h=64, 
                 is_linear=False, 
                 sm_temp=2, 
                 use_softmax=True,
                 activation_function='relu',
                 pretrain_xty_model=True,
                 xty_model=None,
                 ):
        """
        ZNet neural network architecture.
        
        Defines the network structure for learning disentangled C and Z representations,
        treatment prediction, and outcome prediction.
        
        Args:
            input_dim (int): Dimensionality of input features.
            c_dim (int): Dimensionality of confounder representation C.
            z_dim (int): Dimensionality of IV representation Z.
            output_dim (int): Dimensionality of outcome.
            hidden_dim_h (int): Hidden layer dimension. Defaults to 64.
            is_linear (bool): Use linear networks (no activations). Defaults to False.
            sm_temp (float): Temperature for softmax. Defaults to 2.
            use_softmax (bool): Apply softmax to C and Z. Defaults to True.
            activation_function (str): Activation function ('relu', 'leaky_relu', 'sigmoid', 'tanh'). Defaults to 'relu'.
            pretrain_xty_model (bool): Whether X,T,Y model is pretrained. Defaults to True.
            xty_model (nn.Module, optional): Pretrained X,T,Y model. Defaults to None.
        """
        super(ZNetModel, self).__init__()

        if activation_function == 'relu':
            self.nonlinearity = nn.ReLU()
        elif activation_function == 'leaky_relu':
            self.nonlinearity = nn.LeakyReLU()
        elif activation_function == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        elif activation_function == 'tanh':
            self.nonlinearity = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")
        
        if use_softmax and c_dim > 1:
            self.c = nn.Sequential(nn.Linear(input_dim, hidden_dim_h), 
                            self.nonlinearity,
                        nn.Linear(hidden_dim_h, c_dim), 
                        SoftmaxWithTemperature(dim=-1, temperature=sm_temp)
                        )
        else:
            self.c = nn.Sequential(nn.Linear(input_dim, hidden_dim_h), 
                            self.nonlinearity,
                        nn.Linear(hidden_dim_h, c_dim),)
        if use_softmax and z_dim > 1:
            self.z = nn.Sequential(nn.Linear(input_dim, hidden_dim_h),
                                        self.nonlinearity,
                                    nn.Linear(hidden_dim_h, z_dim), 
                                    SoftmaxWithTemperature(dim=-1, temperature=sm_temp))
        else:
            self.z = nn.Sequential(nn.Linear(input_dim, hidden_dim_h),
                                        self.nonlinearity,
                                    nn.Linear(hidden_dim_h, z_dim),)
        
        self.pretrain_xty_model = pretrain_xty_model
        self.x_t_y = xty_model
        
        if is_linear:
            self.t_hat = nn.Sequential(nn.Linear(z_dim, hidden_dim_h), 
                                    nn.Linear(hidden_dim_h, output_dim))
            
            self.c_y = nn.Sequential(
                nn.Linear(c_dim + 1, hidden_dim_h), 
                nn.Linear(hidden_dim_h, output_dim))
            
            if not self.pretrain_xty_model:
                self.x_t_y = nn.Sequential(
                    nn.Linear(input_dim + 1, hidden_dim_h), 
                    nn.Linear(hidden_dim_h, output_dim))
        else:
            self.t_hat = nn.Sequential(nn.Linear(z_dim, hidden_dim_h), 
                                     self.nonlinearity,
                                    nn.Linear(hidden_dim_h, output_dim))
            
            self.c_y = nn.Sequential(
                nn.Linear(c_dim + 1, hidden_dim_h), 
                self.nonlinearity,
                nn.Linear(hidden_dim_h, output_dim))
            
            if not self.pretrain_xty_model:
                self.x_t_y = nn.Sequential(
                    nn.Linear(input_dim + 1, hidden_dim_h), 
                    self.nonlinearity,
                    nn.Linear(hidden_dim_h, output_dim))
        
        self.z_layers = nn.ModuleList([self.z, self.t_hat])
    
    def forward(self, x, t):
        """
        Forward pass through ZNet.
        
        Args:
            x (torch.Tensor): Input features of shape (n_samples, input_dim).
            t (torch.Tensor): Treatment indicators of shape (n_samples, 1).
            
        Returns:
            tuple: (c, z, t_hat, c_y, x_t_y) - All network outputs.
        """
        c = self.c(x) 
        z = self.z(x) 
        t_hat = self.t_hat(z) 

        c_y = self.c_y(torch.concatenate([c, t], dim=-1))
        x_t_y = self.x_t_y(torch.concatenate([x, t], dim=-1)) # We use this to get the error on y given x and t

        return c, z, t_hat, c_y, x_t_y

class ZNetECGModel(nn.Module):
    def __init__(self, 
                 input_dim, 
                 c_dim, 
                 z_dim, 
                 output_dim, 
                 embedded_dim=64,
                 ecg_channels=12,
                 hidden_dim_h=64, 
                 is_linear=False, 
                 sm_temp=2, 
                 use_softmax=True,
                 activation_function='relu',
                 pretrain_xty_model=True,
                 xty_model=None,
                 ):
         """
        ZNet neural network architecture for ECGs. The addition of a ResNet head allows for high dimensional data
        
        Defines the network structure for learning disentangled C and Z representations,
        treatment prediction, and outcome prediction.
        
        Args:
            input_dim (int): Dimensionality of input ECG features.
            c_dim (int): Dimensionality of confounder representation C.
            z_dim (int): Dimensionality of IV representation Z.
            output_dim (int): Dimensionality of outcome.
            embedded_dim (int): ECG embedded dimension after ResNet layers.
            ecg_channels (int): ECG leads (typically 12).
            hidden_dim_h (int): Hidden layer dimension. Defaults to 64.
            is_linear (bool): Use linear networks (no activations). Defaults to False.
            sm_temp (float): Temperature for softmax. Defaults to 2.
            use_softmax (bool): Apply softmax to C and Z. Defaults to True.
            activation_function (str): Activation function ('relu', 'leaky_relu', 'sigmoid', 'tanh'). Defaults to 'relu'.
            pretrain_xty_model (bool): Whether X,T,Y model is pretrained. Defaults to True.
            xty_model (nn.Module, optional): Pretrained X,T,Y model. Defaults to None.
        """
        super(ZNetECGModel, self).__init__()
        
        if activation_function == 'relu':
            self.nonlinearity = nn.ReLU()
        elif activation_function == 'leaky_relu':
            self.nonlinearity = nn.LeakyReLU()
        elif activation_function == 'sigmoid':
            self.nonlinearity = nn.Sigmoid()
        elif activation_function == 'tanh':
            self.nonlinearity = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation function: {activation_function}")
        
        self.resnet = ResNet1D(BasicBlock1D, [2,2, 2], num_classes=embedded_dim, in_channels=ecg_channels)
        
        if use_softmax and c_dim > 1:
            self.c = nn.Sequential(nn.Linear(embedded_dim, hidden_dim_h), 
                            self.nonlinearity,
                        nn.Linear(hidden_dim_h, c_dim), 
                        SoftmaxWithTemperature(dim=-1, temperature=sm_temp)
                        )
        else:
            self.c = nn.Sequential(nn.Linear(embedded_dim, hidden_dim_h), 
                            self.nonlinearity,
                        nn.Linear(hidden_dim_h, c_dim),)
        if use_softmax and z_dim > 1:
            self.z = nn.Sequential(nn.Linear(embedded_dim, hidden_dim_h),
                                        self.nonlinearity,
                                    nn.Linear(hidden_dim_h, z_dim), 
                                    SoftmaxWithTemperature(dim=-1, temperature=sm_temp))
        else:
            self.z = nn.Sequential(nn.Linear(embedded_dim, hidden_dim_h),
                                        self.nonlinearity,
                                    nn.Linear(hidden_dim_h, z_dim),)
        
        self.pretrain_xty_model = pretrain_xty_model
        self.x_t_y = xty_model
        
        if is_linear:
            self.t_hat = nn.Sequential(nn.Linear(z_dim, hidden_dim_h), 
                                    nn.Linear(hidden_dim_h, output_dim))
            
            self.c_y = nn.Sequential(
                nn.Linear(c_dim + 1, hidden_dim_h), 
                nn.Linear(hidden_dim_h, output_dim))
            
            if not self.pretrain_xty_model:
                self.xty_resnet = ResNet1D(BasicBlock1D, [2,2,2], num_classes=embedded_dim+1, in_channels=ecg_channels)
                self.x_t_y = nn.Sequential(self.xty_resnet,
                    nn.Linear(embedded_dim + 1, hidden_dim_h), 
                    nn.Linear(hidden_dim_h, output_dim))
        else:
            self.t_hat = nn.Sequential(nn.Linear(z_dim, hidden_dim_h), 
                                     self.nonlinearity,
                                    nn.Linear(hidden_dim_h, output_dim))
            
            self.c_y = nn.Sequential(
                nn.Linear(c_dim + 1, hidden_dim_h), 
                self.nonlinearity,
                nn.Linear(hidden_dim_h, output_dim))
            
            if not self.pretrain_xty_model:
                self.xty_resnet = ResNet1D(BasicBlock1D, [2,2,2], num_classes=embedded_dim, in_channels=ecg_channels)
                self.x_t_y = nn.Sequential(self.xty_resnet,
                    nn.Linear(embedded_dim + 1, hidden_dim_h), 
                    self.nonlinearity,
                    nn.Linear(hidden_dim_h, output_dim))
        
        self.z_layers = nn.ModuleList([self.z, self.t_hat])
    
    def forward(self, x, t):
        #x = x.to(self.device)
        #t = t.to(self.device)
        x_embedded = self.resnet(x)
        c = self.c(x_embedded) # C
        z = self.z(x_embedded) # Z
        t_hat = self.t_hat(z)  # T hat
        
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        c_y = self.c_y(torch.concatenate([c, t], dim=-1))
        x_t_y = self.x_t_y(x, t) # We use this to get the error on y given x and t
        
        return c, z, t_hat, c_y, x_t_y


class CateLoss(nn.Module):
    def __init__(self, true_cate=None):
        """
        Loss for evaluating Conditional Average Treatment Effect (CATE) estimation.
        
        Args:
            true_cate (array-like, optional): True CATE values for comparison. Defaults to None.
        """
        super(CateLoss, self).__init__()
        self.true_cate = true_cate

    def forward(self, c_y, t):
        """
        Compute CATE loss if true CATE is available.
        
        Args:
            c_y (torch.Tensor): Predicted outcomes from C and t.
            t (torch.Tensor): Treatment indicators.
            
        Returns:
            float or None: Mean squared error between estimated and true CATE, or None if true_cate not available.
        """
        if self.true_cate is not None:
            p0_idx = torch.where(t==0)[0]
            p1_idx = torch.where(t==1)[0]
            cate = c_y[p1_idx] - c_y[p0_idx]
            return torch.mean(self.true_cate - cate).detach().cpu().numpy().item() ** 2
        return None

#######################################################################################
# Separately train a network to learn Y from X,T to calculate epsilon

class X_T_Y_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_h=128, is_linear=False):
        """
        Separate network to learn Y from X and T for computing residuals.
        
        Args:
            input_dim (int): Dimensionality of input features X.
            output_dim (int): Dimensionality of outcome Y.
            hidden_dim_h (int): Hidden layer dimension. Defaults to 128.
            is_linear (bool): Use linear network (no activations). Defaults to False.
        """
        super(X_T_Y_Model, self).__init__()
        if is_linear:
            self.x_t_y = nn.Sequential(nn.Linear(input_dim+1, hidden_dim_h), 
                                       nn.Linear(hidden_dim_h, output_dim))
        else:
            self.x_t_y = nn.Sequential(nn.Linear(input_dim+1, hidden_dim_h), 
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim_h, output_dim))

    def forward(self, x_t_y):
        """
        Forward pass through X,T→Y network.
        
        Args:
            x_t_y (torch.Tensor): Concatenated [X, t] input.
            
        Returns:
            torch.Tensor: Predicted outcome Y.
        """
        return self.x_t_y(x_t_y)
    
    def fit(self, x_t_y, y, verbose=False, epochs=300):
        """
        Fit the X_T_Y_Model with MSE loss.
        
        Trains the network to predict Y from concatenated [X, t] input. This model
        is trained separately when ZNet is instantiated to compute residuals.
        
        Args:
            x_t_y (torch.Tensor): Concatenated [X, t] input features.
            y (torch.Tensor): Target outcome values.
            verbose (bool): Print final loss. Defaults to False.
            epochs (int): Number of training epochs. Defaults to 300.
            
        Returns:
            X_T_Y_Model: The trained model (self) with gradients frozen.
        """ 
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_fn = X_T_Y_Loss()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.forward(x_t_y)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"Final loss after training X_T_Y_Model: {loss.item()}")
        self.eval()  
        for param in self.parameters():
            param.requires_grad = False
        return self

class X_T_Y_ECGModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_h=128, embedded_dim=64, ecg_channels=12, is_linear=False):
        """
        Separate network to learn Y from X and T for computing residuals but for ECG inputs.
        
        Args:
            input_dim (int): Dimensionality of input features X.
            output_dim (int): Dimensionality of outcome Y.
            hidden_dim_h (int): Hidden layer dimension. Defaults to 128.
            embedded_dim (int): ECG embedded dimension after ResNet layers.
            ecg_channels (int): ECG leads (typically 12).
            is_linear (bool): Use linear network (no activations). Defaults to False.
        """
        super(X_T_Y_ECGModel, self).__init__()
        self.xty_resnet = ResNet1D(BasicBlock1D, [2,2,2], num_classes=embedded_dim, in_channels=ecg_channels)
        if is_linear:
            self.x_t_y = nn.Sequential(nn.Linear(embedded_dim+1, hidden_dim_h), 
                                       nn.Linear(hidden_dim_h, output_dim))
        else:
            self.x_t_y = nn.Sequential(nn.Linear(embedded_dim+1, hidden_dim_h), 
                                       nn.ReLU(),
                                       nn.Linear(hidden_dim_h, output_dim))

    def forward(self, X, t):
        x_embedded = self.xty_resnet(X)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        x_t_y = torch.cat([x_embedded, t], dim=-1)
        return self.x_t_y(x_t_y)
    
    def fit(self, train_loader, verbose=False, epochs=20, device='cpu'):
        """
        Fit the X_T_Y_ECGModel to the its data with MSE loss. Allows this model to be trained separately when ZNet is instantiated.
        """ 
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_fn = X_T_Y_Loss()
        for epoch in range(epochs):
            progress_bar = tqdm(train_loader, desc="XTY Training", disable=True)
            for batch in progress_bar:
                sys.stdout.flush()
                X_batch, t_batch, y_batch = batch['X'], batch['t'], batch['y']
                # Move to GPU
                X_batch, t_batch, y_batch = X_batch.to(device), t_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                output = self.forward(X_batch, t_batch)
                loss = loss_fn(output, y_batch)
                loss.backward()
                optimizer.step()
        #if verbose:
        # print(f"Final loss after training X_T_Y_Model: {loss.item()}")
        return self

    
class X_T_Y_Loss(nn.Module):
    def __init__(self, reg_loss=torch.nn.functional.mse_loss):
        """
        Loss function for X,T→Y model training.
        
        Args:
            reg_loss (callable): Regression loss function. Defaults to MSE loss.
        """
        super(X_T_Y_Loss, self).__init__()
        self.reg_loss = reg_loss

    def forward(self, x_t_y, y):
        """
        Compute loss for X,T→Y prediction.
        
        Args:
            x_t_y (torch.Tensor): Predicted outcome from X,T.
            y (torch.Tensor): True outcome values.
            
        Returns:
            torch.Tensor: Regression loss value.
        """
        return self.reg_loss(x_t_y, y)

#######################################################################################
# ZNet Model Helpers

class SoftmaxWithTemperature(nn.Module):
    def __init__(self, dim=1, temperature=1.0):
        """
        Softmax layer with temperature scaling for controlling output sharpness.
        
        Args:
            dim (int): Dimension along which to compute softmax. Defaults to 1.
            temperature (float): Temperature parameter (higher = smoother distribution). Defaults to 1.0.
        """
        super().__init__()
        self.dim = dim
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=dim)
    
    def forward(self, logits):
        """
        Apply temperature-scaled softmax.
        
        Args:
            logits (torch.Tensor): Input logits.
            
        Returns:
            torch.Tensor: Softmax probabilities with temperature scaling and safety clamping.
        """
        # Clamp logits before temperature scaling
        logits = torch.clamp(logits, min=-50, max=50)
        scaled = logits / self.temperature
        output = self.softmax(scaled)
        
        # Additional safety
        output = torch.nan_to_num(output, nan=1.0/logits.shape[self.dim])
        return output
    
class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        """
        Early stopping to halt training when validation loss stops improving.
        
        Args:
            patience (int): Number of epochs to wait for improvement. Defaults to 5.
            delta (float): Minimum change to qualify as improvement. Defaults to 0.
            verbose (bool): Print early stopping messages. Defaults to False.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_loss = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def check_early_stop(self, val_loss):
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss (float): Current validation loss.
            
        Returns:
            None: Updates internal state (stop_training flag).
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            if self.no_improvement_count >= self.patience:
                self.stop_training = True
                if self.verbose:
                    print("Stopping early as no improvement has been observed.")
                    
###########################################################################################               
# ECG building blocks:  
                    
class BasicBlock1D(nn.Module):
    """
    Basic residual block for 1D ResNet architecture.
    
    Implements a residual block with two convolutional layers, batch normalization,
    and skip connections for processing 1D sequential data like ECG signals.
    """
    expansion = 1
 
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initialize BasicBlock1D.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            stride (int): Stride for first convolution. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for skip connection. Defaults to None.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        self.downsample = downsample
 
    def forward(self, x):
        """
        Forward pass through residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, channels, length).
            
        Returns:
            torch.Tensor: Output tensor after residual connection.
        """
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out
 
class ResNet1D(nn.Module):
    """
    Tunable 1D ResNet architecture for processing sequential data like ECG signals.
    
    Implements a configurable ResNet with variable number of stages and blocks per stage.
    """
    def __init__(
        self,
        block,
        layers, 
        num_classes=3,
        in_channels=12,
        base_channels=64
    ):
        """
        Initialize ResNet1D.
        
        Args:
            block (nn.Module): Basic block type (e.g., BasicBlock1D).
            layers (list): Number of blocks per stage, e.g. e.g. [2,2,2,2] or [3,4,6,3] or [2,2,2].
            num_classes (int): Number of output classes/features. Defaults to 3.
            in_channels (int): Number of input channels (e.g., 12 for ECG). Defaults to 12.
            base_channels (int): Base number of channels in first layer. Defaults to 64.
        """
        super().__init__()

        self.in_channels = base_channels

        # Stem
        self.conv1 = nn.Conv1d(
            in_channels,
            base_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.stages = nn.ModuleList()

        out_channels = base_channels
        for i, num_blocks in enumerate(layers):
            stride = 1 if i == 0 else 2
            stage = self._make_layer(
                block,
                out_channels,
                num_blocks,
                stride=stride
            )
            self.stages.append(stage)
            out_channels *= 2

        # Head
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        # final_channels = base_channels * (2 ** (len(layers) - 1))
        self.fc = nn.Linear(self.in_channels, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through ResNet1D.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, length).
            
        Returns:
            torch.Tensor: Output features of shape (batch, num_classes).
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Stages
        for stage in self.stages:
            x = stage(x)

        # Head
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
