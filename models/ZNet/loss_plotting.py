#######################################################################################
# Author: Jenna Fields
# Script: loss_plotting.py
# Function: Create a class to track losses from ZNet training
# Date: 02/06/2026
#######################################################################################
from seed_utils import set_seed

set_seed(42)


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

#######################################################################################
class ZNetLossPlotter:
    def __init__(self):
        """
        Track and visualize ZNet training/validation losses.
        """

        self.all_losses = []
        self.t_hat_losses = []
        self.z_losses = []
        self.c_losses = []
        self.cumulative_matrix_losses = []
        self.mse_c_losses = []
        self.pearson_z_t_losses = []
        self.cate_losses = []
        self.kl_losses_c = []
        self.feature_corr_losses_c = []
        self.kl_losses_z = []
        self.feature_corr_losses_z = []
        self.mse_xt_losses = []

        self.val_all_losses = []
        self.val_t_hat_losses = []
        self.val_z_losses = []
        self.val_c_losses = []
        self.val_cumulative_matrix_losses = []
        self.val_mse_c_losses = []
        self.val_pearson_z_t_losses = []
        self.val_cate_losses = []
        self.val_kl_losses_c = []
        self.val_feature_corr_losses_c = []
        self.val_kl_losses_z = []
        self.val_feature_corr_losses_z = []
        self.val_mse_xt_losses = []

        self.cumulative_loss = []
        self.cumulative_z_loss = []
        self.cumulative_t_hat_loss = []
        self.cumulative_c_loss = []
        self.cumulative_matrix_loss = []
        self.cumulative_mse_c = []
        self.cumulative_pearson_z_t = []
        self.cumulative_cate_loss = []
        self.cumulative_kl_loss_c = []
        self.cumulative_feature_corr_loss_c = []
        self.cumulative_kl_loss_z = []
        self.cumulative_feature_corr_loss_z = []
        self.cumulative_mse_xt = []

    def add_train_batch(self, c_loss, 
                        z_loss, 
                        t_hat_loss, 
                        pearson_matrix_loss, 
                        mse_c, 
                        pearson_z_t, 
                        kl_c,
                        feature_corr_c,
                        kl_z,
                        feature_corr_z,
                        total_loss,
                        mse_xt, 
                        cate_loss=None):
        """
        Add a batch of training losses for epoch aggregation.
        
        Args:
            c_loss (torch.Tensor): C-Y correlation loss.
            z_loss (torch.Tensor): Z-residual correlation loss.
            t_hat_loss (torch.Tensor): Treatment prediction loss.
            pearson_matrix_loss (torch.Tensor): C-Z independence loss.
            mse_c (torch.Tensor): C→Y MSE loss.
            pearson_z_t (torch.Tensor): Z-T correlation loss.
            kl_c (torch.Tensor): KL loss for C.
            feature_corr_c (torch.Tensor): Feature correlation loss for C.
            kl_z (torch.Tensor): KL loss for Z.
            feature_corr_z (torch.Tensor): Feature correlation loss for Z.
            total_loss (torch.Tensor): Total weighted loss.
            mse_xt (torch.Tensor): X,T→Y MSE loss.
            cate_loss (torch.Tensor, optional): CATE loss.
        """
        self.cumulative_c_loss.append((c_loss).detach().cpu().numpy().item())
        self.cumulative_z_loss.append((z_loss).detach().cpu().numpy().item())
        self.cumulative_matrix_loss.append((pearson_matrix_loss).detach().cpu().numpy().item())
        self.cumulative_t_hat_loss.append((t_hat_loss).detach().cpu().numpy().item())
        self.cumulative_mse_c.append((mse_c).detach().cpu().numpy().item())
        self.cumulative_pearson_z_t.append((pearson_z_t).detach().cpu().numpy().item())
        self.cumulative_loss.append(total_loss.detach().cpu().numpy().item())
        self.cumulative_kl_loss_c.append((kl_c).detach().cpu().numpy().item())
        self.cumulative_feature_corr_loss_c.append((feature_corr_c).detach().cpu().numpy().item())
        self.cumulative_kl_loss_z.append((kl_z).detach().cpu().numpy().item())
        self.cumulative_feature_corr_loss_z.append((feature_corr_z).detach().cpu().numpy().item())
        if mse_xt is not None:
            self.cumulative_mse_xt.append((mse_xt).detach().cpu().numpy().item())
        if cate_loss is not None:
            self.cumulative_cate_loss.append(cate_loss.detach().cpu().numpy().item())

    def train_step(self):
        """
        Aggregate batch losses into epoch-level training losses.
        """
        self.all_losses.append(np.mean(self.cumulative_loss))
        self.t_hat_losses.append(np.mean(self.cumulative_t_hat_loss))
        self.z_losses.append(np.mean(self.cumulative_z_loss))
        self.c_losses.append(np.mean(self.cumulative_c_loss))
        self.cumulative_matrix_losses.append(np.mean(self.cumulative_matrix_loss))
        self.mse_c_losses.append(np.mean(self.cumulative_mse_c))
        self.pearson_z_t_losses.append(np.mean(self.cumulative_pearson_z_t))
        self.kl_losses_c.append(np.mean(self.cumulative_kl_loss_c))
        self.feature_corr_losses_c.append(np.mean(self.cumulative_feature_corr_loss_c))
        self.kl_losses_z.append(np.mean(self.cumulative_kl_loss_z))
        self.feature_corr_losses_z.append(np.mean(self.cumulative_feature_corr_loss_z))
        if len(self.cumulative_cate_loss) > 0:
            self.cate_losses.append(np.mean(self.cumulative_cate_loss))
        if len(self.cumulative_mse_xt) > 0:
            self.mse_xt_losses.append(np.mean(self.cumulative_mse_xt))

        self.cumulative_loss = []
        self.cumulative_z_loss = []
        self.cumulative_t_hat_loss = []
        self.cumulative_c_loss = []
        self.cumulative_matrix_loss = []
        self.cumulative_mse_c = []
        self.cumulative_pearson_z_t = []
        self.cumulative_cate_loss = []
        self.cumulative_kl_loss_c = []
        self.cumulative_feature_corr_loss_c = []
        self.cumulative_kl_loss_z = []
        self.cumulative_feature_corr_loss_z = []
        self.cumulative_mse_xt = []

    def val_step(self, c_loss, 
                        z_loss, 
                        t_hat_loss, 
                        pearson_matrix_loss, 
                        mse_c, 
                        pearson_z_t, 
                        kl_c,
                        feature_corr_c,
                        kl_z,
                        feature_corr_z,
                        total_loss,
                        mse_xt,
                        cate_loss=None):
        """
        Record a validation step's loss components.
        
        Args:
            c_loss (torch.Tensor): C-Y correlation loss.
            z_loss (torch.Tensor): Z-residual correlation loss.
            t_hat_loss (torch.Tensor): Treatment prediction loss.
            pearson_matrix_loss (torch.Tensor): C-Z independence loss.
            mse_c (torch.Tensor): C→Y MSE loss.
            pearson_z_t (torch.Tensor): Z-T correlation loss.
            kl_c (torch.Tensor): KL loss for C.
            feature_corr_c (torch.Tensor): Feature correlation loss for C.
            kl_z (torch.Tensor): KL loss for Z.
            feature_corr_z (torch.Tensor): Feature correlation loss for Z.
            total_loss (torch.Tensor): Total weighted loss.
            mse_xt (torch.Tensor): X,T→Y MSE loss.
            cate_loss (torch.Tensor, optional): CATE loss.
        """
        self.val_all_losses.append(np.mean(total_loss.detach().cpu().numpy().item())) 
        self.val_t_hat_losses.append(np.mean(t_hat_loss.detach().cpu().numpy().item()))
        self.val_z_losses.append(np.mean(z_loss.detach().cpu().numpy().item()))
        self.val_c_losses.append(np.mean(c_loss.detach().cpu().numpy().item()))
        self.val_cumulative_matrix_losses.append(np.mean(pearson_matrix_loss.detach().cpu().numpy().item()))
        self.val_mse_c_losses.append(np.mean(mse_c.detach().cpu().numpy().item()))
        self.val_pearson_z_t_losses.append(np.mean(pearson_z_t.detach().cpu().numpy().item()))
        self.val_kl_losses_c.append(np.mean(kl_c.detach().cpu().numpy().item()))
        self.val_feature_corr_losses_c.append(np.mean(feature_corr_c.detach().cpu().numpy().item()))
        self.val_kl_losses_z.append(np.mean(kl_z.detach().cpu().numpy().item()))
        self.val_feature_corr_losses_z.append(np.mean(feature_corr_z.detach().cpu().numpy().item()))
        if mse_xt is not None:
            self.val_mse_xt_losses.append(np.mean(mse_xt.detach().cpu().numpy().item()))
        if cate_loss is not None:
            self.val_cate_losses.append(np.mean(cate_loss.detach().cpu().numpy().item()))
    
    def plot_losses(self, c_mse_loss_alpha, 
                          t_hat_alpha, 
                          z_pearson_loss_alpha, 
                          c_pearson_loss_alpha, 
                          z_t_loss_alpha,
                          kl_loss_alpha,
                          feature_corr_loss_alpha,
                          pearson_matrix_alpha,
                          train_mse_xt = False,
                          true_cate=None,
                          save_path=None):
        """
        Plot training and validation loss curves.
        
        Args:
            c_mse_loss_alpha (float): Weight for C→Y MSE loss.
            t_hat_alpha (float): Weight for treatment prediction loss.
            z_pearson_loss_alpha (float): Weight for Z-residual correlation loss.
            c_pearson_loss_alpha (float): Weight for C-Y correlation loss.
            z_t_loss_alpha (float): Weight for Z-T correlation loss.
            kl_loss_alpha (float): Weight for KL loss.
            feature_corr_loss_alpha (float): Weight for feature correlation loss.
            pearson_matrix_alpha (float): Weight for C-Z independence loss.
            train_mse_xt (bool): Whether X,T→Y loss is tracked. Defaults to False.
            true_cate (float, optional): True CATE for labeling. Defaults to None.
            save_path (str, optional): Directory to save plots. Defaults to None.
        """
        if train_mse_xt: 
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.all_losses, label=f'Total Loss')
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(self.z_losses, label=f'PC X Perp Loss (alpha={z_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.c_losses, label=f'PC X Hat Loss (alpha={c_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.cumulative_matrix_losses, label=f'Matrix Loss (alpha={pearson_matrix_alpha})', alpha=0.7)
        axes[1].plot(self.t_hat_losses, label=f'Treatment Hat Loss (alpha={t_hat_alpha})', alpha=0.7)
        axes[1].plot(self.mse_c_losses, label=f'MSE C Loss (alpha={c_mse_loss_alpha})', alpha=0.7)
        if true_cate is not None:
            axes[1].plot(self.cate_losses, label=f'MSE CATE Loss (cate={true_cate:.2f})', alpha=0.7)
        axes[1].plot(self.pearson_z_t_losses, label=f'PC X Perp - T Loss (alpha={z_t_loss_alpha})', alpha=0.7)
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[1].set_title('C/Z Loss Components')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (pre alpha)')

        axes[2].plot(self.kl_losses_c, label=f'KL C Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.feature_corr_losses_c, label=f'Feature Corr C Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].plot(self.kl_losses_z, label=f'KL Z Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.feature_corr_losses_z, label=f'Feature Corr Z Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[2].set_title('Feature Loss Components')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss (pre alpha)')

        if train_mse_xt:
            axes[3].plot(self.mse_xt_losses, label=f'MSE X T Loss (for error)', alpha=0.7)
            axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=1)
            axes[3].set_title('MSE XT -> Y Loss')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Loss')
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + 'losses.png', bbox_inches='tight')

        if train_mse_xt: 
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.val_all_losses, label=f'Total Loss (Validation)')
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[0].set_title('Total Loss (Validation)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(self.val_z_losses, label=f'PC X Perp Loss (alpha={z_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.val_c_losses, label=f'PC X Hat Loss (alpha={c_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.val_cumulative_matrix_losses, label=f'Matrix Loss (alpha={pearson_matrix_alpha})', alpha=0.7)
        axes[1].plot(self.val_t_hat_losses, label=f'Treatment Hat Loss (alpha={t_hat_alpha})', alpha=0.7)
        axes[1].plot(self.val_mse_c_losses, label=f'MSE C Loss (alpha={c_mse_loss_alpha})', alpha=0.7)
        if true_cate is not None:
            axes[1].plot(self.val_cate_losses, label=f'MSE CATE Loss (cate={true_cate:.2f})', alpha=0.7)
        axes[1].plot(self.val_pearson_z_t_losses, label=f'PC X Perp - T Loss (alpha={z_t_loss_alpha})', alpha=0.7)
        # plt.legend()
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[1].set_title('C/Z Loss Components (Validation)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (pre alpha)')

        axes[2].plot(self.val_kl_losses_c, label=f'KL C Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.val_feature_corr_losses_c, label=f'Feature Corr C Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].plot(self.val_kl_losses_z, label=f'KL Z Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.val_feature_corr_losses_z, label=f'Feature Corr Z Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[2].set_title('Feature Loss Components (Validation)')

        if train_mse_xt:
            axes[3].plot(self.val_mse_xt_losses, label=f'MSE X T Loss (for error)', alpha=0.7)
            axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=1)
            axes[3].set_title('MSE XT -> Y Loss (Validation)')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Loss')

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + 'val_losses.png', bbox_inches='tight')
    
    def save_losses(self, save_path):
        """
        Save training and validation losses to CSV files.
        
        Args:
            save_path (str): Output directory path.
        """
        train_df = pd.DataFrame({
            'all_losses': self.all_losses,
            't_hat_losses': self.t_hat_losses,
            'z_losses': self.z_losses,
            'c_losses': self.c_losses,
            'cumulative_matrix_losses': self.cumulative_matrix_losses,
            'mse_c_losses': self.mse_c_losses,
            'pearson_z_t_losses': self.pearson_z_t_losses,
            'kl_losses_c': self.kl_losses_c,
            'feature_corr_losses_c': self.feature_corr_losses_c,
            'kl_losses_z': self.kl_losses_z,
            'feature_corr_losses_z': self.feature_corr_losses_z,
            'mse_xt_losses': self.mse_xt_losses,
            'cate_losses': self.cate_losses
        })
        val_df = pd.DataFrame({
            'val_all_losses': self.val_all_losses,
            'val_t_hat_losses': self.val_t_hat_losses,
            'val_z_losses': self.val_z_losses,
            'val_c_losses': self.val_c_losses,
            'val_cumulative_matrix_losses': self.val_cumulative_matrix_losses,
            'val_mse_c_losses': self.val_mse_c_losses,
            'val_pearson_z_t_losses': self.val_pearson_z_t_losses,
            'val_kl_losses_c': self.val_kl_losses_c,
            'val_feature_corr_losses_c': self.val_feature_corr_losses_c,
            'val_kl_losses_z': self.val_kl_losses_z,
            'val_feature_corr_losses_z': self.val_feature_corr_losses_z,
            'val_mse_xt_losses': self.val_mse_xt_losses,
            'val_cate_losses': self.val_cate_losses
        })
        train_df.to_csv(save_path + 'train_losses.csv', index=False)
        val_df.to_csv(save_path + 'val_losses.csv', index=False)
        
        
        
class ZNetECGLossPlotter:
    def __init__(self):
        """
        Track and visualize ECG ZNet training/validation losses.
        """

        self.all_losses = []
        self.t_hat_losses = []
        self.z_losses = []
        self.c_losses = []
        self.cumulative_matrix_losses = []
        self.mse_c_losses = []
        self.pearson_z_t_losses = []
        self.cate_losses = []
        self.kl_losses_c = []
        self.feature_corr_losses_c = []
        self.kl_losses_z = []
        self.feature_corr_losses_z = []
        self.mse_xt_losses = []

        self.val_all_losses = []
        self.val_t_hat_losses = []
        self.val_z_losses = []
        self.val_c_losses = []
        self.val_cumulative_matrix_losses = []
        self.val_mse_c_losses = []
        self.val_pearson_z_t_losses = []
        self.val_cate_losses = []
        self.val_kl_losses_c = []
        self.val_feature_corr_losses_c = []
        self.val_kl_losses_z = []
        self.val_feature_corr_losses_z = []
        self.val_mse_xt_losses = []

        self.cumulative_loss = []
        self.cumulative_z_loss = []
        self.cumulative_t_hat_loss = []
        self.cumulative_c_loss = []
        self.cumulative_matrix_loss = []
        self.cumulative_mse_c = []
        self.cumulative_pearson_z_t = []
        self.cumulative_cate_loss = []
        self.cumulative_kl_loss_c = []
        self.cumulative_feature_corr_loss_c = []
        self.cumulative_kl_loss_z = []
        self.cumulative_feature_corr_loss_z = []
        self.cumulative_mse_xt = []

    def add_train_batch(self, c_loss, 
                        z_loss, 
                        t_hat_loss, 
                        pearson_matrix_loss, 
                        mse_c, 
                        pearson_z_t, 
                        kl_c,
                        feature_corr_c,
                        kl_z,
                        feature_corr_z,
                        total_loss,
                        mse_xt, 
                        cate_loss=None):
        """
        Add a batch of training losses for epoch aggregation.
        
        Args:
            c_loss (torch.Tensor): C-Y correlation loss.
            z_loss (torch.Tensor): Z-residual correlation loss.
            t_hat_loss (torch.Tensor): Treatment prediction loss.
            pearson_matrix_loss (torch.Tensor): C-Z independence loss.
            mse_c (torch.Tensor): C→Y MSE loss.
            pearson_z_t (torch.Tensor): Z-T correlation loss.
            kl_c (torch.Tensor): KL loss for C.
            feature_corr_c (torch.Tensor): Feature correlation loss for C.
            kl_z (torch.Tensor): KL loss for Z.
            feature_corr_z (torch.Tensor): Feature correlation loss for Z.
            total_loss (torch.Tensor): Total weighted loss.
            mse_xt (torch.Tensor): X,T→Y MSE loss.
            cate_loss (torch.Tensor, optional): CATE loss.
        """
        self.cumulative_c_loss.append((c_loss).detach().cpu().numpy().item())
        self.cumulative_z_loss.append((z_loss).detach().cpu().numpy().item())
        self.cumulative_matrix_loss.append((pearson_matrix_loss).detach().cpu().numpy().item())
        self.cumulative_t_hat_loss.append((t_hat_loss).detach().cpu().numpy().item())
        self.cumulative_mse_c.append((mse_c).detach().cpu().numpy().item())
        self.cumulative_pearson_z_t.append((pearson_z_t).detach().cpu().numpy().item())
        self.cumulative_loss.append(total_loss.detach().cpu().numpy().item())
        self.cumulative_kl_loss_c.append((kl_c).detach().cpu().numpy().item())
        self.cumulative_feature_corr_loss_c.append((feature_corr_c).detach().cpu().numpy().item())
        self.cumulative_kl_loss_z.append((kl_z).detach().cpu().numpy().item())
        self.cumulative_feature_corr_loss_z.append((feature_corr_z).detach().cpu().numpy().item())
        if mse_xt is not None:
            self.cumulative_mse_xt.append((mse_xt).detach().cpu().numpy().item())
        if cate_loss is not None:
            self.cumulative_cate_loss.append(cate_loss.detach().cpu().numpy().item())

    def train_step(self):
        """
        Aggregate batch losses into epoch-level training losses.
        """
        self.all_losses.append(np.mean(self.cumulative_loss))
        self.t_hat_losses.append(np.mean(self.cumulative_t_hat_loss))
        self.z_losses.append(np.mean(self.cumulative_z_loss))
        self.c_losses.append(np.mean(self.cumulative_c_loss))
        self.cumulative_matrix_losses.append(np.mean(self.cumulative_matrix_loss))
        self.mse_c_losses.append(np.mean(self.cumulative_mse_c))
        self.pearson_z_t_losses.append(np.mean(self.cumulative_pearson_z_t))
        self.kl_losses_c.append(np.mean(self.cumulative_kl_loss_c))
        self.feature_corr_losses_c.append(np.mean(self.cumulative_feature_corr_loss_c))
        self.kl_losses_z.append(np.mean(self.cumulative_kl_loss_z))
        self.feature_corr_losses_z.append(np.mean(self.cumulative_feature_corr_loss_z))
        if len(self.cumulative_cate_loss) > 0:
            self.cate_losses.append(np.mean(self.cumulative_cate_loss))
        if len(self.cumulative_mse_xt) > 0:
            self.mse_xt_losses.append(np.mean(self.cumulative_mse_xt))

        self.cumulative_loss = []
        self.cumulative_z_loss = []
        self.cumulative_t_hat_loss = []
        self.cumulative_c_loss = []
        self.cumulative_matrix_loss = []
        self.cumulative_mse_c = []
        self.cumulative_pearson_z_t = []
        self.cumulative_cate_loss = []
        self.cumulative_kl_loss_c = []
        self.cumulative_feature_corr_loss_c = []
        self.cumulative_kl_loss_z = []
        self.cumulative_feature_corr_loss_z = []
        self.cumulative_mse_xt = []

    def val_step(self, c_loss, 
                        z_loss, 
                        t_hat_loss, 
                        pearson_matrix_loss, 
                        mse_c, 
                        pearson_z_t, 
                        kl_c,
                        feature_corr_c,
                        kl_z,
                        feature_corr_z,
                        total_loss,
                        mse_xt,
                        cate_loss=None):
        """
        Record a validation step's loss components.
        
        Args:
            c_loss (torch.Tensor): C-Y correlation loss.
            z_loss (torch.Tensor): Z-residual correlation loss.
            t_hat_loss (torch.Tensor): Treatment prediction loss.
            pearson_matrix_loss (torch.Tensor): C-Z independence loss.
            mse_c (torch.Tensor): C→Y MSE loss.
            pearson_z_t (torch.Tensor): Z-T correlation loss.
            kl_c (torch.Tensor): KL loss for C.
            feature_corr_c (torch.Tensor): Feature correlation loss for C.
            kl_z (torch.Tensor): KL loss for Z.
            feature_corr_z (torch.Tensor): Feature correlation loss for Z.
            total_loss (torch.Tensor): Total weighted loss.
            mse_xt (torch.Tensor): X,T→Y MSE loss.
            cate_loss (torch.Tensor, optional): CATE loss.
        """
        self.val_all_losses.append(total_loss) # np.mean(total_loss.detach().cpu().numpy().item())) 
        self.val_t_hat_losses.append(t_hat_loss) #np.mean(t_hat_loss.detach().cpu().numpy().item()))
        self.val_z_losses.append(z_loss) #np.mean(z_loss.detach().cpu().numpy().item()))
        self.val_c_losses.append(c_loss) #np.mean(c_loss.detach().cpu().numpy().item()))
        self.val_cumulative_matrix_losses.append(pearson_matrix_loss) #np.mean(pearson_matrix_loss.detach().cpu().numpy().item()))
        self.val_mse_c_losses.append(mse_c) # np.mean(mse_c.detach().cpu().numpy().item()))
        self.val_pearson_z_t_losses.append(pearson_z_t) #np.mean(pearson_z_t.detach().cpu().numpy().item()))
        self.val_kl_losses_c.append(kl_c) #np.mean(kl_c.detach().cpu().numpy().item()))
        self.val_feature_corr_losses_c.append(feature_corr_c) #np.mean(feature_corr_c.detach().cpu().numpy().item()))
        self.val_kl_losses_z.append(kl_z) #np.mean(kl_z.detach().cpu().numpy().item()))
        self.val_feature_corr_losses_z.append(feature_corr_z) #np.mean(feature_corr_z.detach().cpu().numpy().item()))
        if mse_xt is not None:
            self.val_mse_xt_losses.append(mse_xt) #np.mean(mse_xt.detach().cpu().numpy().item()))
        if cate_loss is not None:
            self.val_cate_losses.append(cate_loss) #np.mean(cate_loss.detach().cpu().numpy().item()))
    
    def plot_losses(self, c_mse_loss_alpha, 
                          t_hat_alpha, 
                          z_pearson_loss_alpha, 
                          c_pearson_loss_alpha, 
                          z_t_loss_alpha,
                          kl_loss_alpha,
                          feature_corr_loss_alpha,
                          pearson_matrix_alpha,
                          train_mse_xt = False,
                          true_cate=None,
                          save_path='loss_plots/'):
        """
        Plot training and validation loss curves.
        
        Args:
            c_mse_loss_alpha (float): Weight for C→Y MSE loss.
            t_hat_alpha (float): Weight for treatment prediction loss.
            z_pearson_loss_alpha (float): Weight for Z-residual correlation loss.
            c_pearson_loss_alpha (float): Weight for C-Y correlation loss.
            z_t_loss_alpha (float): Weight for Z-T correlation loss.
            kl_loss_alpha (float): Weight for KL loss.
            feature_corr_loss_alpha (float): Weight for feature correlation loss.
            pearson_matrix_alpha (float): Weight for C-Z independence loss.
            train_mse_xt (bool): Whether X,T→Y loss is tracked. Defaults to False.
            true_cate (float, optional): True CATE for labeling. Defaults to None.
            save_path (str, optional): Directory to save plots. Defaults to 'loss_plots/'.
        """
        if train_mse_xt: 
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.all_losses, label=f'Total Loss')
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(self.z_losses, label=f'PC X Perp Loss (alpha={z_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.c_losses, label=f'PC X Hat Loss (alpha={c_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.cumulative_matrix_losses, label=f'Matrix Loss (alpha={pearson_matrix_alpha})', alpha=0.7)
        axes[1].plot(self.t_hat_losses, label=f'Treatment Hat Loss (alpha={t_hat_alpha})', alpha=0.7)
        axes[1].plot(self.mse_c_losses, label=f'MSE C Loss (alpha={c_mse_loss_alpha})', alpha=0.7)
        if true_cate is not None:
            axes[1].plot(self.cate_losses, label=f'MSE CATE Loss (cate={true_cate:.2f})', alpha=0.7)
        axes[1].plot(self.pearson_z_t_losses, label=f'PC X Perp - T Loss (alpha={z_t_loss_alpha})', alpha=0.7)
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[1].set_title('C/Z Loss Components')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (pre alpha)')

        axes[2].plot(self.kl_losses_c, label=f'KL C Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.feature_corr_losses_c, label=f'Feature Corr C Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].plot(self.kl_losses_z, label=f'KL Z Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.feature_corr_losses_z, label=f'Feature Corr Z Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[2].set_title('Feature Loss Components')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss (pre alpha)')

        if train_mse_xt:
            axes[3].plot(self.mse_xt_losses, label=f'MSE X T Loss (for error)', alpha=0.7)
            axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=1)
            axes[3].set_title('MSE XT -> Y Loss')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Loss')
        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + 'losses.png', bbox_inches='tight')

        if train_mse_xt: 
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].plot(self.val_all_losses, label=f'Total Loss (Validation)')
        axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[0].set_title('Total Loss (Validation)')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')

        axes[1].plot(self.val_z_losses, label=f'PC X Perp Loss (alpha={z_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.val_c_losses, label=f'PC X Hat Loss (alpha={c_pearson_loss_alpha})', alpha=0.7)
        axes[1].plot(self.val_cumulative_matrix_losses, label=f'Matrix Loss (alpha={pearson_matrix_alpha})', alpha=0.7)
        axes[1].plot(self.val_t_hat_losses, label=f'Treatment Hat Loss (alpha={t_hat_alpha})', alpha=0.7)
        axes[1].plot(self.val_mse_c_losses, label=f'MSE C Loss (alpha={c_mse_loss_alpha})', alpha=0.7)
        if true_cate is not None:
            axes[1].plot(self.val_cate_losses, label=f'MSE CATE Loss (cate={true_cate:.2f})', alpha=0.7)
        axes[1].plot(self.val_pearson_z_t_losses, label=f'PC X Perp - T Loss (alpha={z_t_loss_alpha})', alpha=0.7)
        # plt.legend()
        axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[1].set_title('C/Z Loss Components (Validation)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss (pre alpha)')

        axes[2].plot(self.val_kl_losses_c, label=f'KL C Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.val_feature_corr_losses_c, label=f'Feature Corr C Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].plot(self.val_kl_losses_z, label=f'KL Z Loss (alpha={kl_loss_alpha})', alpha=0.7)
        axes[2].plot(self.val_feature_corr_losses_z, label=f'Feature Corr Z Loss (alpha={feature_corr_loss_alpha})', alpha=0.7)
        axes[2].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
        fancybox=True, shadow=True, ncol=1)
        axes[2].set_title('Feature Loss Components (Validation)')

        if train_mse_xt:
            axes[3].plot(self.val_mse_xt_losses, label=f'MSE X T Loss (for error)', alpha=0.7)
            axes[3].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
            fancybox=True, shadow=True, ncol=1)
            axes[3].set_title('MSE XT -> Y Loss (Validation)')
            axes[3].set_xlabel('Epoch')
            axes[3].set_ylabel('Loss')

        fig.tight_layout()
        if save_path is not None:
            plt.savefig(save_path + 'val_losses.png', bbox_inches='tight')
    
    def save_losses(self, save_path):
        """
        Save training and validation losses to CSV files.
        
        Args:
            save_path (str): Output directory path.
        """
        train_df = pd.DataFrame({
            'all_losses': self.all_losses,
            't_hat_losses': self.t_hat_losses,
            'z_losses': self.z_losses,
            'c_losses': self.c_losses,
            'cumulative_matrix_losses': self.cumulative_matrix_losses,
            'mse_c_losses': self.mse_c_losses,
            'pearson_z_t_losses': self.pearson_z_t_losses,
            'kl_losses_c': self.kl_losses_c,
            'feature_corr_losses_c': self.feature_corr_losses_c,
            'kl_losses_z': self.kl_losses_z,
            'feature_corr_losses_z': self.feature_corr_losses_z,
            'mse_xt_losses': self.mse_xt_losses,
            'cate_losses': self.cate_losses
        })
        val_df = pd.DataFrame({
            'val_all_losses': self.val_all_losses,
            'val_t_hat_losses': self.val_t_hat_losses,
            'val_z_losses': self.val_z_losses,
            'val_c_losses': self.val_c_losses,
            'val_cumulative_matrix_losses': self.val_cumulative_matrix_losses,
            'val_mse_c_losses': self.val_mse_c_losses,
            'val_pearson_z_t_losses': self.val_pearson_z_t_losses,
            'val_kl_losses_c': self.val_kl_losses_c,
            'val_feature_corr_losses_c': self.val_feature_corr_losses_c,
            'val_kl_losses_z': self.val_kl_losses_z,
            'val_feature_corr_losses_z': self.val_feature_corr_losses_z,
            'val_mse_xt_losses': self.val_mse_xt_losses,
            'val_cate_losses': self.val_cate_losses
        })
        train_df.to_csv(save_path + 'train_losses.csv', index=False)
        val_df.to_csv(save_path + 'val_losses.csv', index=False)
