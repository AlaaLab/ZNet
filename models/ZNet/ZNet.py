#######################################################################################
# Author: Jenna Fields
# Script: ZNet.py
# Function:  ZNet model wrapper
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

from models.ZNet.model_loss_utils import ZNetLoss, ZNetModel, X_T_Y_Model, EarlyStopping, CateLoss, PearsonLoss, KDEMutualInformation
from models.ZNet.loss_plotting import ZNetLossPlotter
from models.ZNet.pcgrad import PCGrad

#######################################################################################
class ZNet():
    def __init__(self, 
                 input_dim, 
                 c_dim, 
                 z_dim, 
                 output_dim, 
                 lr=0.001, 
                 weight_decay=.1,
                 kl_loss_coeff=0, 
                 feature_corr_loss_coeff = 0, 
                 c_pearson_loss_alpha = 1, 
                 c_mse_loss_alpha = 1, 
                 z_pearson_loss_alpha = 1, 
                 z_t_loss_alpha = 0, 
                 pearson_matrix_alpha = 0, 
                 t_hat_alpha = 1, 
                 true_cate=None, 
                 binary_outcome=False,
                 is_linear=False, 
                 use_sm = True, 
                 sm_temp=2, 
                 use_pcgrad=False, 
                 train_xt_net=False,
                 use_mi_corr_loss=False, 
                 use_mi_matrix_loss=False,
                 device=None,
                 pretrain_xty_model=True
                ):
        """
        Initialize the ZNet model for instrumental variable (IV) generation.
        
        ZNet learns disentangled representations C (confounders) and Z (valid instruments)
        from input features X to predict treatment T and outcome Y.
        
        Args:
            input_dim (int): Dimensionality of input features X.
            c_dim (int): Dimensionality of confounder representation C.
            z_dim (int): Dimensionality of instrumental variable representation Z.
            output_dim (int): Dimensionality of output Y.
            lr (float): Learning rate for optimizers. Defaults to 0.001.
            weight_decay (float): L2 regularization coefficient. Defaults to 0.1.
            kl_loss_coeff (float): Coefficient for KL divergence loss on C and Z distributions. Defaults to 0.
            feature_corr_loss_coeff (float): Coefficient for feature correlation loss. Defaults to 0.
            c_pearson_loss_alpha (float): Weight for correlation between C and Y. Defaults to 1.
            c_mse_loss_alpha (float): Weight for MSE loss on C→Y prediction. Defaults to 1.
            z_pearson_loss_alpha (float): Weight for correlation between Z and residual Y. Defaults to 1.
            z_t_loss_alpha (float): Weight for correlation between Z and T. Defaults to 0.
            pearson_matrix_alpha (float): Weight for independence between C and Z. Defaults to 0.
            t_hat_alpha (float): Weight for treatment prediction loss Z→T. Defaults to 1.
            true_cate (array-like, optional): True conditional average treatment effect for evaluation. Defaults to None.
            binary_outcome (bool): Whether outcome Y is binary. Defaults to False.
            is_linear (bool): Whether to use linear networks (no activation functions). Defaults to False.
            use_sm (bool): Whether to use softmax with temperature on C and Z. Defaults to True.
            sm_temp (float): Temperature parameter for softmax. Defaults to 2.
            use_pcgrad (bool): Whether to use PCGrad for multi-objective optimization. Defaults to False.
            train_xt_net (bool): Whether to train the X,T→Y network jointly. Defaults to False.
            use_mi_corr_loss (bool): Use mutual information instead of Pearson correlation. Defaults to False.
            use_mi_matrix_loss (bool): Use mutual information for C-Z independence. Defaults to False.
            device (str, optional): Device to run model on ('cuda' or 'cpu'). Auto-detects if None.
            pretrain_xty_model (bool): Whether to pretrain X,T,Y model separately. Defaults to True.
            
        Returns:
            None
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.pretrain_xty_model = pretrain_xty_model
        
        if self.pretrain_xty_model:
            self.x_t_y = X_T_Y_Model(input_dim, output_dim, hidden_dim_h=64, is_linear=is_linear).to(self.device)

        self.model = ZNetModel(input_dim, c_dim, z_dim, output_dim, 
                                   is_linear=is_linear, use_softmax=use_sm, 
                                   sm_temp=sm_temp, pretrain_xty_model=pretrain_xty_model,
                                   xty_model=self.x_t_y).to(self.device)
        self.lr = lr
        self.criterion = ZNetLoss(kl_loss_coeff = kl_loss_coeff, 
                                feature_corr_loss_coeff=feature_corr_loss_coeff, 
                                c_pearson_loss_alpha=c_pearson_loss_alpha,
                                c_mse_loss_alpha=c_mse_loss_alpha,
                                z_pearson_loss_alpha= z_pearson_loss_alpha,
                                z_t_loss_alpha=z_t_loss_alpha,
                                pearson_matrix_alpha=pearson_matrix_alpha,
                                t_hat_alpha=t_hat_alpha,
                                binary_outcome=binary_outcome,
                                train_xt_net=train_xt_net,
                                # Add'l to update the correlation loss to MI
                                use_mi_corr_loss = use_mi_corr_loss, 
                                use_mi_matrix_loss = use_mi_matrix_loss
                                )

        self.weight_decay = weight_decay

        self.true_cate = true_cate
        self.use_pcgrad = use_pcgrad

        self.cate_loss = CateLoss(self.true_cate)
        
        
        if self.pretrain_xty_model:
            self.z_optimizer = torch.optim.Adam([{'params':self.model.z.parameters(), 'weight_decay':0}, 
                                            {'params':self.model.t_hat.parameters(), 'weight_decay':self.weight_decay}
                                            ], # TODO: should we separate this out?
                                            lr=self.lr)
        else:
            self.z_optimizer = torch.optim.Adam([{'params':self.model.z.parameters(), 'weight_decay':0}, 
                                            {'params':self.model.t_hat.parameters(), 'weight_decay':self.weight_decay}, 
                                            {'params':self.model.x_t_y.parameters()} 
                                            ], # TODO: should we separate this out?
                                            lr=self.lr)
        self.c_optimizer = torch.optim.Adam([{'params':self.model.c.parameters()},
                                                 {'params':self.model.c_y.parameters()},], 
                                                weight_decay=0, lr=self.lr)
        
        if self.use_pcgrad:
            self.z_optimizer = PCGrad(self.z_optimizer)
            self.c_optimizer = PCGrad(self.c_optimizer)
    
    def _to_device(self, *tensors):
        """
        Move tensors to the model's device.
        
        Args:
            *tensors: Variable number of tensors or other objects to move to device.
            
        Returns:
            list: List of tensors moved to device (non-tensors returned as-is).
        """
        return [t.to(self.device) if isinstance(t, torch.Tensor) else t for t in tensors]
    
    def train_x_t_y(self, x, t, y, verbose=False, epochs=300):
        """
        Fit the X_T_Y_Model when pretrained separately.
        
        Trains a separate neural network to predict Y from X and T, used to compute
        residuals for the Z representation learning. Called from train_znet in train_models.py.
        
        Args:
            x (torch.Tensor): Input features of shape (n_samples, input_dim).
            t (torch.Tensor): Treatment indicators of shape (n_samples, 1).
            y (torch.Tensor): Outcome values of shape (n_samples, 1).
            verbose (bool): Whether to print training progress. Defaults to False.
            epochs (int): Number of training epochs. Defaults to 300.
            
        Returns:
            X_T_Y_Model: The trained X_T_Y model.
        """ 
        x_t_y = torch.concatenate([x, t], dim=-1)
        return self.x_t_y.fit(x_t_y, y, verbose=verbose, epochs=epochs)
    
    def calculate_loss(self, X_batch, t_batch, w_batch, y_batch, eval_mode=False):
        """
        Calculate all loss components for a batch of data.
        
        Args:
            X_batch (torch.Tensor): Input features.
            t_batch (torch.Tensor): Treatment indicators.
            w_batch (torch.Tensor): Additional weights or variables.
            y_batch (torch.Tensor): Outcome values.
            eval_mode (bool): Whether to set model to evaluation mode. Defaults to False.
            
        Returns:
            tuple: Loss components from criterion forward pass.
        """
        if eval_mode:
            self.model.eval()

        output_phi, output_h0, output_h1 = self.model(X_batch)

        return self.criterion.forward(output_phi, output_h0, output_h1, t_batch, w_batch, y_batch)

    def step_optimizers(self, c_loss : torch.Tensor, z_loss : torch.Tensor, 
                        pearson_matrix_loss : torch.Tensor, t_hat_loss : torch.Tensor, 
                        kl_c : torch.Tensor, kl_z : torch.Tensor, 
                        mse_c : torch.Tensor, pearson_z_t : torch.Tensor, mse_xt : torch.Tensor = None):
        """
        Perform backward pass and optimizer steps for all loss components.
        
        Handles multi-objective optimization using either PCGrad (if enabled) or
        standard backpropagation through all losses. Manages gradients for both
        C and Z networks separately.
        
        Args:
            c_loss (torch.Tensor): Correlation loss for C and Y.
            z_loss (torch.Tensor): Correlation loss for Z and residual Y.
            pearson_matrix_loss (torch.Tensor): Independence loss between C and Z.
            t_hat_loss (torch.Tensor): Treatment prediction loss from Z.
            kl_c (torch.Tensor): KL divergence loss for C.
            kl_z (torch.Tensor): KL divergence loss for Z.
            mse_c (torch.Tensor): MSE loss for C→Y prediction.
            pearson_z_t (torch.Tensor): Correlation between Z and T.
            mse_xt (torch.Tensor, optional): MSE loss for X,T→Y prediction. Defaults to None.
            
        Returns:
            None
        """

        self.z_optimizer.zero_grad()
        self.c_optimizer.zero_grad()

        if self.use_pcgrad:
            z_loss_list = []
            c_loss_list = []
            if self.criterion.c_pearson_loss_alpha > 0:
                c_loss_list.append(c_loss)
            if self.criterion.z_pearson_loss_alpha > 0:
                z_loss_list.append(z_loss)
            if self.criterion.pearson_matrix_alpha > 0:
                # Do we want both here?
                z_loss_list.append(pearson_matrix_loss)
                c_loss_list.append(pearson_matrix_loss)
            if self.criterion.t_hat_alpha > 0:
                z_loss_list.append(t_hat_loss)
            if self.criterion.c_mse_loss_alpha > 0:
                c_loss_list.append(mse_c)
            if self.criterion.z_t_loss_alpha > 0:
                z_loss_list.append(pearson_z_t)
            if self.criterion.feature_loss.kl_loss_coeff > 0 or self.criterion.feature_loss.feature_corr_loss_coeff > 0:
                z_loss_list.append(kl_z)
                c_loss_list.append(kl_c) 
            if self.criterion.train_xt_net:
                z_loss_list.append(mse_xt)

            self.z_optimizer.pc_backward(z_loss_list)
            self.c_optimizer.pc_backward(c_loss_list)
        else:
            if self.criterion.c_mse_loss_alpha > 0: mse_c.backward(retain_graph=True)
            if self.criterion.z_pearson_loss_alpha > 0: z_loss.backward(retain_graph=True)
            if self.criterion.feature_loss.kl_loss_coeff > 0 or self.criterion.feature_loss.feature_corr_loss_coeff > 0: kl_c.backward(retain_graph=True)
            if self.criterion.pearson_matrix_alpha > 0: pearson_matrix_loss.backward(retain_graph=True)
            if self.criterion.t_hat_alpha > 0: t_hat_loss.backward(retain_graph=True)
            if self.criterion.z_t_loss_alpha > 0: pearson_z_t.backward(retain_graph=True)
            if self.criterion.z_pearson_loss_alpha > 0: z_loss.backward(retain_graph=True)
            if self.criterion.feature_loss.kl_loss_coeff > 0 or self.criterion.feature_loss.feature_corr_loss_coeff > 0: kl_z.backward()
            if self.criterion.train_xt_net: mse_xt.backward()

        # print(f"Gradients - C: {[p.grad for p in self.model.c.parameters() if p.grad is not None]}, Z: {[p.grad for p in self.model.z.parameters() if p.grad is not None]}, T_hat: {[p.grad for p in self.model.t_hat.parameters() if p.grad is not None]}")
        # Check if gradients are none
        # if any(torch.isnan(p.grad).any() for p in self.model.c.parameters()) or any(torch.isnan(p.grad).any() for p in self.model.z.parameters()) or any(torch.isnan(p.grad).any() for p in self.model.t_hat.parameters()):
        #     # import pdb; pdb.set_trace()
        #     raise ValueError("Some gradients are NaN")
        self.c_optimizer.step()
        self.z_optimizer.step()

    def val_step(self, val_X, val_t, val_y):
        """
        Perform a validation step to compute all losses on validation data.
        
        Evaluates the model on validation data without updating parameters,
        computing all loss components and CATE loss if true_cate is available.
        
        Args:
            val_X (torch.Tensor): Validation input features.
            val_t (torch.Tensor): Validation treatment indicators.
            val_y (torch.Tensor): Validation outcome values.
            
        Returns:
            tuple: (c_loss, z_loss, t_hat_loss, pearson_matrix_loss, mse_c, 
                   pearson_z_t, kl_c, feature_loss_c, kl_z, feature_loss_z, 
                   total_loss, val_cate_loss, mse_xt)
        """
        self.model.eval()
        val_X, val_t, val_y = self._to_device(val_X, val_t, val_y)
        c, z, t_hat, c_y, x_t_y = self.model(val_X, val_t)
        c_loss, z_loss, pearson_matrix_loss, t_hat_loss, kl_c, feature_loss_c, kl_z, feature_loss_z, mse_c, pearson_z_t, mse_xt = self.criterion.forward(c, z, 
                                                                                                                t_hat, c_y, x_t_y,
                                                                                                                val_X, val_y, val_t)
        total_loss = self.criterion.c_pearson_loss_alpha * c_loss + \
                             self.criterion.z_pearson_loss_alpha * z_loss + \
                             self.criterion.pearson_matrix_alpha * pearson_matrix_loss + \
                             self.criterion.t_hat_alpha * t_hat_loss + \
                             self.criterion.feature_loss.kl_loss_coeff * kl_c + \
                             self.criterion.feature_loss.feature_corr_loss_coeff * feature_loss_c + \
                             self.criterion.feature_loss.kl_loss_coeff * kl_z + \
                             self.criterion.feature_loss.feature_corr_loss_coeff * feature_loss_z + \
                             self.criterion.c_mse_loss_alpha * mse_c + \
                             self.criterion.z_t_loss_alpha * pearson_z_t
        
        val_cate_loss = self.cate_loss(c_y, val_t)

        self.model.train()
        return c_loss, z_loss, t_hat_loss, pearson_matrix_loss, mse_c, pearson_z_t, kl_c, feature_loss_c, kl_z, feature_loss_z, total_loss, val_cate_loss, mse_xt


    def fit(self, X, t, y, num_epochs=50, batch_size = 50, plot_losses=False, 
            val_X = None, val_t = None, val_y = None, use_early_stopping=False):
        """
        Train the ZNet model on the provided data.
        
        Trains both the X,T→Y pretraining network and the main ZNet model with
        disentangled C and Z representations. Supports validation monitoring,
        early stopping, and loss plotting.
        
        Args:
            X (torch.Tensor): Training input features of shape (n_samples, input_dim).
            t (torch.Tensor): Training treatment indicators of shape (n_samples, 1).
            y (torch.Tensor): Training outcome values of shape (n_samples, 1).
            num_epochs (int): Number of training epochs. Defaults to 50.
            batch_size (int): Batch size for training. Defaults to 50.
            plot_losses (bool): Whether to plot loss curves after training. Defaults to False.
            val_X (torch.Tensor, optional): Validation input features. Defaults to None.
            val_t (torch.Tensor, optional): Validation treatment indicators. Defaults to None.
            val_y (torch.Tensor, optional): Validation outcome values. Defaults to None.
            use_early_stopping (bool): Whether to use early stopping based on validation loss. Defaults to False.
            
        Returns:
            tuple: (all_losses, z_losses, cumulative_matrix_losses, t_hat_losses, 
                   mse_c_losses, pearson_z_t_losses) - Training history for each loss component.
        """
        X, t, y = self._to_device(X, t, y)
        self.train_x_t_y(X, t, y, verbose=True, epochs=num_epochs)
        self.model.train()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, delta=.001, verbose=False)

        loss_plotter = ZNetLossPlotter()
        for epoch in range(num_epochs):
            indices = torch.randperm(X.shape[0], device=X.device) 
            
            self.model.train()
            
            for i in range(0, X.shape[0], batch_size):
                batch_X, batch_y, batch_t = X[indices[i:i+batch_size]], y[indices[i:i+batch_size]], t[indices[i:i+batch_size]]
                # batch_X, batch_t, batch_y = self._to_device(batch_X, batch_t, batch_y)
                c, z, t_hat, c_y, x_t_y = self.model(batch_X, batch_t)
                c_loss, z_loss, pearson_matrix_loss, t_hat_loss, kl_c, feature_loss_c, kl_z, feature_loss_z, mse_c, pearson_z_t, mse_xt = self.criterion.forward(c, z, t_hat, c_y, x_t_y, batch_X, batch_y, batch_t)

                # print(f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size+1}, C Loss: {c_loss.item()}, Z Loss: {z_loss.item()}, T Hat Loss: {t_hat_loss.item()}, "
                #       f"Pearson Matrix Loss: {pearson_matrix_loss.item()}, MSE C: {mse_c.item()}, "
                #       f"Pearson Z T: {pearson_z_t.item()}, KL C: {kl_c.item()}, KL Z: {kl_z.item()}")
                self.step_optimizers(self.criterion.c_pearson_loss_alpha * c_loss, # Correlation loss for C + Y (maximize)
                                     self.criterion.z_pearson_loss_alpha * z_loss, # Correlation loss for Z + (Y - Y|X, T) (minimize)
                                     self.criterion.pearson_matrix_alpha * pearson_matrix_loss, # Correlation loss for C and Z (minimize)
                                     self.criterion.t_hat_alpha * t_hat_loss, # Cross entropy loss for T_hat (maximize)
                                     self.criterion.feature_loss.kl_loss_coeff * kl_c + self.criterion.feature_loss.feature_corr_loss_coeff * feature_loss_c, # KL + feature correlation loss for C
                                     self.criterion.feature_loss.kl_loss_coeff * kl_z + self.criterion.feature_loss.feature_corr_loss_coeff * feature_loss_z, # KL + feature correlation loss for Z
                                     self.criterion.c_mse_loss_alpha * mse_c, # MSE loss for C + Y (maximize)
                                     self.criterion.z_t_loss_alpha * pearson_z_t, # Correlation loss for Z and T (maximize)
                                     mse_xt) # MSE loss for X and T (maximize if train_xt_net is True)

                # Calculate train cate loss
                train_cate_loss = self.cate_loss(c_y, batch_t)

                # Calculate total loss
                total_loss = self.criterion.c_pearson_loss_alpha * c_loss + \
                             self.criterion.z_pearson_loss_alpha * z_loss + \
                             self.criterion.pearson_matrix_alpha * pearson_matrix_loss + \
                             self.criterion.t_hat_alpha * t_hat_loss + \
                             self.criterion.feature_loss.kl_loss_coeff * kl_c + \
                             self.criterion.feature_loss.feature_corr_loss_coeff * feature_loss_c + \
                             self.criterion.feature_loss.kl_loss_coeff * kl_z + \
                             self.criterion.feature_loss.feature_corr_loss_coeff * feature_loss_z + \
                             self.criterion.c_mse_loss_alpha * mse_c + \
                             self.criterion.z_t_loss_alpha * pearson_z_t
                
                # Track batch losses
                loss_plotter.add_train_batch(c_loss, 
                                             z_loss, 
                                             t_hat_loss, 
                                             pearson_matrix_loss, 
                                             mse_c, pearson_z_t, 
                                             kl_c, 
                                             feature_loss_c, 
                                             kl_z,
                                             feature_loss_z, 
                                             total_loss, 
                                             mse_xt,
                                             train_cate_loss)
            # print(f"Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size+1}, Total Loss: {total_loss.item()}")
            # print(f"  C Loss: {c_loss.item()}, Z Loss: {z_loss.item()}, T Hat Loss: {t_hat_loss.item()}, "
            #         f"Pearson Matrix Loss: {pearson_matrix_loss.item()}, MSE C: {mse_c.item()}, "
            #         f"Pearson Z T: {pearson_z_t.item()}, KL C: {kl_c.item()}, KL Z: {kl_z.item()}")   
            # print()
            # Track epoch losses
            loss_plotter.train_step()
            if val_X is not None:
                val_X, val_t, val_y = self._to_device(val_X, val_t, val_y)
                c_loss, z_loss, t_hat_loss, pearson_matrix_loss, mse_c, pearson_z_t, kl_c, feature_loss_c, kl_z, feature_loss_z, total_loss, val_cate_loss, mse_xt = self.val_step(val_X, val_t, val_y)
                loss_plotter.val_step(c_loss, z_loss, t_hat_loss, pearson_matrix_loss, mse_c, pearson_z_t, kl_c, feature_loss_c, kl_z, feature_loss_z, total_loss, mse_xt, val_cate_loss)

            early_stopping.check_early_stop(np.mean(total_loss.detach().cpu().numpy().item()))

            if use_early_stopping and early_stopping.stop_training:
                if plot_losses:
                    print(f"Early stopping at epoch {epoch}")
                break

        if plot_losses:
            loss_plotter.plot_losses(self.criterion.c_mse_loss_alpha, 
                                     self.criterion.t_hat_alpha, 
                                     self.criterion.z_pearson_loss_alpha, 
                                     self.criterion.c_pearson_loss_alpha, 
                                     self.criterion.z_t_loss_alpha, 
                                     self.criterion.feature_loss.kl_loss_coeff,
                                     self.criterion.feature_loss.feature_corr_loss_coeff,
                                     self.criterion.pearson_matrix_alpha, 
                                     self.criterion.train_xt_net,
                                     self.true_cate)

        return loss_plotter.all_losses, loss_plotter.z_losses, loss_plotter.cumulative_matrix_losses, loss_plotter.t_hat_losses, loss_plotter.mse_c_losses, loss_plotter.pearson_z_t_losses

    def get_generated_data(self, X, t):
        """
        Generate C and Z representations for given input data.
        
        Runs forward pass through the trained ZNet model to extract learned
        confounder (C) and instrumental variable (Z) representations.
        
        Args:
            X (np.ndarray or torch.Tensor): Input features of shape (n_samples, input_dim).
            t (np.ndarray or torch.Tensor): Treatment indicators of shape (n_samples, 1).
            
        Returns:
            tuple: (c, z, t_hat, c_y, x_t_y) as numpy arrays:
                - c: Confounder representation (n_samples, c_dim)
                - z: Instrumental variable representation (n_samples, z_dim)
                - t_hat: Predicted treatment from Z (n_samples, 1)
                - c_y: Predicted outcome from C and t (n_samples, 1)
                - x_t_y: Predicted outcome from X and t (n_samples, 1)
        """
        self.model.eval()
        if type(X) == np.ndarray:
            X = torch.from_numpy(X.astype(np.float32))
        if type(t) == np.ndarray:
            t = torch.from_numpy(t.astype(np.float32))
        X, t = self._to_device(X, t)
        with torch.no_grad():
            c, z, t_hat, c_y, x_t_y = self.model(X, t)
        return c.detach().cpu().numpy(), z.detach().cpu().numpy(), t_hat.detach().cpu().numpy(), c_y.detach().cpu().numpy(), x_t_y.detach().cpu().numpy()
    
