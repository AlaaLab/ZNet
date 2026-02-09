#######################################################################################
# Author: Jenna Fields, Franny Dean
# Script: ZNet_ECG.py
# Function: ZNet model wrapper for ECGs
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

from models.ZNet.model_loss_utils import ZNetLoss, ZNetECGModel, X_T_Y_Model, X_T_Y_ECGModel, EarlyStopping, CateLoss, PearsonLoss, KDEMutualInformation
from models.ZNet.loss_plotting import ZNetECGLossPlotter
from models.ZNet.pcgrad import PCGrad

#######################################################################################
class ZNetECG():
    def __init__(self, 
                 input_dim, 
                 c_dim, 
                 z_dim, 
                 output_dim, 
                 embedded_dim=64,
                 ecg_channels=12,
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
        Initialize the ZNet ECG model for instrumental variable learning. The only difference from base ZNet is the data set up for high dimensional inputs.
        
        Learns confounder (C) and instrument (Z) representations from ECG inputs
        and predicts treatment and outcome via disentangled objectives.
        
        Args:
            input_dim (int): ECG input dimensionality.
            c_dim (int): Confounder representation dimension.
            z_dim (int): Instrument representation dimension.
            output_dim (int): Outcome dimension.
            embedded_dim (int): ECG embedding dimension. Defaults to 64.
            ecg_channels (int): Number of ECG channels. Defaults to 12.
            lr (float): Learning rate. Defaults to 0.001.
            weight_decay (float): L2 regularization. Defaults to 0.1.
            kl_loss_coeff (float): KL loss coefficient. Defaults to 0.
            feature_corr_loss_coeff (float): Feature correlation loss coefficient. Defaults to 0.
            c_pearson_loss_alpha (float): C-Y correlation weight. Defaults to 1.
            c_mse_loss_alpha (float): C→Y MSE weight. Defaults to 1.
            z_pearson_loss_alpha (float): Z-residual correlation weight. Defaults to 1.
            z_t_loss_alpha (float): Z-T correlation weight. Defaults to 0.
            pearson_matrix_alpha (float): C-Z independence weight. Defaults to 0.
            t_hat_alpha (float): Treatment prediction weight. Defaults to 1.
            true_cate (array-like, optional): True CATE for evaluation. Defaults to None.
            binary_outcome (bool): Whether outcome is binary. Defaults to False.
            is_linear (bool): Whether to use linear heads. Defaults to False.
            use_sm (bool): Use softmax on C and Z. Defaults to True.
            sm_temp (float): Softmax temperature. Defaults to 2.
            use_pcgrad (bool): Use PCGrad optimization. Defaults to False.
            train_xt_net (bool): Train X,T→Y network. Defaults to False.
            use_mi_corr_loss (bool): Use mutual information correlation loss. Defaults to False.
            use_mi_matrix_loss (bool): Use MI for C-Z independence. Defaults to False.
            device (str, optional): Device override (e.g., 'cuda'). Defaults to None.
            pretrain_xty_model (bool): Pretrain X,T→Y model separately. Defaults to True.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
            
        self.pretrain_xty_model = pretrain_xty_model
        
        if self.pretrain_xty_model:
            self.x_t_y = X_T_Y_ECGModel(input_dim, output_dim, hidden_dim_h=64, is_linear=is_linear).to(self.device)

        self.model = ZNetECGModel(input_dim, c_dim, z_dim, output_dim, 
                                  embedded_dim=embedded_dim, ecg_channels=ecg_channels,
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
            raise NotImplementedError(
                "PCGrad is not supported for this module."
            )
    
    def train_x_t_y(self, train_loader, verbose=False, device='cpu'):
        """
        Fit the ECG X,T→Y model when pretrained separately.
        
        Args:
            train_loader (DataLoader): ECG training loader with keys 'X', 't', 'y'.
            verbose (bool): Whether to print training progress. Defaults to False.
            device (str): Device for training. Defaults to 'cpu'.
        """ 
        self.x_t_y.train()
        self.x_t_y.fit(train_loader, verbose=verbose, device=device)            
        self.x_t_y.eval()  # Set the model to evaluation mode
        for param in self.x_t_y.parameters():
            param.requires_grad = False
            
        #if verbose:
        #    print(f"Final loss after training X_T_Y_Model: {loss.item()}")
    
    def calculate_loss(self, X_batch, t_batch, w_batch, y_batch, eval_mode=False):
        """
        Compute loss components for a batch.
        
        Args:
            X_batch (torch.Tensor): ECG inputs.
            t_batch (torch.Tensor): Treatments.
            w_batch (torch.Tensor): Sample weights.
            y_batch (torch.Tensor): Outcomes.
            eval_mode (bool): Set model to eval mode if True. Defaults to False.
        
        Returns:
            tuple: Loss components from criterion.
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
        Backpropagate and update optimizers for all loss components.
        
        Args:
            c_loss (torch.Tensor): C-Y correlation loss.
            z_loss (torch.Tensor): Z-residual correlation loss.
            pearson_matrix_loss (torch.Tensor): C-Z independence loss.
            t_hat_loss (torch.Tensor): Treatment prediction loss.
            kl_c (torch.Tensor): KL loss for C.
            kl_z (torch.Tensor): KL loss for Z.
            mse_c (torch.Tensor): C→Y MSE loss.
            pearson_z_t (torch.Tensor): Z-T correlation loss.
            mse_xt (torch.Tensor, optional): X,T→Y MSE loss.
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

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.c_optimizer.step()
        self.z_optimizer.step()

    def val_step(self, val_loader):
        """
        Run a validation epoch over a loader.
        
        Args:
            val_loader (DataLoader): Validation loader with ECG batches.
        
        Returns:
            tuple: Aggregated loss components for validation.
        """
        self.model.eval()
        c_loss_sum = 0.0
        z_loss_sum = 0.0
        t_hat_loss_sum = 0.0
        pearson_matrix_loss_sum = 0.0
        mse_c_sum = 0.0
        pearson_z_t_sum = 0.0
        kl_c_sum = 0.0
        feature_loss_c_sum = 0.0
        kl_z_sum = 0.0
        feature_loss_z_sum = 0.0
        total_loss_sum = 0.0
        val_cate_loss_sum = 0.0
        mse_xt_sum = 0.0

        n_batches = 0

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation", disable=True)
            for batch in progress_bar:
                
                val_X, val_t, val_y = batch['X'], batch['t'], batch['y']
           
                val_X, val_t, val_y = val_X.to(self.device), val_t.to(self.device),val_y.to(self.device)
                
                if val_t.dim() == 1:
                    val_t = val_t.unsqueeze(-1)
                if val_y.dim() == 1:
                    val_y = val_y.unsqueeze(-1)
                c, z, t_hat, c_y, x_t_y = self.model(val_X, val_t)

                c_loss, z_loss, pearson_matrix_loss, t_hat_loss, kl_c, feature_loss_c, kl_z, feature_loss_z, mse_c, pearson_z_t, mse_xt = self.criterion.forward(c, z, t_hat, c_y, x_t_y,val_X, val_y, val_t)
        
        
       
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
                # print('val_cate_loss, c_y, val_t: ', val_cate_loss, c_y, val_t)
            
                # Accumulate
                c_loss_sum += c_loss.item()
                z_loss_sum += z_loss.item()
                t_hat_loss_sum += t_hat_loss.item()
                pearson_matrix_loss_sum += pearson_matrix_loss.item()
                mse_c_sum += mse_c.item()
                pearson_z_t_sum += pearson_z_t.item()
                kl_c_sum += kl_c.item()
                feature_loss_c_sum += feature_loss_c.item()
                kl_z_sum += kl_z.item()
                feature_loss_z_sum += feature_loss_z.item()
                total_loss_sum += total_loss.item()
                if val_cate_loss is not None:
                    val_cate_loss_sum += val_cate_loss.item()

                if mse_xt is not None:
                    mse_xt_sum += mse_xt.item()
                
                n_batches += 1

        c_loss = c_loss_sum / n_batches
        z_loss = z_loss_sum / n_batches
        t_hat_loss = t_hat_loss_sum / n_batches
        pearson_matrix_loss = pearson_matrix_loss_sum / n_batches
        mse_c = mse_c_sum / n_batches
        pearson_z_t = pearson_z_t_sum / n_batches
        kl_c = kl_c_sum / n_batches
        feature_loss_c = feature_loss_c_sum / n_batches
        kl_z = kl_z_sum / n_batches
        feature_loss_z = feature_loss_z_sum / n_batches
        total_loss = total_loss_sum / n_batches
        try:
            val_cate_loss = val_cate_loss_sum / n_batches
        except:
            val_cate_loss = 0
        mse_xt = mse_xt_sum / n_batches if self.criterion.train_xt_net else None

        self.model.train()
        return c_loss, z_loss, t_hat_loss, pearson_matrix_loss, mse_c, pearson_z_t, kl_c, feature_loss_c, kl_z, feature_loss_z, total_loss, val_cate_loss, mse_xt

    def fit(self, train_loader, num_epochs=50, batch_size = 50, plot_losses=False, 
            val_loader=None, use_early_stopping=False):
        """
        Train the ECG ZNet model.
        
        Args:
            train_loader (DataLoader): Training loader.
            num_epochs (int): Number of epochs. Defaults to 50.
            batch_size (int): Batch size (unused for loader-based training). Defaults to 50.
            plot_losses (bool): Plot training curves. Defaults to False.
            val_loader (DataLoader, optional): Validation loader. Defaults to None.
            use_early_stopping (bool): Enable early stopping. Defaults to False.
        
        Returns:
            tuple: Loss history arrays.
        """
        
        self.train_x_t_y(train_loader, verbose=True, device=self.device)
        self.model.train()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=5, delta=.001, verbose=False)

        loss_plotter = ZNetECGLossPlotter()
        for epoch in range(num_epochs):
            # indices = torch.randperm(X.shape[0], device=X.device) 
            
            # self.model.train()
            
            progress_bar = tqdm(train_loader, desc="Training", disable=True)
            for batch in progress_bar:
                batch_X, batch_t, batch_y = batch['X'], batch['t'], batch['y']
                if batch_t.dim() == 1:
                    batch_t = batch_t.unsqueeze(-1)
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(-1)
    
                batch_X, batch_t, batch_y = batch_X.to(self.device), batch_t.to(self.device), batch_y.to(self.device)
                
                c, z, t_hat, c_y, x_t_y = self.model(batch_X, batch_t)
                c_loss, z_loss, pearson_matrix_loss, t_hat_loss, kl_c, feature_loss_c, kl_z, feature_loss_z, mse_c, pearson_z_t, mse_xt = self.criterion.forward(c, z, t_hat, c_y, x_t_y, batch_X, batch_y, batch_t)

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
                
            # Track epoch losses
            loss_plotter.train_step()
            if val_loader is not None:
                c_loss, z_loss, t_hat_loss, pearson_matrix_loss, mse_c, pearson_z_t, kl_c, feature_loss_c, kl_z, feature_loss_z, total_loss, val_cate_loss, mse_xt = self.val_step(val_loader)
                loss_plotter.val_step(c_loss, z_loss, t_hat_loss, pearson_matrix_loss, mse_c, pearson_z_t, kl_c, feature_loss_c, kl_z, feature_loss_z, total_loss, mse_xt, val_cate_loss)

            early_stopping.check_early_stop(total_loss) #np.mean(total_loss.detach().cpu().numpy().item()))

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

    def get_generated_data(self, data_loader):
        """
        Generate C/Z representations for an ECG loader.
        
        Args:
            data_loader (DataLoader): Loader yielding ECG batches with 'X' and 't'.
        
        Returns:
            tuple: (c, z, t_hat, c_y, x_t_y) as numpy arrays.
        """
        self.model.eval()

        c_list = []
        z_list = []
        t_hat_list = []
        c_y_list = []
        x_t_y_list = []

        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Generating data", disable=True)
            for batch in progress_bar:
                X, t = batch['X'], batch['t']
                X, t  = X.to(self.device), t.to(self.device)

                c, z, t_hat, c_y, x_t_y = self.model(X, t)

                c_list.append(c.detach().cpu())
                z_list.append(z.detach().cpu())
                t_hat_list.append(t_hat.detach().cpu())
                c_y_list.append(c_y.detach().cpu())
                x_t_y_list.append(x_t_y.detach().cpu())

        self.model.train()

        return (
            torch.cat(c_list, dim=0).numpy(),
            torch.cat(z_list, dim=0).numpy(),
            torch.cat(t_hat_list, dim=0).numpy(),
            torch.cat(c_y_list, dim=0).numpy(),
            torch.cat(x_t_y_list, dim=0).numpy(),
        )
