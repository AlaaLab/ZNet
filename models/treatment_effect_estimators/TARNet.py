#######################################################################################
# Author: Jenna Fields, Franny Dean
# Script: TARNet.py
# Function: Run TARNet model on datasets
# Date: 02/06/2026
#######################################################################################
from seed_utils import set_seed

set_seed(42)

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from models.treatment_effect_estimators.parent_class import DownstreamParent

#######################################################################################
class TARNetModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_phi=200, hidden_dim_h=100):
        """
        TARNet representation and outcome heads.
        
        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Outcome dimension.
            hidden_dim_phi (int): Hidden dimension for representation. Defaults to 200.
            hidden_dim_h (int): Hidden dimension for outcome heads. Defaults to 100.
        """
        super(TARNetModel, self).__init__()
        self.nonlinearity = nn.ELU()
        self.phi = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_phi), 
            self.nonlinearity,
            nn.Linear(hidden_dim_phi, hidden_dim_phi), 
            self.nonlinearity,
            nn.Linear(hidden_dim_phi, input_dim),
            nn.BatchNorm1d(input_dim, affine=False))

        self.h_0 = nn.Sequential(nn.Linear(input_dim, hidden_dim_h), 
                                 self.nonlinearity,
                                 nn.Linear(hidden_dim_h, hidden_dim_h), 
                                 self.nonlinearity,
                                 nn.Linear(hidden_dim_h, output_dim))

        self.h_1 = nn.Sequential(nn.Linear(input_dim, hidden_dim_h), 
                                 self.nonlinearity,
                                 nn.Linear(hidden_dim_h, hidden_dim_h), 
                                 self.nonlinearity,
                                 nn.Linear(hidden_dim_h, output_dim))
        
        self.h_layers = nn.ModuleList([self.h_0, self.h_1])

    def forward(self, input):
        """
        Forward pass through TARNet.
        
        Args:
            input (torch.Tensor): Input features.
        
        Returns:
            tuple: (phi, h0, h1) representation and outcome heads.
        """
        output = self.phi(input)
        return output, self.h_0(output), self.h_1(output)

class TARNetLoss(nn.Module):
    def __init__(self, alpha, use_t_hat, set_eps=.5):
        """
        TARNet loss with optional IPM regularization.
        
        Args:
            alpha (float): IPM loss weight.
            use_t_hat (bool): Use learned threshold for treatment indicator.
            set_eps (float): Initial threshold for treatment split. Defaults to 0.5.
        """
        super(TARNetLoss, self).__init__()
        self.L = nn.MSELoss(reduction='none')
        self.use_t_hat = use_t_hat
        # if use_t_hat:
        self.epsilon = nn.Parameter(torch.tensor(set_eps)) if use_t_hat else torch.tensor(set_eps)
        self.epsilon_coeff = nn.Parameter(torch.tensor(100.0)) if use_t_hat else torch.tensor(100.0)
        self.alpha = alpha

    def forward(self, batch_phi, batch_h0, batch_h1, t, w, y) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute factual loss and IPM regularization.
        
        Args:
            batch_phi (torch.Tensor): Representation outputs.
            batch_h0 (torch.Tensor): Control outcome head outputs.
            batch_h1 (torch.Tensor): Treated outcome head outputs.
            t (torch.Tensor): Treatment indicators.
            w (torch.Tensor): Sample weights.
            y (torch.Tensor): Outcomes.
        
        Returns:
            tuple: (factual_loss, ipm_loss)
        """

        if self.use_t_hat:
            t = torch.sigmoid(self.epsilon_coeff * (t - self.epsilon))
            # t = torch.where(t<.5, 0, 1)


            batch_h0 = batch_h0 * (1 - t)
            batch_h1 = batch_h1 * t
        else:
            batch_h0 = batch_h0 * (torch.where(t < self.epsilon, 1, 0))
            batch_h1 =  batch_h1 * (torch.where(t >= self.epsilon, 1, 0))

        # Calculate IPM
        p0_idx = torch.where(t < .5)[0]
        p1_idx = torch.where(t >= .5)[0]
        batch_phi0 =  batch_phi[p0_idx] if len(p0_idx) > 0 else torch.tensor(0, dtype=torch.float)
        batch_phi1 =  batch_phi[p1_idx] if len(p1_idx) > 0 else torch.tensor(0, dtype=torch.float)
        
        norm = torch.linalg.vector_norm(torch.mean(batch_phi0, dim=0) - torch.mean(batch_phi1, dim=0))
        IPM_loss = (2 * norm) ** 2
        
        loss = torch.mean(w * self.L(batch_h0 + batch_h1, y))

        return loss, self.alpha * IPM_loss

class TARNetTrainer():
    def __init__(self, input_dim, output_dim, alpha=0, lr=0.001, 
                 weight_decay=.1, use_t_hat=True, eps_opt_info = [0.01, 0.01], set_eps=.5, 
                 true_cate = None):
        """
        Trainer for TARNet model.
        
        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Outcome dimension.
            alpha (float): IPM loss weight. Defaults to 0.
            lr (float): Learning rate. Defaults to 0.001.
            weight_decay (float): L2 regularization. Defaults to 0.1.
            use_t_hat (bool): Use learned treatment threshold. Defaults to True.
            eps_opt_info (list): [lr, weight_decay] for epsilon optimizer.
            set_eps (float): Initial treatment threshold. Defaults to 0.5.
            true_cate (array-like, optional): True CATE for evaluation.
        """
        self.model = TARNetModel(input_dim, output_dim)
        self.lr = lr
        # If alpha > 0, then we will use the IPM Loss component 
        self.criterion = TARNetLoss(alpha, use_t_hat=use_t_hat, set_eps=set_eps)
        self.weight_decay = weight_decay
        self.phi_optimizer = torch.optim.Adam(self.model.phi.parameters(), lr=self.lr)
        self.h_optimizer = torch.optim.Adam(self.model.h_layers.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.use_t_hat = use_t_hat
        self.true_cate = true_cate
        if use_t_hat:
            self.epsilon_optimizer = torch.optim.Adam([self.criterion.epsilon, self.criterion.epsilon_coeff], lr=eps_opt_info[0], weight_decay=eps_opt_info[1])
        
    
    def calculate_loss(self, X_batch, t_batch, w_batch, y_batch, eval_mode=False):
        """
        Compute TARNet loss components for a batch.
        
        Args:
            X_batch (torch.Tensor): Input features.
            t_batch (torch.Tensor): Treatments.
            w_batch (torch.Tensor): Weights.
            y_batch (torch.Tensor): Outcomes.
            eval_mode (bool): Set model to eval mode if True.
        
        Returns:
            tuple: (factual_loss, ipm_loss)
        """
        if eval_mode:
            self.model.eval()

        output_phi, output_h0, output_h1 = self.model(X_batch)

        return self.criterion.forward(output_phi, output_h0, output_h1, t_batch, w_batch, y_batch)
    
    def step_optimizers(self, loss : torch.Tensor, IPM_loss : torch.Tensor):
        """
        Backpropagate and update TARNet optimizers.
        
        Args:
            loss (torch.Tensor): Factual loss.
            IPM_loss (torch.Tensor): IPM regularization loss.
        
        Returns:
            torch.Tensor: Total loss.
        """
        self.phi_optimizer.zero_grad()
        self.h_optimizer.zero_grad()
        if self.use_t_hat:
            self.epsilon_optimizer.zero_grad()

        loss.backward(retain_graph=True)

        self.h_optimizer.step()
        if self.use_t_hat:
            self.epsilon_optimizer.step()

        IPM_loss.backward()

        self.phi_optimizer.step()


        return loss + IPM_loss
    
    def calc_w(self, t):
        """
        Compute sample weights for treatment imbalance.
        
        Args:
            t (torch.Tensor): Treatment indicators.
        
        Returns:
            torch.Tensor: Weights for each sample.
        """
        u = torch.mean(t)
        w = t / (2 * u) + (1 - t) / (2 * (1 - u))
        self.w = w
        return w

    def fit(self, X, t, y, num_epochs=50, batch_size = 50, plot_losses=False,
            val_X = None, val_t = None, val_y = None):
        """
        Train TARNet on input data.
        
        Args:
            X (torch.Tensor): Input features.
            t (torch.Tensor): Treatments.
            y (torch.Tensor): Outcomes.
            num_epochs (int): Number of epochs. Defaults to 50.
            batch_size (int): Batch size. Defaults to 50.
            plot_losses (bool): Plot training curves. Defaults to False.
            val_X (torch.Tensor, optional): Validation features.
            val_t (torch.Tensor, optional): Validation treatments.
            val_y (torch.Tensor, optional): Validation outcomes.
        
        Returns:
            tuple: (losses, epsilon, epsilon_coeff) over training.
        """
        self.model.train()

        all_losses = []
        all_losses_val = []
        epsilon_over_time = []
        epsilon_coeff_over_time = []
        all_cate_loss = []
        all_cate_loss_val = []
        w = self.calc_w(t)
        if val_t is not None:
            val_w = self.calc_w(val_t)
        for epoch in range(num_epochs):
            cumulative_loss = []
            cumulative_cate_loss = []
            indices = torch.randperm(X.shape[0]) 
            for i in range(0, X.shape[0], batch_size):
                output_phi, output_h0, output_h1 = self.model(X[indices[i:i+batch_size]])
                loss, IPM_loss = self.criterion.forward(output_phi, output_h0, output_h1, 
                                                        t[indices[i:i+batch_size]], w[indices[i:i+batch_size]], 
                                                        y[indices[i:i+batch_size]])
                total_loss = self.step_optimizers(loss, IPM_loss)
                
                cumulative_loss.append((total_loss).detach().cpu().numpy().item())
                if self.use_t_hat:
                    epsilon_over_time.append(self.criterion.epsilon.detach().cpu().tolist())
                    epsilon_coeff_over_time.append(self.criterion.epsilon_coeff.detach().cpu().tolist())
                
                if self.true_cate is not None:
                    cate_est = output_h1 - output_h0
                    cate_loss = torch.mean((self.true_cate - cate_est) ** 2)
                    cumulative_cate_loss.append(cate_loss.detach().cpu().numpy().item())

                
            all_losses.append(np.mean(cumulative_loss))
            all_cate_loss.append(np.mean(cumulative_cate_loss))
            if val_X is not None:
                output_phi, output_h0, output_h1 = self.model(val_X)
                val_loss, _ = self.criterion.forward(output_phi, output_h0, output_h1, 
                                                    val_t, val_w, 
                                                    val_y)
                
                all_losses_val.append((val_loss).detach().cpu().numpy().item())

                if self.true_cate is not None:
                    cate_est = output_h1 - output_h0
                    cate_loss = torch.mean((self.true_cate - cate_est) ** 2)
                    all_cate_loss_val.append(cate_loss.detach().cpu().numpy().item())

        if plot_losses:
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].plot(all_losses, label='All Loss')
            axes[0].plot(all_losses_val, label='All Loss (Validation)')
            axes[0].legend()
            axes[1].plot(all_cate_loss, label='CATE Loss')
            axes[1].plot(all_cate_loss_val, label='CATE Loss (Validation)')
            axes[1].legend()
            plt.show()

        return all_losses, epsilon_over_time, epsilon_coeff_over_time
    
    def e_PEHE_loss(self, X, ite):
        """
        Compute PEHE error on ITE predictions.
        
        Args:
            X (torch.Tensor): Input features.
            ite (torch.Tensor): True ITE values.
        
        Returns:
            torch.Tensor: PEHE loss.
        """
        self.model.eval()
        _, output_h0, output_h1 = self.model(X)
        tau_hat = output_h1 - output_h0
        return torch.sqrt(torch.mean(((tau_hat - ite) ** 2)))
    
    def CATE_estimate(self, X):
        """
        Estimate CATE from TARNet outputs.
        
        Args:
            X (torch.Tensor): Input features.
        
        Returns:
            torch.Tensor: Mean treatment effect.
        """
        self.model.eval()
        _, output_h0, output_h1 = self.model(X)
        tau_hat = output_h1 - output_h0
        return torch.mean(tau_hat)
    
    def e_ATE_loss(self, X, ite):
        """
        Compute ATE error against true ITE.
        
        Args:
            X (torch.Tensor): Input features.
            ite (torch.Tensor): True ITE values.
        
        Returns:
            torch.Tensor: Absolute ATE error.
        """
        self.model.eval()
        return torch.abs(self.CATE_estimate(X) - torch.mean(ite))
    
    def e_ATT_loss(self, X, t, outcome):
        """
        Compute ATT error against TARNet estimate.
        
        Args:
            X (torch.Tensor): Input features.
            t (torch.Tensor): Treatments.
            outcome (torch.Tensor): Outcomes.
        
        Returns:
            torch.Tensor: Absolute ATT error.
        """
        self.model.eval()

        p0_idx = torch.where(t==0)[0]
        p1_idx = torch.where(t==1)[0]
        X_1 =  X[p1_idx] if len(p1_idx) > 0 else torch.tensor(0, dtype=torch.float)
        outcome_0 =  outcome[p0_idx] if len(p0_idx) > 0 else torch.tensor(0, dtype=torch.float)
        outcome_1 =  outcome[p1_idx] if len(p1_idx) > 0 else torch.tensor(0, dtype=torch.float)

        att = torch.mean(outcome_1) - torch.mean(outcome_0)

        e_att = torch.abs(att - self.CATE_estimate(X_1))
        return e_att
    
    def CATE_z_score(self, X):
        """
        Compute z-score and p-value for CATE estimate.
        
        Args:
            X (torch.Tensor): Input features.
        
        Returns:
            tuple: (ate, z_score, p_value_two_tailed)
        """
        self.model.eval()
        
        _, output_h0, output_h1 = self.model(X)
        ite = output_h1 - output_h0
        ate = torch.mean(ite)

        z_score =(ate / (torch.std(ite) / np.sqrt(len(X)))).detach().cpu().numpy()

        p_value_two_tailed = 2 * (1 - stats.norm.cdf(abs(z_score)))
        # print(ate, z_score, p_value_two_tailed)
        return ate, z_score, p_value_two_tailed

    def evaluate(self, X, t, y, ite):
        """
        Evaluate TARNet with ATE and PEHE metrics.
        
        Args:
            X (torch.Tensor): Input features.
            t (torch.Tensor): Treatments.
            y (torch.Tensor): Outcomes.
            ite (torch.Tensor): True ITE values.
        
        Returns:
            dict: Metrics including ATE, PEHE, and p-values.
        """
        ate, z_score, p_value_two_tailed = self.CATE_z_score(X)
        pehe_loss = self.e_PEHE_loss(X, ite)
        ate_loss = self.e_ATE_loss(X, ite)
        return {'ate' : ate.detach().cpu().item(), 'ate_z_score': z_score, 'ate_p_value' : p_value_two_tailed, 
                'pehe_loss' : pehe_loss.detach().cpu().item(), 'ate_loss' : ate_loss.detach().cpu().item()}
    
    def predict_ite(self, x):
        """
        Predict individual treatment effects.
        
        Args:
            x (np.ndarray): Input features.
        
        Returns:
            np.ndarray: Predicted ITEs.
        """
        self.model.eval()
        x = torch.from_numpy(x.astype(np.float32))
        _, Y_hat0, Y_hat1 = self.model(x)
        tau_hat = Y_hat1 - Y_hat0
        return tau_hat.detach().cpu().numpy()
    
    def predict_outcome(self, x, t):
        """
        Predict outcomes under observed treatment.
        
        Args:
            x (np.ndarray): Input features.
            t (np.ndarray): Treatment indicators.
        
        Returns:
            np.ndarray: Predicted outcomes.
        """
        self.model.eval()
        x = torch.from_numpy(x.astype(np.float32))
        t = torch.from_numpy(t.astype(np.float32))
        _, Y_hat0, Y_hat1 = self.model(x)

        return torch.where(t == 1, Y_hat1, Y_hat0).detach().cpu().numpy()
    
    def factual_loss(self, x, z, t, y):
        """
        Compute factual loss for downstream evaluation.
        
        Args:
            x (np.ndarray): Input features.
            z (np.ndarray): Instruments (unused).
            t (np.ndarray): Treatments.
            y (np.ndarray): Outcomes.
        
        Returns:
            float: Factual loss value.
        """
        x = torch.from_numpy(x.astype(np.float32))
        t = torch.from_numpy(t.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32)) 
        w = self.calc_w(t)
        return self.calculate_loss(x, t, w, y, eval_mode=True)[0].detach().cpu().numpy().item()
    

class TARNet(DownstreamParent):
    def __init__(self, data, config):
        """
        Train TARNet and wrap in DownstreamParent interface.
        
        Args:
            data (ParentDataset): Dataset providing train/val splits.
            config (dict): Model and training parameters with tarnet prefixes.
        """
        train_params = {k : v for k, v in config.items() if k.startswith('tarnet_params_train')}
        model_params = {k : v for k, v in config.items() if k.startswith('tarnet_params_model')}
        model = TARNetTrainer(len(data.x_cols), 
                                  data.y.shape[-1], 
                                  **model_params
                                  )
        train_data = data.train(data_type='torch')
        val_data = data.val(data_type='torch')
        model.fit(train_data.x, train_data.t, train_data.y, 
                val_X = val_data.x, val_t= val_data.t, val_y = val_data.y,
                **train_params)
        super().__init__('tarnet', model)
        
