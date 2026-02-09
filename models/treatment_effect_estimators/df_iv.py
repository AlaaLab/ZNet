#######################################################################################
# Original Author: Frauen, Dennis, and Stefan Feuerriegel. 
#          "Estimating individual treatment effects under unobserved confounding using binary instruments." arXiv preprint arXiv:2208.08544 (2022).
# Editors: Jenna Fields
# Script: df_iv.py
# Function: Implementation of DFIV (see Xu 2021 paper)
# Date: 02/06/2026
#######################################################################################
from datetime import datetime
from seed_utils import set_seed

set_seed(42)

import torch
import torch.nn as nn
from models.treatment_effect_estimators import helper
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping
from DGP.dataset_class import ParentDataset
# from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.loggers import CSVLogger
from models.treatment_effect_estimators.parent_class import DownstreamParent

#######################################################################################


class DFIVModel(pl.LightningModule):
    def __init__(self, config, xdim, zdim):
        super().__init__()
        self.lambda1 = config["lambda1"]
        self.lambda2 = config["lambda2"]
        self.dropout = nn.Dropout(config["dropout"])
        self.xdim = xdim
        self.zdim = zdim
        # First stage neural network to build joint instrument+covariate representation
        self.psi = nn.Sequential(
            nn.Linear(xdim + zdim, config["hidden_size_psi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_psi"], config["hidden_size_psi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_psi"], config["hidden_size_psi"]),
            nn.ReLU()
        )
        # Second stage neural network to build treatment representation
        self.phi = nn.Sequential(
            nn.Linear(1, config["hidden_size_phi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_phi"], config["hidden_size_phi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_phi"], config["hidden_size_phi"]),
            nn.ReLU()
        )
        # Third stage neural network to build covariate representation
        self.xi = nn.Sequential(
            nn.Linear(xdim, config["hidden_size_xi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_xi"], config["hidden_size_xi"]),
            nn.ReLU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["hidden_size_xi"], config["hidden_size_xi"]),
            nn.ReLU()
        )

        # Optimization
        self.automatic_optimization = False
        self.optimizer_psi = torch.optim.Adam(self.psi.parameters(), lr=config["lr1"])
        self.optimizer_phi = torch.optim.Adam(self.phi.parameters(), lr=config["lr2"])
        self.optimizer_xi = torch.optim.Adam(self.xi.parameters(), lr=config["lr2"])
        self.save_hyperparameters(config)
        # Target function
        self.mu = None

    # Batchwise outer product vecctor of two vectors
    @staticmethod
    def batch_outer_vec(a, b):
        batch_size = a.size(0)
        dim_a = a.size(1)
        dim_b = b.size(1)
        outer = torch.bmm(torch.unsqueeze(a, 2), torch.unsqueeze(b, 1))
        outer_vec = torch.reshape(outer, (batch_size, dim_a * dim_b))
        return outer_vec

    def configure_optimizers(self):
        return self.optimizer_psi, self.optimizer_phi, self.optimizer_xi

    def compute_V(self, psi, phi):
        batch_size = psi.size(0)
        dim_psi = psi.size(1)
        bracket = torch.matmul(torch.transpose(psi, 0, 1), psi)
        bracket += self.lambda1 * batch_size * torch.eye(n=dim_psi).type_as(bracket)
        try:
            V_hat = torch.matmul(torch.matmul(torch.transpose(phi, 0, 1), psi), torch.inverse(bracket))
        except:
            V_hat = torch.matmul(torch.matmul(torch.transpose(phi, 0, 1), psi), torch.linalg.pinv(bracket))
        return V_hat

    def compute_mu(self, V, psi, xi, Y):
        batch_size = psi.size(0)
        outer_dim = V.size(0) * xi.size(1)
        # Outerproduct of V*psi and xi
        V_psi = torch.transpose(torch.matmul(V, torch.transpose(psi, 0, 1)), 0, 1)
        outer_vec = DFIVModel.batch_outer_vec(V_psi, xi)
        # Plug into ridge regression formula
        bracket = torch.matmul(torch.transpose(outer_vec, 0, 1), outer_vec)
        bracket += self.lambda2 * batch_size * torch.eye(n=outer_dim).type_as(bracket)
        if bracket.device.type == 'mps': # Prevent MPS errors
            mu_hat = torch.matmul(torch.inverse(bracket.cpu()).to(outer_vec.device), torch.transpose(outer_vec, 0, 1))
        else:
            mu_hat = torch.matmul(torch.inverse(bracket), torch.transpose(outer_vec, 0, 1))
        mu_hat = torch.matmul(mu_hat, Y)
        return mu_hat

    def loss_1(self, V_hat, phi, psi):
        batch_size = psi.size(0)
        psi_t = torch.transpose(psi, 0, 1)
        phi_t = torch.transpose(phi, 0, 1)
        l1 = torch.sum(torch.square(phi_t - torch.matmul(V_hat, psi_t))) / batch_size
        l1 += self.lambda1 * torch.sum(torch.square(V_hat))
        return l1

    def loss_2(self, mu_hat, V_hat, psi, xi, Y):
        V_psi = torch.transpose(torch.matmul(V_hat, torch.transpose(psi, 0, 1)), 0, 1)
        outer_vec = DFIVModel.batch_outer_vec(V_psi, xi)
        rhs = torch.squeeze(torch.matmul(torch.unsqueeze(mu_hat, 0), torch.transpose(outer_vec, 0, 1)))
        l2 = torch.mean(torch.square(Y - rhs))
        l2 += self.lambda2 * torch.sum(torch.square(mu_hat))
        return l2

    def training_step(self, train_batch, batch_idx):
        self.train()
        batch1 = train_batch["first_stage"]
        batch2 = train_batch["second_stage"]

        # First stage
        repr_psi1 = self.psi(batch1[:, 2:-1])
        repr_phi1 = self.phi(batch1[:, 1:2])

        V_hat1 = self.compute_V(phi=repr_phi1, psi=repr_psi1)
        loss1_psi = self.loss_1(V_hat=V_hat1, phi=repr_phi1.detach(), psi=repr_psi1)
        # Optimizer step + zero grad
        self.optimizer_psi.zero_grad()
        self.manual_backward(loss1_psi)
        self.optimizer_psi.step()

        #Second stage
        repr_psi2 = self.psi(batch2[:, 2:-1]).detach()
        repr_phi2 = self.phi(batch2[:, 1:2])
        repr_xi2 = self.xi(batch2[:, (2 + self.zdim):-1])

        V_hat2 = self.compute_V(phi=repr_phi2, psi=repr_psi2)
        mu_hat2 = self.compute_mu(V=V_hat2, psi=repr_psi2, xi=repr_xi2, Y=batch2[:, 0])

        # Update treatment network
        loss2_phi = self.loss_2(mu_hat=mu_hat2, V_hat=V_hat2, psi=repr_psi2, xi=repr_xi2.detach(), Y=batch2[:, 0])
        self.optimizer_phi.zero_grad()
        self.manual_backward(loss2_phi, retain_graph=True)
        self.optimizer_psi.step()

        # Update covariate network
        loss2_xi = self.loss_2(mu_hat=mu_hat2, V_hat=V_hat2.detach(), psi=repr_psi2, xi=repr_xi2, Y=batch2[:, 0])
        # Optimization
        self.optimizer_xi.zero_grad()
        self.manual_backward(loss2_xi)
        self.optimizer_xi.step()

        # x_np = batch2[:, (2 + self.zdim):-1].detach().cpu().numpy()
        # y_hat0 = self.predict_cf(x_np, 0, mu=mu_hat2.detach())
        # y_hat1 = self.predict_cf(x_np, 1, mu=mu_hat2.detach())
        # tau_hat = y_hat1 - y_hat0
        # cate_loss = np.mean((batch2[:, -1].detach().cpu().numpy() - tau_hat) ** 2)
        # Logging
        self.log('loss1', loss1_psi.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('loss2', loss2_phi.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        # self.log('cate_loss', cate_loss.item(), logger=True, on_epoch=True, on_step=False)

    def compute_final_mu(self, data_np):
        self.eval()
        data_torch = torch.from_numpy(data_np.astype(np.float32))
        # Compute representations
        repr_psi = self.psi(data_torch[:, 2:-1])
        repr_phi = self.phi(data_torch[:, 1:2])
        repr_xi = self.xi(data_torch[:, (2 + self.zdim):-1])
        V_hat = self.compute_V(phi=repr_phi, psi=repr_psi)
        self.mu = self.compute_mu(V=V_hat, psi=repr_psi, xi=repr_xi, Y=data_torch[:, 0]).detach()

    def predict_cf(self, x_np, nr):
        self.eval()
        n = x_np.shape[0]
        x = torch.from_numpy(x_np.astype(np.float32))
        # x = x.to(self.device)
        repr_xi = self.xi(x)
        repr_phi = self.phi(torch.full((n, 1), float(nr))) #.to(self.device))
        # Outerproduct of psi and xi
        outer = DFIVModel.batch_outer_vec(repr_phi, repr_xi)
        # if mu is None:
        # Counterfactual outcomes
        y_hat = torch.squeeze(torch.matmul(torch.unsqueeze(self.mu, 0), torch.transpose(outer, 0, 1)))
        # else:
        #     y_hat = torch.squeeze(torch.matmul(torch.unsqueeze(mu, 0), torch.transpose(outer, 0, 1)))
        return y_hat.detach().cpu().numpy() 

    def predict_outcome(self, x_np, t_np):
        self.eval()
        n = x_np.shape[0]
        x = torch.from_numpy(x_np.astype(np.float32))
        # x = x.to(self.device)
        repr_xi = self.xi(x)
        # print(torch.from_numpy(t_np.astype(np.float32)).shape)
        repr_phi = self.phi(torch.from_numpy(t_np.astype(np.float32))) #.to(self.device))
        # Outerproduct of psi and xi
        outer = DFIVModel.batch_outer_vec(repr_phi, repr_xi)
        # if mu is None:
        # Counterfactual outcomes
        y_hat = torch.squeeze(torch.matmul(torch.unsqueeze(self.mu, 0), torch.transpose(outer, 0, 1)))
        # else:
        #     y_hat = torch.squeeze(torch.matmul(torch.unsqueeze(mu, 0), torch.transpose(outer, 0, 1)))
        return y_hat.detach().cpu().numpy() 

    def predict_ite(self, x_np):
        y_hat0 = self.predict_cf(x_np, 0)
        y_hat1 = self.predict_cf(x_np, 1)
        tau_hat = y_hat1 - y_hat0
        return tau_hat
    
    def factual_loss(self, x_np, z_np, y_np):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        Z = torch.from_numpy(z_np.astype(np.float32))
        T = torch.full((n, 1), 1.0)
        Y = torch.from_numpy(y_np.astype(np.float32))

        repr_psi2 = self.psi(torch.concat((Z, X), dim=1))
        repr_phi2 = self.phi(T)
        repr_xi2 = self.xi(X)
        V_hat2 = self.compute_V(phi=repr_phi2, psi=repr_psi2)
        mu_hat2 = self.compute_mu(V=V_hat2, psi=repr_psi2, xi=repr_xi2, Y=Y)
        loss2 = self.loss_2(mu_hat=mu_hat2.T, V_hat=V_hat2, psi=repr_psi2, xi=repr_xi2.detach(), Y=Y)
        train_obj = loss2.detach().cpu().numpy().item()
        return train_obj

class DFIV(DownstreamParent):
    def __init__(self, data : ParentDataset, config):
        if 'log_file' not in config:
            config['log_file'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if 'epochs' not in config:
            config['epochs'] = 100
        if 'early_stopping' not in config:
            config['early_stopping'] = False
        if 'patience' not in config:
            config['patience'] = 10
        if 'logging' not in config:
            config['logging'] = False
        self.data = data
        self.config = config
        self.epochs = config['epochs']
        self.logging = config['logging']
        self.early_stopping = config['early_stopping']
        self.patience = config['patience']
        model = self.train_dfiv()
        super().__init__("DFIV", model)

    def train_dfiv(self):

        Y, A, Z, X = helper.split_data(self.data)

        data1 = np.concatenate((Y.reshape(-1, 1), A.reshape(-1, 1), Z, X, self.data.ite.reshape(-1, 1)), 1)
        data1 = data1[self.data.train_indices]
        # Split data for first/ second stage
        d_first, d_second = train_test_split(data1, test_size=0.5, shuffle=False)
        d_first = torch.from_numpy(d_first.astype(np.float32))
        d_second = torch.from_numpy(d_second.astype(np.float32))
        # Dataloaders for both stages
        first_loader = DataLoader(dataset=d_first, batch_size=self.config["batch_size"],
                                shuffle=True)
        second_loader = DataLoader(dataset=d_second, batch_size=self.config["batch_size"],
                                shuffle=True)
        loaders = {"first_stage": first_loader, "second_stage": second_loader}
        # print("Defining DFIV model...")
        # Create DFIV model
        dfiv = DFIVModel(config=self.config, xdim=X.shape[1], zdim=Z.shape[1]) #d_first.size(1) - 3)
        # Check for available GPUs
        # if torch.cuda.is_available():
        #     gpu = -1
        # else:
        #     gpu = 0

        # Train model
        if self.logging:
            logger = CSVLogger(save_dir="logs/dfiv/", name=f"{self.config['log_file']}")
        else:
            logger = False
            # Trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=False, #gpus=gpu,
            #                      enable_model_summary=False, logger=False, enable_checkpointing=False)
        
        if self.early_stopping:
            early_stopping_callback = EarlyStopping(
                monitor="loss2", 
                patience=self.patience, 
                mode="min"
            )
            Trainer = pl.Trainer(max_epochs=self.epochs, enable_progress_bar=False, #gpus=gpu,
                                enable_model_summary=False, logger=logger, 
                                callbacks=[early_stopping_callback], log_every_n_steps=1)
        else: 
            Trainer = pl.Trainer(max_epochs=self.epochs, enable_progress_bar=False, #gpus=gpu,
                                enable_model_summary=False, logger=logger)
        # print("Training DFIV model...")
        
        Trainer.fit(dfiv, train_dataloaders=loaders)
        # Compute structural function after training using entire dataset
        dfiv.compute_final_mu(data1)
        return dfiv
