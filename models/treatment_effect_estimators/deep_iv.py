#######################################################################################
# Original Author: Frauen, Dennis, and Stefan Feuerriegel. 
#        "Estimating individual treatment effects under unobserved confounding using binary instruments." arXiv preprint arXiv:2208.08544 (2022).
# Editors: Jenna Fields
# Script: deep_iv.py
# Function: DeepIV implementation for binary treatments (see Hartford paper)
# Date: 02/06/2026
#######################################################################################
from pyexpat import model
from seed_utils import set_seed

set_seed(42)

import torch
import torch.nn as nn
import torch.nn.functional as fctnl
import pytorch_lightning as pl
import numpy as np
from models.treatment_effect_estimators import helper
from datetime import datetime
from DGP.dataset_class import ParentDataset
from models.treatment_effect_estimators.parent_class import DownstreamParent

#######################################################################################

#Feed forward neural network for second stage regression
class second_stage_nn(pl.LightningModule):
    def __init__(self, config, input_size, first_stage_nn):
        super().__init__()
        self.layer1 = nn.Linear(input_size, config["hidden_size2"])
        self.layer2 = nn.Linear(config["hidden_size2"], config["hidden_size2"])
        self.layer3 = nn.Linear(config["hidden_size2"], 1)
        self.dropout = nn.Dropout(config["dropout2"])
        # self.cate = config["cate"]
        self.z_dim = config["z_dim"]
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr2"])
        self.first_stage_nn = first_stage_nn
        self.all_cate_loss = []
        self.val_cate_loss = []
        self.save_hyperparameters(config)

    def configure_optimizers(self):
        return self.optimizer

    def format_input(self, batch_torch):
        n = batch_torch.shape[0]
        Y = batch_torch[:, 0]
        x0 = torch.concat((torch.zeros(n, 1).type_as(batch_torch), batch_torch[:, (2 + self.z_dim):-1]), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(batch_torch), batch_torch[:, (2 + self.z_dim):-1]), dim=1)
        ite = batch_torch[:, -1]
        return [Y, x0, x1, ite]

    def obj(self, Y_hat0, Y_hat1, data):
        pi = self.first_stage_nn.forward(data[:, 2:-1]).detach()
        loss = torch.mean((data[:, 0] - (1 - pi) * Y_hat0 - pi*Y_hat1)**2)
        return loss

    def forward(self, x0, x1):
        out0 = self.dropout(fctnl.relu(self.layer1(x0)))
        out0 = self.dropout(fctnl.relu(self.layer2(out0)))
        out0 = torch.squeeze(self.layer3(out0))
        out1 = self.dropout(fctnl.relu(self.layer1(x1)))
        out1 = self.dropout(fctnl.relu(self.layer2(out1)))
        out1 = torch.squeeze(self.layer3(out1))
        return out0, out1

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Format data
        [Y, x0, x1, ite] = self.format_input(train_batch)
        # Forward pass
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        # Loss
        loss = self.obj(Y_hat0, Y_hat1, train_batch)
        tau_hat = Y_hat1 - Y_hat0
        cate_loss = torch.mean((ite - tau_hat)**2)
        self.all_cate_loss.append(cate_loss.detach().cpu().numpy().item())
        # Logging
        self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('cate_loss', cate_loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Formnat data
        [Y, x0, x1, ite] = self.format_input(train_batch)
        # Forward pass
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        # Loss
        loss = self.obj(Y_hat0, Y_hat1, train_batch)
        tau_hat = Y_hat1 - Y_hat0
        cate_loss = torch.mean((ite - tau_hat)**2)
        self.val_cate_loss.append(cate_loss.detach().cpu().numpy().item())
        # Logging
        self.log('val_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        self.log('val_cate_loss', cate_loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        return loss

    def predict_cf(self, x_np, nr):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        x0 = torch.concat((torch.zeros(n, 1).type_as(X), X), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(X), X), dim=1)
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        if nr==1:
            return Y_hat1.detach().cpu().numpy()
        else:
            return Y_hat0.detach().cpu().numpy()

    def predict_outcome(self, x_np, a_np):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        A = torch.from_numpy(a_np.astype(np.float32))
        x0 = torch.concat((torch.zeros(n, 1).type_as(X), X), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(X), X), dim=1)
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        
        return torch.where(A==1, Y_hat1, Y_hat0).detach().cpu().numpy()
    
    def predict_outcome_z(self, x_np, z_np):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        # A = torch.from_numpy(a_np.astype(np.float32))
        Z = torch.from_numpy(z_np.astype(np.float32))
        pi = self.first_stage_nn.forward(torch.concat((Z, X), dim=1)).detach()

        x0 = torch.concat((torch.zeros(n, 1).type_as(X), X), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(X), X), dim=1)

        Y_hat0, Y_hat1 = self.forward(x0, x1)
        # (1 - pi) * Y_hat0 + pi*Y_hat1
        return (1 - pi) * Y_hat0 + pi*Y_hat1 #torch.where(A==1, Y_hat1, Y_hat0).detach().numpy()
    
    def predict_ite(self, x_np):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        x0 = torch.concat((torch.zeros(n, 1).type_as(X), X), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(X), X), dim=1)
        Y_hat0, Y_hat1 = self.forward(x0, x1)
        tau_hat = Y_hat1 - Y_hat0
        return tau_hat.detach().cpu().numpy()
    
    def factual_loss(self, x_np, z_np, y_np):
        self.eval()
        n = x_np.shape[0]
        X = torch.from_numpy(x_np.astype(np.float32))
        Z = torch.from_numpy(z_np.astype(np.float32))
        T = torch.full((n, 1), 1.0)
        Y = torch.from_numpy(y_np.astype(np.float32))
        
        x0 = torch.concat((torch.zeros(n, 1).type_as(X), X), dim=1)
        x1 = torch.concat((torch.ones(n, 1).type_as(X), X), dim=1)
        y_hat0, y_hat1 = self.forward(x0, x1)
        pi = self.first_stage_nn.forward(torch.concat((Z, X), dim=1))
        loss = torch.mean((Y - (1 - pi) * y_hat0 - pi * y_hat1) ** 2)
        train_obj = loss.detach().cpu().numpy().item()
        return train_obj

class DeepIV(DownstreamParent):
    def __init__(self, data : ParentDataset, config):
        if 'log_file' not in config:
            config['log_file'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if 'epochs1' not in config:
            config['epochs1'] = 100
        if 'epochs2' not in config: 
            config['epochs2'] = 200
        if 'early_stopping' not in config:
            config['early_stopping'] = False
        if 'patience' not in config:
            config['patience'] = 10
        if 'logging' not in config:
            config['logging'] = False
        if 'validation' not in config:
            config['validation'] = True
        self.data = data
        self.config = config
        self.validation = config['validation']
        self.logging = config['logging']
        self.epochs = config['epochs1']
        self.epochs2 = config['epochs2']
        self.early_stopping = config['early_stopping']
        self.patience = config['patience']
        model = self.train_DeepIV()
        super().__init__('DeepIV', model)

    def train_DeepIV(self):
        # Y, A, Z, X = helper.split_data(self.data)
        # self.data = np.concatenate((np.expand_dims(Y, 1), np.expand_dims(A, 1), Z, X), 1)
        first_stage = self.train_first_stage()
        second_stage = self.train_second_stage(first_stage)
        return second_stage
    
    def train_first_stage(self):
        Y, A, Z, X = helper.split_data(self.data)
        config1 = self.config.copy()
        if 'log_file' not in self.config:
            config1['log_file'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config1['log_file'] += "/first_stage"
        data1 = np.concatenate((A.reshape(-1, 1), Z, X), 1)
        first_stage, _ = helper.train_nn(data=data1, config=config1, model_class= helper.ffnn, input_size=X.shape[1] + Z.shape[1],
                                    validation=self.validation, logging=self.logging, output_type="binary", epochs=self.epochs, train_indices = self.data.train_indices, val_indices=self.data.val_indices)
        return first_stage

    def train_second_stage(self, first_stage):
        Y, A, Z, X = helper.split_data(self.data)
        config2 = self.config.copy()
        config2["batch_size"] = self.config["batch_size2"]
        if 'log_file' not in self.config:
            config2['log_file'] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        config2['log_file'] += "/second_stage"

        data1 = np.concatenate((Y.reshape(-1, 1), A.reshape(-1, 1), Z, X, self.data.ite.reshape(-1, 1)), 1)
        # data1 = np.concatenate((np.expand_dims(Y, 1), np.expand_dims(A, 1), Z, X), 1)
        config2['z_dim'] = Z.shape[1]
        second_stage, _ = helper.train_nn(data=data1, config=config2, model_class=second_stage_nn, input_size=X.shape[1] + 1,
                                    validation=self.validation, logging=self.logging, first_stage_nn=first_stage, epochs=self.epochs2,
                                    early_stopping=self.early_stopping, patience=self.patience, train_indices=self.data.train_indices, val_indices=self.data.val_indices)
        return second_stage
