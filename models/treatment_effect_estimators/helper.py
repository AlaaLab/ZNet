#######################################################################################
# Original Author: Frauen, Dennis, and Stefan Feuerriegel. 
#          "Estimating individual treatment effects under unobserved confounding using binary instruments." arXiv preprint arXiv:2208.08544 (2022).
# Editor: Jenna Fields
# Script: helper.py
# Function: Downstream estimator helper functions
# Date: 02/06/2026
#######################################################################################
from seed_utils import set_seed

set_seed(42)

from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fctnl
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer

#######################################################################################

def rmse(y_hat, y, scaler=1):
    return np.sqrt(np.mean(((y - (y_hat*scaler)) ** 2)))


def create_loaders(data, batch_size, validation=True, train_indices=None, val_indices=None):
    if train_indices is not None and val_indices is not None:
        d_train = torch.from_numpy(data[train_indices].astype(np.float32))
        d_val = torch.from_numpy(data[val_indices].astype(np.float32))
    else:
        d_train, d_val = train_test_split(data, test_size=0.1, shuffle=False)
        d_train = torch.from_numpy(d_train.astype(np.float32))
        d_val = torch.from_numpy(d_val.astype(np.float32))
    if not validation:
        d_train = np.concatenate((d_train, d_val), axis=0)

    train_loader = DataLoader(dataset=d_train, batch_size=int(batch_size), shuffle=False, num_workers=0)
    if validation:
        val_loader = DataLoader(dataset=d_val, batch_size=int(batch_size), shuffle=False, num_workers=0)
    else:
        val_loader = None
    return train_loader, val_loader


def train_nn(data, config, model_class, epochs=50, validation=True, 
             logging=True, early_stopping = False, patience=10, train_indices = None, val_indices = None, 
             **kwargs):
    # Data
    train_loader, val_loader = create_loaders(data, config["batch_size"], validation=validation, train_indices=train_indices, val_indices=val_indices)
    # Model
    model = model_class(config=config, **kwargs)
    # Train
    if logging:
        logger = CSVLogger(save_dir="logs/", name=f"deepiv_{config['log_file']}")
    else:
        logger = False
        
    if early_stopping:
        if validation:
            early_stopping_callback = EarlyStopping(
                monitor="val_loss", 
                patience=patience, 
                mode="min"
            )
        else:
            early_stopping_callback = EarlyStopping(
                monitor="train_loss", 
                patience=patience, 
                mode="min"
            )

        Trainer1 = Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False, 
                                    logger=logger, enable_checkpointing=False, callbacks=[early_stopping_callback], log_every_n_steps=1)
    else:   
        Trainer1 = Trainer(max_epochs=epochs, enable_progress_bar=False, enable_model_summary=False,
                                logger=logger, enable_checkpointing=False, log_every_n_steps=1)

    if validation:
        Trainer1.fit(model, train_loader, val_loader)
        # Validation error after training
        val_results = Trainer1.validate(model=model, dataloaders=val_loader, verbose=False)
        val_err = val_results[0]['val_loss']
    else:
        Trainer1.fit(model, train_loader)
        val_err = None

    return model, val_err


def split_data(data):
    Y = data.y
    T = data.t
    Z = data.z
    X = data.x
    return Y, T, Z, X


# Feed forward neural network, either binary or continuous output
class ffnn(pl.LightningModule):
    def __init__(self, config, input_size, output_type, weights=None):
        super().__init__()
        self.save_hyperparameters(config)
        self.layer1 = nn.Linear(input_size, config["hidden_size"])
        self.layer2 = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.layer3 = nn.Linear(config["hidden_size"], 1)
        self.dropout = nn.Dropout(config["dropout"])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=config["lr"])
        self.output_type = output_type
        if weights is None:
            self.weights = 1
        else:
            self.weights = torch.from_numpy(weights.astype(np.float32))

    def configure_optimizers(self):
        return self.optimizer

    def format_input(self, batch_torch):
        Y = batch_torch[:, 0]
        X = batch_torch[:, 1:]
        return [Y, X]

    def obj(self, y_hat, y):
        if self.output_type == "continuous":
            loss_y = torch.mean(((y_hat.view_as(y) - y) * self.weights) ** 2)
        else:
            loss_y = fctnl.binary_cross_entropy(y_hat.view_as(y), y, reduction='mean')
        return loss_y

    def forward(self, x):
        out = self.dropout(fctnl.relu(self.layer1(x)))
        out = self.dropout(fctnl.relu(self.layer2(out)))
        out = torch.squeeze(self.layer3(out))
        if self.output_type == "binary":
            out = torch.sigmoid(out)
        return out

    def training_step(self, train_batch, batch_idx):
        self.train()
        # Format data
        [Y, X] = self.format_input(train_batch)
        # Forward pass
        y_hat = self.forward(X)
        # Loss
        loss = self.obj(y_hat, Y)
        # Logging
        try:
            self.log('train_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        except:
            self.log('train_loss', loss, logger=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, train_batch, batch_idx):
        self.eval()
        # Format data
        [Y, X] = self.format_input(train_batch)
        # Forward pass
        y_hat = self.forward(X)
        # Loss
        loss = self.obj(y_hat, Y)
        # Logging
        try:
            self.log('val_loss', loss.detach().cpu().numpy().item(), logger=True, on_epoch=True, on_step=False)
        except:
            self.log('val_loss', loss, logger=True, on_epoch=True, on_step=False)
        return loss

    def predict(self, x_np):
        self.eval()
        X = torch.from_numpy(x_np.astype(np.float32))
        tau_hat = self.forward(X)
        return tau_hat.detach().cpu().numpy()

    def validation_mse(self, d_val):
        y_hat = self.predict(d_val[:, 1:])
        return np.mean((y_hat - d_val[:, 0])**2)


def train_base_model(model_name, d_train, params=None, validation=False, logging=False):
    import models.deep_iv as deepiv
    model = None
    if model_name == "deepiv":
        model = deepiv.train_DeepIV(d_train, params, validation=validation, logging=logging)
    return model