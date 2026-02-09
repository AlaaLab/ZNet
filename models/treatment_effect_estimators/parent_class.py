#######################################################################################
# Author: Jenna Fields
# Script: parent_class.py
# Function: Downstream estimator parent wrapper
# Date: 02/06/2026
#######################################################################################


class DownstreamParent():
    def __init__(self, model_name, model):
        """
        Parent class for downstream treatment effect estimators.
        
        Provides a unified interface for different treatment effect estimation methods
        (TARNet, DeepIV, DFIV) that consume generated instrumental variables.
        
        Args:
            model_name (str): Name of the downstream model ('tarnet', 'deep_iv', 'df_iv').
            model: The underlying model object.
        """
        self.model_name = model_name
        self.model = model

    def predict_ite(self, x):
        """
        Predict individual treatment effects.
        
        Args:
            x (np.ndarray): Input features.
            
        Returns:
            np.ndarray: Predicted ITEs.
        """
        return self.model.predict_ite(x)

    def predict_outcome(self, x, t):
        """
        Predict outcomes for given features and treatment.
        
        Args:
            x (np.ndarray): Input features.
            t (np.ndarray): Treatment indicators.
            
        Returns:
            np.ndarray: Predicted outcomes.
        """
        return self.model.predict_outcome(x, t)

    def factual_loss(self, x, z, t, y):
        """
        Compute factual loss for training.
        
        Args:
            x (np.ndarray): Confounder representation.
            z (np.ndarray): Instrument representation.
            t (np.ndarray): Treatment indicators.
            y (np.ndarray): Outcome values.
            
        Returns:
            float: Factual loss value.
        """
        if self.model_name == "tarnet":
            return self.model.factual_loss(x, z, t, y)
        return self.model.factual_loss(x, z, y)
