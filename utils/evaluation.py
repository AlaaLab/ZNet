#######################################################################################
# Author: Franny Dean, Jenna Fields
# Script: evaluation.py
# Function:  Functions for evaluating Z and CATE?
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed

set_seed(42)

import numpy as np
import statsmodels.api as sm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

#######################################################################################
# Model Evaluation 

def cross_val_score_two_stage_model(X, y, t, model_to_try, cv=5):
    """
    Perform cross-validation for two-stage IV models.
    
    Evaluates model performance using k-fold cross-validation and a custom
    scoring function based on F-statistics.
    
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Outcome values.
        t (np.ndarray): Treatment indicators.
        model_to_try: Two-stage model object with fit and predict/evaluate methods.
        cv (int): Number of cross-validation folds. Defaults to 5.
        
    Returns:
        float: Mean cross-validation score.
    """
    scores = []
    for train_index, test_index in KFold(n_splits=cv).split(X): 
        X_train, X_test = X[train_index], X[test_index]
        t_train, t_test = t[train_index], t[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model_to_try.fit(X_train, y_train, t_train)
        try:
            X_hat_train, X_perp_train, T_hat_train, _, _ = model_to_try.predict(X_train, t_train, y_train)
            X_hat_test, X_perp_test, T_hat_test, _, _ = model_to_try.predict(X_test, t_test, y_test)
        except:
            X_hat_train, X_perp_train, T_hat_train, _ = model_to_try.evaluate(X_train, y_train, y_train)
            X_hat_test, X_perp_test, T_hat_test, _ = model_to_try.evaluate(X_test, t_test, y_test)

        scores.append(custom_score_function(X_perp_test, X_hat_test, t_test, y_test))
    return np.mean(scores)

def custom_score_function(X_perp_test, X_hat_test, T_test, Y_test):
    """
    Compute F-statistics to evaluate instrumental variable quality.
    
    Calculates F-statistics for:
    1. Relevance: T on X_perp controlling for X_hat
    2. Predictive power: Y on X_perp controlling for X_hat
    
    Args:
        X_perp_test (torch.Tensor): Instrument representation (Z).
        X_hat_test (torch.Tensor): Confounder representation (C).
        T_test (torch.Tensor): Treatment values.
        Y_test (torch.Tensor): Outcome values.
        
    Returns:
        tuple: (f_stat_T, f_stat_Y) - F-statistics for treatment and outcome.
    """

    # Fit linear regression of T on X_perp controlling for X_hat
    input = torch.concat([X_hat_test, X_perp_test], dim=1).detach().cpu().numpy()
    model_T = sm.OLS(T_test.numpy(), sm.add_constant(input, has_constant='add')).fit()
    f_stat_T = model_T.fvalue  # Get F-statistic
    
    # Fit linear regression of Y on X_perp controlling for X_hat
    input = torch.concat([X_hat_test, X_perp_test], dim=1).detach().cpu().numpy()
    model_Y = sm.OLS(Y_test.numpy(), sm.add_constant(input, has_constant='add')).fit()
    f_stat_Y = model_Y.fvalue  # Get F-statistic

    return f_stat_T, f_stat_Y



#######################################################################################
# Z Evaluation

# Plot the t-sne - are X_hat and X_perp separated?
def plot_tsne(x_hat, x_perp, n_components=2, title=""):
        """
        Plot t-SNE visualization of confounder and instrument representations.
        
        Visualizes the separation between C (confounder) and Z (instrument)
        representations using t-SNE dimensionality reduction.
        
        Args:
            x_hat (torch.Tensor): Confounder representation C.
            x_perp (torch.Tensor): Instrument representation Z.
            n_components (int): Number of t-SNE dimensions (1 or 2). Defaults to 2.
            title (str): Plot title suffix. Defaults to "".
            
        Returns:
            None: Displays the plot.
        """

        # Run separate t-SNE on control and treatment representations
        tsne_control = TSNE(n_components=n_components, random_state=42)
        X_tsne_x_hat = tsne_control.fit_transform(x_hat.detach().cpu().numpy())
        
        tsne_treatment = TSNE(n_components=n_components, random_state=42)

        X_tsne_X_perp = tsne_treatment.fit_transform(x_perp.detach().cpu().numpy())

        # Plot t-SNE
        if n_components == 1:
            plt.scatter(X_tsne_x_hat, np.zeros_like(X_tsne_x_hat), color='blue', label='Control')
            plt.scatter(X_tsne_X_perp, np.zeros_like(X_tsne_X_perp), color='red', label='Treatment')
            plt.xlabel("Dimension 1")
            plt.yticks([])  
        else:
            plt.scatter(X_tsne_x_hat[:, 0], X_tsne_x_hat[:, 1], color='blue', label='Control')
            plt.scatter(X_tsne_X_perp[:, 0], X_tsne_X_perp[:, 1], color='red', label='Treatment')
            plt.xlabel("Dimension 1")
            plt.ylabel("Dimension 2")
        
        plt.title(f"t-SNE Visualization {title}")
        plt.legend()
        plt.show()

# Exogneity/Exclusion Restriction Assumption
def evaluate_exogeneity(X_perp_train, X_hat_train, T_train, Y_train, Y_hat=None, continuous_outcome=True, verbose=True):
    """
    Evaluate the exogeneity/exclusion restriction assumption for instruments.
    
    Tests whether instruments (Z/X_perp) are independent of outcome residuals,
    which is required for valid instrumental variables. Computes correlations
    and covariances between residuals and instrument features.
    
    Args:
        X_perp_train (torch.Tensor): Instrument representation Z.
        X_hat_train (torch.Tensor): Confounder representation C.
        T_train (torch.Tensor): Treatment values.
        Y_train (torch.Tensor): Outcome values.
        Y_hat (np.ndarray, optional): Predicted outcomes. Computed if None. Defaults to None.
        continuous_outcome (bool): Whether outcome is continuous. Defaults to True.
        verbose (bool): Print detailed results. Defaults to True.
        
    Returns:
        tuple: (model_Y, covariances, correlations) - Regression model and independence measures.
    """

    # Calculate residuals
    if Y_hat is None:
        input = torch.concat([X_hat_train, T_train.reshape(-1, 1)], dim=1).detach().cpu().numpy()
        if continuous_outcome:
            # Fit using OLS from statsmodels
            X_input_with_const = sm.add_constant(input, has_constant='add')  # Add constant for intercept
            model_Y = sm.OLS(Y_train.numpy(), X_input_with_const).fit()
            Y_hat = model_Y.predict(X_input_with_const)
        else:
            # For logistic regression, use sm.Logit
            X_input_with_const = sm.add_constant(input, has_constant='add')  # Add constant for intercept
            model_Y = sm.Logit(Y_train.numpy(), X_input_with_const).fit()
            Y_hat = model_Y.predict(X_input_with_const)

    residuals_Y = Y_train.reshape(-1, 1) - Y_hat.reshape(-1, 1)
    input = torch.concat([X_hat_train, X_perp_train], dim=1).detach().cpu().numpy()
    model_Y = sm.OLS(residuals_Y.numpy(), sm.add_constant(input, has_constant='add')).fit()

    covariances = []
    correlations = []
    for i in range(X_perp_train.numpy().shape[1]):
        # Flatten the current column of X_perp_train
        X_column = X_perp_train.numpy()[:, i]
        # Compute covariance and correlation
        covariance = np.cov(residuals_Y.numpy().flatten(), X_column)[0, 1]
        correlation = covariance / (np.std(residuals_Y.numpy().flatten()) * np.std(X_column))
        
        correlations.append(abs(correlation))
        covariances.append(abs(covariance))
        
    average_correlation = np.mean(correlations)
    average_covariance = np.mean(covariances)
    
    if verbose:
        print("Average |correlation| of residuals of Y with each column of X_perp:")
        print(average_correlation)
        print("Average |covariance| of residuals of Y with each column of X_perp:")
        print(average_covariance)

    return model_Y, covariances, correlations

# Relevance
def evaluate_relevance(X_perp_train, X_hat_train, T_train, continuous_treatment=True, verbose=True):
    """Relevance requires that X_perp is predictive of T_train (and ideally X_hat is not).
    Run a logistic regression predicting T from X_perp, X_hat, and show model.
    
    We *want* the coefficients in front of the X_hat columns to be close to 0 and/or not statistically significant.
    But those in front of the X_perp columns should be statistically significant.
    
    We also run an F statistic.
    
    Args:        
        X_perp_train (torch.Tensor): Instrument representation Z.
        X_hat_train (torch.Tensor): Confounder representation C.
        T_train (torch.Tensor): Treatment values.
        continuous_treatment (bool): Whether treatment is continuous. Defaults to True.
        verbose (bool): Print detailed results. Defaults to True.

    Returns:
        tuple: (model_T, covariances, correlations, f_stat) - Regression model and relevance measures.
    """

    X_input_with_const = sm.add_constant(X_perp_train.detach().cpu().numpy(), has_constant='add')  # Add constant for intercept
    if continuous_treatment:
        model_T = sm.OLS(T_train.numpy(), X_input_with_const).fit()
    else:
        model_T = sm.Logit(T_train.numpy(), X_input_with_const).fit()
    
    f_stat = model_T.fvalue  # Get F-statistic
    if verbose:
        print("F-Statistic:")
        print(f_stat)

        print("Covariance of T on X_perp:")
    covariances = []
    correlations = []
    for i in range(X_perp_train.numpy().shape[1]):
        # Flatten the current column of X_perp_train
        X_column = X_perp_train.numpy()[:, i]
        
        # Compute covariance and correlation
        covariance = np.cov(T_train.numpy().flatten(), X_column)[0, 1]
        correlation = covariance / (np.std(T_train.numpy().flatten()) * np.std(X_column))
        
        correlations.append(abs(correlation))
        covariances.append(abs(covariance))

    average_correlation = np.mean(correlations)
    average_covariance = np.mean(covariances)
    if verbose:
        print("Average |correlation| of T with each column of X_perp:")
        print(average_correlation)
        print("Average |covariance| of T with each column of X_perp:")
        print(average_covariance)

    return model_T, covariances, correlations, f_stat


# Endogeneity of T
def evaluate_endogeneity_t(T_train, X_train, Y_train, continous_outcome=True, verbose=True):
    """Endogenity of T requires that T_train is predictive of Y.
    Run a logistic regression predicting Y from X, T and show model.
    
    We *want* in front of the T columns to be statistically significant.
    
    Args:
        T_train (torch.Tensor): Treatment values.
        X_train (torch.Tensor): Input features.
        Y_train (torch.Tensor): Outcome values.
        continous_outcome (bool): Whether outcome is continuous. Defaults to True.
        verbose (bool): Print detailed results. Defaults to True.

    Returns:
        tuple: (model_Y, covariances, correlations) - Regression model and endogeneity measures.
    """

    input = torch.concat([T_train.reshape(-1, 1), X_train], dim=1).numpy()
    X_input_with_const = sm.add_constant(input, has_constant='add')  # Add constant for intercept
    if continous_outcome:
        model_Y = sm.OLS(Y_train.numpy(), X_input_with_const).fit()
    else:
        model_Y = sm.Logit(Y_train.numpy(), X_input_with_const).fit()

    if verbose:
        print("Logistic Regression of Y on [T, X]:")
        print(model_Y.summary())

        print("Covariance of Y on T:")
    # Compute covariance and correlation
    covariance = np.cov(Y_train.numpy().flatten(), T_train.numpy().flatten())[0, 1]
    correlation = covariance / (np.std(T_train.numpy().flatten()) * np.std(T_train.numpy().flatten()))
    
    if verbose:
        print("|correlation| of Y with T:")
        print(abs(correlation))
        print("|covariance| of Y with T:")
        print(abs(covariance))
    return model_Y, covariance, correlation


def znet_z_effect(t_layers, X_hat_train, X_perp_train, T_train, z_first=True, add_sigmoid=False):
    """Testing whether our T layers predict worse when we use a random/zero X perp

    Args:
        t_layers: The T layers of the ZNet model.
        X_hat_train: The X_hat representation from ZNet.
        X_perp_train: The X_perp representation from ZNet.
        T_train: The treatment values.
        z_first: Whether the ZNet architecture is z_first or not (i.e. whether we concatenate X_perp before or after X_hat). Defaults to True.
        add_sigmoid: Whether to add a sigmoid activation to the output of the T layers. Defaults to False.
    
    Returns:        
        tuple: (t_hat_score, t_rand_score, t_zero_score) - AUC scores for T predictions with true, random, and zero X_perp.
    """
    # First get the true prediction
    if z_first:
        input = torch.concat([X_perp_train, X_hat_train], dim=1)
    else:
        input = torch.concat([X_hat_train, X_perp_train], dim=1)
    
    t_hat = t_layers(input)

    # Now what happens if we mess it up?
    X_perp_rand = torch.rand_like(X_perp_train)
    X_perp_zeros = torch.zeros_like(X_perp_train)

    if z_first:
        input_rand = torch.concat([X_perp_rand, X_hat_train], dim=1)
        input_zeros = torch.concat([X_perp_zeros, X_hat_train], dim=1)
    else:
        input_rand = torch.concat([X_hat_train, X_perp_rand], dim=1)
        input_zeros = torch.concat([X_perp_zeros, X_perp_rand], dim=1)
    
    t_hat_rand = t_layers(input_rand)
    t_hat_zeros = t_layers(input_zeros)

    if add_sigmoid:
        t_hat_rand = torch.sigmoid(t_hat_rand)
        t_hat_zeros = torch.sigmoid(t_hat_zeros)
        t_hat = torch.sigmoid(t_hat)

    t_hat_score = roc_auc_score(T_train.detach().cpu().numpy(), t_hat.detach().cpu().numpy())
    t_rand_score = roc_auc_score(T_train.detach().cpu().numpy(), t_hat_rand.detach().cpu().numpy())
    t_zero_score = roc_auc_score(T_train.detach().cpu().numpy(), t_hat_zeros.detach().cpu().numpy())

    print(f"T AUC with calculated X perp: {t_hat_score}")
    print(f"T AUC with random X perp: {t_rand_score}")
    print(f"T AUC with zero X perp: {t_zero_score}")



    return t_hat_score, t_rand_score, t_zero_score

def evaluate_independence(X_hat_train, X_perp_train, verbose=True):
    """Check if X_hat and X_perp are independent.
    Calculate covariance and correlation.
    
    Args: 
        X_hat_train (torch.Tensor): Confounder representation C.
        X_perp_train (torch.Tensor): Instrument representation Z.
        verbose (bool): Print detailed results. Defaults to True.
    
    Returns:
        tuple: (covariances, correlations) - Lists of covariances and correlations between X_hat and X_perp features.
    """

    if verbose:
        print("Covariance of X_perp on X_hat:")
    covariances = []
    correlations = []
    for i in range(X_hat_train.numpy().shape[1]):
        for j in range(X_perp_train.numpy().shape[1]):
            # Flatten the current column of X_hat_train, X_perp_train
            X_column = X_hat_train.numpy()[:, i]
            X_perp_column = X_perp_train.numpy()[:, j]
            
            # Compute covariance and correlation
            covariance = np.cov(X_perp_column, X_column)[0, 1]
            correlation = covariance / (np.std(X_perp_column) * np.std(X_column))
            
            correlations.append(abs(correlation))
            covariances.append(abs(covariance))
        
    average_correlation = np.mean(correlations)
    average_covariance = np.mean(covariances)
    if verbose:
        print("Average |correlation| of X_perp with each column of X_hat:")
        print(average_correlation)
        print("Average |covariance| of X_perp with each column of X_hat:")
        print(average_covariance)

    return covariances, correlations


#######################################################################################
# CATE Evaluation
   
def compute_PEHE_ATE_metrics(control_output, treatment_output, t, y, ite=None):
    """Compute comparison metrics
    
    Args:
        control_output (torch.Tensor): Model output for control group.
        treatment_output (torch.Tensor): Model output for treatment group.
        t (torch.Tensor): Treatment assignment.
        y (torch.Tensor): Observed outcomes.
        ite (torch.Tensor, optional): Individual treatment effects. Defaults to None.
    
    Returns:
        tuple: (pehe, ate_error, ite_pred, ate_pred) - PEHE, ATE error, predicted ITE, and predicted ATE.
    """
    ite_pred = treatment_output - control_output 
    if ite is not None:
        ite_true = ite 
    else:
        ite_true = torch.where(t == 1, y, -y)
    pehe = torch.mean((ite_pred - ite_true) ** 2)
    
    ate_pred = torch.mean(ite_pred)  
    ate_true = torch.mean(ite_true)  
    ate_error = torch.abs(ate_pred - ate_true)
    
    return pehe.item(), ate_error.item(), ite_pred, ate_pred 
