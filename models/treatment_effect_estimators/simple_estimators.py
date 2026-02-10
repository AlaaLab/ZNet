#######################################################################################
# Author: Jenna Fields
# Script: simple_estimators.py
# Function: Treatment effects estimated with OLS, diff in means, TSLS
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed

set_seed(42)

import statsmodels.api as sm

#######################################################################################

def diff_in_means(df, verbose=True):
    """
    Compute ATE via naive difference-in-means.
    
    Args:
        df (pd.DataFrame): Dataset with columns 't' and 'y'.
        verbose (bool): Print estimate if True. Defaults to True.
    
    Returns:
        float: Average treatment effect estimate.
    """
    treated = df[df['t'] == 1]['y'].mean()
    control = df[df['t'] == 0]['y'].mean()
    ate = treated - control

    if verbose: 
        print("ATE (Diff in Means):", ate)

    return ate

def TSLS_splits(train_df, val_df, test_df, x_cols, z_cols, verbose=True):
    """
    Run two-stage least squares (2SLS) on train/val/test splits.
    
    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        x_cols (list): Covariate column names.
        z_cols (list): Instrument column names.
        verbose (bool): Print estimates if True. Defaults to True.
    
    Returns:
        tuple: (train_cate, val_ate, test_ate) estimates.
    """

    y = train_df['y']

    # Independent variables (including endogenous)
    X = train_df[x_cols]

    # Combined
    XZ = train_df[x_cols + z_cols]   

    # First Stage: Regress endogenous variable on instruments and covariates
    # Add constant to the instruments
    XZ = sm.add_constant(XZ, has_constant='add')

    # Fit the first stage model
    model_stage1 = sm.OLS(train_df['t'], XZ).fit()

    # Get predicted values from the first stage
    endogenous_predicted = model_stage1.predict()

    # Second Stage: Regress outcome on predicted endogenous variable and covariates
    # Add constant to the independent variables
    X = sm.add_constant(X, has_constant='add')

    # Replace the endogenous variable with the predicted values
    X['t_hat'] = endogenous_predicted

    # Fit the second stage model
    model_stage2 = sm.OLS(y, X).fit()
    cate_2sls = model_stage2.params['t_hat']
    
    # Evaluate on val set
    X_val = val_df[x_cols]
    X_val = sm.add_constant(X_val, has_constant='add')
    XZ_val = val_df[x_cols + z_cols]
    XZ_val = sm.add_constant(XZ_val, has_constant='add')
    model_stage1 = sm.OLS(val_df['t'], XZ_val).fit()
    X_val = sm.add_constant(X_val, has_constant='add')
    X_val['t_hat'] = model_stage1.predict(XZ_val)
    model_stage2 = sm.OLS(val_df['y'], X_val).fit()
    ate_val = model_stage2.params['t_hat']

    # Evaluate on test set
    X_test = test_df[x_cols]
    XZ_test = test_df[x_cols + z_cols]
    XZ_test = sm.add_constant(XZ_test, has_constant='add')
    model_stage1 = sm.OLS(test_df['t'], XZ_test).fit()
    X_test = sm.add_constant(X_test, has_constant='add')
    X_test['t_hat'] = model_stage1.predict(XZ_test)
    X_test = sm.add_constant(X_test, has_constant='add')

    model_stage2 = sm.OLS(test_df['y'], X_test).fit()
    ate_test = model_stage2.params['t_hat']

    if verbose:
        print(f"Train 2SLS CATE:", cate_2sls)
        print(f"Val 2SLS ATE:", ate_val)
        print(f"Test 2SLS ATE:", ate_test)

    return cate_2sls, ate_val, ate_test

def TSLS_df(df, x_cols, z_cols, verbose=True):
    """
    Run two-stage least squares (2SLS) on a full dataset.
    
    Args:
        df (pd.DataFrame): Dataset with columns 't' and 'y'.
        x_cols (list): Covariate column names.
        z_cols (list): Instrument column names.
        verbose (bool): Print estimate if True. Defaults to True.
    
    Returns:
        float: 2SLS CATE estimate.
    """
    y = df['y']

    # Independent variables (including endogenous)
    X = df[x_cols]

    # Combined
    XZ = df[x_cols + z_cols]   
    
    # First Stage: Regress endogenous variable on instruments and covariates
    # Add constant to the instruments
    XZ = sm.add_constant(XZ, has_constant='add')

    # Fit the first stage model
    model_stage1 = sm.OLS(df['t'], XZ).fit()

    # Get predicted values from the first stage
    endogenous_predicted = model_stage1.predict()

    # Second Stage: Regress outcome on predicted endogenous variable and covariates
    # Add constant to the independent variables
    X = sm.add_constant(X, has_constant='add')

    # Replace the endogenous variable with the predicted values
    X['t_hat'] = endogenous_predicted

    # Fit the second stage model
    model_stage2 = sm.OLS(y, X).fit()
    cate_2sls = model_stage2.params['t_hat']
    if verbose:
        print(f"2SLS CATE:", cate_2sls)
    return cate_2sls

def OLS_splits(train_df, val_df, test_df, x_cols, z_cols, verbose=True):
    """
    Run OLS with treatment included on train/val/test splits.
    
    Args:
        train_df (pd.DataFrame): Training split.
        val_df (pd.DataFrame): Validation split.
        test_df (pd.DataFrame): Test split.
        x_cols (list): Covariate column names.
        z_cols (list): Instrument column names.
        verbose (bool): Print estimates if True. Defaults to True.
    
    Returns:
        tuple: (train_cate, val_ate, test_ate) estimates.
    """
    XZT = train_df[x_cols + z_cols]
    y = train_df['y']
    # First compare to one stage
    XZT = sm.add_constant(XZT, has_constant='add')

    XZT['t'] = train_df[['t']]

    model_one_stage = sm.OLS(y, XZT).fit()
    cate_ols = model_one_stage.params['t']

    # Get val ATE
    XZT_val = val_df[x_cols + z_cols]
    XZT_val = sm.add_constant(XZT_val, has_constant='add')
    XZT_val['t'] = val_df[['t']]
    val_pred = model_one_stage.predict(XZT_val)
    t_val = val_df['t'].values
    # get ate on val set
    treated_val = val_pred[t_val == 1].mean()
    control_val = val_pred[t_val == 0].mean()
    ate_val = treated_val - control_val

    # Get test ATE
    XZT_test = test_df[x_cols + z_cols]
    XZT_test = sm.add_constant(XZT_test, has_constant='add')
    XZT_test['t'] = test_df[['t']]
    test_pred = model_one_stage.predict(XZT_test)
    t_test = test_df['t'].values
    # get ate on test set
    treated_test = test_pred[t_test == 1].mean()
    control_test = test_pred[t_test == 0].mean()
    ate_test = treated_test - control_test

    if verbose:
        print("OLS CATE:", cate_ols)
        print(f"Val OLS ATE:", ate_val)
        print(f"Test OLS ATE:", ate_test)

    return cate_ols, ate_val, ate_test

def OLS_df(df, x_cols, z_cols, verbose=True):
    """
    Run OLS with treatment included on a full dataset.
    
    Args:
        df (pd.DataFrame): Dataset with columns 't' and 'y'.
        x_cols (list): Covariate column names.
        z_cols (list): Instrument column names.
        verbose (bool): Print estimate if True. Defaults to True.
    
    Returns:
        float: OLS CATE estimate.
    """
    XZT = df[x_cols + z_cols]
    y = df['y']
    # First compare to one stage
    XZT = sm.add_constant(XZT, has_constant='add')

    XZT['t'] = df[['t']]

    model_one_stage = sm.OLS(y, XZT).fit()
    cate_ols = model_one_stage.params['t']
    if verbose:
        print("OLS CATE:", cate_ols)
    return cate_ols 
