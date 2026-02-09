#######################################################################################
# Author: Jenna Fields
# Script: evaluate_models.py
# Function:  Evaluate the generated IV dataset and downstream models on various metrics.
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed

set_seed(42)

import numpy as np
from utils.evaluation import evaluate_exogeneity, evaluate_independence, evaluate_relevance
from models.treatment_effect_estimators.simple_estimators import OLS_splits, TSLS_splits, diff_in_means
from models.treatment_effect_estimators.deep_iv import DeepIV
from models.treatment_effect_estimators.df_iv import DFIV
from models.treatment_effect_estimators.TARNet import TARNet
from models.treatment_effect_estimators.parent_class import DownstreamParent
from DGP.dataset_class import GeneratedIVDataset, ParentDataset
from sklearn.neighbors import NearestNeighbors
import warnings

def run_second_stage_model_evaluations(data : GeneratedIVDataset, model : DownstreamParent, model_type, bootstrap=False, bootstrap_seed=42):
    """
    Evaluate a downstream model on train/val/test splits.
    
    Args:
        data (GeneratedIVDataset): Dataset with generated IV representations.
        model (DownstreamParent): Trained downstream estimator.
        model_type (str): Label for downstream model type (e.g., 'deep_iv').
        bootstrap (bool): Whether to bootstrap test split. Defaults to False.
        bootstrap_seed (int): Seed for bootstrap sampling. Defaults to 42.
    
    Returns:
        dict: Metrics for MSE, ATE, PEHE, and objective values.
    """
    
    train_data = data.train()
    val_data = data.val()
    test_data = data.test()
    
    if bootstrap:
        set_seed(bootstrap_seed)
        # Bootstrap test data
        n = len(test_data.x)
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        test_data.x = test_data.x[bootstrap_indices]
        test_data.t = test_data.t[bootstrap_indices]
        test_data.y = test_data.y[bootstrap_indices]
        test_data.z = test_data.z[bootstrap_indices]
        set_seed(42)

    train_ite = model.predict_ite(train_data.x)
    val_ite = model.predict_ite(val_data.x)
    test_ite = model.predict_ite(test_data.x)

    train_pehe_loss_df_iv = np.sqrt(np.mean(((train_ite - train_data.ite) ** 2)))
    val_pehe_loss_df_iv = np.sqrt(np.mean(((val_ite - val_data.ite) ** 2)))
    test_pehe_loss_df_iv = np.sqrt(np.mean(((test_ite - test_data.ite) ** 2)))

    train_ate = np.mean(train_ite)
    train_ate_std = np.std(train_ite)
    val_ate = np.mean(val_ite)
    val_ate_std = np.std(val_ite)
    test_ate = np.mean(test_ite)
    test_ate_std = np.std(test_ite)

    Y_hat_train = model.predict_outcome(train_data.x, train_data.t)
    Y_hat_val = model.predict_outcome(val_data.x, val_data.t)
    Y_hat_test = model.predict_outcome(test_data.x, test_data.t)

    train_mse = np.mean((Y_hat_train - train_data.y) ** 2)
    val_mse = np.mean((Y_hat_val - val_data.y) ** 2)
    test_mse = np.mean((Y_hat_test - test_data.y) ** 2)

    train_obj = model.factual_loss(train_data.x, train_data.z, train_data.t, train_data.y)
    val_obj = model.factual_loss(val_data.x, val_data.z, val_data.t, val_data.y)
    test_obj = model.factual_loss(test_data.x, test_data.z, test_data.t, test_data.y)

    # SET BACK TO DEFAULT SEED
    set_seed(42)

    return {f'train_mse_{model_type}': train_mse,
            f'val_mse_{model_type}': val_mse,
            f'test_mse_{model_type}': test_mse,
            f'train_ate_{model_type}': train_ate,
            f'val_ate_{model_type}': val_ate,
            f'test_ate_{model_type}': test_ate,
            f'train_ate_std_{model_type}': train_ate_std,
            f'val_ate_std_{model_type}': val_ate_std,
            f'test_ate_std_{model_type}': test_ate_std,
            f'train_pehe_loss_{model_type}': train_pehe_loss_df_iv,
            f'val_pehe_loss_{model_type}': val_pehe_loss_df_iv,
            f'test_pehe_loss_{model_type}': test_pehe_loss_df_iv,
            f'train_obj_{model_type}': train_obj,
            f'val_obj_{model_type}': val_obj,
            f'test_obj_{model_type}': test_obj,}


def evaluate_generatediv_dataset(data : GeneratedIVDataset,
                                 eval_params, 
                                 bootstrap=False, 
                                 bootstrap_seed=42
                                 ):
    """
    Evaluate a generated IV dataset across multiple metrics and models.
    
    Args:
        data (GeneratedIVDataset): Dataset with generated IV representations.
        eval_params (dict): Evaluation configuration (methods, model params, verbose).
        bootstrap (bool): Whether to bootstrap test split. Defaults to False.
        bootstrap_seed (int): Seed for bootstrap sampling. Defaults to 42.
    
    Returns:
        dict: Aggregated evaluation results.
    """
    if 'verbose' not in eval_params:
        eval_params['verbose'] = False
    results = {}
    train_df = data.df[data.df['split'] == 'train']
    val_df = data.df[data.df['split'] == 'val']
    test_df = data.df[data.df['split'] == 'test']
    train_tensor = data.train(data_type='torch')
    val_tensor = data.val(data_type='torch')
    test_tensor = data.test(data_type='torch')
    def print_split(split):
        if eval_params['verbose']:
            print()
            print(f"Evaluating {split} split:")
    if bootstrap:
        set_seed(bootstrap_seed)
        # Bootstrap test data
        n = len(data.test().x)
        bootstrap_indices = np.random.choice(n, size=n, replace=True)
        test_df = test_df.reset_index(drop=True).iloc[bootstrap_indices]
        test_tensor.x = test_tensor.x[bootstrap_indices]
        test_tensor.t = test_tensor.t[bootstrap_indices]
        test_tensor.y = test_tensor.y[bootstrap_indices]
        test_tensor.z = test_tensor.z[bootstrap_indices]
        test_tensor.ite = test_tensor.ite[bootstrap_indices]
        set_seed(42) # reset seed

    if 'tsls' in eval_params['methods']:
        train_ate, val_ate, test_ate = TSLS_splits(train_df, val_df, test_df, data.x_cols, data.z_cols, verbose=eval_params['verbose'])
        results['train_TSLS_ATE'] = train_ate
        results['val_TSLS_ATE'] = val_ate
        results['test_TSLS_ATE'] = test_ate
    if 'ols' in eval_params['methods']:
        train_ate, val_ate, test_ate = OLS_splits(train_df, val_df, test_df, data.x_cols, data.z_cols, verbose=eval_params['verbose'])
        results['train_OLS_ATE'] = train_ate
        results['val_OLS_ATE'] = val_ate
        results['test_OLS_ATE'] = test_ate
    if 'diff_in_means' in eval_params['methods']:
        print_split('train')
        results['train_diff_in_means_ATE'] = diff_in_means(train_df, verbose=eval_params['verbose'])
        print_split('val')
        results['val_diff_in_means_ATE'] = diff_in_means(val_df, verbose=eval_params['verbose'])
        print_split('test')
        results['test_diff_in_means_ATE'] = diff_in_means(test_df, verbose=eval_params['verbose'])
    if 'exogeneity' in eval_params['methods']:
        print_split('train')
        _, covariances, corrs = evaluate_exogeneity(train_tensor.z, train_tensor.x, train_tensor.t, train_tensor.y, verbose=eval_params['verbose'])
        results['train_exogeneity_cov'] = np.mean(covariances)
        results['train_exogeneity_corr'] = np.mean(corrs)
        print_split('val')
        _, covariances, corrs = evaluate_exogeneity(val_tensor.z, val_tensor.x, val_tensor.t, val_tensor.y, verbose=eval_params['verbose'])
        results['val_exogeneity_cov'] = np.mean(covariances)
        results['val_exogeneity_corr'] = np.mean(corrs)
        print_split('test')
        _, covariances, corrs = evaluate_exogeneity(test_tensor.z, test_tensor.x, test_tensor.t, test_tensor.y, verbose=eval_params['verbose'])
        results['test_exogeneity_cov'] = np.mean(covariances)
        results['test_exogeneity_corr'] = np.mean(corrs)
    if 'independence' in eval_params['methods']:
        print_split('train')
        cov, corr = evaluate_independence(train_tensor.x, train_tensor.z, verbose=eval_params['verbose'])
        results['train_independence_cov'] = np.mean(cov)
        results['train_independence_corr'] = np.mean(corr)
        print_split('val')
        cov, corr = evaluate_independence(val_tensor.x, val_tensor.z, verbose=eval_params['verbose'])
        results['val_independence_cov'] = np.mean(cov)
        results['val_independence_corr'] = np.mean(corr)
        print_split('test')
        cov, corr = evaluate_independence(test_tensor.x, test_tensor.z, verbose=eval_params['verbose'])
        results['test_independence_cov'] = np.mean(cov)
        results['test_independence_corr'] = np.mean(corr)
    if 'relevance' in eval_params['methods']:
        print_split('train')
        _, cov, corr, f_stat = evaluate_relevance(train_tensor.z, train_tensor.x, train_tensor.t, verbose=eval_params['verbose']) #continous_treatment=False, v
        results['train_relevance_cov'] = np.mean(cov)
        results['train_relevance_corr'] = np.mean(corr)
        results['train_relevance_f_stat'] = f_stat
        print_split('val')
        _, cov, corr, f_stat = evaluate_relevance(val_tensor.z, val_tensor.x, val_tensor.t, verbose=eval_params['verbose']) #continous_treatment=False, v
        results['val_relevance_cov'] = np.mean(cov)
        results['val_relevance_corr'] = np.mean(corr)
        results['val_relevance_f_stat'] = f_stat
        print_split('test')
        _, cov, corr, f_stat = evaluate_relevance(test_tensor.z, test_tensor.x, test_tensor.t, verbose=eval_params['verbose']) # continous_treatment=False, 
        results['test_relevance_cov'] = np.mean(cov)
        results['test_relevance_corr'] = np.mean(corr)
        results['test_relevance_f_stat'] = f_stat
    if 'u_z_independence' in eval_params['methods']:
        print_split('train')
        cov, corr = data.evaluate_u_z(verbose=eval_params['verbose'], split='train')
        results['train_u_z_independence_cov'] = np.mean(cov)
        results['train_u_z_independence_corr'] = np.mean(corr)
        print_split('val')
        cov, corr = data.evaluate_u_z(verbose=eval_params['verbose'], split='val')
        results['val_u_z_independence_cov'] = np.mean(cov)
        results['val_u_z_independence_corr'] = np.mean(corr)
        print_split('test')
        cov, corr = data.evaluate_u_z(verbose=eval_params['verbose'], split='test')
        results['test_u_z_independence_cov'] = np.mean(cov)
        results['test_u_z_independence_corr'] = np.mean(corr)
    if 'deep_iv' in eval_params['methods']:
        if len(data.z_cols) == 0:
            warnings.warn("No valid instrumental variables found for Deep IV evaluation. Running Deep IV with empty instrument may lead to unreliable results. You've been warned...(or there is a bug \U0001F41E)", UserWarning)
        if 'deep_iv_trained_model' in eval_params:
            model = eval_params['deep_iv_trained_model']
        else:
            model = DeepIV(data, eval_params['deep_iv_model_params'])
        results.update(run_second_stage_model_evaluations(data, model, 'deep_iv', bootstrap=bootstrap, bootstrap_seed=bootstrap_seed))
    if 'df_iv' in eval_params['methods']:
        if len(data.z_cols) == 0:
            warnings.warn("No valid instrumental variables found for DF IV evaluation. Running DF IV with empty instrument may lead to unreliable results. You've been warned...(or there is a bug \U0001F41E)", UserWarning)
        if 'df_iv_trained_model' in eval_params:
            model = eval_params['df_iv_trained_model']
        else:
            model = DFIV(data, eval_params['df_iv_model_params'])
        results.update(run_second_stage_model_evaluations(data, model, 'df_iv', bootstrap=bootstrap, bootstrap_seed=bootstrap_seed))
    if 'tarnet' in eval_params['methods']:
        if len(data.z_cols) != 0:
            warnings.warn("Instrumental variables found for TARNet evaluation. TARNet does not use instrumental variables, so this may lead to unreliable results. You've been warned...(or there is a bug \U0001F41E)", UserWarning)
        # If you have a pre-trained TARNet model, use it
        if 'tarnet_trained_model' in eval_params:
            model = eval_params['tarnet_trained_model']
        else:
            model = TARNet(data, eval_params['tarnet_model_params'])
        results.update(run_second_stage_model_evaluations(data, model, 'tarnet', bootstrap=bootstrap, bootstrap_seed=bootstrap_seed))

    return results


def nearest_neighbors(data : ParentDataset, split = 'val', verbose=False):
    """
    Estimate CATE using nearest neighbors in covariate space.
    
    Args:
        data (ParentDataset): Dataset with splits.
        split (str): Which split to use ('train', 'val', 'test'). Defaults to 'val'.
        verbose (bool): Print summary if True. Defaults to False.
    
    Returns:
        float: Nearest-neighbor CATE estimate.
    """
    # Fit the NearestNeighbors model
    df = data.df[data.df['split'] == split]
    X = df[data.x_cols].values
    nbrs = NearestNeighbors(n_neighbors=5, algorithm='auto').fit(X)

    # Find the nearest neighbors for each point
    distances, indices = nbrs.kneighbors(X)

    cate_cf = []
    no_found = 0
    cate_diff = []
    for i, (ind, dist) in enumerate(zip(indices, distances)):
        t_i = df.iloc[i]['t']
        nn_outcomes = []
        treat_eff_ests = []
        for d, j in zip(dist, ind):
            t_j = df.iloc[j]['t']
            if t_i != t_j: 
                nn_outcomes.append(df.iloc[j]['y'])
                if t_j == 1:
                    treat_eff_ests.append(df.iloc[j]['y'] - df.iloc[i]['y'])
                else:
                    treat_eff_ests.append(df.iloc[i]['y'] - df.iloc[j]['y'])
        if len(treat_eff_ests) == 0:
            no_found += 1
        else:
            est_cate = np.mean(treat_eff_ests)
            cate_cf.append(est_cate)
            cate_diff.append(np.abs(est_cate - df.iloc[i]['ite']))

    if verbose:
        print('NN not found: ', no_found / len(df))
        print('NN Est: ', np.mean(cate_cf))
        print('True CATE: ', df['ite'].mean())
    nn_cate_estimate = np.mean(cate_cf)
    return nn_cate_estimate