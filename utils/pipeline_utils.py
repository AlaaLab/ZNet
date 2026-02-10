#######################################################################################
# Author: Jenna Fields
# Script: pipeline_utils.py
# Function: Utilities for running the full ZNet pipeline, including grid search, bayesian search, and nearest neighbor evaluation.
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed
from utils.bayesian_search.multi_obj_search import MultiObjectiveHyperparameterTuner
from utils.bayesian_search.single_obj_search import BotorchOptimizer

set_seed(42)

from glob import glob
import json
from DGP.dataset_class import ParentDataset
from DGP.generate_datasets import generate_datasets, generate_named_dataset, load_object_dataset
import pandas as pd
import numpy as np
from utils.train_models import train_autoiv, train_giv, train_viv, train_znet, ecg_full_train
from utils.evaluate_models import evaluate_generatediv_dataset
from utils.evaluate_models import nearest_neighbors
from models.treatment_effect_estimators.deep_iv import DeepIV
from models.treatment_effect_estimators.df_iv import DFIV
from models.treatment_effect_estimators.TARNet import TARNet

import os
import json
import tempfile
import shutil

#######################################################################################

# Parameters for different models for searching
AUTOIV_PARAMS_INT = ['autoiv_params_emb_dim', 'autoiv_params_rep_dim', 'autoiv_params_epochs']
AUTOIV_PARAMS_FLOAT = ['autoiv_params_sigma', 'autoiv_params_lrate', 'autoiv_params_coef_cx2y', 'autoiv_params_coef_zc2x', 'autoiv_params_coef_lld_zx',
                 'autoiv_params_coef_lld_zy', 'autoiv_params_coef_lld_cx', 'autoiv_params_coef_lld_cy',
                 'autoiv_params_coef_lld_zc', 
                 'autoiv_params_coef_reg', 'autoiv_params_dropout'
                 ]
ZNET_PARAMS_BOOL =  ['znet_params_use_pcgrad', 'znet_params_is_linear', 'znet_params_use_sm', 'znet_params_use_mi_corr_loss', 'znet_params_use_mi_matrix_loss']
ZNET_PARAMS_INT = ['znet_params_sm_temp', 'znet_train_params_batch_size', 'znet_train_params_num_epochs', 'dim_options_c_dim', 'dim_options_z_dim']

GIV_PARAMS_INT = ['giv_params_num_clusters', 'giv_params_batch_size', 'giv_params_epoch']
GIV_PARAMS_FLOAT = ['giv_params_beta1', 'giv_params_beta2', 'giv_params_lr']

VIV_PARAMS_INT = ['viv_params_epochs', 'viv_params_bs', 'viv_params_d']
VIV_PARAMS_FLOAT = ['viv_params_lrate', 'viv_params_lrate_min', 'viv_params_loss_y', 'viv_params_loss_t', 'viv_params_loss_x', 'viv_params_kl_loss', 'viv_params_ad_loss']

def extract_param_set(combos, param_name):
    """
    Extract a specific parameter set from a combination dictionary.
    
    Args:
        combos (dict): Full parameter combination.
        param_name (str): Prefix to filter keys.
    
    Returns:
        dict: Filtered parameter subset without prefix.
    """
    return {f'{k[len(param_name):]}': v for k, v in combos.items() if k.startswith(param_name)}


def run_combination(combos, dataset, bootstrap=False, eval_params=None, save_data=True, dir_name=None, seed=42):
    """
    Run one grid-search combination for ZNet and downstream evaluation.
    
    Args:
        combos (dict): Parameter combination.
        dataset (ParentDataset): Dataset to run on.
        bootstrap (bool): Whether to use bootstrap sampling. Defaults to False.
        eval_params (dict, optional): Evaluation configuration.
        save_data (bool): Save generated data to disk. Defaults to True.
        dir_name (str, optional): Output directory name. Defaults to None.
        seed (int): Random seed when bootstrapping. Defaults to 42.
    
    Returns:
        dict: Combined parameters and evaluation results.
    """

    if not bootstrap:
        set_seed(42)
    else:
        set_seed(seed)
    
    # Collect inputs
    param_combination = extract_param_set(combos, 'znet_params_')
    train_param_combination = extract_param_set(combos, 'znet_train_params_')

    # Downstream estimators
    deep_iv_param_combination = extract_param_set(combos, 'deep_iv_params_')
    df_iv_params_combinations = extract_param_set(combos, 'df_iv_params_')
    tarnet_params_combination = extract_param_set(combos, 'tarnet_params_')
    dim_option_combination = extract_param_set(combos, 'dim_options_')
    
    _, znet_data, save_data_path = train_znet(dataset, 
                                                param_combination,
                                                train_param_combination,
                                                dim_option_combination, 
                                                save_data=save_data,
                                                dir_name=dir_name
                                                )
    if eval_params is None or 'methods' not in eval_params:
        eval_params = {
            'methods': ['tsls', 'ols', 'diff_in_means', 'exogeneity', 'independence', 'relevance', 'deep_iv', 'df_iv'],
        }
    eval_params['deep_iv_model_params'] = deep_iv_param_combination
    eval_params['df_iv_model_params'] = df_iv_params_combinations
    eval_params['tarnet_model_params'] = tarnet_params_combination
    eval_params['verbose'] = False

    eval_results = evaluate_generatediv_dataset(znet_data, eval_params, bootstrap=bootstrap, bootstrap_seed=seed)
    eval_results['znet_data_path'] = save_data_path

    if eval_results is None:
        print("No evaluation results found.")
        return None

    return {**combos, **eval_results}

def run_bootstrap_combination(combos, dataset, num_bootstrap, eval_params=None):
    """
    Run bootstrap evaluation for a parameter combination.
    
    Args:
        combos (dict): Parameter combinations. To specify for downstream models, use keys with prefixes like 'deep_iv_params_' and 'df_iv_params_'.
        dataset (GeneratedIVDataset): Generated IV dataset.
        num_bootstrap (int): Number of bootstrap samples.
        eval_params (dict, optional): Evaluation configuration. 
    
    Returns:
        dict: Bootstrap results for each metric.
    """
    
    # Collect inputs
    deep_iv_param_combination = extract_param_set(combos, 'deep_iv_params_')
    df_iv_params_combinations = extract_param_set(combos, 'df_iv_params_')
    tarnet_params_combination = extract_param_set(combos, 'tarnet_params_')

    if eval_params is None or 'methods' not in eval_params:
        eval_params = {
            'methods': ['tsls', 'ols', 'diff_in_means', 'exogeneity', 'independence', 'relevance', 'deep_iv', 'df_iv'],
        }
    
    eval_params['deep_iv_model_params'] = deep_iv_param_combination
    eval_params['df_iv_model_params'] = df_iv_params_combinations
    eval_params['tarnet_model_params'] = tarnet_params_combination
    eval_params['verbose'] = False

    if 'deep_iv' in eval_params['methods']:
        eval_params['deep_iv_trained_model'] = DeepIV(dataset, deep_iv_param_combination)
    if 'df_iv' in eval_params['methods']:
        eval_params['df_iv_trained_model'] = DFIV(dataset, df_iv_params_combinations)
    if 'tarnet' in eval_params['methods']:
        eval_params['tarnet_trained_model'] = TARNet(dataset, tarnet_params_combination)
    results_dict = {}

    for i in range(num_bootstrap):
        bootstrap_seed = 42 + i
        eval_results = evaluate_generatediv_dataset(dataset, eval_params, bootstrap=True, bootstrap_seed=bootstrap_seed)
        if eval_results is None:
            print("No evaluation results found for bootstrap seed {}.".format(bootstrap_seed))
            continue
        eval_results = {**combos, **eval_results}
        for k in eval_results:
            if k not in results_dict:
                results_dict[k] = []
            results_dict[k].append(eval_results[k])

    return results_dict

def run_z_combination(combos, dataset, bootstrap=False, eval_params=None, save_data=False, dir_name=None):
    """
    Run a grid-search combination for ZNet only (no downstream models).
    
    Args:
        combos (dict): Parameter combinations. To specify for ZNet, use keys with prefix 'znet_params_', 'znet_train_params_', and 'dim_options_'.
        dataset (ParentDataset): Dataset to run on.
        bootstrap (bool): Whether to bootstrap evaluation. Defaults to False.
        eval_params (dict, optional): Evaluation configuration.
        save_data (bool): Save generated data to disk. Defaults to False.
        dir_name (str, optional): Output directory name. Defaults to None.
    
    Returns:
        dict: Combined parameters and evaluation results.
    """

    if not bootstrap:
        set_seed(42)
    
    # Collect inputs
    param_combination = extract_param_set(combos, 'znet_params_')
    train_param_combination = extract_param_set(combos, 'znet_train_params_')
    dim_option_combination = extract_param_set(combos, 'dim_options_')
    
    _, znet_data, save_data_path = train_znet(dataset, 
                                                param_combination,
                                                train_param_combination,
                                                dim_option_combination, 
                                                save_data=save_data,
                                                dir_name=dir_name
                                                )
    if eval_params is None or 'methods' not in eval_params:
        eval_params = {
            'methods': ['tsls', 'ols', 'diff_in_means', 'exogeneity', 'independence', 'relevance'],
        }
    eval_params['verbose'] = False

    eval_results = evaluate_generatediv_dataset(znet_data, eval_params)
    eval_results['znet_data_path'] = save_data_path

    if eval_results is None:
        print("No evaluation results found.")
        return None

    return {**combos, **eval_results}

def run_downstream_combination(combos, dataset, bootstrap=False, eval_params=None):
    """
    Run downstream estimators on an existing generated IV dataset.
    
    Args:
        combos (dict): Parameter combination.
        dataset (GeneratedIVDataset): Generated IV dataset.
        bootstrap (bool): Whether to bootstrap evaluation. Defaults to False.
        eval_params (dict, optional): Evaluation configuration.
    
    Returns:
        dict: Evaluation results.
    """

    if not bootstrap:
        set_seed(42)
    
    # Collect inputs
    deep_iv_param_combination = extract_param_set(combos, 'deep_iv_params_')
    df_iv_params_combinations = extract_param_set(combos, 'df_iv_params_')
    tarnet_params_combination = extract_param_set(combos, 'tarnet_params_')
    
    if eval_params is None or 'methods' not in eval_params:
        eval_params = {
            'methods': ['deep_iv', 'df_iv', 'tarnet'],
        }

    eval_params['deep_iv_model_params'] = deep_iv_param_combination
    eval_params['df_iv_model_params'] = df_iv_params_combinations
    eval_params['tarnet_model_params'] = tarnet_params_combination
    eval_params['verbose'] = False

    eval_results = evaluate_generatediv_dataset(dataset, eval_params, bootstrap)

    if eval_results is None:
        print("No evaluation results found.")
        return None

    return {
            **combos, 
            **eval_results}

class NpEncoder(json.JSONEncoder):
    """
    JSON encoder that converts NumPy types to native Python types.
    """
    def default(self, obj):
        if isinstance(obj, np.integer) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.floating) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)

# Save dictionary to a file
def save_dict_to_json(data, filepath):
    """
    Save a dictionary to JSON.
    
    Args:
        data (dict): Data to serialize.
        filepath (str): Destination path.
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, cls=NpEncoder)

def choose_option(prompt, options):
    """
    Prompt the user to select an option from a list.
    
    Args:
        prompt (str): Prompt text.
        options (list): List of selectable options.
    
    Returns:
        str: Selected option.
    """
    while True:
        choice = input(prompt).lower()  # Get input and convert to lowercase for case-insensitivity
        if choice in [opt.lower() for opt in options]:
            return choice
        else:
            print("Invalid choice. Please pick from:", ", ".join(options))


def run_nearest_neighbors_eval(data : ParentDataset, 
                               grid_search_path, 
                               split = 'val',
                               verbose=False, 
                               save_filename = None,
                               manual_choose=False):
    """
    Select best grid-search params using nearest-neighbor CATE estimate.
    
    Args:
        data (ParentDataset): Dataset with splits.
        grid_search_path (str): Path to grid search CSV.
        split (str): Split to compare against. Defaults to 'val'.
        verbose (bool): Print additional output. Defaults to False.
        save_filename (str, optional): Directory to save best params.
        manual_choose (bool): Prompt user when multiple best. Defaults to False.
    
    Returns:
        tuple: (best_params_index, best_param_dict)
    """
    
    nn_cate_estimate = nearest_neighbors(data, split=split, verbose=verbose)
    grid_results = pd.read_csv(grid_search_path)
    ate_cols = [i for i in grid_results.columns if 'ate' in i.lower() and 'val' in i.lower() and 'std' not in i.lower() and 'ols' not in i.lower() and 'diff_in_means' not in i.lower()]
    print('NN estimate', nn_cate_estimate)
    print()
    best_values = {}
    fstat = grid_results[f'{split}_relevance_f_stat'].values
    check_number = min(5, len(fstat))
    top_five = np.argpartition(fstat, -check_number)[-check_number:]
    for i in top_five:
        ate_est = grid_results.iloc[i][ate_cols].values
        abs_dist = np.sum(np.square(ate_est - nn_cate_estimate), axis=-1)
        best_values[i] = abs_dist
        
    min_val = min(best_values.values())

    best_params = [k for k, v in best_values.items() if v == min_val]
    if len(best_params) == 1:
        best_params = best_params[0]
    else:
        if manual_choose:
            print()
            print('Multiple best parameter combinations found. You need to choose between: ')
            print()
            for i in best_params:
                print(f"Parameter combination {i}:")
                print(grid_results.iloc[i][ate_cols + ['train_relevance_f_stat', 'val_relevance_f_stat']])
                print()
            best_params = choose_option("Please enter the index of the best parameter combination: ", [str(i) for i in best_params])
        else:
            # pick a random parameter set among the best
            try:
                best_params = best_params[np.randint(0, len(best_params)).item()]
            except:
                best_params = best_params[int(np.random.randint(0, len(best_params)))]
    print()
    print("Index of best parameters: ", best_params)
    print(grid_results.iloc[best_params][ate_cols + ['train_relevance_f_stat', 'val_relevance_f_stat']])
    print()
    best_param_dict = {'znet_params' : {}, 
                       'znet_train_params' : {},
                       'deep_iv_params' : {},
                       'df_iv_params' : {}, 
                       'dim_options' : {}}
    for col in grid_results.columns:
        if col.startswith('znet_params_'):
            best_param_dict['znet_params'][col[len('znet_params_'):]] = grid_results.iloc[int(best_params)][col]
        elif col.startswith('znet_train_params_'):
            best_param_dict['znet_train_params'][col[len('znet_train_params_'):]] = grid_results.iloc[int(best_params)][col]
        elif col.startswith('deep_iv_params_'):
            best_param_dict['deep_iv_params'][col[len('deep_iv_params_'):]] = grid_results.iloc[int(best_params)][col]
        elif col.startswith('df_iv_params_'):
            best_param_dict['df_iv_params'][col[len('df_iv_params_'):]] = grid_results.iloc[int(best_params)][col]
        elif col.startswith('dim_options_'):
            best_param_dict['dim_options'][col[len('dim_options_'):]] = grid_results.iloc[int(best_params)][col]

    # Save the best parameters to JSON files
    if save_filename is not None:
        save_dict_to_json(best_param_dict['znet_params'], f'{save_filename}/znet_params.json')
        save_dict_to_json(best_param_dict['znet_train_params'], f'{save_filename}/znet_train_params.json')
        save_dict_to_json(best_param_dict['dim_options'], f'{save_filename}/dim_options.json')
        save_dict_to_json(best_param_dict['deep_iv_params'], f'{save_filename}/deep_iv_params.json')
        save_dict_to_json(best_param_dict['df_iv_params'], f'{save_filename}/df_iv_params.json')

    return best_params, best_param_dict

def load_best_param_dict(save_filename):
    """
    Load all saved parameter JSON files into the same format as saved in function above.
    
    Args:
        save_filename (str): The directory where the JSON files were saved.
    
    Returns:
        dict: Parameter dictionary with znet/deep_iv/df_iv/tarnet sections.
    """
    best_param_dict = {
        'znet_params': {},
        'znet_train_params': {},
        'deep_iv_params': {},
        'df_iv_params': {},
        'dim_options': {}
    }
    
    # Helper to load if file exists
    def load_json_if_exists(filepath):
        if os.path.exists(filepath):
            with open(filepath, "r") as f:
                return json.load(f)
        return {}

    best_param_dict['znet_params']       = load_json_if_exists(os.path.join(save_filename, "znet_params.json"))
    best_param_dict['znet_train_params'] = load_json_if_exists(os.path.join(save_filename, "znet_train_params.json"))
    best_param_dict['deep_iv_params']    = load_json_if_exists(os.path.join(save_filename, "deep_iv_params.json"))
    best_param_dict['df_iv_params']      = load_json_if_exists(os.path.join(save_filename, "df_iv_params.json"))
    best_param_dict['dim_options']       = load_json_if_exists(os.path.join(save_filename, "dim_options.json"))
    print(best_param_dict)
    return best_param_dict

def run_nearest_neighbors_downstream_eval(data : ParentDataset, 
                                          grid_search_path,
                                          split = 'val',
                                          verbose=False, 
                                          save_filename = None):
    """
    Choose downstream params using nearest-neighbor CATE estimate.
    
    Args:
        data (ParentDataset): Dataset with splits.
        grid_search_path (str or dict): Path(s) to downstream grid results.
        split (str): Split to compare against. Defaults to 'val'.
        verbose (bool): Print additional output. Defaults to False.
        save_filename (str, optional): Directory to save best params.
    
    Returns:
        tuple: (best_param_indices, best_param_dict)
    """
    param_tuple = ()
    nn_cate_estimate = nearest_neighbors(data, split=split, verbose=verbose)
    print(f"NN estimate: {nn_cate_estimate}")
    grid_results = {}
    if type(grid_search_path) is not str:
        grid_results = pd.merge(pd.read_csv(grid_search_path['deep_iv']), pd.read_csv(grid_search_path['df_iv']), how='outer')
        print(grid_results)
    else:
        grid_results = pd.read_csv(grid_search_path)

    if f'{split}_ate_deep_iv' in grid_results:
        deep_iv_ate_diff = np.square(grid_results[f'{split}_ate_deep_iv'].fillna(np.inf).values - nn_cate_estimate)
        deepiv_best_params = np.argmin(deep_iv_ate_diff)
        print('Deep IV best params: ', deepiv_best_params, grid_results[f'{split}_ate_deep_iv'].values[deepiv_best_params]) #np.min(deep_iv_ate_diff))
        param_tuple += (deepiv_best_params,)

    if f'{split}_ate_df_iv' in grid_results:
        df_iv_ate_diff = np.square(grid_results[f'{split}_ate_df_iv'].fillna(np.inf).values - nn_cate_estimate)
        dfiv_best_params = np.argmin(df_iv_ate_diff)
        print('DF IV best params: ', dfiv_best_params, grid_results[f'{split}_ate_df_iv'].values[dfiv_best_params]) #np.min(df_iv_ate_diff))
        param_tuple += (dfiv_best_params,)

    if f'{split}_ate_tarnet' in grid_results:
        tarnet_ate_diff = np.square(grid_results[f'{split}_ate_tarnet'].fillna(np.inf).values - nn_cate_estimate)
        tarnet_best_params = np.argmin(tarnet_ate_diff)
        print('TARNet best params: ', tarnet_best_params, grid_results[f'{split}_ate_tarnet'].values[tarnet_best_params]) #np.min(tarnet_ate_diff)
        param_tuple += (tarnet_best_params,)

    best_param_dict = {
                       'deep_iv_params' : {},
                       'df_iv_params' : {}, 
                       'tarnet_train_params' : {},
                       'tarnet_params' : {}
                      }
    print(grid_results.columns)
    for col in grid_results.columns:
        if col.startswith('deep_iv_params_'):
            best_param_dict['deep_iv_params'][col[len('deep_iv_params_'):]] = grid_results.iloc[int(deepiv_best_params)][col]
        elif col.startswith('df_iv_params_'):
            best_param_dict['df_iv_params'][col[len('df_iv_params_'):]] = grid_results.iloc[int(dfiv_best_params)][col]
        elif col.startswith('tarnet_params_train'):
            best_param_dict['tarnet_train_params'][col[len('tarnet_params_train_'):]] = grid_results.iloc[int(tarnet_best_params)][col]
        elif col.startswith('tarnet_params_model'):
            best_param_dict['tarnet_params'][col[len('tarnet_params_model_'):]] = grid_results.iloc[int(tarnet_best_params)][col]
    print(best_param_dict)

    if len(best_param_dict['tarnet_train_params']) > 0:
        best_param_dict['tarnet_train_params']['batch_size']  = int(best_param_dict['tarnet_train_params']['batch_size'])
        best_param_dict['tarnet_train_params']['num_epochs'] = int(best_param_dict['tarnet_train_params']['num_epochs'])
    # Save the best parameters to JSON files
    if save_filename is not None:
        if len(best_param_dict['deep_iv_params']) > 0: save_dict_to_json(best_param_dict['deep_iv_params'], f'{save_filename}/deep_iv_params.json')
        if len(best_param_dict['df_iv_params']) > 0: save_dict_to_json(best_param_dict['df_iv_params'], f'{save_filename}/df_iv_params.json')
        if len(best_param_dict['tarnet_params']) > 0: save_dict_to_json(best_param_dict['tarnet_params'], f'{save_filename}/tarnet_params.json')
        if len(best_param_dict['tarnet_train_params']) > 0: save_dict_to_json(best_param_dict['tarnet_train_params'], f'{save_filename}/tarnet_train_params.json')

    return param_tuple, best_param_dict

def dataset_setup(args):
    """
    Load or generate datasets based on CLI arguments.
    
    Args:
        args (argparse.Namespace): Parsed arguments.
    
    Returns:
        dict: Mapping of dataset names to dataset objects.
    """
    if args.dataset_name is None:
        datasets = generate_datasets()
    elif args.generate_data:
        datasets = {d: generate_named_dataset(d) for d in args.dataset_name}
    else:
        if args.dataset_dir is None:
            raise ValueError("If dataset_name is provided and generate data is false, dataset_dir must also be provided.")
        datasets = {d: load_object_dataset(args.dataset_dir, d) for d in args.dataset_name}

    return datasets

def find_grid_search_path(dataset_name):
    """
    Find the latest grid-search directory for a dataset.
    
    Args:
        dataset_name (str): Dataset identifier.
    
    Returns:
        str: Path to grid-search results directory.
    """
    dir_paths = glob(f'grid_search_results/{dataset_name}_grid_search*')
    if not dir_paths:
        raise FileNotFoundError(f"No grid search results found for dataset: {dataset_name}")
    if len(dir_paths) > 1:
        print(f"Multiple grid search results found for dataset {dataset_name}. Using the first one: {dir_paths[0]}")
    return dir_paths[0]

def find_iv_datasets(dataset_names):
    """
    Locate generated IV datasets for the provided names.
    
    Args:
        dataset_names (list): Dataset identifiers.
    
    Returns:
        dict: Mapping of dataset name to loaded dataset.
    """
    iv_datasets = {}
    for dataset_name in dataset_names:
        znet_data_path = glob(f'znet_generated_data/{dataset_name}_*/znet_dataset.pkl')
        if not znet_data_path:
            raise FileNotFoundError(f"No ZNet generated data found for dataset: {dataset_name}")
        iv_datasets[dataset_name] = load_object_dataset(znet_data_path[0])
    return iv_datasets

def convert_bootstrap_params(bootstrap_params):
    """
    Normalize bootstrap parameter dictionaries via JSON serialization.
    
    Args:
        bootstrap_params (dict): Bootstrap parameter map.
    
    Returns:
        dict: Normalized parameter map.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        # Use the temporary directory
        temp_file = os.path.join(temp_dir, "bootstrap_params.json")
        save_dict_to_json(bootstrap_params, temp_file)

        # Your code here
        bootstrap_params = json.load(open(temp_file))
    finally:
        # Manually clean up
        shutil.rmtree(temp_dir)
    return bootstrap_params


# -------- Bayesian search related functions --------
def convert_param_downstream(value, key):
        """
        Convert downstream parameter types based on key.
        
        Args:
            value: Parameter value.
            key (str): Parameter name.
        
        Returns:
            Converted value.
        """
        if key in ['use_early_stopping']:
            return bool(round(value))
        elif key in ['batch_size', 'batch_size2', 'epochs', 'epochs1', 'epochs2', 'num_epochs', 'patience', 'hidden_size_phi', 'hidden_size_psi', 'hidden_size_xi', 'hidden_size', 'hidden_size2']:
            return int(round(value))
        else:
            return value
        
def downstream_ate_objective(params, dataset, method):
    """
    Objective function for downstream ATE optimization.
    
    Args:
        params (dict): Parameter set.
        dataset (GeneratedIVDataset): Dataset.
        method (str): Downstream method name.
    
    Returns:
        float: Objective value.
    """
    if method not in ['df_iv', 'deep_iv', 'tarnet']:
        raise ValueError("Method must be 'df_iv' or 'deep_iv' or 'tarnet'")

    if method == 'tarnet':
        params = extract_param_set(params, 'tarnet_params_')
        params['alpha'] = 0
    eval_params = {'methods': [method], f'{method}_model_params': {k: convert_param_downstream(v, k) for k, v in params.items()}}

    eval_params['verbose'] = False
    eval_params[f'{method}_model_params']['logging'] = False
    eval_results = evaluate_generatediv_dataset(dataset, eval_params)
    
    return -1. * eval_results[f'val_mse_{method}'] # model objective loss

def downstream_multi_objective(params, dataset, method):
    """
    Multi-objective function for downstream model tuning.
    
    Args:
        params (dict): Parameter set.
        dataset (GeneratedIVDataset): Dataset.
        method (str): Downstream method name.
    
    Returns:
        tuple: Objective values for multi-objective tuning.
    """
    if method not in ['df_iv', 'deep_iv', 'tarnet']:
        raise ValueError("Method must be 'df_iv' or 'deep_iv' or 'tarnet'")

    if method == 'tarnet':
        params = extract_param_set(params, 'tarnet_params_')
        params['alpha'] = 0
    eval_params = {'methods': [method], f'{method}_model_params': {k: convert_param_downstream(v, k) for k, v in params.items()}}
    
    nn_cate_estimate = nearest_neighbors(dataset.original_dataset, split='val')
    eval_params['verbose'] = False
    eval_params[f'{method}_model_params']['logging'] = False
    eval_results = evaluate_generatediv_dataset(dataset, eval_params)
    ate_mse = np.abs(nn_cate_estimate - eval_results[f'val_ate_{method}']) #** 2

    return -1 * ate_mse, -1. * eval_results[f'val_mse_{method}'] # model objective loss

def convert_param_multi_obj(value, key, params):
    """
    Convert multi-objective parameter types based on key.
    
    Args:
        value: Parameter value.
        key (str): Parameter name.
        params (dict): Full parameter dictionary.
    
    Returns:
        Converted value.
    """
    
    if key in ZNET_PARAMS_BOOL:
        return bool(round(value))
    elif key == 'znet_params_sm_temp' and bool(round(params['znet_params_use_sm'])):
        return 1
    elif key in ZNET_PARAMS_INT + AUTOIV_PARAMS_INT + GIV_PARAMS_INT + VIV_PARAMS_INT:
        return int(round(value))
    else:
        return value
        
def multi_objective_function(params, dataset, save_results=None, model_type='znet'):
    """
    Multi-objective function for generative IV tuning.
    
    Args:
        params (dict): Parameter set.
        dataset (ParentDataset): Dataset.
        save_results (str, optional): Output path for results.
        model_type (str): Generative model type. Defaults to 'znet'.
    
    Returns:
        tuple: Objective values for multi-objective tuning.
    """
    def save_intermediate_results(params, results, save_results):
        if save_results is not None:
            combined = {**params, **results}
            with open(save_results, 'a') as f:
                f.write(json.dumps(combined) + '\n')
    
    if model_type == 'znet':
        model_params = {k[len('znet_params_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("znet_params_")}
        train_params = {k[len('znet_train_params_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("znet_train_params_")}
        dim_options = {k[len('dim_options_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("dim_options_")}
        train_params['use_early_stopping'] = False 
        model_params['weight_decay'] = 0
        dim_options['y_dim'] = 1
        _, gen_data, _ = train_znet(dataset, model_params, train_params, dim_options)
    if model_type == 'znet_ecg':
        model_params = {k[len('znet_params_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("znet_params_")}
        train_params = {k[len('znet_train_params_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("znet_train_params_")}
        dim_options = {k[len('dim_options_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("dim_options_")}
        train_params['use_early_stopping'] = False 
        model_params['weight_decay'] = 0
        dim_options['y_dim'] = 1
        _, gen_data, _ = ecg_full_train(dataset, model_params, train_params, dim_options)
    elif model_type == 'autoiv':
        model_params = {k[len('autoiv_params_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("autoiv_params_")}
        train_params = {}
        dim_options = {}
        _, gen_data, _ = train_autoiv(dataset, model_params)
    elif model_type == 'giv':
        model_params = {k[len('giv_params_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("giv_params_")}
        train_params = {}
        dim_options = {}
        _, gen_data, _ = train_giv(dataset, model_params)
    elif model_type == 'viv':
        model_params = {k[len('viv_params_'):]: convert_param_multi_obj(v, k, params) for k, v in params.items() if k.startswith("viv_params_")}
        train_params = {}
        dim_options = {}
        _, gen_data, _ = train_viv(dataset, model_params, prepend_path=os.getenv('TMPDIR', ''))
    else:
        raise ValueError("model_type must be 'znet', 'autoiv', 'giv', or 'viv'")
    

    if np.isnan(gen_data.x).any():
        print("Generated data contains NaNs. Returning large negative scores.")
        print("Params: ", {**model_params, **train_params, **dim_options})
    eval_results = evaluate_generatediv_dataset(gen_data, 
                                 {'methods': ['tsls', 'ols', 'diff_in_means', 'exogeneity', 'independence', 'relevance']})
    fstat = eval_results['val_relevance_f_stat']
    ideal_fstat = 40
    fstat_score = (ideal_fstat - fstat) ** 2

    independence_zx = eval_results['val_independence_corr']
    if save_results is not None:
        save_intermediate_results({
                **model_params,
                **train_params,
                **dim_options
            }, eval_results, save_results)
    return -1. * fstat_score, -1. * independence_zx
    

def run_geniv_bayesian_search(param_bounds, dataset, n_calls=20, n_initial_points=5, save_results=None, dir_name=None, model_type='znet'):
    """
    Run Bayesian search for generative IV models.
    
    Args:
        param_bounds (dict): Parameter bounds.
        dataset (ParentDataset): Dataset.
        n_calls (int): Optimization iterations. Defaults to 20.
        n_initial_points (int): Random initial points. Defaults to 5.
        save_results (str, optional): Output path for results.
        dir_name (str, optional): Output directory name.
        model_type (str): Generative model type. Defaults to 'znet'.
    
    Returns:
        tuple: (best_params, pareto_params, pareto_results)
    """
    obj_names = ['Negative F-Statistic MSE', 'Negative Independence Corr']
    tuner = MultiObjectiveHyperparameterTuner(
        param_bounds=param_bounds,
        objective=lambda x: multi_objective_function(x, dataset, save_results=save_results, model_type=model_type),
        objective_names=obj_names,
        dir_name=dir_name, 
        verbose=False
    )
    tuner.optimize(n_iterations=n_calls, n_initial=n_initial_points)

    pareto_mask, pareto_Y, pareto_params = tuner.get_pareto_front_indices()
    
    # Read in all results
    # Instead of reading from file, we can get results directly from tuner
    pareto_results = pd.DataFrame([dict(zip(obj_names, o['objectives'])) for i, o in enumerate(tuner.trial_history) if pareto_mask[i]])
    return pareto_Y, pareto_params, pareto_results

def run_downstream_bayesian_search(param_bounds, dataset, method, n_calls=20, n_initial_points=5, dir_name=None, multi_objective=True):
    """
    Run Bayesian search for downstream estimators.
    
    Args:
        param_bounds (dict): Parameter bounds.
        dataset (GeneratedIVDataset): Dataset.
        method (str): Downstream method name.
        n_calls (int): Optimization iterations. Defaults to 20.
        n_initial_points (int): Random initial points. Defaults to 5.
        dir_name (str, optional): Output directory name.
    
    Returns:
        dict: Best parameters.
    """
    
    # For multi-objective optimization:
    if multi_objective:
        obj_names = ['Negative ATE MSE', 'Negative Y MSE']
        tuner = MultiObjectiveHyperparameterTuner(
            param_bounds=param_bounds,
            objective=lambda x: downstream_multi_objective(x, dataset, method=method),
            objective_names=obj_names,
            dir_name=dir_name, 
            verbose=False
        )
        tuner.optimize(n_iterations=n_calls, n_initial=n_initial_points)

        pareto_mask, pareto_Y, pareto_params = tuner.get_pareto_front_indices()
        
        # Read in all results
        # Instead of reading from file, we can get results directly from tuner
        pareto_results = pd.DataFrame([dict(zip(obj_names, o['objectives'])) for i, o in enumerate(tuner.trial_history) if pareto_mask[i]])
        # best_params = pareto_params[pareto_results['Negative Y MSE'].idxmax()]
        best_params = pareto_params[pareto_results['Negative ATE MSE'].idxmax()]

    # For single-objective optimization:
    else:
        optimizer = BotorchOptimizer(lambda x: downstream_ate_objective(x, dataset, method), 
                                    param_bounds, 
                                    minimize=False, 
                                    beta=1,
                                    dir_name=dir_name)

        results = optimizer.run_optimization(n_iterations=n_calls, acquisition_type='UCB', n_init=n_initial_points, verbose=False)

        best_params = results[-1]['best_params']

    return best_params