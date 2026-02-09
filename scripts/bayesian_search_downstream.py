#######################################################################################
# Author: Jenna Fields
# Script: bayesian_search_downstream.py
# Function: Bayesian search
# Date: 02/06/2026
#######################################################################################
import os
from seed_utils import set_seed
set_seed(42)

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from DGP.generate_datasets import load_object_dataset
from utils.pipeline_utils import find_iv_datasets, convert_param_downstream, run_downstream_bayesian_search
from datetime import datetime
import argparse
import json
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

#######################################################################################

# Load dictionary from a file
def load_dict_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def convert_numpy_recursive(obj):
    """Recursively convert numpy types in nested structures"""
    if isinstance(obj, dict):
        return {k: convert_numpy_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_recursive(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
#######################################################################################


def main(all_params, datasets, return_params=False, ncalls=20, n_initial_points=5, output_path=None):
    """
    Run Bayesian hyperparameter search for downstream treatment effect estimators.
    
    Optimizes hyperparameters for DeepIV, DFIV, and TARNet models using generated
    IV representations from upstream methods.
    
    Args:
        all_params (dict): Dictionary containing parameter bounds for each estimator.
        datasets (dict): Dictionary mapping dataset names to generated IV datasets.
        return_params (bool): Whether to return best parameters. Defaults to False.
        ncalls (int): Number of Bayesian optimization calls per estimator. Defaults to 20.
        n_initial_points (int): Number of random initial points. Defaults to 5.
        output_path (str, optional): Directory for saving results. Defaults to None.
        
    Returns:
        dict: Best parameters for each dataset and estimator if return_params=True.
    """
    deep_iv_bounds = all_params['deep_iv_params'] if 'deep_iv_params' in all_params else {}
    df_iv_bounds = all_params['df_iv_params'] if 'df_iv_params' in all_params else {}
    tarnet_params = all_params['tarnet_params'] if 'tarnet_params' in all_params else {}
    tarnet_train_params = all_params['tarnet_train_params'] if 'tarnet_train_params' in all_params else {}
    # --------------------------------------------------------------------------------
    
    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # --------------------------------------------------------------------------------
    tarnet_params_labelled = {f'tarnet_params_model_{k}': v for k, v in tarnet_params.items()}
    tarnet_train_params_labelled = {f'tarnet_params_train_{k}': v for k, v in tarnet_train_params.items()}

    tarnet_bounds = {**tarnet_params_labelled, **tarnet_train_params_labelled} if len({**tarnet_params_labelled, **tarnet_train_params_labelled}) > 0 else {}

    # --------------------------------------------------------------------------------
    results_dict = {}
    
    # --------------------------------------------------------------------------------

    print(f"Running downstream bayesian search with {len(datasets)} datasets.")
    all_best_params = {}
    for dataset_name, current_dataset in datasets.items():
        print(f"Running downstream bayesian search for dataset: {dataset_name}")

        # Create a unique directory for each dataset
        file_name = f"{dataset_name}_downstream_bayesian_search"
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_dir = os.path.join(output_path, 'downstream_bayesian_search_results')
        else:
            output_dir = "downstream_bayesian_search_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Create a directory for the results
        os.makedirs(f"{output_dir}/{file_name}_{time}", exist_ok=True)

        results_dict = {}
        if len(deep_iv_bounds) > 0:
            print('Running DeepIV Bayesian Search')
            deepiv_best_params = run_downstream_bayesian_search(
                                deep_iv_bounds,
                                current_dataset, 
                                'deep_iv',
                                n_calls=ncalls,
                                n_initial_points=n_initial_points,
                                dir_name = None) 
        if len(df_iv_bounds) > 0:
            print('Running DFIV Bayesian Search')
            dfiv_best_params = run_downstream_bayesian_search(
                                df_iv_bounds,
                                current_dataset, 
                                'df_iv',
                                n_calls=ncalls,
                                n_initial_points=n_initial_points,
                                dir_name = None) 
        if len(tarnet_bounds) > 0:
            print('Running TARNet Bayesian Search')
            tarnet_best_params = run_downstream_bayesian_search(
                                tarnet_bounds,
                                current_dataset, 
                                'tarnet',
                                n_calls=ncalls,
                                n_initial_points=n_initial_points,
                                dir_name = None) 
        results_dict['deep_iv_params'] = {k: convert_param_downstream(v, k) for k, v in deepiv_best_params.items()} if len(deep_iv_bounds) > 0 else {}
        results_dict['df_iv_params'] = {k: convert_param_downstream(v, k) for k, v in dfiv_best_params.items()} if len(df_iv_bounds) > 0 else {}
        results_dict['tarnet_params'] = {k[len('tarnet_params_model_'):]: convert_param_downstream(v, k) for k, v in tarnet_best_params.items() if k.startswith('tarnet_params_model_')} if len(tarnet_bounds) > 0 else {}
        results_dict['tarnet_train_params'] = {k[len('tarnet_params_train_'):]: convert_param_downstream(v, k) for k, v in tarnet_best_params.items() if k.startswith('tarnet_params_train_')} if len(tarnet_bounds) > 0 else {}
        json.dump(convert_numpy_recursive(results_dict), open(f"{output_dir}/{file_name}_{time}/best_params.json", 'w'), indent=4)
        print(f"Results saved to directory: {f'{output_dir}/{file_name}_{time}/'}")

        print("Done!")
        print(f"Completed bayesian search for dataset: {dataset_name}")

        if return_params:
            all_best_params[dataset_name] = convert_numpy_recursive(results_dict)

        print(f"Best parameters for {dataset_name}: {results_dict}")
    return all_best_params
#######################################################################################


if __name__ == '__main__':
    # --------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Bayesian Search")

    parser.add_argument("--param_json", type=str, help="Path to the JSON file containing bayesian search parameters")
    parser.add_argument("--dataset_name", type=list, default=None, help="List of datasets to run bayesian search on", required=True)
    parser.add_argument("--dataset_path", type=list, default=None, help="Paths to the datasets")
    parser.add_argument("--ncalls", type=int, default=20, help="Number of calls for the bayesian search")

    args = parser.parse_args()
    
    if args.dataset_path is None:
        datasets = find_iv_datasets(args.dataset_name)
    else:
        datasets = {args.dataset_name: load_object_dataset(args.dataset_path)}

    all_params = json.load(open(args.param_json))
    main(all_params, datasets, ncalls=args.ncalls)
    
        
#######################################################################################

