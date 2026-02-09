#######################################################################################
# Author: Jenna Fields, Franny Dean
# Script: bayesian_search_ivgen.py
# Function: Bayesian hyperparameter tuning for models using BoTorch.
# Date: 02/06/2026
#######################################################################################


import json
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from utils.train_models import train_znet, ecg_full_train
from utils.pipeline_utils import dataset_setup, convert_param_multi_obj, run_geniv_bayesian_search

from datetime import datetime
import argparse
from os import makedirs

#######################################################################################

def main(bounds, datasets, gen_iv_method, return_data=False, n_calls=20, n_initial_points=5, output_path=None):
    """
    Run Bayesian hyperparameter search for IV generation methods.
    
    Performs Bayesian optimization to find optimal hyperparameters for generative IV methods
    (ZNet, AutoIV, GIV, VIV) and generates IV datasets with the best parameters.
    
    Args:
        bounds (dict): Dictionary of hyperparameter bounds for Bayesian search.
        datasets (dict): Dictionary mapping dataset names to dataset objects.
        gen_iv_method (str): IV generation method ('znet', 'autoiv', 'giv', or 'viv').
        return_data (bool): Whether to return generated data. Defaults to False.
        n_calls (int): Number of Bayesian optimization calls. Defaults to 20.
        n_initial_points (int): Number of random initial points. Defaults to 5.
        output_path (str, optional): Directory for saving results. Defaults to None.
        
    Returns:
        tuple: (all_gen_data, all_best_params) if return_data=True, else None.
    """

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    all_gen_data = {}
    all_best_params = {}
    for dataset_name, current_dataset in datasets.items():
        print(f"Running bayesian search for dataset: {dataset_name} for method: {gen_iv_method}")

        # Create a unique directory for each dataset
        file_name = f"{dataset_name}_bayesian_search"
        file_name += f"_{gen_iv_method}"
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            output_dir = os.path.join(output_path, 'bayesian_search_results')
        else:
            output_dir = "bayesian_search_results"
        # Create a directory for the results
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        makedirs(f"{output_dir}/{file_name}_{time}", exist_ok=True)

        if "ECG" in dataset_name:
            gen_iv_method_input = gen_iv_method + "_ecg"
        else:
            gen_iv_method_input = gen_iv_method
        _, pareto_params, pareto_results = run_geniv_bayesian_search(bounds, 
                                                        current_dataset, 
                                                        n_calls=n_calls, 
                                                        n_initial_points=n_initial_points, 
                                                        save_results=None, #f"{output_dir}/{file_name}_{time}/bayesian_search_results.jsonl",
                                                        dir_name=None, #f"{output_dir}/{file_name}_{time}", 
                                                        model_type=gen_iv_method_input)
    
        best_param_dict = pareto_params[pareto_results['Negative F-Statistic MSE'].idxmax()]
        print(f"Completed bayesian search for dataset: {dataset_name} for method: {gen_iv_method}")
        best_param_dict = {k: convert_param_multi_obj(v, k, best_param_dict) for k, v in best_param_dict.items()}
        print(f"Best parameters found: {best_param_dict}")

        json.dump(best_param_dict, open(f"{output_dir}/{file_name}_{time}/best_params.json", 'w'), indent=4)
        if gen_iv_method == 'znet':
            # Generate a dataset with the best parameters
            if "ECG" not in dataset_name:
                _, znet_data, save_data_path = train_znet(current_dataset, 
                                                        {k[len('znet_params_'):]: v for k, v in best_param_dict.items() if k.startswith('znet_params_')},
                                                        {k[len('znet_train_params_'):]: v for k, v in best_param_dict.items() if k.startswith('znet_train_params_')},
                                                        {k[len('dim_options_'):]: v for k, v in best_param_dict.items() if k.startswith('dim_options_')},
                                                        save_data=True,
                                                        prepend_path=output_path if output_path is not None else '',
                                                        dir_name=f"{dataset_name}_{time}"
                                                        )
            else:
                
                # Generate a dataset with the best parameters
                _, znet_data, save_data_path = ecg_full_train(current_dataset, 
                                                        {k[len('znet_params_'):]: v for k, v in best_param_dict.items() if k.startswith('znet_params_')},
                                                        {k[len('znet_train_params_'):]: v for k, v in best_param_dict.items() if k.startswith('znet_train_params_')},
                                                        {k[len('dim_options_'):]: v for k, v in best_param_dict.items() if k.startswith('dim_options_')},
                                                        save_data=True,
                                                        prepend_path=output_path if output_path is not None else '',
                                                        dir_name=f"{dataset_name}_{time}"
                                                        )
        elif gen_iv_method == 'autoiv':
            from utils.train_models import train_autoiv
            _, znet_data, save_data_path = train_autoiv(current_dataset, 
                                                    best_param_dict,
                                                    save_data=True,
                                                    prepend_path=output_path if output_path is not None else '',
                                                    dir_name=f"{dataset_name}_{time}"
                                                    )
        elif gen_iv_method == 'giv':
            from utils.train_models import train_giv
            _, znet_data, save_data_path = train_giv(current_dataset, 
                                                    best_param_dict,
                                                    save_data=True,
                                                    prepend_path=output_path if output_path is not None else '',
                                                    dir_name=f"{dataset_name}_{time}"
                                                    )
        elif gen_iv_method == 'viv':
            from utils.train_models import train_viv
            _, znet_data, save_data_path = train_viv(current_dataset, 
                                                    best_param_dict,
                                                    save_data=True,
                                                    prepend_path=output_path if output_path is not None else '',
                                                    dir_name=f"{dataset_name}_{time}"
                                                    )
        else:
            raise ValueError(f"Invalid gen_iv_method: {gen_iv_method}")

        print(f"{gen_iv_method} data generated and saved to: {save_data_path}")
        if return_data:
            all_gen_data[dataset_name] = znet_data
            all_best_params[dataset_name] = best_param_dict
    return all_gen_data, all_best_params


if __name__ == '__main__':
    
    # --------------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Grid Search")
    
    parser.add_argument("--param_json", type=str, help="Path to the JSON file containing grid search parameters")
    parser.add_argument("--dataset_name", type=list, default=None, help="List of datasets to run grid search on (if None, will generate and run on all datasets)")
    parser.add_argument("--generate_data", action='store_true', help="Flag to generate data or use existing data (if no flag, will use data args.dataset_dir)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory to load datasets from")
    parser.add_argument("--gen_iv_method", type=str, default='znet', help="Which method to use for generating IV data")
    args = parser.parse_args()

    datasets = dataset_setup(args)

    all_params = json.load(open(args.param_json))
    main(all_params, datasets, args.gen_iv_method)

    # -----------------------------------------------------------------
