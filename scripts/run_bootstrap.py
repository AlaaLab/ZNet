#######################################################################################
# Author: Jenna Fields, Franny Dean
# Script: run_bootstrap.py
# Function: Run bootstrapping to evaluate stability of results across random data shuffling. Can be used to evaluate both ZNet and downstream estimator performance across bootstraps.
# Date: 02/06/2026
#######################################################################################

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import multiprocessing
import argparse
from utils.pipeline_utils import run_bootstrap_combination, dataset_setup
from datetime import datetime
import json

#######################################################################################

def load_dict_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def reshuffle_bootstrap(current_dataset, params, num_bootstraps, dir_name, results_dict, eval_params):
    print(f"Launching data shuffling bootstrap for {num_bootstraps} bootstraps.")
    result = run_bootstrap_combination(params, current_dataset, num_bootstraps, eval_params)
    print("Bootstrap complete. Aggregating results...")
    for k in result:
        if k not in results_dict:
            results_dict[k] = []
        results_dict[k].extend(result[k])

    df = pd.DataFrame(results_dict)
    df.to_csv(f"{dir_name}/final.csv", index=False)
    print(f"Results saved to directory: {f'{dir_name}/'}")

    print("Done!")

def main(all_params, datasets, num_bootstraps, output_path=None):
    multiprocessing.freeze_support()
    multiprocessing.set_start_method("spawn", force=True)

    time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    eval_params = {}
    all_params_for_bootstrap = {}
    for dataset_name in datasets:
        znet_params = all_params[dataset_name]['znet_params'] if 'znet_params' in all_params[dataset_name] else {}
        deep_iv_params = all_params[dataset_name]['deep_iv_params'] if 'deep_iv_params' in all_params[dataset_name] else {}
        df_iv_params = all_params[dataset_name]['df_iv_params'] if 'df_iv_params' in all_params[dataset_name] else {}
        znet_train_params = all_params[dataset_name]['znet_train_params'] if 'znet_train_params' in all_params[dataset_name] else {}
        tarnet_params = all_params[dataset_name]['tarnet_params'] if 'tarnet_params' in all_params[dataset_name] else {}
        tarnet_train_params = all_params[dataset_name]['tarnet_train_params'] if 'tarnet_train_params' in all_params[dataset_name] else {}
        dim_options = all_params[dataset_name]['dim_options'] if 'dim_options' in all_params[dataset_name] else {}

        # --------------------------------------------------------------------------------
        znet_params_labelled = {f'znet_params_{k}': v for k, v in znet_params.items()}
        deep_iv_params_labelled = {f'deep_iv_params_{k}': v for k, v in deep_iv_params.items()}
        df_iv_params_labelled = {f'df_iv_params_{k}': v for k, v in df_iv_params.items()}
        tarnet_params_labelled = {f'tarnet_params_model_{k}': v for k, v in tarnet_params.items()}
        tarnet_train_params_labelled = {f'tarnet_params_train_{k}': v for k, v in tarnet_train_params.items()}
        znet_train_params_labelled = {f'znet_train_params_{k}': v for k, v in znet_train_params.items()}
        dim_options_labelled = {f'dim_options_{k}': v for k, v in dim_options.items()}
        methods = ['tsls', 'ols', 'diff_in_means', 'exogeneity', 'independence', 'relevance', 'u_z_independence']
        if len(deep_iv_params) > 0:
            methods.append('deep_iv')
        if len(df_iv_params) > 0:
            methods.append('df_iv')
        if len(tarnet_params) > 0:
            methods.append('tarnet')
        eval_params[dataset_name] = {'methods' : methods}
        all_params_for_bootstrap[dataset_name] = {**znet_params_labelled,
                                    **znet_train_params_labelled,
                                    **deep_iv_params_labelled,
                                    **df_iv_params_labelled,
                                    **dim_options_labelled,
                                    **tarnet_params_labelled,
                                    **tarnet_train_params_labelled
                                    }
    
    results_dirs = {}
    for dataset_name, current_dataset in datasets.items():
        print(f"Running bootstrap for dataset: {dataset_name}")
        results_dict = {}
        if output_path is not None:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            dir_name = os.path.join(output_path, 'bootstrap_results', f"{dataset_name}_{time}")
        else:
            dir_name = f"bootstrap_results/{dataset_name}_{time}"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        reshuffle_bootstrap(current_dataset, all_params_for_bootstrap[dataset_name], 
                                num_bootstraps=num_bootstraps, 
                                dir_name=dir_name,
                                results_dict=results_dict,
                                eval_params=eval_params[dataset_name]
                                )

        print(f"Completed bootstrap for dataset: {dataset_name}")
        results_dirs[dataset_name] = dir_name

    return results_dirs

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Bootstrap")

    parser.add_argument("--param_json", type=str, help="Path to the JSON file containing grid search parameters")
    parser.add_argument("--dataset_name", type=list, default=None, help="List of datasets to run grid search on (if None, will generate and run on all datasets)")
    parser.add_argument("--generate_data", action='store_true', help="Flag to generate data or use existing data (if no flag, will use data args.dataset_dir)")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Directory to load datasets from")
    parser.add_argument("--num_bootstraps", type=int, default=100, help="Number of bootstraps to run")
    args = parser.parse_args()

    datasets = dataset_setup(args)

    all_params = json.load(open(args.param_json))
    main(all_params, datasets, args.num_bootstraps)
