#######################################################################################
# Author: Jenna Fields
# Script:  main_driver_bayesian.py
# Function: Runs entire pipeline for ZNet and downstream models, including bayesian search and bootstrap.
# Date: 02/06/2026
#######################################################################################

import os
import sys

print("Current working directory:", os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print("Current working directory:", os.getcwd())

from datetime import datetime
from bayesian_search_ivgen import main as bayesian_search_main
from run_bootstrap import main as bootstrap_main
from bayesian_search_downstream import main as downstream_bayesian_search_main
from DGP.generate_datasets import generate_datasets, load_object_dataset
from DGP.dataset_class import TrueIVDataset
from utils.pipeline_utils import convert_bootstrap_params

import numpy as np
import json 
import argparse
from os import makedirs
import pandas as pd 

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
    
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="Main driver script")
    parser.add_argument("--param_json", type=str, default="params.json", help="Path to the JSON file containing bounds parameters")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated list of datasets to run bayesian search on (if None, will generate and run on all datasets)")
    parser.add_argument("--num_bootstraps", type=int, default=1, help="Number of bootstrap samples to generate")
    parser.add_argument("--tarnet_param_json", type=str, default=None, help="Path to the JSON file containing parameters for tarnet search. Will only be used if 'tarnet_params' is in param_json or if --run_all_with_tarnet is set.")
    parser.add_argument("--compare_methods", type=str, default='autoiv,giv,viv,trueiv', help="Comma-separated list of methods to compare")
    parser.add_argument("--run_all_with_tarnet", action='store_true', help="Flag to indicate if all methods should be run with TARNet. Only relevant if 'tarnet_params' is in param_json.")
    parser.add_argument("--ncalls_bayesian", type=int, default=10, help="Number of calls for the bayesian search")
    parser.add_argument("--only_search_znet", action='store_true', help="Flag to indicate if we should not search all generative IV methods (if no flag, will use default parameters for non-ZNet methods)")
    parser.add_argument("--n_initial_points_bayesian", type=int, default=5, help="Number of initial points for the bayesian search")
    parser.add_argument("--skip_znet", action='store_true', help="Flag to skip ZNet and only run other methods")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--existing_znet_data", type=str, default=None, help="Path to json containing paths to existing generative IV data to use instead of generating new data")
    parser.add_argument("--dataset_version", type=int, default=None, help="Version of datasets to use (defaults to original)")
    args = parser.parse_args()
    other_methods = args.compare_methods.split(',')
    if args.dataset_version is None:
        name_edit = ""
    else:
        name_edit =  f"_v{args.dataset_version}"
    all_datasets = generate_datasets(version=args.dataset_version)
    if args.datasets is not None:
        datasets = {k + name_edit: all_datasets[k + name_edit][1] for k in args.datasets.split(",") if k + name_edit in all_datasets}
    else:
        datasets = {k + name_edit: all_datasets[k + name_edit][1] for k in all_datasets if k + name_edit in all_datasets}
    if args.existing_znet_data is not None:
        existing_data_info = json.load(open(args.existing_znet_data))
    else:
        existing_data_info = {}
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Check that requirements for parameter combos are met
    
    all_params = json.load(open(args.param_json))

    if args.run_all_with_tarnet:
        if args.tarnet_param_json is None and 'tarnet_params' not in all_params:
            raise ValueError("TARNet parameter JSON file or tarnet_params in param_json must be specified when running all methods with TARNet.")
    if 'tarnet_params' in all_params and 'tarnet_train_params' not in all_params:
        raise ValueError("If 'tarnet_params' is specified in param_json, 'tarnet_train_params' must also be specified.")
    if 'tarnet_train_params' in all_params and 'tarnet_params' not in all_params:
        raise ValueError("If 'tarnet_train_params' is specified in param_json, 'tarnet_params' must also be specified.")
    if args.run_all_with_tarnet:
        if 'tarnet_params' not in all_params:
            tarnet_params = json.load(open(args.tarnet_param_json))
            all_params = {**all_params, **tarnet_params}
    summary_results = {}
    for dataset_name, dataset in datasets.items():
        summary_results[dataset_name] = {'true_ate': datasets[dataset_name].ite.mean()}
    makedirs(f"{args.output_dir}/bootstrap_summary_results", exist_ok=True)
    pd.DataFrame(summary_results).T.to_csv(f"{args.output_dir}/bootstrap_summary_results/bootstrap_summary_results_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    ##################################################
    # ZNet
    if not args.skip_znet:
        print('Running ZNet')
        datasets_to_run = {}
        for dataset_name in datasets:
            if dataset_name not in existing_data_info:
                datasets_to_run[dataset_name] = datasets[dataset_name]
        if len(datasets_to_run) > 0:
            znet_bounds = {**{f'znet_params_{k}': v for k, v in all_params['znet_params'].items()},
                            **{f'znet_train_params_{k}': v for k, v in all_params['znet_train_params'].items()},
                            **{f'dim_options_{k}': v for k, v in all_params['dim_options'].items()}}
            all_znet_data, all_best_znet_params = bayesian_search_main(znet_bounds, datasets_to_run, 'znet', return_data=True, n_calls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
        else:
            all_znet_data = {}
            all_best_znet_params = {}
        for dataset_name in datasets:
            if dataset_name in existing_data_info:
                # Load existing ZNet data
                znet_data = load_object_dataset(existing_data_info[dataset_name], 'znet_dataset')
                all_znet_data[dataset_name] = znet_data
                all_best_znet_params[dataset_name] = {}
        all_best_params_downstream = downstream_bayesian_search_main(all_params, all_znet_data, return_params=True, ncalls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
        bootstrap_params = {}
        # print(all_best_znet_params)
        for dataset_name in datasets:
            bootstrap_params[dataset_name] = {
                'znet_params': {k[len('znet_params_'):]: v for k, v in all_best_znet_params[dataset_name].items() if k.startswith('znet_params_')},
                'deep_iv_params': all_best_params_downstream[dataset_name]['deep_iv_params'],
                'df_iv_params': all_best_params_downstream[dataset_name]['df_iv_params'],
                'znet_train_params': {k[len('znet_train_params_'):]: v for k, v in all_best_znet_params[dataset_name].items() if k.startswith('znet_train_params_')},
                'tarnet_params': all_best_params_downstream[dataset_name]['tarnet_params'],
                'tarnet_train_params': all_best_params_downstream[dataset_name]['tarnet_train_params'],
                'dim_options': {k[len('dim_options_'):]: v for k, v in all_best_znet_params[dataset_name].items() if k.startswith('dim_options_')},
            }
        
        bootstrap_params = convert_numpy_recursive(bootstrap_params)

        results_dirs = bootstrap_main(bootstrap_params, all_znet_data, args.num_bootstraps, output_path=args.output_dir)
        for dataset_name, dir_name in results_dirs.items():
            df_final = pd.read_csv(f"{dir_name}/final.csv")
            for col in df_final.columns:
                if 'ate' in col.lower() and 'std' not in col.lower():
                    mean_col = df_final[col].mean()
                    median_col = df_final[col].median()
                    std_col = df_final[col].std()
                    se_col = std_col / np.sqrt(len(df_final[col]))
                    summary_results[dataset_name][f"znet_{col}_mean"] = mean_col
                    summary_results[dataset_name][f"znet_{col}_median"] = median_col
                    summary_results[dataset_name][f"znet_{col}_std"] = std_col
                    summary_results[dataset_name][f"znet_{col}_se"] = se_col

        pd.DataFrame(summary_results).T.to_csv(f"{args.output_dir}/bootstrap_summary_results/bootstrap_summary_results_checkpoint_znet_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    ##################################################
    # AUTOIV
    if 'autoiv' in other_methods:
        print('Running AutoIV')
        from utils.train_models import train_autoiv
        if not args.only_search_znet:
            autoiv_bounds = {f'autoiv_params_{k}': v for k, v in all_params['autoiv_params'].items()}
            all_autoiv_data, all_best_autoiv_params = bayesian_search_main(autoiv_bounds, datasets, 'autoiv', return_data=True, n_calls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
            all_autoiv_data = {f'autoiv_{k}': v for k, v in all_autoiv_data.items()}
            all_best_autoiv_params = {f'autoiv_{k}': v for k, v in all_best_autoiv_params.items()}
        else:
            all_autoiv_data = {}
            for dataset_name in datasets:
                _, autoiv_data, _ = train_autoiv(datasets[dataset_name])
                all_autoiv_data[f'autoiv_{dataset_name}'] = autoiv_data
        autoiv_params_downstream = downstream_bayesian_search_main(all_params, all_autoiv_data, return_params=True, ncalls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
        autoiv_bootstrap_params = {}
        for dataset_name in all_autoiv_data:
            autoiv_bootstrap_params[dataset_name] = {
                'autoiv_params': all_best_autoiv_params[dataset_name] if not args.only_search_znet else {},
                'deep_iv_params': autoiv_params_downstream[dataset_name]['deep_iv_params'],
                'df_iv_params': autoiv_params_downstream[dataset_name]['df_iv_params'],
                'tarnet_params': autoiv_params_downstream[dataset_name]['tarnet_params'],
                'tarnet_train_params': autoiv_params_downstream[dataset_name]['tarnet_train_params'],
            } 
        autoiv_bootstrap_params = convert_bootstrap_params(autoiv_bootstrap_params)
        results_dirs = bootstrap_main(autoiv_bootstrap_params, all_autoiv_data, args.num_bootstraps, output_path=args.output_dir)
        for dataset_name in summary_results:
            df_final = pd.read_csv(f"{results_dirs[f'autoiv_{dataset_name}']}/final.csv")
            for col in df_final.columns:
                if 'ate' in col.lower() and 'std' not in col.lower():
                    mean_col = df_final[col].mean()
                    median_col = df_final[col].median()
                    std_col = df_final[col].std()
                    se_col = std_col / np.sqrt(len(df_final[col]))
                    summary_results[dataset_name][f"autoiv_{col}_mean"] = mean_col
                    summary_results[dataset_name][f"autoiv_{col}_median"] = median_col
                    summary_results[dataset_name][f"autoiv_{col}_std"] = std_col
                    summary_results[dataset_name][f"autoiv_{col}_se"] = se_col
        pd.DataFrame(summary_results).T.to_csv(f"{args.output_dir}/bootstrap_summary_results/bootstrap_summary_results_checkpoint_autoiv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    ##################################################
    # GIV
    if 'giv' in other_methods:
        print('Running GIV')
        from utils.train_models import train_giv
        if not args.only_search_znet:
            giv_bounds = {f'giv_params_{k}': v for k, v in all_params['giv_params'].items()}
            all_giv_data, all_best_giv_params = bayesian_search_main(giv_bounds, datasets, 'giv', return_data=True, n_calls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
            all_giv_data = {f'giv_{k}': v for k, v in all_giv_data.items()}
            all_best_giv_params = {f'giv_{k}': v for k, v in all_best_giv_params.items()}
        else:
            all_giv_data = {}
            for dataset_name in datasets:
                _, giv_data, _ = train_giv(datasets[dataset_name], model_params={'num_clusters': 5})
                all_giv_data[f'giv_{dataset_name}'] = giv_data
        giv_params_downstream = downstream_bayesian_search_main(all_params, all_giv_data, return_params=True, ncalls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
        giv_bootstrap_params = {}
        for dataset_name in all_giv_data:
            giv_bootstrap_params[dataset_name] = {
                'giv_params': all_best_giv_params[dataset_name] if not args.only_search_znet else {},
                'deep_iv_params': giv_params_downstream[dataset_name]['deep_iv_params'],
                'df_iv_params': giv_params_downstream[dataset_name]['df_iv_params'],
                'tarnet_params': giv_params_downstream[dataset_name]['tarnet_params'],
                'tarnet_train_params': giv_params_downstream[dataset_name]['tarnet_train_params'],
            }
        giv_bootstrap_params = convert_bootstrap_params(giv_bootstrap_params)
        results_dirs = bootstrap_main(giv_bootstrap_params, all_giv_data, args.num_bootstraps, output_path=args.output_dir)
        for dataset_name in summary_results:
            df_final = pd.read_csv(f"{results_dirs[f'giv_{dataset_name}']}/final.csv")
            for col in df_final.columns:
                if 'ate' in col.lower() and 'std' not in col.lower():
                    mean_col = df_final[col].mean()
                    median_col = df_final[col].median()
                    std_col = df_final[col].std()
                    se_col = std_col / np.sqrt(len(df_final[col]))
                    summary_results[dataset_name][f"giv_{col}_mean"] = mean_col
                    summary_results[dataset_name][f"giv_{col}_median"] = median_col
                    summary_results[dataset_name][f"giv_{col}_std"] = std_col
                    summary_results[dataset_name][f"giv_{col}_se"] = se_col
        pd.DataFrame(summary_results).T.to_csv(f"{args.output_dir}/bootstrap_summary_results/bootstrap_summary_results_checkpoint_giv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    ##################################################
    # VIV
    if 'viv' in other_methods:
        print('Running VIV')
        from utils.train_models import train_viv
        if not args.only_search_znet:
            viv_bounds = {f'viv_params_{k}': v for k, v in all_params['viv_params'].items()}
            all_viv_data, all_best_viv_params = bayesian_search_main(viv_bounds, datasets, 'viv', return_data=True, n_calls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
            all_viv_data = {f'viv_{k}': v for k, v in all_viv_data.items()}
            all_best_viv_params = {f'viv_{k}': v for k, v in all_best_viv_params.items()}
        else:
            all_viv_data = {}
            for dataset_name in datasets:
                _, viv_data, _ = train_viv(datasets[dataset_name])
                all_viv_data[f'viv_{dataset_name}'] = viv_data
        viv_params_downstream = downstream_bayesian_search_main(all_params, all_viv_data, return_params=True, ncalls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
        viv_bootstrap_params = {}
        for dataset_name in all_viv_data:
            viv_bootstrap_params[dataset_name] = {
                'viv_params': all_best_viv_params[dataset_name] if not args.only_search_znet else {},
                'deep_iv_params': viv_params_downstream[dataset_name]['deep_iv_params'],
                'df_iv_params': viv_params_downstream[dataset_name]['df_iv_params'],
                'tarnet_params': viv_params_downstream[dataset_name]['tarnet_params'],
                'tarnet_train_params': viv_params_downstream[dataset_name]['tarnet_train_params'],
            }
        viv_bootstrap_params = convert_bootstrap_params(viv_bootstrap_params)
        results_dirs = bootstrap_main(viv_bootstrap_params, all_viv_data, args.num_bootstraps, output_path=args.output_dir)
        for dataset_name in summary_results:
            df_final = pd.read_csv(f"{results_dirs[f'viv_{dataset_name}']}/final.csv")
            for col in df_final.columns:
                if 'ate' in col.lower() and 'std' not in col.lower():
                    mean_col = df_final[col].mean()
                    median_col = df_final[col].median()
                    std_col = df_final[col].std()
                    se_col = std_col / np.sqrt(len(df_final[col]))
                    summary_results[dataset_name][f"viv_{col}_mean"] = mean_col
                    summary_results[dataset_name][f"viv_{col}_median"] = median_col
                    summary_results[dataset_name][f"viv_{col}_std"] = std_col
                    summary_results[dataset_name][f"viv_{col}_se"] = se_col
        pd.DataFrame(summary_results).T.to_csv(f"{args.output_dir}/bootstrap_summary_results/bootstrap_summary_results_checkpoint_viv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    ##################################################
    # True IV
    if 'trueiv' in other_methods:
        print('Running TrueIV')
        all_true_iv_data = {}
        for dataset_name in datasets:
            # We don't want to run TRUE IV inference for no_cand datasets
            if 'no_cand' in dataset_name:
                continue
            true_iv_data = TrueIVDataset(datasets[dataset_name])
            all_true_iv_data[f'true_iv_{dataset_name}'] = true_iv_data
        true_iv_params_downstream = downstream_bayesian_search_main(all_params, all_true_iv_data, return_params=True, ncalls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
        true_iv_bootstrap_params = {}
        for dataset_name in all_true_iv_data:
            true_iv_bootstrap_params[dataset_name] = {
                'deep_iv_params': true_iv_params_downstream[dataset_name]['deep_iv_params'],
                'df_iv_params': true_iv_params_downstream[dataset_name]['df_iv_params'],
                'tarnet_params': true_iv_params_downstream[dataset_name]['tarnet_params'],
                'tarnet_train_params': true_iv_params_downstream[dataset_name]['tarnet_train_params'],
            }
        true_iv_bootstrap_params = convert_bootstrap_params(true_iv_bootstrap_params)
        results_dirs = bootstrap_main(true_iv_bootstrap_params, all_true_iv_data, args.num_bootstraps, output_path=args.output_dir)
        for dataset_name in summary_results:
            # We don't want to run TRUE IV inference for no_cand datasets
            if 'no_cand' in dataset_name:
                continue
            df_final = pd.read_csv(f"{results_dirs[f'true_iv_{dataset_name}']}/final.csv")
            for col in df_final.columns:
                if 'ate' in col.lower() and 'std' not in col.lower():
                    mean_col = df_final[col].mean()
                    median_col = df_final[col].median()
                    std_col = df_final[col].std()
                    se_col = std_col / np.sqrt(len(df_final[col]))
                    summary_results[dataset_name][f"true_iv_{col}_mean"] = mean_col
                    summary_results[dataset_name][f"true_iv_{col}_median"] = median_col
                    summary_results[dataset_name][f"true_iv_{col}_std"] = std_col
                    summary_results[dataset_name][f"true_iv_{col}_se"] = se_col

        
        # Run the remaining with TARNet if there are tarnet params
        if args.tarnet_param_json is not None or 'tarnet_params' in all_params:
            no_cand_true_iv_data = {}
            if args.tarnet_param_json is not None:
                tarnet_params = json.load(open(args.tarnet_param_json))
            else:
                tarnet_params = {'tarnet_params' : all_params['tarnet_params'],'tarnet_train_params' : all_params['tarnet_train_params']}
            no_cand_all_params = {'znet_params': all_params['znet_params'], 'znet_train_params': all_params['znet_train_params'], 'dim_options': all_params['dim_options']}
            no_cand_all_params = {**no_cand_all_params, **tarnet_params}
            for dataset_name in datasets:
                # We don't want to run TRUE IV inference for no_cand datasets
                if 'no_cand' not in dataset_name:
                    continue
                true_iv_data = TrueIVDataset(datasets[dataset_name])
                no_cand_true_iv_data[f'true_iv_{dataset_name}'] = true_iv_data
            true_iv_params_downstream = downstream_bayesian_search_main(no_cand_all_params, no_cand_true_iv_data, return_params=True, ncalls=args.ncalls_bayesian, n_initial_points=args.n_initial_points_bayesian, output_path=args.output_dir)
            true_iv_bootstrap_params = {}
            for dataset_name in no_cand_true_iv_data:
                true_iv_bootstrap_params[dataset_name] = {
                    'tarnet_params': true_iv_params_downstream[dataset_name]['tarnet_params'],
                    'tarnet_train_params': true_iv_params_downstream[dataset_name]['tarnet_train_params'],
                }
            true_iv_bootstrap_params = convert_bootstrap_params(true_iv_bootstrap_params)
            results_dirs = bootstrap_main(true_iv_bootstrap_params, no_cand_true_iv_data, args.num_bootstraps, output_path=args.output_dir)
            for dataset_name in summary_results:
                # We don't want to run TRUE IV inference for no_cand datasets
                if 'no_cand' not in dataset_name:
                    continue
                df_final = pd.read_csv(f"{results_dirs[f'true_iv_{dataset_name}']}/final.csv")
                for col in df_final.columns:
                    if 'ate' in col.lower() and 'std' not in col.lower():
                        mean_col = df_final[col].mean()
                        median_col = df_final[col].median()
                        std_col = df_final[col].std()
                        se_col = std_col / np.sqrt(len(df_final[col]))
                        summary_results[dataset_name][f"true_iv_{col}_mean"] = mean_col
                        summary_results[dataset_name][f"true_iv_{col}_median"] = median_col
                        summary_results[dataset_name][f"true_iv_{col}_std"] = std_col
                        summary_results[dataset_name][f"true_iv_{col}_se"] = se_col
        pd.DataFrame(summary_results).T.to_csv(f"{args.output_dir}/bootstrap_summary_results/bootstrap_summary_results_checkpoint_trueiv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(summary_results).T.to_csv(f"{args.output_dir}/bootstrap_summary_results/final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
