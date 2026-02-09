#######################################################################################
# Author: Jenna Fields
# Script: multi_obj_search.py
# Function: Multi-objective functions for bayesian tuning
# Date: 02/06/2026
#######################################################################################

import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.models.transforms import Standardize
from botorch.acquisition.multi_objective import qLogExpectedHypervolumeImprovement, qNoisyExpectedHypervolumeImprovement, qLogNoisyExpectedHypervolumeImprovement
from botorch.optim import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.utils.sampling import draw_sobol_samples
from botorch.sampling import SobolQMCNormalSampler
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize

from seed_utils import set_seed, reset_to_random
# Set dtype for better numerical stability
torch.set_default_dtype(torch.float32)

class MultiObjectiveHyperparameterTuner:
    """
    Multiobjective Bayesian hyperparameter tuning using BoTorch.
    Optimizes multiple conflicting objectives simultaneously.
    """
    
    def __init__(self, param_bounds, objective, objective_names=None, dir_name=None, verbose=True):
        """
        Initialize the tuner.
        
        Args:
            param_bounds: dict with parameter names as keys and (min, max) tuples as values
            objective_names: list of objective names for plotting/logging
        """
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_params = len(self.param_names)
        self.objective_names = objective_names or [f"Objective_{i}" for i in range(2)]
        self.objective_function = objective
        # Convert bounds to BoTorch format
        self.bounds = torch.stack([
            torch.tensor([bounds[0] for bounds in param_bounds.values()], dtype=torch.float64),
            torch.tensor([bounds[1] for bounds in param_bounds.values()], dtype=torch.float64)
        ])
        
        # Storage for trials
        self.X = torch.empty(0, self.n_params)
        self.Y = torch.empty(0, 2)  # Assuming 2 objectives
        self.trial_history = []
        self.dir_name = dir_name
        self.verbose = verbose

    def params_from_tensor(self, x_tensor):
        """Convert normalized tensor to parameter dictionary."""
        params = {}
        for i, name in enumerate(self.param_names):
            value = x_tensor[i].item()
            params[name] = value 
        return params
    
    def evaluate_parameters(self, x_tensor):
        """Evaluate parameters and return multiobjective values."""
        params = self.params_from_tensor(x_tensor)
        objectives = self.objective_function(params)
        
        # Store trial
        self.trial_history.append({
            'params': params.copy(),
            'objectives': objectives
        })
        if self.verbose:
            print(f"Trial {len(self.trial_history)}: {params}")
            print(f"  Objectives: {self.objective_names[0]}={objectives[0]:.4f}, "
                f"{self.objective_names[1]}={objectives[1]:.4f}")
        
        return torch.tensor(objectives, dtype=torch.float64)
    
    def get_initial_points(self, n_initial=5):
        """Generate initial random points using Sobol sampling."""
        reset_to_random()
        X_init = draw_sobol_samples(bounds=self.bounds, n=n_initial, q=1).squeeze(-2)
        set_seed(42)
        Y_init = torch.stack([self.evaluate_parameters(x) for x in X_init])

        self.X = X_init
        self.Y = Y_init

        return X_init, Y_init   

    def fit_model(self):
        """Fit Gaussian Process model to current data."""
        norm_X = normalize(self.X, self.bounds)
        std_Y = standardize(self.Y)
        # Create multi-output GP
        model = SingleTaskGP(
            norm_X, 
            std_Y, 
            outcome_transform=Standardize(m=std_Y.shape[-1])
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)
        
        return model, norm_X, std_Y
    
    def get_next_point(self, model, norm_X, std_Y):
        """Get next point to evaluate using multiobjective acquisition function."""
        # Create hypervolume improvement acquisition function
        ref_point = std_Y.min(dim=0)[0] - 0.1 * (std_Y.max(dim=0)[0] - std_Y.min(dim=0)[0])

        sampler = SobolQMCNormalSampler(sample_shape=torch.Size([128]))

        # Use current Pareto front for reference
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=std_Y,
        )
        
        acq_func = qLogNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            X_baseline=norm_X,  # Use all observed points as baseline
            sampler=sampler,
            prune_baseline=True,
            cache_pending=True
        )
        
        # Optimize acquisition function
        candidates, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=self.bounds,
            q=1,
            num_restarts=20,
            raw_samples=512,
            options={"batch_limit": 5, "maxiter": 200},
        )
        
        return candidates[0]
    
    def optimize(self, n_iterations=20, n_initial=5):
        """Run the multiobjective Bayesian optimization."""
        if self.verbose:
            print("Starting Multiobjective Bayesian Optimization...")
            print("="*50)

        # Get initial points
        if self.verbose:
            print(f"Generating {n_initial} initial points...")
        self.get_initial_points(n_initial)
        
        # Main optimization loop
        for i in range(n_iterations):
            if self.verbose:
                print(f"\nIteration {i+1}/{n_iterations}")
                print("-" * 30)

            # Fit GP model
            model, norm_X, std_Y = self.fit_model()

            # Get next point to evaluate
            next_x = self.get_next_point(model, norm_X, std_Y)
            next_y = self.evaluate_parameters(next_x)
            
            # Add to dataset
            self.X = torch.cat([self.X, next_x.unsqueeze(0)])
            self.Y = torch.cat([self.Y, next_y.unsqueeze(0)])

            # Save intermediate results
            if self.dir_name is not None:
                import pandas as pd
                df = pd.DataFrame(self.trial_history)
                df.to_csv(f"{self.dir_name}/intermediate_results.csv", index=False)


    def get_pareto_front(self):
        """Extract the Pareto front from all evaluations."""
        pareto_mask = torch.ones(len(self.Y), dtype=torch.bool)
        
        for i, y_i in enumerate(self.Y):
            for j, y_j in enumerate(self.Y):
                if i != j and torch.all(y_j >= y_i) and torch.any(y_j > y_i):
                    pareto_mask[i] = False
                    break
        
        pareto_X = self.X[pareto_mask]
        pareto_Y = self.Y[pareto_mask]
        pareto_params = [self.trial_history[i]['params'] for i in range(len(pareto_mask)) if pareto_mask[i]]
        
        return pareto_X, pareto_Y, pareto_params
    
    def plot_results(self):
        """Plot the optimization results."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Objective space with Pareto front
        Y_np = self.Y.numpy()
        ax1.scatter(Y_np[:, 0], Y_np[:, 1], alpha=0.6, s=50, label='All points')
        
        # Highlight Pareto front
        _, pareto_Y, _ = self.get_pareto_front()
        pareto_Y_np = pareto_Y.numpy()
        ax1.scatter(pareto_Y_np[:, 0], pareto_Y_np[:, 1], 
                   color='red', s=100, label='Pareto front', marker='*')
        
        ax1.set_xlabel(self.objective_names[0])
        ax1.set_ylabel(self.objective_names[1])
        ax1.set_title('Multiobjective Optimization Results')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Convergence over iterations
        iterations = range(1, len(Y_np) + 1)
        ax2_twin = ax2.twinx()
        ax2.plot(iterations, Y_np[:, 0], 'b-o', label=self.objective_names[0], markersize=4)
        ax2_twin.plot(iterations, Y_np[:, 1], 'r-s', label=self.objective_names[1], markersize=4)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Objective Value')
        ax2.set_title('Convergence Plot')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig
    
    def print_pareto_solutions(self):
        """Print the Pareto optimal solutions."""
        _, pareto_Y, pareto_params = self.get_pareto_front()
        
        print("\n" + "="*60)
        print("PARETO OPTIMAL SOLUTIONS")
        print("="*60)
        
        for i, (params, objectives) in enumerate(zip(pareto_params, pareto_Y)):
            print(f"\nSolution {i+1}:")
            print(f"  Parameters: {params}")
            print(f"  {self.objective_names[0]}: {objectives[0].item():.4f}")
            print(f"  {self.objective_names[1]}: {objectives[1].item():.4f}")

    def get_pareto_front_indices(self):
        """Extract the Pareto front from all evaluations."""
        pareto_mask = torch.ones(len(self.Y), dtype=torch.bool)
        
        for i, y_i in enumerate(self.Y):
            for j, y_j in enumerate(self.Y):
                if i != j and torch.all(y_j >= y_i) and torch.any(y_j > y_i):
                    pareto_mask[i] = False
                    break
        
        pareto_Y = self.Y[pareto_mask]
        pareto_params = [self.trial_history[i]['params'] for i in range(len(pareto_mask)) if pareto_mask[i]]
        
        return pareto_mask, pareto_Y, pareto_params
