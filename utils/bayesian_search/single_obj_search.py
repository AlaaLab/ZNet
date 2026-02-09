#######################################################################################
# Author: Jenna Fields
# Script: single_obj_search.py
# Function: single-objective functions for bayesian tuning
# Date: 02/06/2026
#######################################################################################

import torch
import numpy as np
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.transforms import standardize, normalize
from seed_utils import set_seed, reset_to_random

class BotorchOptimizer:
    """Simple Botorch-based Bayesian Optimizer"""

    def __init__(self, objective_func, param_bounds, device=None, minimize=True, beta=0.1, dir_name=None):
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.objective = objective_func
        self.device = device or torch.device("cpu")
        
        # Move bounds to device
        self.bounds = torch.stack([
            torch.tensor([bounds[0] for bounds in param_bounds.values()], dtype=torch.float64),
            torch.tensor([bounds[1] for bounds in param_bounds.values()], dtype=torch.float64)
        ])

        
        # Storage for observations
        self.X = torch.empty(0, len(self.bounds[0]), dtype=torch.float64, device=self.device)
        self.Y = torch.empty(0, 1, dtype=torch.float64, device=self.device)
        self.minimize = minimize
        self.beta = beta
        self.dir_name = dir_name

    def params_from_tensor(self, x_tensor):
        """Convert normalized tensor to parameter dictionary."""
        params = {}
        for i, name in enumerate(self.param_names):
            value = x_tensor[i].item()
            params[name] = value 
        return params
    
    def generate_initial_data(self, n_init=5):
        """Generate initial random observations"""
        # Sample random points within bounds
        # Reset random seed for initial points
        reset_to_random()
        X_init = torch.rand(n_init, self.bounds.shape[1], device=self.device)
        X_init = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * X_init
        set_seed(42)  # Reset seed for reproducibility in optimization
        # Evaluate objective function
        Y_init = torch.tensor(
            [[self.objective(self.params_from_tensor(x.cpu().numpy()))] for x in X_init],
            dtype=torch.float64,
            device=self.device
        )
        
        self.X = X_init
        self.Y = Y_init
        
    def fit_model(self):
        """Fit a Gaussian Process model to current data"""
        # Normalize inputs and standardize outputs
        X_norm = normalize(self.X, self.bounds)
        Y_std = standardize(self.Y)
        
        # Create and fit GP model
        model = SingleTaskGP(X_norm, Y_std)
        model = model.to(self.device)
        
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        # Check model state
        return model, X_norm, Y_std
    
    
    def optimize_acquisition(self, model, acquisition_func, bounds_norm):
        """Optimize the acquisition function to find next candidate"""
        candidate, acq_value = optimize_acqf(
            acq_function=acquisition_func,
            bounds=bounds_norm,
            q=1,  # Number of candidates to generate
            num_restarts=20,
            raw_samples=256,  # Number of raw samples for initialization
            sequential=True,
            options={"maxiter": 100, "ftol":1e-6, "factr":None},
        )
        return candidate
    
    def run_optimization(self, n_iterations=10, acquisition_type='EI', n_init=5, verbose=True):
        """Run Bayesian optimization loop"""
        # Generate initial data
        self.generate_initial_data(n_init=n_init)

        # Normalized bounds for acquisition optimization
        bounds_norm = torch.stack([
            torch.zeros(self.bounds.shape[1], device=self.device),
            torch.ones(self.bounds.shape[1], device=self.device)
        ])
        
        results = []
        
        for i in range(n_iterations):
            print(f"Iteration {i+1}/{n_iterations}")
            
            # Fit GP model
            model, X_norm, Y_std = self.fit_model()
            
            # Choose acquisition function
            if acquisition_type == 'EI':
                acq_func = LogExpectedImprovement(model, best_f=Y_std.max()) 
            elif acquisition_type == 'UCB':
                acq_func = UpperConfidenceBound(model, beta=self.beta) 
            else:
                raise ValueError("Unknown acquisition function")
            
            # Optimize acquisition function
            candidate_norm = self.optimize_acquisition(model, acq_func, bounds_norm)
            
            # Convert back to original scale
            candidate = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * candidate_norm
            
            # Evaluate objective
            y_new = torch.tensor(
                [[self.objective(self.params_from_tensor(candidate.squeeze().cpu().numpy()))]],
                dtype=torch.float64,
                device=self.device
            )
            
            # Add to dataset
            self.X = torch.cat([self.X, candidate])
            self.Y = torch.cat([self.Y, y_new])
            
            best_idx = self.Y.argmax()
            results.append({
                'iteration': i + 1,
                'best_x': self.X[best_idx].cpu().numpy(),
                'best_params': self.params_from_tensor(self.X[best_idx].cpu().numpy()),
                'best_y': self.Y[best_idx].item(),
                'current_x': candidate.squeeze().cpu().numpy(),
                'current_y': y_new.item()
            })
            if verbose:
                print(f"  Current: f({candidate.squeeze().cpu().numpy()}) = {y_new.item():.4f}")
                print(f"  Best so far: f({self.X[best_idx].cpu().numpy()}) = {self.Y[best_idx].item():.4f}")
            if self.dir_name is not None:
                import json
                # Check if file exists
                import os
                if not os.path.exists(self.dir_name):
                    os.makedirs(self.dir_name)
                # Save progress after each iteration
                with open(f"{self.dir_name}/bayesian_optimization_progress.json", "a") as f:
                    json.dump(results[-1]['best_params'], f, indent=4)
        
        return results