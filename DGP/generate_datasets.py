#######################################################################################
# Author: Jenna Fields, edits by Franny Dean
# Script: generate_datasets.py
# Function: Create semi-synthetic data for evaluation
# Date: 02/06/2026
#######################################################################################

from DGP.dataset_class import ECG_DGPDataset
from seed_utils import set_seed

set_seed(42)

from glob import glob
import pandas as pd
import pyreadr
from DGP.phi_generation import PhiGeneration
import os
import pickle
import numpy as np

#######################################################################################

def check_zero_add(arr, epsilon=0.1, even_out=False):
    """
    Ensure no zero values in coefficient arrays and optionally balance positive/negative values.
    
    Adds epsilon to zero values and can rebalance the number of positive/negative
    coefficients to ensure identifiability in DGP.
    
    Args:
        arr (list or np.ndarray): Array of coefficients.
        epsilon (float): Value to add to zeros. Defaults to 0.1.
        even_out (bool): Balance positive/negative coefficient counts. Defaults to False.
        
    Returns:
        list: Modified coefficient array.
    """
    for i, a in enumerate(arr):
        if isinstance(a, list):
            for j, b in enumerate(a):
                if b == 0:
                    arr[i][j] += epsilon
        else:
            if a == 0:
                arr[i] += epsilon

    if even_out:
        original_shape = np.array(arr).shape
        arr = np.array(arr).flatten()
        new_arr = np.array(arr).copy()

        positives = [(p, v) for p, v in enumerate(arr) if v > 0]
        negatives = [(p, v) for p, v in enumerate(arr) if v < 0]
        while (len(positives) - len(negatives) > 0 and len(arr) % 2 == 0) or (len(positives) - len(negatives) > 1 and len(arr) % 2 == 1):
            # Turn min positive idx to negative
            min_pos = min([v for i, v in positives])
            min_idx = [i for i, v in positives if v == min_pos][0]
            positives = [i for i in positives if i[0] != min_idx]
            negatives = negatives + [(min_idx, -min_pos)]
            new_arr[min_idx] = -min_pos
        while (len(negatives) - len(positives) > 0 and len(arr) % 2 == 0) or (len(negatives) - len(positives) > 1 and len(arr) % 2 == 1):  
            # Turn max negative idx to positive
            max_neg = max([v for i, v in negatives])
            max_idx = [i for i, v in negatives if v == max_neg][0]
            negatives = [i for i in negatives if i[0] != max_idx]
            positives = positives + [(max_idx, -max_neg)]
            new_arr[max_idx] = -max_neg
        arr = np.reshape(new_arr, original_shape).tolist()     

    return arr

def generate_linear_disjoint(df, seed=None):
    """
    Generate synthetic dataset with linearly disjoint confounders and instruments.
    
    Creates a data generating process where:
    - Confounders affect both treatment and outcome
    - Instruments affect treatment but not outcome directly
    - Treatment effect is linear
    
    Args:
        df (pd.DataFrame): Base dataframe (e.g., IHDP data).
        seed (int, optional): Random seed for reproducibility. Uses preset values if None.
        
    Returns:
        DGPDataset: Generated dataset with known confounders and instruments.
    """
    select_covariates = [
                        'twin',
                        'mom.hs',
                        'mom.scoll',
                        'cig',
                        'first',
                        'booze',
                        'drugs',
                        'work.dur',
                        'prenatal',
                        'ein',
                        'sex',
                        'tex',
                        'bw', 
                        'b.head', 
                        'preterm', 
                        'birth.o'] 
        
    u_cols=['momage', 'ark', 'mom.lths',]
    ux_cols=[ 'booze', 'cig', 'first', 'work.dur','prenatal', 'drugs']
    xt_cols=['bw',  'b.head', 'preterm']
    xy_cols=['booze','work.dur','prenatal', 'first', 'drugs', 'sex']
    if seed is None:
        e_1=[[.3], [.4], [-.2]]
        e_2=[[-.8], [2], [.3]]
        xt_betas=[-1, -1, 4]
        xy_betas=[-1, 1, 3, 1, .3, 1]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-1, 1, 0.1), (3, 1)).tolist())
        e_2 = check_zero_add(np.random.choice(np.arange(-1, 1, 0.1), (3, 1)).tolist())
        xt_betas = check_zero_add(np.random.choice(np.arange(-1.5, 2.5, 0.5), 3).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-1, 3, 0.5), 6).tolist(), 0.5)
        phi_seed = seed 

    phi_1='random_additive'
    treatment_effect='linear'
    
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                 x_cols=select_covariates, 
                                 u_cols=u_cols,
                                 ux_cols=ux_cols,
                                 xt_cols=xt_cols,
                                 xy_cols=xy_cols,
                                 e_1=e_1,
                                 e_2=e_2,
                                 xt_betas=xt_betas,
                                 xy_betas=xy_betas,
                                 phi_1=phi_1,
                                 seed=phi_seed,
                                 treatment_effect=treatment_effect)
    return z_candidates_gen.gen_data()

def generate_linear_mixed(df, seed=None):
    """
    Generate synthetic dataset with linear mixed structure (partial overlap between confounders and instruments).
    
    Creates a DGP where some variables act as both confounders and instruments,
    representing a challenging identification scenario.
    
    Args:
        df (pd.DataFrame): Base dataframe (e.g., IHDP data).
        seed (int, optional): Random seed for reproducibility. Uses preset values if None.
        
    Returns:
        DGPDataset: Generated dataset with mixed confounders and instruments.
    """
    select_covariates = [
                'twin',
                'mom.hs',
                'mom.scoll',
                'cig',
                'first',
                'booze',
                'drugs',
                'work.dur',
                'prenatal',
                'ein',
                'sex',
                'tex',
                'bw', 
                'b.head', 
                'preterm', 
                'birth.o'] 
    if seed is None:
        e_1=[[1], [2], [-2]]
        xt_betas=[-2, -3, 2, 3, -4]
        xy_betas=[-1, 2, 1, -2]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 2, 0.2), (3, 1)).tolist())
        xt_betas = check_zero_add(np.random.choice(np.arange(-1, 3, 0.5), 5).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-1, 4, 1), 4).tolist(), 0.5)
        phi_seed = seed

    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                x_cols=select_covariates, 
                                u_cols=['momage', 'ark', 'mom.lths',],
                                ux_cols=[ 'booze', 'cig', 'first', 'work.dur','prenatal'], 
                                xt_cols=['bw',  'b.head', 'preterm', 'booze','work.dur',],
                                xy_cols=['booze','work.dur','prenatal', 'drugs'], 
                                treatment_effect='linear', 
                                phi_1 = 'simple_additive', 
                                e_1=e_1, 
                                xt_betas=xt_betas,
                                xy_betas=xy_betas, 
                                seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_linear_no_cand(df, seed=None):
    select_cov = [
            'twin',
            'mom.hs',
            'mom.scoll',
            'cig',
            'first',
            'booze',
            'drugs',
            'work.dur',
            'prenatal',
            'ein',
            'ark',
            'tex',
            'bw', 
            'b.head', 
            'preterm', 
            'birth.o'] 
    
    if seed is None:
        xt_betas = [1, 1, -3, 2]
        phi_seed = 42
    else:
        np.random.seed(seed)
        xt_betas = check_zero_add(np.random.choice(np.arange(-1, 3, 0.5), 4).tolist(), 0.5, even_out=True)
        phi_seed = seed

    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                     x_cols=select_cov,
                                     u_cols=['preterm', 'momwhite', 'tex'], 
                                     ux_cols=[ 'booze', 'cig', 'first', 'work.dur','prenatal'], 
                                     xt_cols=['bw',  'b.head','prenatal', 'drugs'], 
                                     xy_cols=['booze','work.dur','prenatal', 'drugs', 'bw',  'b.head'], 
                                     treatment_effect='linear', 
                                     phi_1 = 'random_additive',
                                     xt_betas=xt_betas, 
                                     seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_disjoint(df, seed=None):
    select_cov = [
                'twin',
                'mom.hs',
                'mom.scoll',
                'cig',
                'first',
                'booze',
                'drugs',
                'work.dur',
                'prenatal',
                'ein',
                'sex',
                'tex',
                'bw', 
                'b.head', 
                'preterm', 
                'birth.o'] 
    if seed is None:
        e_1 = [[1], [2], [-1]]
        xt_betas = [1, 5, -3]
        xy_betas=[-1, 2, 1, -2]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 2, 0.2), (3, 1)).tolist())
        xt_betas = check_zero_add(np.random.choice(np.arange(-1, 4, 0.5), 3).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-4, 4, 1), 4).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                x_cols=select_cov, 
                                u_cols=['momage', 'ark', 'mom.lths',],
                                ux_cols=[ 'booze', 'cig', 'first', 'work.dur','prenatal'], 
                                xt_cols=['bw',  'b.head', 'preterm'],
                                xy_cols=['booze','work.dur','prenatal', 'drugs'], 
                                treatment_effect='sigmoid', 
                                phi_1 = 'tx_nonadditive_pos', 
                                e_1=e_1, 
                                xt_betas=xt_betas,
                                xy_betas=xy_betas,
                                seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_mixed(df, seed=None):
    # Define the columns to be used in the phi generation
    select_cov = [
    'twin',
    'mom.hs',
    'mom.scoll',
    'cig',
    'first',
    'booze',
    'drugs',
    'work.dur',
    'prenatal',
    'ein',
    'sex',
    'tex',
    'bw', 
    'b.head', 
    'preterm', 
    'birth.o'] 
    if seed is None:
        xt_betas = [-2, 3, 2, 3, -4]
        xy_betas = [-1, 2, 1, -3]
        phi_seed = 42
    else:
        np.random.seed(seed)
        xt_betas = check_zero_add(np.random.choice(np.arange(-1, 3, 0.5), 5).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-4, 4, 1), 4).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                x_cols=select_cov, 
                                u_cols=['momage', 'ark', 'mom.lths',],
                                ux_cols=[ 'booze', 'cig', 'first', 'work.dur','prenatal'], 
                                xt_cols=['bw',  'b.head', 'preterm', 'booze','work.dur',],
                                xy_cols=['booze','work.dur','prenatal', 'drugs'], 
                                treatment_effect='linear', 
                                phi_1 = 'sin_tx', 
                                xt_betas=xt_betas,
                                xy_betas=xy_betas,
                                seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_no_cand(df, seed=None):
    select_cov = [
            'twin',
            'mom.hs',
            'mom.scoll',
            'cig',
            'first',
            'booze',
            'drugs',
            'momage',
            'work.dur',
            'prenatal',
            'ein',
            'ark',
            'tex',
            'bw', 
            'b.head', 
            'birth.o'] 
    if seed is None:
        e_1 = [[5], [3], [4]]
        e_2 = [[-1], [2], [3]]
        xt_betas = [-3, 1, -3, 2]
        xy_betas = [-1, 2, 1, -2, 3, 1]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 4, 0.2), (3, 1)).tolist())
        e_2 = check_zero_add(np.random.choice(np.arange(-2, 4, 0.2), (3, 1)).tolist())
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 3, 0.5), 4).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-4, 4, 1), 6).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, 
                                 phi_3='x',
                                 x_cols=select_cov, 
                                 u_cols=['preterm', 'momwhite', 'tex'], 
                                 ux_cols=[ 'booze', 'cig', 'first', 'work.dur','prenatal', 'drugs', 'bw',  'b.head'],
                                xt_cols=['bw',  'b.head', 'booze', 'prenatal'], 
                                xy_cols=['booze','work.dur','prenatal', 'drugs', 'bw',  'b.head'], 
                                treatment_effect='linear', 
                                phi_1='sin_tx', 
                                e_1=e_1, 
                                e_2=e_2, 
                                xt_betas=xt_betas, 
                                xy_betas=xy_betas,
                                seed=phi_seed)
    return z_candidates_gen.gen_data()


def generate_linear_giv(df, seed=None):
    select_cov = [
        'twin',
        'mom.hs',
        'mom.scoll',
        'cig',
        'first',
        'booze',
        'drugs',
        'work.dur',
        'prenatal',
        'ein',
        'sex',
        'tex',
        'bw', 
        'b.head', 
        'preterm', 
        'birth.o'] 
    giv_x = (df['preterm'] + df['birth.o']).values
    num_groups = 5
    range_for_group = (giv_x.max() - giv_x.min()) / num_groups
    for i in range(num_groups):
        # Create a mask for the current group
        mask = (giv_x >= (i * range_for_group) + giv_x.min() - 1) & (giv_x < ((i + 1) * range_for_group + giv_x.min() + 1))
        # Assign the group number to the corresponding rows in the DataFrame
        df.loc[mask, 'group_instr'] = i + 1
    
    if seed is None:
        e_3 = check_zero_add(np.random.choice(np.arange(-4, 4, 0.001), (3, 5)).tolist())
        e_2=[[3], [-4], [-3]]
        giv_coeffs=np.array([[ 1 ],
                            [4 ],
                            [2 ],
                            [-4],
                            [5]])
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_3 = check_zero_add(np.random.choice(np.arange(-4, 4, 0.001), (3, 5)).tolist(), 0.002)
        e_2 = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (3, 1)).tolist(), 0.5)
        giv_coeffs = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (5, 1)).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, phi_3='x_u',
                                 x_cols=select_cov, 
                                 u_cols='normal',
                                 u_cols_count=3,
                                 e_3=e_3, 
                                 ux_cols=[ 'booze', 'cig', 'work.dur','prenatal', 'drugs'],
                                 xt_cols=['birth.o', 'preterm', 'bw',  'b.head'], 
                                group_instr='group_instr',
                                 xy_cols=['booze','work.dur','prenatal', 'drugs', 'sex', 'ark', 'tex'], 
                                 e_2=e_2, 
                                 xt_betas=[0, 0, 0, 0], 
                                giv_coeffs=giv_coeffs,
                                 phi_1='random_additive',
                                 phi_2 = 'group_instrument',
                                 treatment_effect='linear', 
                                 seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_giv(df, seed=None):
    # Define the columns to be used in the phi generation
    select_cov = [
    'twin',
    'mom.hs',
    'mom.scoll',
    'cig',
    'first',
    'booze',
    'drugs',
    'work.dur',
    'prenatal',
    'ein',
    'sex',
    'tex',
    'bw', 
    'b.head', 
    'preterm', 
    'birth.o']
    giv_x = (df['preterm'] + df['birth.o']).values
    num_groups = 5
    range_for_group = (giv_x.max() - giv_x.min()) / num_groups
    for i in range(num_groups):
        # Create a mask for the current group
        mask = (giv_x >= (i * range_for_group) + giv_x.min() - 1) & (giv_x < ((i + 1) * range_for_group + giv_x.min() + 1))
        # Assign the group number to the corresponding rows in the DataFrame
        df.loc[mask, 'group_instr'] = i + 1 
    if seed is None:
        e_2 = [[5], [-2], [-3]]
        giv_coeffs = np.array([[ 6 ],
                               [-3 ],
                               [2 ],
                               [-2],
                               [2]])
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_2 = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (3, 1)).tolist(), 0.5)
        giv_coeffs = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (5, 1)).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                 x_cols=select_cov, 
                                 u_cols=['momage', 'ark', 'mom.lths',], 
                                 ux_cols=[ 'booze', 'cig', 'work.dur','prenatal', 'drugs'],
                                 xt_cols=['birth.o', 'preterm', 'bw',  'b.head'], 
                                group_instr='group_instr',
                                 xy_cols=['booze','work.dur','prenatal', 'drugs', 'sex', 'ark', 'tex'], 
                                 e_2=e_2, 
                                 xt_betas=[0, 0, 0, 0], 
                                giv_coeffs=giv_coeffs,
                                 xy_betas=np.random.uniform(-2, 2, 6),
                                 phi_1='GIV_y',
                                 phi_2 = 'group_instrument',
                                 treatment_effect='sin',
                                 seed=phi_seed)
    return z_candidates_gen.gen_data()

## WITHOUT U \to X
def generate_linear_disjoint_no_U_to_X(df, seed=None):
    select_covariates = [
                        'twin',
                        'mom.hs',
                        'mom.scoll',
                        'cig',
                        'first',
                        'booze',
                        'drugs',
                        'work.dur',
                        'prenatal',
                        'ein',
                        'sex',
                        'tex',
                        'bw', 
                        'b.head', 
                        'preterm', 
                        'birth.o'] 
        
    u_cols=['momage', 'ark', 'mom.lths',]
    ux_cols=[]
    xt_cols=['bw',  'b.head', 'preterm']
    xy_cols=['booze','work.dur','prenatal', 'first', 'drugs', 'sex']
    if seed is None:
        e_1=[[3], [4], [-2]]
        e_2=[[-1], [-2], [3]]
        xt_betas=[-1, -1, 4]
        xy_betas=[-1, 1, 3, 1, .3, 1]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        e_2 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 3, 0.5), 3).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-2, 4, 1), 6).tolist(), 0.5)
        phi_seed = seed
    phi_1='random_additive'
    treatment_effect='linear'
        
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                 x_cols=select_covariates, 
                                 u_cols_count=3,
                                 u_cols="normal", 
                                 ux_cols=ux_cols,
                                 xt_cols=xt_cols,
                                 xy_cols=xy_cols,
                                 e_1=e_1,
                                 e_2=e_2,
                                 xt_betas=xt_betas,
                                 xy_betas=xy_betas,
                                 phi_1=phi_1,
                                 treatment_effect=treatment_effect, 
                                 seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_linear_mixed_no_U_to_X(df, seed=None):
    select_covariates = [
                'twin',
                'mom.hs',
                'mom.scoll',
                'cig',
                'first',
                'booze',
                'drugs',
                'work.dur',
                'prenatal',
                'ein',
                'sex',
                'tex',
                'bw', 
                'b.head', 
                'preterm', 
                'birth.o']
    
    if seed is None:
        e_1=[[1], [2], [-2]]
        xt_betas=[-2, -3, 2, 3, -4]
        xy_betas=[-1, 2, 1, -2]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 3, 0.5), 5).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-4, 4, 1), 4).tolist(), 0.5)
        phi_seed = seed
    
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                x_cols=select_covariates, 
                                u_cols="normal",
                                u_cols_count=3,
                                ux_cols=[], 
                                xt_cols=['bw',  'b.head', 'preterm', 'booze','work.dur',],
                                xy_cols=['booze','work.dur','prenatal', 'drugs'], 
                                treatment_effect='linear', 
                                phi_1 = 'simple_additive', 
                                e_1=e_1,
                                xt_betas=xt_betas,
                                xy_betas=xy_betas, 
                                seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_linear_no_cand_no_U_to_X(df, seed=None):
    select_cov = [
            'twin',
            'mom.hs',
            'mom.scoll',
            'cig',
            'first',
            'booze',
            'drugs',
            'work.dur',
            'prenatal',
            'ein',
            'ark',
            'tex',
            'bw', 
            'b.head', 
            'preterm', 
            'birth.o'] 
    
    if seed is None:
        xt_betas=[1, 1, -3, 2]
        phi_seed = 42
    else:
        np.random.seed(seed)
        xt_betas = check_zero_add(np.random.choice(np.arange(-1, 3, 0.5), 4).tolist(), 0.5, even_out=True)
        phi_seed = seed
    
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                     x_cols=select_cov,
                                     u_cols="normal", 
                                     u_cols_count=3,
                                     ux_cols=[], 
                                     xt_cols=['bw',  'b.head','prenatal', 'drugs'], 
                                     xy_cols=['booze','work.dur','prenatal', 'drugs', 'bw',  'b.head'], 
                                     treatment_effect='linear', 
                                     phi_1 = 'random_additive',
                                     xt_betas=xt_betas,
                                     seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_disjoint_no_U_to_X(df, seed=None):
    select_cov = [
                'twin',
                'mom.hs',
                'mom.scoll',
                'cig',
                'first',
                'booze',
                'drugs',
                'work.dur',
                'prenatal',
                'ein',
                'sex',
                'tex',
                'bw', 
                'b.head', 
                'preterm', 
                'birth.o'] 
    if seed is None:
        e_1=[[1], [2], [-1]]
        e_2=[[-1], [3], [3]]
        xt_betas=[1, 5, -3]
        xy_betas=[-1, 2, 1, -2]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        e_2 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 3, 0.5), 3).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-4, 4, 1), 4).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                x_cols=select_cov, 
                                u_cols="normal", 
                                u_cols_count=3,
                                ux_cols=[], 
                                xt_cols=['bw',  'b.head', 'preterm'],
                                xy_cols=['booze','work.dur','prenatal', 'drugs'], 
                                treatment_effect='sigmoid', 
                                phi_1 = 'tx_nonadditive_pos', 
                                e_1=e_1,
                                e_2=e_2,
                                xt_betas=xt_betas,
                                xy_betas=xy_betas,
                                seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_mixed_no_U_to_X(df, seed=None):
    # Define the columns to be used in the phi generation
    select_cov = [
    'twin',
    'mom.hs',
    'mom.scoll',
    'cig',
    'first',
    'booze',
    'drugs',
    'work.dur',
    'prenatal',
    'ein',
    'sex',
    'tex',
    'bw', 
    'b.head', 
    'preterm', 
    'birth.o'] 
    if seed is None:
        e_1=[[2], [2], [-1]]
        e_2=[[-2], [-1.5], [-3]]
        xt_betas=[-2, 3, 2, 3, -4]
        xy_betas=[-1, 2, 1, -3]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        e_2 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 4.5, 0.5), 5).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-4, 4, 1), 4).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                x_cols=select_cov, 
                                u_cols="normal", 
                                u_cols_count=3,
                                ux_cols=[], 
                                xt_cols=['bw',  'b.head', 'preterm', 'booze','work.dur',],
                                xy_cols=['booze','work.dur','prenatal', 'drugs'], 
                                treatment_effect='linear', 
                                phi_1 = 'sin_tx', 
                                e_1=e_1, 
                                e_2=e_2, 
                                xt_betas=xt_betas,
                                xy_betas=xy_betas,
                                seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_no_cand_no_U_to_X(df, seed=None):
    select_cov = [
            'twin',
            'mom.hs',
            'mom.scoll',
            'cig',
            'first',
            'booze',
            'drugs',
            'momage',
            'work.dur',
            'prenatal',
            'ein',
            'ark',
            'tex',
            'bw', 
            'b.head', 
            'birth.o'] 
    if seed is None:
        e_1=[[5], [3], [4]]
        e_2=[[-1], [2], [3]]
        xt_betas=[-3, 1, -3, 2]
        xy_betas=[-1, 2, 1, -2, 3, 1]
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_1 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        e_2 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 4.5, 0.5), 4).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-2, 4, 1), 6).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, 
                                 phi_3='x',
                                 x_cols=select_cov, 
                                 u_cols="normal",
                                 u_cols_count=3,
                                 ux_cols=[],
                                xt_cols=['bw',  'b.head', 'booze', 'prenatal'], 
                                xy_cols=['booze','work.dur','prenatal', 'drugs', 'bw',  'b.head'], 
                                treatment_effect='linear', 
                                phi_1='sin_tx', 
                                e_1=e_1, 
                                e_2=e_2, 
                                xt_betas=xt_betas, 
                                xy_betas=xy_betas,
                                seed=phi_seed)
    return z_candidates_gen.gen_data()


def generate_linear_giv_no_U_to_X(df, seed=None):
    select_cov = [
        'twin',
        'mom.hs',
        'mom.scoll',
        'cig',
        'first',
        'booze',
        'drugs',
        'work.dur',
        'prenatal',
        'ein',
        'sex',
        'tex',
        'bw', 
        'b.head', 
        'preterm', 
        'birth.o'] 
    giv_x = (df['preterm'] + df['birth.o']).values
    num_groups = 5
    range_for_group = (giv_x.max() - giv_x.min()) / num_groups
    for i in range(num_groups):
        # Create a mask for the current group
        mask = (giv_x >= (i * range_for_group) + giv_x.min() - 1) & (giv_x < ((i + 1) * range_for_group + giv_x.min() + 1))
        # Assign the group number to the corresponding rows in the DataFrame
        df.loc[mask, 'group_instr'] = i + 1
    
    if seed is None:
        e_2 = [[3], [-4], [-3]]
        giv_coeffs = np.array([[ 1 ],
                              [4 ],
                              [2 ],
                              [-4],
                              [5]])
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_2 = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (3, 1)).tolist(), 0.5)
        giv_coeffs = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (5, 1)).tolist(), 0.5)
        phi_seed = seed

    z_candidates_gen = PhiGeneration(df, phi_3='x',
                                 x_cols=select_cov, 
                                 u_cols='normal',
                                 u_cols_count=3,
                                 e_3=np.random.choice(np.arange(-4, 4, 0.001), (3, 5)), 
                                 ux_cols=[],
                                 xt_cols=['birth.o', 'preterm', 'bw',  'b.head'], 
                                group_instr='group_instr',
                                 xy_cols=['booze','work.dur','prenatal', 'drugs', 'sex', 'ark', 'tex'], 
                                 e_2=e_2, 
                                 xt_betas=[0, 0, 0, 0], 
                                giv_coeffs=giv_coeffs,
                                 phi_1='random_additive',
                                 phi_2 = 'group_instrument',
                                 treatment_effect='linear', 
                                 seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_giv_no_U_to_X(df, seed=None):
    # Define the columns to be used in the phi generation
    select_cov = [
    'twin',
    'mom.hs',
    'mom.scoll',
    'cig',
    'first',
    'booze',
    'drugs',
    'work.dur',
    'prenatal',
    'ein',
    'sex',
    'tex',
    'bw', 
    'b.head', 
    'preterm', 
    'birth.o']
    giv_x = (df['preterm'] + df['birth.o']).values
    num_groups = 5
    range_for_group = (giv_x.max() - giv_x.min()) / num_groups
    for i in range(num_groups):
        # Create a mask for the current group
        mask = (giv_x >= (i * range_for_group) + giv_x.min() - 1) & (giv_x < ((i + 1) * range_for_group + giv_x.min() + 1))
        # Assign the group number to the corresponding rows in the DataFrame
        df.loc[mask, 'group_instr'] = i + 1 

    if seed is None:
        e_2 = [[5], [-2], [-3]]
        giv_coeffs = np.array([[ 6 ],
                              [-3 ],
                              [2 ],
                              [-2],
                              [2]])
        phi_seed = 42
    else:
        np.random.seed(seed)
        e_2 = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (3, 1)).tolist(), 0.5)
        giv_coeffs = check_zero_add(np.random.choice(np.arange(-4, 4, 1), (5, 1)).tolist(), 0.5)
        phi_seed = seed

    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                 x_cols=select_cov, 
                                 u_cols="normal", 
                                 u_cols_count=3,
                                 ux_cols=[],
                                 xt_cols=['birth.o', 'preterm', 'bw',  'b.head'], 
                                group_instr='group_instr',
                                 xy_cols=['booze','work.dur','prenatal', 'drugs', 'sex', 'ark', 'tex'], 
                                 e_2=e_2, 
                                 xt_betas=[0, 0, 0, 0], 
                                giv_coeffs=giv_coeffs,
                                 xy_betas=np.random.uniform(-2, 2, 6),
                                 phi_1='GIV_y',
                                 phi_2 = 'group_instrument',
                                 treatment_effect='sin',
                                 seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_linear_no_cand_no_U(df, seed=None):
    select_cov = [
            'twin',
            'mom.hs',
            'mom.scoll',
            'cig',
            'first',
            'booze',
            'drugs',
            'work.dur',
            'prenatal',
            'ein',
            'ark',
            'tex',
            'bw', 
            'b.head', 
            'preterm', 
            'birth.o'] 
    
    if seed is None:
        xt_betas = [1, 1, -3, 2]
        phi_seed = 42
    else:
        np.random.seed(seed)
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 3, 0.5), 4).tolist(), 0.5, even_out=True)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, phi_3='x', 
                                     x_cols=select_cov,
                                     u_cols_count=0,
                                     ux_cols=[], 
                                     xt_cols=['bw',  'b.head','prenatal', 'drugs'], 
                                     xy_cols=['booze','work.dur','prenatal', 'drugs', 'bw',  'b.head'], 
                                     treatment_effect='linear', 
                                     phi_1 = 'random_additive',
                                     xt_betas=xt_betas, 
                                     seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_nonlinear_no_cand_no_U(df, seed=None):
    select_cov = [
            'twin',
            'mom.hs',
            'mom.scoll',
            'cig',
            'first',
            'booze',
            'drugs',
            'momage',
            'work.dur',
            'prenatal',
            'ein',
            'ark',
            'tex',
            'bw', 
            'b.head', 
            'birth.o'] 
    
    if seed is None:
        xt_betas = [-3, 1, -3, 2]
        xy_betas = [-1, 2, 1, -2, 3, 1]
        phi_seed = 42
    else:
        np.random.seed(seed)
        xt_betas = check_zero_add(np.random.choice(np.arange(-2, 3, 0.5), 4).tolist(), 0.5, even_out=True)
        xy_betas = check_zero_add(np.random.choice(np.arange(-4, 4, 1), 6).tolist(), 0.5)
        phi_seed = seed
    z_candidates_gen = PhiGeneration(df, 
                                 phi_3='x', 
                                 x_cols=select_cov, 
                                 u_cols_count=0,
                                 ux_cols=[],
                                xt_cols=['bw',  'b.head', 'booze', 'prenatal'], 
                                xy_cols=['booze','work.dur','prenatal', 'drugs', 'bw',  'b.head'], 
                                treatment_effect='linear', 
                                phi_1='sin_tx', 
                                xt_betas=xt_betas, 
                                xy_betas=xy_betas,
                                seed=phi_seed)
    return z_candidates_gen.gen_data()

def generate_ecg(df):
     
    df = df[df['filepath'].notna()]

    z_cols = [col for col in df.columns if col.startswith('z')]
    x_cols = [col for col in df.columns if col.startswith('x')]
    u_cols = [col for col in df.columns if col.startswith('u')]
    generated_dataset = ECG_DGPDataset(df, 
                                x_cols=x_cols,
                                c_cols=x_cols,
                                u_cols=u_cols,
                                z_cols=z_cols)

    return df, generated_dataset

# TODO: add a new dataset follow this format:
# def generate_linear_disjoint(df, seed=None):
#     select_covariates = [
#                         'twin',
#                         'mom.hs',
#                         'mom.scoll',
#                         'cig',
#                         'first',
#                         'booze',
#                         'drugs',
#                         'work.dur',
#                         'prenatal',
#                         'ein',
#                         'sex',
#                         'tex',
#                         'bw', 
#                         'b.head', 
#                         'preterm', 
#                         'birth.o'] # Any covariates from IHDP that you want to include
        
#     u_cols=['momage', 'ark', 'mom.lths',] # Unobserved confounders 
#     ux_cols=[ 'booze', 'cig', 'first', 'work.dur','prenatal', 'drugs'] # Unobserved confounders that also affect x
#     xt_cols=['bw',  'b.head', 'preterm'] # Covariates that affect treatment
#     xy_cols=['booze','work.dur','prenatal', 'first', 'drugs', 'sex'] # Covariates that affect outcome
#     if seed is None:
#         e_1=[[3], [4], [-2]] # Error term for treatment
#         e_2=[[-1], [-2], [3]] # Error term for outcome
#         xt_betas=[-1, -1, 4] # Coefficients for treatment model
#         xy_betas=[-1, 1, 3, 1, .3, 1] # Coefficients for outcome model
#         phi_seed = 42
#     else:
#         np.random.seed(seed)
#         e_1 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
#         e_2 = check_zero_add(np.random.choice(np.arange(-2, 4, 1), (3, 1)).tolist(), 0.5)
#         xt_betas = check_zero_add(np.random.choice(np.arange(-2, 3, 0.5), 3).tolist(), 0.5, even_out=True)
#         xy_betas = check_zero_add(np.random.choice(np.arange(-2, 4, 1), 6).tolist(), 0.5)
#         phi_seed = seed
#     phi_1='random_additive' 
#     treatment_effect='linear' 
#     z_candidates_gen = PhiGeneration(df, phi_3='x', 
#                                  x_cols=select_covariates, 
#                                  u_cols=u_cols,
#                                  ux_cols=ux_cols,
#                                  xt_cols=xt_cols,
#                                  xy_cols=xy_cols,
#                                  e_1=e_1,
#                                  e_2=e_2,
#                                  xt_betas=xt_betas,
#                                  xy_betas=xy_betas,
#                                  phi_1=phi_1,
#                                  treatment_effect=treatment_effect)
#     return z_candidates_gen.gen_data()

def generate_named_dataset(dataset_name, data_file= "ihdp", output_dir=None, version=None):
    # Read the data
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.RData'):
        df = pyreadr.read_r(data_file)["ihdp"]
    elif data_file == 'ihdp':
        # df = pyreadr.read_r('datafiles/ihdp.RData')["ihdp"]
        files = glob('**/ihdp.RData', recursive=True)
        if len(files) == 0:
            raise FileNotFoundError("No ihdp.RData file found in datafiles directory.")
        df = pyreadr.read_r(files[0])["ihdp"]
    if version is None:
        seed = None
        name_edit = ""
    else:
        seed = 42 + version
        name_edit = f"_v{version}"
    if dataset_name == 'linear_disjoint':
        generated_df, dataset =  generate_linear_disjoint(df, seed=seed)
    elif dataset_name == 'linear_mixed':
        generated_df, dataset =  generate_linear_mixed(df, seed=seed)
    elif dataset_name == 'linear_no_cand':
        generated_df, dataset =  generate_linear_no_cand(df, seed=seed)
    elif dataset_name == 'nonlinear_disjoint':
        generated_df, dataset =  generate_nonlinear_disjoint(df, seed=seed)
    elif dataset_name == 'nonlinear_mixed':
        generated_df, dataset =  generate_nonlinear_mixed(df, seed=seed)
    elif dataset_name == 'nonlinear_no_cand':
        generated_df, dataset =  generate_nonlinear_no_cand(df, seed=seed)
    elif dataset_name == 'linear_giv':
        generated_df, dataset =  generate_linear_giv(df, seed=seed)
    elif dataset_name == 'nonlinear_giv':
        generated_df, dataset =  generate_nonlinear_giv(df, seed=seed)
    
    # Datasets with no U \to X
    elif dataset_name == 'linear_disjoint_no_U_to_X':
        generated_df, dataset =  generate_linear_disjoint_no_U_to_X(df, seed=seed)
    elif dataset_name == 'linear_mixed_no_U_to_X':
        generated_df, dataset =  generate_linear_mixed_no_U_to_X(df, seed=seed)
    elif dataset_name == 'linear_no_cand_no_U_to_X':
        generated_df, dataset =  generate_linear_no_cand_no_U_to_X(df, seed=seed)
    elif dataset_name == 'nonlinear_disjoint_no_U_to_X':
        generated_df, dataset =  generate_nonlinear_disjoint_no_U_to_X(df, seed=seed)
    elif dataset_name == 'nonlinear_mixed_no_U_to_X':
        generated_df, dataset =  generate_nonlinear_mixed_no_U_to_X(df, seed=seed)
    elif dataset_name == 'nonlinear_no_cand_no_U_to_X':
        generated_df, dataset =  generate_nonlinear_no_cand_no_U_to_X(df, seed=seed)
    elif dataset_name == 'linear_giv_no_U_to_X':
        generated_df, dataset =  generate_linear_giv_no_U_to_X(df, seed=seed)
    elif dataset_name == 'nonlinear_giv_no_U_to_X':
        generated_df, dataset =  generate_nonlinear_giv_no_U_to_X(df, seed=seed)

    elif dataset_name == 'linear_no_cand_no_U':
        generated_df, dataset =  generate_linear_no_cand_no_U(df, seed=seed)
    elif dataset_name == 'nonlinear_no_cand_no_U':
        generated_df, dataset =  generate_nonlinear_no_cand_no_U(df, seed=seed)
    elif dataset_name == 'ECG_data': # ECG data file is processed separately for privacy.
        files = glob('**/physionet_ecg_data.csv', recursive=True)
        if len(files) == 0:
            raise FileNotFoundError("No physionet_ecg_data.csv file found in datafiles directory.")
        df = pd.read_csv(files[0])
        generated_df, dataset = generate_ecg(df)
    # TODO: When you add a new dataset, add a new elif statement here
    # for example:
    # elif dataset_name == 'your_new_dataset_name':
    #     generated_df, dataset =  your_new_dataset_function(df)
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}")

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/csvs', exist_ok=True)
        os.makedirs(f'{output_dir}/objects', exist_ok=True)
        # Save the generated DataFrame
        generated_df.to_csv(f"{output_dir}/csvs/{dataset_name}.csv", index=False)
        # Save the dataset object
        with open(f"{output_dir}/objects/{dataset_name}.pkl", "wb") as f:
            pickle.dump(dataset, f)

    return generated_df, dataset

def generate_datasets(data_file = "ihdp", output_dir = None, version=None):
    # Read the data
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.RData'):
        df = pyreadr.read_r(data_file)["ihdp"]
    elif data_file == 'ihdp':
        # df = pyreadr.read_r('datafiles/ihdp.RData')["ihdp"]
        files = glob('**/ihdp.RData', recursive=True)
        if len(files) == 0:
            raise FileNotFoundError("No ihdp.RData file found in datafiles directory.")
        df = pyreadr.read_r(files[0])["ihdp"]

    if version is None:
        seed = None
        name_edit = ""
    else:
        seed = 42 + version
        name_edit =  f"_v{version}"
    # Generate datasets
    linear_disjoint_df, ld_dataset = generate_linear_disjoint(df, seed=seed)
    linear_mixed_df, lm_dataset = generate_linear_mixed(df, seed=seed)
    linear_no_cand_df, lnc_dataset = generate_linear_no_cand(df, seed=seed)
    nonlinear_disjoint_df, nld_dataset = generate_nonlinear_disjoint(df, seed=seed)
    nonlinear_mixed_df, nlm_dataset = generate_nonlinear_mixed(df, seed=seed)
    nonlinear_no_cand_df, nlcd_dataset = generate_nonlinear_no_cand(df, seed=seed)
    linear_giv_df, lg_dataset = generate_linear_giv(df, seed=seed)
    nonlinear_giv_df, ng_dataset = generate_nonlinear_giv(df, seed=seed)
    
    # No U \to X datasets
    linear_disjoint_df_no_U_to_X, ld_no_U_to_X_dataset = generate_linear_disjoint_no_U_to_X(df, seed=seed)
    linear_mixed_df_no_U_to_X, lm_no_U_to_X_dataset = generate_linear_mixed_no_U_to_X(df, seed=seed)
    linear_no_cand_df_no_U_to_X, lnc_no_U_to_X_dataset = generate_linear_no_cand_no_U_to_X(df, seed=seed)
    nonlinear_disjoint_df_no_U_to_X, nld_no_U_to_X_dataset = generate_nonlinear_disjoint_no_U_to_X(df, seed=seed)
    nonlinear_mixed_df_no_U_to_X, nlm_no_U_to_X_dataset = generate_nonlinear_mixed_no_U_to_X(df, seed=seed)
    nonlinear_no_cand_df_no_U_to_X, nlcd_no_U_to_X_dataset = generate_nonlinear_no_cand_no_U_to_X(df, seed=seed)
    linear_giv_df_no_U_to_X, lg_no_U_to_X_dataset = generate_linear_giv_no_U_to_X(df, seed=seed)
    nonlinear_giv_df_no_U_to_X, ng_no_U_to_X_dataset = generate_nonlinear_giv_no_U_to_X(df, seed=seed)
    
    # Re-read the data
    if data_file.endswith('.csv'):
        df = pd.read_csv(data_file)
    elif data_file.endswith('.RData'):
        df = pyreadr.read_r(data_file)["ihdp"]
    elif data_file == 'ihdp':
        # df = pyreadr.read_r('datafiles/ihdp.RData')["ihdp"]
        files = glob('**/ihdp.RData', recursive=True)
        if len(files) == 0:
            raise FileNotFoundError("No ihdp.RData file found in datafiles directory.")
        df = pyreadr.read_r(files[0])["ihdp"]
    # No U datasets
    linear_no_cand_df_no_U, lnc_no_U_dataset = generate_linear_no_cand_no_U(df, seed=seed)
    nonlinear_no_cand_df_no_U, nlcd_no_U_dataset = generate_nonlinear_no_cand_no_U(df, seed=seed)

    # ECG_data dataset
    ecg_files = glob('**/physionet_ecg_data.csv', recursive=True)
    if len(ecg_files) == 0:
        raise FileNotFoundError("No physionet_ecg_data.csv file found in datafiles directory.")
    ecg_df = pd.read_csv(ecg_files[0])
    ecg_df, ecg_dataset = generate_ecg(ecg_df)
    
    # TODO: When you add a new dataset, add a new line here
    # for example:
    # {new name}_df, {new name}_dataset = generate_{new name}(df)

    # Save datasets to a dictionary
    datasets = {
        f"linear_disjoint{name_edit}": (linear_disjoint_df, ld_dataset),
        f"linear_mixed{name_edit}": (linear_mixed_df, lm_dataset),
        f"linear_no_cand{name_edit}": (linear_no_cand_df, lnc_dataset),
        f"nonlinear_disjoint{name_edit}": (nonlinear_disjoint_df, nld_dataset),
        f"nonlinear_mixed{name_edit}": (nonlinear_mixed_df, nlm_dataset),
        f"nonlinear_no_cand{name_edit}": (nonlinear_no_cand_df, nlcd_dataset),
        f"linear_giv{name_edit}": (linear_giv_df, lg_dataset),
        f"nonlinear_giv{name_edit}": (nonlinear_giv_df, ng_dataset),
        # No U \to X
        f"linear_disjoint_no_U_to_X{name_edit}": (linear_disjoint_df_no_U_to_X, ld_no_U_to_X_dataset),
        f"linear_mixed_no_U_to_X{name_edit}": (linear_mixed_df_no_U_to_X, lm_no_U_to_X_dataset),
        f"linear_no_cand_no_U_to_X{name_edit}": (linear_no_cand_df_no_U_to_X, lnc_no_U_to_X_dataset),
        f"nonlinear_disjoint_no_U_to_X{name_edit}": (nonlinear_disjoint_df_no_U_to_X, nld_no_U_to_X_dataset),
        f"nonlinear_mixed_no_U_to_X{name_edit}": (nonlinear_mixed_df_no_U_to_X, nlm_no_U_to_X_dataset),
        f"nonlinear_no_cand_no_U_to_X{name_edit}": (nonlinear_no_cand_df_no_U_to_X, nlcd_no_U_to_X_dataset),
        f"linear_giv_no_U_to_X{name_edit}": (linear_giv_df_no_U_to_X, lg_no_U_to_X_dataset),
        f"nonlinear_giv_no_U_to_X{name_edit}": (nonlinear_giv_df_no_U_to_X, ng_no_U_to_X_dataset),
        # No U
        f"linear_no_cand_no_U{name_edit}": (linear_no_cand_df_no_U, lnc_no_U_dataset),
        f"nonlinear_no_cand_no_U{name_edit}": (nonlinear_no_cand_df_no_U, nlcd_no_U_dataset),
        # ECG
        f"ECG_data{name_edit}": (ecg_df, ecg_dataset),
        # TODO: add your new dataset here
        # "{new name}_name": ({new name}_df, {new name}_dataset)
    }
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f'{output_dir}/csvs', exist_ok=True)
        os.makedirs(f'{output_dir}/objects', exist_ok=True)
        # Save each dataset to a CSV file
        for name, (df, dataset) in datasets.items():
            # Save the dataset
            df.to_csv(f"{output_dir}/csvs/{name}.csv", index=False)
            # Save the dataset object
            with open(f"{output_dir}/objects/{name}.pkl", "wb") as f:
                pickle.dump(dataset, f)
    
    return datasets

def load_csv_datasets(output_dir):
    datasets = {}
    for filename in os.listdir(f'{output_dir}/csvs'):
        if filename.endswith('.csv'):
            name = filename[:-4]
            df = pd.read_csv(f"{output_dir}/csvs/{filename}")
            datasets[name] = df
    return datasets

def load_object_dataset(dataset_dir, dataset_name):
    file_path = f"{dataset_dir}/{dataset_name}.pkl"
    with open(file_path, "rb") as f:
        dataset = pickle.load(f)
    return dataset

def load_object_datasets(output_dir):
    datasets = {}
    for filename in os.listdir(f'{output_dir}/objects'):
        if filename.endswith('.pkl'):
            name = filename[:-4]
            with open(f"{output_dir}/objects/{filename}", "rb") as f:
                dataset = pickle.load(f)
            datasets[name] = dataset
    return datasets
