#######################################################################################
# Author: Jenna Fields
# Code adapted from: https://github.com/anpwu/Meta-EM/blob/94214ab2218bf64d98c76587a14478e456f159f3/Generator.py
# Script: phi_generation.py
# Function: Generating synthetic outcomes and treatments from structural causal models
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed

set_seed(42)

import numpy as np
import pandas as pd
import random
import torch
from DGP.dataset_class import DGPDataset

#######################################################################################

def cat(data_list, axis=1):
    try:
        output=torch.cat(data_list,axis)
    except:
        output=np.concatenate(data_list,axis)

    return output

class PhiGeneration(object):
    def __init__(self, df : pd.DataFrame, 
                        phi_1 = 'simple_additive', # Outcome
                        phi_2 = 'linear', # Determining T
                        phi_3 = 'x', # Determining U
                        treatment_effect = 'linear', 
                        e_1 = None, # Outcome - U coeff
                        e_2 = None, # Determining T - U coeff
                        e_3 = None, # U influence on X
                        seed = 42, 
                        x_cols = None, 
                        u_cols = None, 
                        binary_t = True, 
                        xt_betas = None,
                        xy_betas = None, 
                        ux_cols = None, 
                        xt_cols = None, 
                        xy_cols = None, 
                        u_cols_count=None, 
                        group_instr=None, 
                        giv_coeffs=None, 
                        use_giv_as_int=False):
        self.seed = seed
        self.phi_1 = phi_1 # we can use a string to use a pre-defined, or we can pass in lamba x, t, u, e : return y, g
        self.phi_2 = phi_2
        self.phi_3 = phi_3 # We can use the default, or pass in a lambda x, u : return x style function
        self.use_giv_as_int = use_giv_as_int
        self.treatment_effect = treatment_effect

        self.binary_t = binary_t
        self.data_df = df
        self.xy_betas = xy_betas
        self.mean = None
        self.e_3 = e_3

        np.random.seed(seed)
        random.seed(seed)

        # Get column names for dataframe that we want
        self.x_cols = x_cols if x_cols is not None else [i for i in self.data_df.columns if i.lower().startswith('x')]

        if u_cols == "normal":
            # Example: generate U columns from N(0,1) - need u_cols_count to be defined
            num_rows = self.data_df.shape[0]
            for i in range(u_cols_count):
                self.data_df[f"U{i+1}"] = np.random.normal(loc=0, scale=1, size=num_rows)
            self.u_cols = [f"U{i+1}" for i in range(u_cols_count)]
        else:
            self.u_cols = u_cols if u_cols is not None else [i for i in self.data_df.columns if i.lower().startswith('u')]
        self.group_instr = None
        self.giv_coeffs = None
        if self.phi_2 == 'group_instrument':
            self.group_instr = df[group_instr].values if group_instr is not None else np.random.choice(5, len(df), p=[0.05, 0.1, 0.25, 0.5, 0.1])
            self.giv_coeffs = giv_coeffs if giv_coeffs is not None else np.random.choice(np.arange(-5, 5, .5).round(2), size=(len(np.unique(self.group_instr)), 1))

        self.e_1 = e_1 if e_1 is not None else np.random.choice(np.arange(-5, 5, 0.1).round(2), size=(len(self.u_cols), 1))
        self.e_2 = e_2 if e_2 is not None else np.random.choice(np.arange(-5, 5, 0.1).round(2), size=(len(self.u_cols), 1))


        # ux_cols are the columns of columns x we want to be affected by u. xt_cols are columns of x we want to be used to calculate t 
        self.ux_cols = np.array([i for i, elem in enumerate(self.x_cols) if elem in ux_cols]) if ux_cols is not None else ux_cols
        self.xt_cols = np.array([i for i, elem in enumerate(self.x_cols) if elem in xt_cols]) if xt_cols is not None else xt_cols
        self.xy_cols = np.array([i for i, elem in enumerate(self.x_cols) if elem in xy_cols]) if xy_cols is not None else xy_cols

        # set x_dim
        self.xt_betas = xt_betas

    def generate_x(self, x, u):
        if self.phi_3 == 'x': #either we want no u, or there is already a u that x has a correlation with
            return x
        if self.phi_3 == 'x_u':
            if self.ux_cols is None: 
                self.ux_cols = np.where(np.random.choice([0, 1], size=(x.shape[1]), p=[0.75, 0.25]))[0]
            # x[:, self.ux_cols] = x[:, self.ux_cols] + u
            x[:, self.ux_cols] = x[:, self.ux_cols] + u @ self.e_3
            return x
        if self.phi_3 == 'x_rand_u':
            if self.ux_cols is None: 
                self.ux_cols = np.where(np.random.choice([0, 1], size=x.shape[1], p=[0.75, 0.25]))[0]
            if self.e_3 is None:
                self.e_3 = np.random.choice(np.arange(-.1, .1, 0.001), (u.shape[1], self.ux_cols.shape[0]))
            x[:, self.ux_cols] = x[:, self.ux_cols] + u @ self.e_3
            return x
        else:
            return self.phi_3(x, u)
        
    def generate_t(self, x, u, e_t):
        if self.phi_2 == 'linear':
            if self.xt_cols is None: 
                self.xt_cols = np.where(np.random.choice([0, 1], size=x.shape[1], p=[0.5, 0.5]))[0]
            if self.xt_betas is None:
                self.xt_betas = np.random.choice(np.arange(-1, 1, 0.1).round(2), size=self.xt_cols.shape)
            return np.expand_dims((x[:, self.xt_cols] @ self.xt_betas), -1) +  u @ self.e_2 + e_t
        if self.phi_2 == 'group_instrument': 
            givs = np.unique(self.group_instr)
            base_t = u @ self.e_2 + e_t
            if self.xt_cols is not None and len(self.xt_cols) > 0: 
                if self.xt_betas is None:
                    self.xt_betas = np.random.choice(np.arange(-1, 1, 0.1).round(2), size=self.xt_cols.shape)
                base_t += np.expand_dims((x[:, self.xt_cols] @ self.xt_betas), -1)
            for i in range(len(givs)):
                if self.use_giv_as_int:
                    base_t[self.group_instr == givs[i]] += self.giv_coeffs[i] * givs[i]
                else:
                    base_t[self.group_instr == givs[i]] += self.giv_coeffs[i]
            return base_t            
        else:
            return self.phi_2(x, u, e_t)

    def generate_treatment_effect(self, t):
        func = self.treatment_effect
        if func=='abs':
            return np.abs(t)
        elif func=='2dpoly':
            return -3 * t + .9 * (t**2)
        elif func=='sigmoid':
            return 1/(1+np.exp(-1*t))
        elif func=='sin':
            return np.sin(t)
        elif func=='cos':
            return np.cos(t)
        elif func=='step':
            return 1. * (t<0) + 2.5 * (t>=0)
        elif func=='3dpoly':
            return -1.5 * t + .9 * (t**2) + t**3
        elif func=='linear':
            return t
        elif func=='3_linear':
            return 3 * t
        elif func=='negative_linear':
            return -3 * t
        elif func=='rand_pw':
            pw_linear = self._generate_random_pw_linear()
            return np.reshape(np.array([pw_linear(x_i) for x_i in t.flatten()]), t.shape)
        else:
            raise NotImplementedError()
        
    def generate_y(self, x, t, u, e):
        # Get treatment effect 
        g = self.generate_treatment_effect(t)
        if self.xy_cols is None:
            self.xy_cols = np.where(np.random.choice([0, 1], size=x.shape[1], p=[0.5, 0.5]))[0]

        if self.phi_1 == 'simple_additive':
            return g + 2 * np.sum(x[:, self.xy_cols], 1, keepdims=True) / self.xy_cols.shape + u @ self.e_1 + e, g
        if self.phi_1 == 'random_additive':
            if self.xy_betas is None:
                self.xy_betas = random.choices(np.arange(0, 1, 0.1).round(2), k=self.xy_cols.shape[0])
            return g + np.sum(x[:, self.xy_cols] * self.xy_betas, 1, keepdims=True) + u @ self.e_1 + e, g
        if self.phi_1 == 'GIV_y':
            return g + 2 * np.sum(x[:, self.xy_cols], 1, keepdims=True) / self.xy_cols.shape + u @ self.e_1 + e - np.abs(x[:,0:1]*x[:,1:2])- np.sin(10+x[:,2:3]*x[:,2:3]), g
        if self.phi_1 == 'tx_nonadditive_pos':
            if self.xy_betas is None:
                self.xy_betas = random.choices(np.arange(0, 1, 0.1).round(2), k=self.xy_cols.shape[0])
            return np.abs(np.sum(x[:, self.xy_cols] * self.xy_betas, 1, keepdims=True)) * (g ** 2) + 2 * g - 3 * np.abs(x[:,0:1]*x[:,1:2]) + u @ self.e_1 + e, g
        if self.phi_1 == 'sin_tx':
            if self.xy_betas is None:
                self.xy_betas = random.choices(np.arange(0, 1, 0.1).round(2), k=self.xy_cols.shape[0])
            return np.sin(np.abs(np.sum(x[:, self.xy_cols] * self.xy_betas, 1, keepdims=True)) * g) + g * np.abs(x[:,0:1]*x[:,1:2]) + g + u @ self.e_1 + e, g
        if self.phi_1 == 'super_nonlinear':
            if self.xy_betas is None:
                self.xy_betas = random.choices(np.arange(0, 1, 0.1).round(2), k=self.xy_cols.shape[0])
            return np.abs(np.sum((x[:, self.xy_cols[:2]] * self.xy_betas[:2] * g), 1, keepdims=True)) + np.abs(np.sum((x[:, self.xy_cols] * self.xy_betas), 1, keepdims=True)) + g ** 2 - 3 * np.abs(x[:,0:1]*x[:,1:2]) + u @ self.e_1 + e, g
        if self.phi_1 == 'tx_nonadditive_pos_1':
            if self.xy_betas is None:
                self.xy_betas = random.choices(np.arange(0, 1, 0.1).round(2), k=self.xy_cols.shape[0])
            return np.abs(np.sum(x[:, self.xy_cols] * self.xy_betas, 1, keepdims=True)) * (g ** 2) + g - 3 * np.abs(x[:,0:1]*x[:,1:2]) + u @ self.e_1 + e, g
        
        else:
            return self.phi_1(x, t, u, e) # THIS ASSUMES YOU OVERRIDE GENERATE TREATMENT EFFECT
            
    def normalize(self, y):
        return (y - self.mean) / self.std

    def denormalize(self, y):
        return y*self.std + self.mean

    def gen_t_cf(self, t, x, u, e2):
        y, g = self.generate_y(x, 1 - t, u, e2)

        y = self.normalize(y)
        g = self.normalize(g)

        return y, g

    def gen_data(self):
        num = len(self.data_df)        

        x = self.data_df[self.x_cols].values

        u = self.data_df[self.u_cols].values 
        u = (u - u.min(axis=0, keepdims=True)) / (u.max(axis=0, keepdims=True) - u.min(axis=0, keepdims=True))
        x = self.generate_x(x, u)

        # Normalize x
        x = (x - x.min(axis=0, keepdims=True)) / (x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True))

        e_y = np.random.normal(0, .1, size=(num, 1))
        e_t = np.random.normal(0, .1, size=(num, 1))
                    
        t = self.generate_t(x, u, e_t) 
        
        # Make T binary 
        if self.binary_t:
            sigmoid = lambda x: 1/(1+np.exp(-1*x))
            t = (np.random.rand(*t.shape) < sigmoid(t)).astype(int)

        y, g = self.generate_y(x, t, u, e_y)

        if self.mean is None:
            self.mean = y.mean()
            self.std = y.std()

        y = self.normalize(y)
        g = self.normalize(g)

        y_cf, g_cf = self.gen_t_cf(t,x,u,e_y)

        x_cols = ['x{}'.format(i+1) for i in range(x.shape[1])]
        if self.group_instr is not None:
            data_df = pd.DataFrame(np.concatenate([x, u, t, y, 1 - t, y_cf, g_cf, np.expand_dims(self.group_instr, -1)], 1), 
                                                    columns=x_cols + 
                                                            ['u{}'.format(i+1) for i in range(u.shape[1])] + 
                                                            ['t', 'y', 't_cf', 'y_cf', 'g_cf', 'group_instr'])
        else:
            data_df = pd.DataFrame(np.concatenate([x, u, t, y, 1 - t, y_cf, g_cf], 1), 
                                    columns=x_cols + 
                                            ['u{}'.format(i+1) for i in range(u.shape[1])] + 
                                            ['t', 'y', 't_cf', 'y_cf', 'g_cf'])
        
        data_df['ite'] = np.where(data_df['t'] == 1, data_df['y'], data_df['y_cf']) - np.where(data_df['t'] == 0, data_df['y'], data_df['y_cf'])

        if self.group_instr is not None:
            c_cols = [i for i in x_cols if (int(i[1:]) in self.xy_cols or int(i[1:]) in self.xt_cols)]
            self.generated_dataset = DGPDataset(data_df,
                                                x_cols=x_cols, 
                                                true_c_cols=c_cols, 
                                                true_z_cols=['group_instr'], 
                                                u_cols=['u{}'.format(i+1) for i in range(u.shape[1])], 
                                                train_size=0.7, 
                                                valid_size=0.2)
        else:                                
            z_cols = [f'x{i + 1}' for i in self.xt_cols if i not in self.xy_cols]
            c_cols = [i for i in x_cols if i not in z_cols and (int(i[1:]) in self.xy_cols or int(i[1:]) in self.xt_cols)]
            self.generated_dataset = DGPDataset(data_df,
                                    x_cols=x_cols, 
                                    true_c_cols=c_cols, 
                                    true_z_cols=z_cols, 
                                    u_cols=['u{}'.format(i+1) for i in range(u.shape[1])], 
                                    train_size=0.7, 
                                    valid_size=0.2)
        
        return data_df, self.generated_dataset
    

    
