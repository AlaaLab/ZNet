# ZNet: Causal Effect Estimation with Learned Instrument Representations

ZNet is a neural network framework for learning instrumental variable (IV) representations from observational data. The model learns the structural causal model of an IV with moment based constraints to force the IV conditoins. The result is separating observed data into confounders and an instrumental component to enable robust causal effect estimation in the presence of unobserved confounding.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
- [Usage Examples](#usage-examples)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Evaluation](#evaluation)
- [Citation](#citation)

## Overview

ZNet addresses the challenge of instrumental variable selection by learning two disentangled representations:
- **C (Confounders)**: Features that predict the outcome Y
- **Z (Instruments)**: Features that predict treatment T but are independent of unobserved confounders

### Key Features

- **Multi-objective Optimization**: Balances multiple moment conditions as loss objectives which are used to constrain the learned representations
- **Flexible Architecture**: Supports linear and non-linear networks, softmax representations
- **Advanced Optimization**: Optional PCGrad for gradient conflict resolution
- **Mutual Information**: KDE-based MI estimation as alternative to Pearson correlation
- **Comparison Methods**: Includes AutoIV, GIV, VIV implementations for benchmarking
- **Downstream Integration**: Compatible with DeepIV, DFIV, and TARNet estimators
- **ECG Support**: Includes version of architecture for ECGs as an example of high-dimensional data inputs

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn (visualization)
- BoTorch (Bayesian optimization)
- TensorFlow 2.x (for baseline comparisons)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/jennaef/znet-iv.git
cd znet-iv
```

2. Create conda environment:
```bash
conda env create -f znet.yml
conda activate znet
```

## Quick Start

### Basic ZNet Training

```python
from models.ZNet.ZNet import ZNet
from DGP.generate_datasets import generate_datasets

# Load or generate dataset
datasets = generate_datasets()
data = datasets['linear_disjoint'][1] 

# Initialize ZNet
znet = ZNet(
    input_dim=len(data.x_cols),
    c_dim=10,  # Confounder dimension
    z_dim=5,   # Instrument dimension
    output_dim=1,
    c_pearson_loss_alpha=1.0, # Select loss coefficients
    z_pearson_loss_alpha=1.0,
    t_hat_alpha=1.0
)

# Train
train_data = data.train(data_type='torch')
val_data = data.val(data_type='torch')

znet.fit(
    train_data.x, train_data.t, train_data.y,
    val_X=val_data.x, val_t=val_data.t, val_y=val_data.y,
    num_epochs=50,
    batch_size=64
)

# Extract representations
c, z, t_hat, c_y, x_t_y = znet.get_generated_data(data.x, data.t)
```

### Running Full Pipeline

```bash
# Run full pipeline with Bayesian hyperparameter search
python scripts/main_driver_bayesian.py \
    --param_json params.json \
    --datasets linear_disjoint \
    --num_bootstraps 50 \
    --ncalls_bayesian 10
```

## Project Structure

```
znet-iv/
├── models/
│   ├── ZNet/
│   │   ├── ZNet.py                  # Main ZNet model
│   │   ├── ZNet_ECG.py              # ECG-specific variant
│   │   ├── model_loss_utils.py      # Loss functions & architecture
│   │   ├── pcgrad.py                # PCGrad optimizer
│   │   └── loss_plotting.py         # Training visualization
│   ├── gen_IV_comparisons/
│   │   ├── AutoIV/                  # AutoIV baseline
│   │   ├── GIV/                     # GIV baseline
│   │   └── VIV/                     # VIV baseline
│   └── treatment_effect_estimators/
│       ├── deep_iv.py               # DeepIV estimator
│       ├── df_iv.py                 # DFIV estimator
│       ├── TARNet.py                # TARNet estimator
│       └── simple_estimators.py     # OLS, TSLS baselines
├── DGP/
│   ├── dataset_class.py             # Dataset classes
│   ├── generate_datasets.py         # Synthetic data generation
│   └── phi_generation.py            # DGP parameter generation
├── utils/
│   ├── train_models.py              # Training utilities
│   ├── evaluate_models.py           # Evaluation metrics
│   ├── evaluation.py                # IV quality assessment
│   ├── pipeline_utils.py            # Pipeline helpers
│   └── bayesian_search/
│       ├── single_obj_search.py     # Single-objective optimization
│       └── multi_obj_search.py      # Multi-objective optimization
├── scripts/
│   ├── main_driver.py               # Grid search pipeline
│   ├── main_driver_bayesian.py     # Bayesian search pipeline
│   ├── bayesian_search_ivgen.py    # IV generation tuning
│   ├── bayesian_search_downstream.py # Downstream tuning
│   ├── run_bootstrap.py             # Bootstrap analysis
│   └── run_grid_search_*.py         # Grid search scripts
├── seed_utils.py                    # Reproducibility utilities
└── README.md                        # This file
```

## Key Components

### ZNet Model

The core `ZNet` class in `models/ZNet/ZNet.py` implements the disentangled IV learning framework:

**Loss Components:**
- `c_pearson_loss_alpha`: Weight for C-Y correlation (maximize)
- `z_pearson_loss_alpha`: Weight for Z-residual correlation (minimize)
- `pearson_matrix_alpha`: Weight for C-Z independence (minimize)
- `t_hat_alpha`: Weight for treatment prediction Z→T (maximize)
- `c_mse_loss_alpha`: Weight for outcome prediction C→Y (minimize)
- `z_t_loss_alpha`: Weight for Z-T correlation (maximize)
- `kl_loss_coeff`: KL divergence on C and Z distributions
- `feature_corr_loss_coeff`: Feature independence within C and Z

**Architecture Options:**
- `is_linear`: Use linear networks (no activation functions)
- `use_sm`: Apply softmax with temperature to C and Z
- `sm_temp`: Temperature parameter for softmax
- `use_pcgrad`: Enable PCGrad multi-task optimization
- `use_mi_corr_loss`: Use mutual information instead of Pearson
- `use_mi_matrix_loss`: Use MI for C-Z independence

### Data Generating Processes

The `DGP/` directory contains utilities for creating synthetic datasets with known causal structure. See our paper for a description of these.

### Dataset Classes

- **`ParentDataset`**: Base class with train/val/test splits
- **`DGPDataset`**: Synthetic data with known confounders/instruments
- **`GeneratedIVDataset`**: Wrapper for ZNet/AutoIV/GIV/VIV outputs
- **`ECG_DGPDataset`**: ECG signal data support

### Downstream Estimators

After generating C and Z representations, use downstream models to estimate treatment effects:

- **DeepIV**: Two-stage deep learning approach
- **DFIV**: Deep feature instrumental variables
- **TARNet**: Treatment-agnostic representation network

## Usage Examples

### Example 1: Training with Custom Hyperparameters

```python
from utils.train_models import train_znet

# Define hyperparameters
model_params = {
    'lr': 0.001,
    'weight_decay': 0.1,
    'c_pearson_loss_alpha': 0.8,
    'z_pearson_loss_alpha': 0.6,
    't_hat_alpha': 0.5,
    'pearson_matrix_alpha': 0.3,
    'use_pcgrad': True
}

train_params = {
    'num_epochs': 100,
    'batch_size': 128,
    'plot_losses': True
}

dim_params = {
    'c_dim': 15,
    'z_dim': 8
}

# Train and generate IV dataset
znet_model, znet_data, save_path = train_znet(
    data=my_dataset,
    model_params=model_params,
    train_params=train_params,
    gen_data_params=dim_params,
    save_data=True
)
```

### Example 2: Evaluating IV Quality

```python
from utils.evaluation import evaluate_exogeneity, custom_score_function

# Extract representations
train_data = znet_data.train(data_type='torch')

# Evaluate exogeneity (Z ⊥ U)
model_y, covariances, correlations = evaluate_exogeneity(
    train_data.z,  # Instruments
    train_data.x,  # Confounders
    train_data.t,
    train_data.y,
    verbose=True
)

# Compute F-statistics for relevance
f_stat_t, f_stat_y = custom_score_function(
    train_data.z,
    train_data.x,
    train_data.t,
    train_data.y
)

print(f"Treatment relevance F-stat: {f_stat_t:.2f}")
print(f"Outcome predictive power F-stat: {f_stat_y:.2f}")
```

### Example 3: Bootstrap Analysis

```python
from scripts.run_bootstrap import main as bootstrap_main

# Define bootstrap parameters
bootstrap_params = {
    'my_dataset': {
        'znet_params': model_params,
        'deep_iv_params': {...},
        'df_iv_params': {...},
        'znet_train_params': train_params,
        'dim_options': dim_params
    }
}

# Run bootstrap
results_dirs = bootstrap_main(
    bootstrap_params=bootstrap_params,
    datasets={'my_dataset': znet_data},
    num_bootstraps=100
)
```

## Hyperparameter Tuning

### Bayesian Optimization

ZNet supports automated hyperparameter tuning using Bayesian optimization with BoTorch:

```python
from scripts.bayesian_search_ivgen import main as bayesian_search

# Define search bounds
bounds = {
    'znet_params_lr': [0.0001, 0.01],
    'znet_params_c_pearson_loss_alpha': [0.0, 1.0],
    'znet_params_z_pearson_loss_alpha': [0.0, 1.0],
    'znet_params_t_hat_alpha': [0.0, 1.0],
    'dim_options_c_dim': [5, 20],
    'dim_options_z_dim': [1, 15]
}

# Run multi-objective Bayesian search
all_data, best_params = bayesian_search(
    bounds=bounds,
    datasets={'my_dataset': data},
    gen_iv_method='znet',
    n_calls=30,
    n_initial_points=10,
    return_data=True
)
```

The Bayesian search optimizes multiple objectives. It can also be selected to optimize for a single objective. We use:
1. **F-statistic**: Treatment relevance (maximize)
2. **Exogeneity**: Independence between confounders and instruments (maximize)

## Evaluation

### IV Quality Metrics

ZNet includes comprehensive evaluation metrics for instrumental variable quality:

1. **Relevance**: F-statistic for Z→T relationship
2. **Exclusion restriction**: Correlation between Z and C, Z and residuals
3. **Unconfoundedness**: Correlation between Z and hidden confounders

### Treatment Effect Estimation

Evaluate ATE estimation accuracy:

```python
from utils.evaluate_models import evaluate_cate

# Train downstream model
from models.treatment_effect_estimators.deep_iv import DeepIV

deep_iv = DeepIV(x_dim=c_dim, z_dim=z_dim, y_dim=1)
deep_iv.fit(train_data.x, train_data.z, train_data.t, train_data.y)

# Evaluate
test_data = znet_data.test(data_type='np')
ate_estimate = deep_iv.predict_ate(test_data.x, test_data.z)
true_ate = test_data.ite.mean()

print(f"ATE Error: {abs(ate_estimate - true_ate):.4f}")
```

### Visualization

```python
from utils.evaluation import plot_tsne

# Visualize C and Z separation
plot_tsne(
    x_hat=train_data.x,  # Confounders
    x_perp=train_data.z,  # Instruments
    n_components=2,
    title="ZNet Representations"
)
```

### ECG/Sequential Data

For high-dimensional data (e.g., ECG signals):

```python
from models.ZNet.ZNet_ECG import ZNetECG

znet_ecg = ZNetECG(
    input_dim=2500, 
    c_dim=10,
    z_dim=5,
    output_dim=1,
    ecg_channels=12
)
```

## Comparison Methods

ZNet can be benchmarked against other IV generation methods:

- **AutoIV**: Meta-learning approach with mutual information
- **GIV**: Generative adversarial IV selection
- **VIV**: Variational IV with ELBO optimization
- **True IV**: Oracle method using known valid instruments

Run comparisons:

```bash
python scripts/main_driver_bayesian.py \
    --compare_methods autoiv,giv,viv,trueiv \
    --datasets {dataset}
```
## Citation

Please cite our paper: [Causal Effect Estimation with Learned Instrument Representations](archive)

## Acknowledgments

- VIV implementation adapted from [VIV](https://github.com/xinshu-li/VIV/tree/master)
- GIV, AutoIV implementation based on [Meta-EM framework](https://github.com/causal-machine-learning-lab/meta-em)
- DeepIV, DFIV implementation adapted from [MRIV-Net](https://github.com/DennisFrauen/MRIV-Net/tree/main)
- PCGrad implementation adapted from [WeiCheng Tseng](https://github.com/WeiChengTseng/Pytorch-PCGrad)