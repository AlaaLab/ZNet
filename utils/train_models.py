#######################################################################################
# Author: Jenna Fields, edits Franny Dean
# Script: train_models.py
# Function:  
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed

set_seed(42)
import sys
import torch   
from models.ZNet.ZNet import ZNet
from models.ZNet.ZNet_ECG import ZNetECG
from models.gen_IV_comparisons.AutoIV.auto_iv_trainer import generate_IV as train_AutoIV
from models.gen_IV_comparisons.GIV.GIV import generate_IV as train_GIV
from models.gen_IV_comparisons.VIV.viv import generate_IV as train_VIV
from datetime import datetime
from DGP.dataset_class import *
import os
from utils.ecg_utils import *
from torch.utils.data import DataLoader, random_split
import pickle

#######################################################################################
## Training for IV Generation Methods

def train_znet_inner(data : ParentDataset, 
               model_params, 
               train_params, 
               gen_data_params,
               save_data = False, 
               dir_name=None,
               load_model_path=None
              ):
    """
    Train ZNet model without pretraining (internal training function).
    
    Args:
        data (ParentDataset): Dataset containing X, t, y data.
        model_params (dict): ZNet model hyperparameters.
        train_params (dict): Training hyperparameters (epochs, batch_size, etc.).
        gen_data_params (dict): Generated data dimensions (c_dim, z_dim, y_dim).
        save_data (bool): Whether to save generated IV data. Defaults to False.
        dir_name (str, optional): Directory name for saving data. Defaults to None.
        load_model_path (str, optional): Path to pretrained model weights. Defaults to None.
        
    Returns:
        tuple: (znet, znet_data, save_data_path) - Trained model, generated dataset, and save path.
    """
    if 'y_dim' not in gen_data_params:
        gen_data_params['y_dim'] = data.y.shape[-1]
    if 'c_dim' not in gen_data_params:
        gen_data_params['c_dim'] = 10
    if 'z_dim' not in gen_data_params:
        gen_data_params['z_dim'] = 1
    znet = ZNet(len(data.x_cols), 
                gen_data_params['c_dim'], 
                gen_data_params['z_dim'], 
                gen_data_params['y_dim'],
                **model_params)
    if load_model_path is not None:
        print(f">>> Loading pretrained ZNet from {load_model_path}")
        znet.model.load_state_dict(torch.load(load_model_path))
    train_data = data.train(data_type='torch')
    val_data = data.val(data_type='torch')

    znet.fit(train_data.x, train_data.t, train_data.y, 
             val_X = val_data.x, val_t= val_data.t, val_y = val_data.y,
             **train_params)
    
    znet_x, znet_z, _, _, _ = znet.get_generated_data(data.x, data.t)
    znet_data = ZNetDataset(data, znet_x, znet_z)
    if save_data:
        if not os.path.exists("znet_generated_data/"):
            os.makedirs("znet_generated_data/")
        if dir_name is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(f"znet_generated_data/{dir_name}/", exist_ok=True)
        znet_data.save_csv(f"znet_generated_data/{dir_name}/znet_df.csv")
        znet_data.get_combined_dataset().to_csv(f"znet_generated_data/{dir_name}/combined_znet_df.csv")
        with open(f"znet_generated_data/{dir_name}/znet_dataset.pkl", "wb") as f:
            pickle.dump(znet_data, f)
        save_data_path = f"znet_generated_data/{dir_name}/"
    else:
        save_data_path = None

    return znet, znet_data, save_data_path

def train_znet(data : ParentDataset, 
               model_params, 
               train_params, 
               gen_data_params,
               save_data = False, 
               dir_name=None,
               load_model_path=None,
               pretrain=True, 
               prepend_path= ''
                 ):
    """
    Train ZNet model with optional pretraining stage.
    
    Implements a two-stage training procedure:
    1. Optional pretraining focusing on C→Y and Z→T predictions
    2. Full training with all loss components for disentangled representation learning
    
    Args:
        data (ParentDataset): Dataset containing X, t, y data.
        model_params (dict): ZNet model hyperparameters.
        train_params (dict): Training hyperparameters (epochs, batch_size, etc.).
        gen_data_params (dict): Generated data dimensions (c_dim, z_dim, y_dim).
        save_data (bool): Whether to save generated IV data. Defaults to False.
        dir_name (str, optional): Directory name for saving data. Defaults to None.
        load_model_path (str, optional): Path to pretrained model weights. Defaults to None.
        pretrain (bool): Whether to perform pretraining stage. Defaults to True.
        prepend_path (str): Path prefix for save directory. Defaults to ''.
        
    Returns:
        tuple: (znet, znet_data, save_data_path) - Trained model, generated dataset, and save path.
    """
    
    if 'y_dim' not in gen_data_params:
        gen_data_params['y_dim'] = data.y.shape[-1]
    if 'c_dim' not in gen_data_params:
        gen_data_params['c_dim'] = 10
    if 'z_dim' not in gen_data_params:
        gen_data_params['z_dim'] = 1
    znet = ZNet(len(data.x_cols), 
                gen_data_params['c_dim'], 
                gen_data_params['z_dim'], 
                gen_data_params['y_dim'],
                **model_params)
    
    if pretrain:
        model_pretrain_params = {
            'weight_decay' : 0,
            'lr' : 0.0001,
            'kl_loss_coeff' : 0,
            'feature_corr_loss_coeff' : 0,
            'c_pearson_loss_alpha' : 0,
            'c_mse_loss_alpha' : 1, 
            'z_pearson_loss_alpha' : 0,
            'z_t_loss_alpha' : 0,
            'pearson_matrix_alpha' : 0,
            't_hat_alpha' : 1, 
            'use_pcgrad': False,
            'is_linear' : True,
            'use_sm' : True,
            'sm_temp' : 1,
            'train_xt_net' : False,  
        }
        znet_pretrained, _, _ = train_znet_inner(
            data,
            model_pretrain_params, 
            train_params, 
            gen_data_params
        )
        pretrained_weights = znet_pretrained.model.state_dict()

    if load_model_path is not None:
        print(f">>> Loading pretrained ZNet from {load_model_path}")
        znet.model.load_state_dict(torch.load(load_model_path))
    
    if pretrain:
        print(f">>> Loading pretrained ZNet (C\to Y, Z\to T training)")
        znet.model.load_state_dict(pretrained_weights, strict=False)
    
    train_data = data.train(data_type='torch')
    val_data = data.val(data_type='torch')
    
    znet.fit(train_data.x, train_data.t, train_data.y, 
             val_X = val_data.x, val_t= val_data.t, val_y = val_data.y,
             **train_params)
    
    znet_x, znet_z, _, _, _ = znet.get_generated_data(data.x, data.t)
    znet_data = ZNetDataset(data, znet_x, znet_z)
    if save_data:
        if not prepend_path.endswith('/') and prepend_path != '':
            prepend_path += '/'
        if not os.path.exists(f"{prepend_path}znet_generated_data/"):
            os.makedirs(f"{prepend_path}znet_generated_data/")
        if dir_name is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(f"{prepend_path}znet_generated_data/{dir_name}/", exist_ok=True)
        znet_data.save_csv(f"{prepend_path}znet_generated_data/{dir_name}/znet_df.csv")
        znet_data.get_combined_dataset().to_csv(f"{prepend_path}znet_generated_data/{dir_name}/combined_znet_df.csv")
        with open(f"{prepend_path}znet_generated_data/{dir_name}/znet_dataset.pkl", "wb") as f:
            pickle.dump(znet_data, f)
        save_data_path = f"{prepend_path}znet_generated_data/{dir_name}/"
    else:
        save_data_path = None

    return znet, znet_data, save_data_path


def train_autoiv(data : ParentDataset, 
                 model_params = None,
                 save_data = False,
                 dir_name = None,
                 prepend_path= ''
                 ):
    if model_params is None:
        model_params = {} 
    model = train_AutoIV(data, model_params)
    autoiv_z = model.get_rep_z(data)
    autoiv_data = AutoIVDataset(data, data.x, autoiv_z)

    if save_data:
        if not prepend_path.endswith('/') and prepend_path != '':
            prepend_path += '/'
        if not os.path.exists(f"{prepend_path}autoiv_generated_data/"):
            os.makedirs(f"{prepend_path}autoiv_generated_data/")
        if dir_name is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(f"{prepend_path}autoiv_generated_data/{dir_name}/", exist_ok=True)
        autoiv_data.save_csv(f"{prepend_path}autoiv_generated_data/{dir_name}/autoiv_df.csv")
        autoiv_data.get_combined_dataset().to_csv(f"{prepend_path}autoiv_generated_data/{dir_name}/combined_autoiv_df.csv")
        save_data_path = f"{prepend_path}autoiv_generated_data/{dir_name}/"
    else:
        save_data_path = None
    
    return model, autoiv_data, save_data_path

def train_giv(data : ParentDataset, 
              model_params,
              verbose = False,
              save_data = False,
              dir_name = None,
              prepend_path= ''
              ):
    if 'num_clusters' not in model_params:
        model_params['num_clusters'] = 5
    cluster_final, net = train_GIV(data, model_params)
    giv_data = GIVDataset(data, data.x, cluster_final)
    
    if save_data:
        if not prepend_path.endswith('/') and prepend_path != '':
            prepend_path += '/'
        if not os.path.exists(f"{prepend_path}giv_generated_data/"):
            os.makedirs(f"{prepend_path}giv_generated_data/")
        if dir_name is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(f"{prepend_path}giv_generated_data/{dir_name}/", exist_ok=True)
        giv_data.save_csv(f"{prepend_path}giv_generated_data/{dir_name}/giv_df.csv")
        giv_data.get_combined_dataset().to_csv(f"{prepend_path}giv_generated_data/{dir_name}/combined_giv_df.csv")
        save_data_path = f"{prepend_path}giv_generated_data/{dir_name}/"
    else:
        save_data_path = None
    return net, giv_data, save_data_path

def train_viv(data : ParentDataset, 
                 model_params = None,
                 save_data = False,
                 dir_name = None,
                 prepend_path= ''
                 ):
    if model_params is None:
        model_params = {'exp' : 1} 
    if 'exp' not in model_params:
        model_params['exp'] = 1
    if not prepend_path.endswith('/') and prepend_path != '':
            prepend_path += '/'
    # Make a temporary directory for VIV's intermediate files
    temp_viv = f'{prepend_path}temp_viv_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S.%f")[:-3]}'
    os.makedirs(temp_viv, exist_ok=True)
    viv_z, model = train_VIV(data, dir_name, model_params['exp'], model_params, temp_viv)
    import shutil
    shutil.rmtree(temp_viv)
    viv_data = VIVDataset(data, data.x, viv_z)

    if save_data:
        if not os.path.exists(f"{prepend_path}viv_generated_data/"):
            os.makedirs(f"{prepend_path}viv_generated_data/")
        if dir_name is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(f"{prepend_path}viv_generated_data/{dir_name}/", exist_ok=True)
        viv_data.save_csv(f"{prepend_path}viv_generated_data/{dir_name}/viv_df.csv")
        viv_data.get_combined_dataset().to_csv(f"{prepend_path}viv_generated_data/{dir_name}/combined_viv_df.csv")
        save_data_path = f"{prepend_path}viv_generated_data/{dir_name}/"
    else:
        save_data_path = None

    return model, viv_data, save_data_path

def train_ecg_znet(data,
               train_loader, val_loader, full_dataloader,
               model_params, 
               train_params, 
               gen_data_params,
               train_indices,
               val_indices,
               test_indices,
               znet_model=None,
               save_data = False, 
               dir_name=None,
               load_model_path=None,
               prepend_path = ''):
    
    if 'y_dim' not in gen_data_params:
        gen_data_params['y_dim'] = 1
    if 'c_dim' not in gen_data_params:
        gen_data_params['c_dim'] = 10
    if 'z_dim' not in gen_data_params:
        gen_data_params['z_dim'] = 1
    
    if znet_model is not None:
        znet = znet_model
    else:
        znet = ZNetECG(5000, 
                gen_data_params['c_dim'], 
                gen_data_params['z_dim'], 
                gen_data_params['y_dim'],
                **model_params)
    
    znet.fit(train_loader=train_loader, val_loader=val_loader,
             **train_params)
    sys.stdout.flush()

    znet_x, znet_z, _, _, _ = znet.get_generated_data(full_dataloader)
    znet_data = ZNetECGDataset(data.full_data, znet_x, znet_z, 
                            train_indices, val_indices, test_indices)
    
    if save_data:
        if not prepend_path.endswith('/') and prepend_path != '':
            prepend_path += '/'
        if not os.path.exists(f"{prepend_path}znet_generated_data/"):
            os.makedirs(f"{prepend_path}znet_generated_data/")
        if dir_name is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(f"{prepend_path}znet_generated_data/{dir_name}/", exist_ok=True)
        znet_data.save_csv(f"{prepend_path}znet_generated_data/{dir_name}/znet_df.csv")
        # Skip combining og dataset for ecgs...
        # znet_data.get_combined_dataset().to_csv(f"{prepend_path}znet_generated_data/{dir_name}/combined_znet_df.csv")
        with open(f"{prepend_path}znet_generated_data/{dir_name}/znet_dataset.pkl", "wb") as f:
            pickle.dump(znet_data, f)
        save_data_path = f"{prepend_path}znet_generated_data/{dir_name}/"
    else:
        save_data_path = None


    return znet, znet_data, save_data_path

def ecg_full_train(data : ParentDataset,
               model_params, 
               train_params, 
               gen_data_params,
               save_data = False, 
               dir_name=None,
               load_model_path=None,
               pretrain=True,
               prepend_path = ''
              ):
    
    full_dataset = ECGDataset(data.full_data) 

    train_size = int(0.7 * len(full_dataset))
    val_size   = int(0.15 * len(full_dataset))
    test_size  = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split( 
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(402)  # reproducible split
    )

    train_indices = train_dataset.indices
    val_indices   = val_dataset.indices
    test_indices  = test_dataset.indices

    assert len(train_indices) == train_size
    assert len(val_indices)   == val_size
    assert len(test_indices)  == test_size

    # Ensure no overlap
    assert set(train_indices).isdisjoint(val_indices)
    assert set(train_indices).isdisjoint(test_indices)
    assert set(val_indices).isdisjoint(test_indices)

    train_loader = DataLoader(train_dataset, batch_size=train_params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=train_params['batch_size'], shuffle=False)
    full_dataloader = DataLoader(full_dataset, batch_size=train_params['batch_size'], shuffle=False)

    
    if pretrain:
        model_pretrain_params = {
            'weight_decay' : 0,
            'lr' : 0.0001,
            'kl_loss_coeff' : 0,
            'feature_corr_loss_coeff' : 0,
            'c_pearson_loss_alpha' : 0,
            'c_mse_loss_alpha' : 1, 
            'z_pearson_loss_alpha' : 0,
            'z_t_loss_alpha' : 0,
            'pearson_matrix_alpha' : 0,
            't_hat_alpha' : 1, 
            'use_pcgrad': False,
            'is_linear' : True,
            'use_sm' : True,
            'sm_temp' : 1,
            'train_xt_net' : False,  
        }
        pretrain_params = {
            'batch_size' : 64,
            'num_epochs' : 5, 
            'plot_losses' : False, 
            'use_early_stopping' : False,
        }
        znet_pretrained, _, _ = train_ecg_znet(data,
            train_loader, val_loader, full_dataloader,
            model_pretrain_params, 
            pretrain_params, 
            gen_data_params,
            train_indices,
            val_indices,
            test_indices
        )
        pretrained_weights = znet_pretrained.model.state_dict()
        
        print('Finished pretrain.')
    
    znet = ZNetECG(5000, 
                gen_data_params['c_dim'], 
                gen_data_params['z_dim'], 
                gen_data_params['y_dim'],
                **model_params)
    
    if pretrain:
        print(f"Loading pretrained ZNet: (C\to Y, Z\to T training)")
        znet.model.load_state_dict(pretrained_weights, strict=False)
        
    znet, znet_data, save_data_path = train_ecg_znet(data,
                                     train_loader, val_loader, full_dataloader,
                                     model_params, 
                                     train_params, 
                                     gen_data_params, 
                                     train_indices,
                                     val_indices,
                                     test_indices,
                                     znet_model=znet,
                                     dir_name=dir_name,
                                     save_data=save_data,
                                     prepend_path=prepend_path)
    print('Finished training ZNet.')
    sys.stdout.flush()
    
    return znet, znet_data, save_data_path
    
