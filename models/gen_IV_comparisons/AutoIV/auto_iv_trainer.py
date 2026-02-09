#######################################################################################
# Author: https://github.com/causal-machine-learning-lab/meta-em/blob/main/GenerateIV/AutoIV/model.py
# Edited by: Jenna Fields
# Script: auto_iv_trainer.py
# Function: AutoIV implementation compatible with ZNet data classes
# Date: 02/06/2026
#######################################################################################

try:
    import tensorflow as tf
except:
    pass
import random
import os
import numpy as np

# from DGP.dataset_class import ParentDataset, SplitDataset
from .auto_iv import AutoIV

# Make sure eager execution is disabled (since the model uses graph mode)
tf.compat.v1.disable_eager_execution()

#######################################################################################

def get_tf_var(names):
    _vars = []
    for na_i in range(len(names)):
        _vars = _vars + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=names[na_i])
    return _vars

def get_opt(lrate, NUM_ITER_PER_DECAY, lrate_decay, loss, _vars):
    """
    Create optimizer using TensorFlow 1.x compatible API
    """
    # Define the global step variable
    global_step = tf.compat.v1.Variable(0, trainable=False, name='global_step')
    
    # Create learning rate with decay
    learning_rate = tf.compat.v1.train.exponential_decay(
        learning_rate=lrate,
        global_step=global_step,
        decay_steps=NUM_ITER_PER_DECAY,
        decay_rate=lrate_decay,
        staircase=True
    )
    
    # Create optimizer
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    
    # Create training operation with global step update
    train_op = optimizer.minimize(loss, var_list=_vars, global_step=global_step)
    
    return train_op

def get_IV(data, resultDir, exp):
    autoiv_savepath = resultDir + 'autoiv/'
    load_z = np.load(autoiv_savepath+f'z_{exp}.npz')
    rep_z = load_z['rep_z']
    data.train.z = rep_z
    return rep_z

def generate_IV(data, train_dict):
    # Clear any existing graph
    tf.compat.v1.reset_default_graph()
    
    # Set default configuration settings if not provided
    if 'seed' not in train_dict:
        train_dict['seed'] = 42
    if 'emb_dim' not in train_dict:
        train_dict['emb_dim'] = 1
    if 'rep_dim' not in train_dict:
        train_dict['rep_dim'] = 1
    train_dict['coefs'] = {
                'coef_cx2y': train_dict.get('coef_cx2y', 1), 
                'coef_zc2x': train_dict.get('coef_zc2x', 1), 
                'coef_lld_zx': train_dict.get('coef_lld_zx', 1),
                'coef_lld_zy': train_dict.get('coef_lld_zy', 1), 
                'coef_lld_cx': train_dict.get('coef_lld_cx', 1), 
                'coef_lld_cy': train_dict.get('coef_lld_cy', 1),
                'coef_lld_zc': train_dict.get('coef_lld_zc', 1), 
                'coef_bound_zx': train_dict.get('coef_bound_zx', 1), 
                'coef_bound_zy': train_dict.get('coef_bound_zy', 1),
                'coef_bound_cx': train_dict.get('coef_bound_cx', 1), 
                'coef_bound_cy': train_dict.get('coef_bound_cy', 1), 
                'coef_bound_zc': train_dict.get('coef_bound_zc', 1), 
                'coef_reg': train_dict.get('coef_reg', 0.001)
            }
    if 'dropout' not in train_dict:
        train_dict['dropout'] = 0.0
    if 'rep_net_layer' not in train_dict:
        train_dict['rep_net_layer'] = 2
    if 'x_net_layer' not in train_dict:
        train_dict['x_net_layer'] = 2
    if 'emb_net_layer' not in train_dict:
        train_dict['emb_net_layer'] = 2
    if 'y_net_layer' not in train_dict:
        train_dict['y_net_layer'] = 2
    if 'sigma' not in train_dict:
        train_dict['sigma'] = 0.1
    if 'lrate' not in train_dict:
        train_dict['lrate'] = 1e-3
    if 'opt_lld_step' not in train_dict:
        train_dict['opt_lld_step'] = 1
    if 'opt_bound_step' not in train_dict:
        train_dict['opt_bound_step'] = 1
    if 'opt_2stage_step' not in train_dict:
        train_dict['opt_2stage_step'] = 1
    if 'epochs' not in train_dict:
        train_dict['epochs'] = 1000
    if 'interval' not in train_dict:
        train_dict['interval'] = 10
    
    # Add the 'num' parameter which is required by the model (batch size)
    if 'num' not in train_dict:
        # Set num to the batch size from data or default to the training data size
        train_dict['num'] = data.train().t.shape[0]

    # Set random seeds for reproducibility
    random.seed(train_dict['seed'])
    np.random.seed(train_dict['seed'])
    tf.compat.v1.set_random_seed(train_dict['seed'])
    os.environ['PYTHONHASHSEED'] = str(train_dict['seed'])

    # Get dimensions
    dim_x, dim_v, dim_y = data.train().t.shape[1], data.train().x.shape[1], data.train().y.shape[1]
    # print(f"Dimensions - X: {dim_x}, V: {dim_v}, Y: {dim_y}")
    # Create the model
    model = AutoIV(train_dict, dim_x, dim_v, dim_y)

    """ Get trainable variables. """
    zx_vars = get_tf_var(['zx'])
    zy_vars = get_tf_var(['zy'])
    cx_vars = get_tf_var(['cx'])
    cy_vars = get_tf_var(['cy'])
    zc_vars = get_tf_var(['zc'])
    rep_vars = get_tf_var(['rep/rep_z', 'rep/rep_c'])
    x_vars = get_tf_var(['x'])
    emb_vars = get_tf_var(['emb'])
    y_vars = get_tf_var(['y'])

    vars_lld = zx_vars + zy_vars + cx_vars + cy_vars + zc_vars
    vars_bound = rep_vars
    vars_2stage = rep_vars + x_vars + emb_vars + y_vars

    """ Set optimizer. """
    # Using TF1.x compatible optimizers
    with tf.compat.v1.variable_scope('opt_lld'):
        train_opt_lld = get_opt(
            lrate=train_dict['lrate'], 
            NUM_ITER_PER_DECAY=100,
            lrate_decay=0.95, 
            loss=model.loss_lld, 
            _vars=vars_lld
        )

    with tf.compat.v1.variable_scope('opt_bound'):
        train_opt_bound = get_opt(
            lrate=train_dict['lrate'], 
            NUM_ITER_PER_DECAY=100,
            lrate_decay=0.95, 
            loss=model.loss_bound, 
            _vars=vars_bound
        )

    with tf.compat.v1.variable_scope('opt_2stage'):
        train_opt_2stage = get_opt(
            lrate=train_dict['lrate'], 
            NUM_ITER_PER_DECAY=100,
            lrate_decay=0.95, 
            loss=model.loss_2stage, 
            _vars=vars_2stage
        )

    train_opts = [train_opt_lld, train_opt_bound, train_opt_2stage]
    train_steps = [
        train_dict['opt_lld_step'], 
        train_dict['opt_bound_step'], 
        train_dict['opt_2stage_step']
    ]

    # Initialize variables
    model.sess.run(tf.compat.v1.global_variables_initializer())

    """ Training, validation, and test dict. """
    dict_train_true = {
        model.v: data.train().x, 
        model.x: data.train().t, 
        model.y: data.train().y, 
        model.train_flag: True
    }
    
    dict_train = {
        model.v: data.train().x, 
        model.x: data.train().t, 
        model.x_pre: data.train().t, 
        model.y: data.train().y, 
        model.train_flag: False
    }
    
    dict_valid = {
        model.v: data.val().x, 
        model.x: data.val().t, 
        model.x_pre: data.val().t, 
        model.y: data.val().y, 
        model.train_flag: False
    }
    
    dict_test = {
        model.v: data.test().x, 
        model.x_pre: data.test().t, 
        model.y: data.test().y, 
        model.train_flag: False
    }

    epochs = train_dict['epochs']
    intt = train_dict['epochs'] // train_dict['interval']
    
    for ep_th in range(epochs):
        if (ep_th % intt == 0) or (ep_th == epochs - 1):
            loss = model.sess.run([
                model.loss_cx2y,
                model.loss_zc2x,
                model.lld_zx,
                model.lld_zy,
                model.lld_cx,
                model.lld_cy,
                model.lld_zc,
                model.bound_zx,
                model.bound_zy,
                model.bound_cx,
                model.bound_cy,
                model.bound_zc,
                model.loss_reg
            ], feed_dict=dict_train)
            
            y_pre_train = model.sess.run(model.y_pre, feed_dict=dict_train)
            y_pre_valid = model.sess.run(model.y_pre, feed_dict=dict_valid)
            y_pre_test = model.sess.run(model.y_pre, feed_dict=dict_test)

            mse_train = np.mean(np.square(y_pre_train - data.train().y))
            mse_valid = np.mean(np.square(y_pre_valid - data.val().y))
            mse_test = np.mean(np.square(y_pre_test - data.test().y))

            # print("Epoch {}: {} - {}".format(ep_th, mse_train, mse_valid))
            
        for i in range(len(train_opts)):  # optimizer to train
            for j in range(train_steps[i]):  # steps of optimizer
                model.sess.run(train_opts[i], feed_dict=dict_train_true)

    # def get_rep_z(data : SplitDataset):
    #     dict_data = {
    #         model.v: data.x, 
    #         model.x: data.t,  # Changed from x_pre to x based on the model architecture
    #         model.x_pre: data.t, 
    #         model.y: data.y, 
    #         model.train_flag: False
    #     }
    #     data_z = model.sess.run(model.rep_z, feed_dict=dict_data)
    #     return data_z

    # rep_z = get_rep_z(data.train())
    # data.train().z = rep_z

    # rep_z_val = get_rep_z(data.val())
    # data.val().z = rep_z_val

    # rep_z_test = get_rep_z(data.test())
    # data.test().z = rep_z_test
    
    # combined_z = np.zeros((data.train().x.shape[0] + data.val().x.shape[0] + data.test().x.shape[0], rep_z.shape[1]))
    # combined_z[data.train_indices, :] = rep_z
    # combined_z[data.val_indices, :] = rep_z_val
    # combined_z[data.test_indices, :] = rep_z_test
    # # Create AutoIVDataset with the combined z
    # autoiv_data = AutoIVDataset(data, data.x, combined_z)
    return model #rep_z, rep_z_val, rep_z_test
# try:

#     import tensorflow as tf
# except:
#     pass
# import random
# import os
# import numpy as np
# from .auto_iv import AutoIV

# def get_tf_var(names):
#     _vars = []
#     for na_i in range(len(names)):
#         _vars = _vars + tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=names[na_i])
#     return _vars

# def get_opt(lrate, NUM_ITER_PER_DECAY, lrate_decay, loss, _vars):
#     global_step = tf.Variable(0, trainable=False)
#     lr = tf.keras.optimizers.schedules.ExponentialDecay(lrate, NUM_ITER_PER_DECAY, lrate_decay, staircase=True)
#     opt = tf.keras.optimizers.Adam(lr)
#     train_opt = opt.minimize(loss, global_step=global_step, var_list=_vars)
#     return train_opt


# def get_IV(data, resultDir, exp):
#     autoiv_savepath = resultDir + 'autoiv/'
#     load_z = np.load(autoiv_savepath+f'z_{exp}.npz')
#     rep_z = load_z['rep_z']
#     data.train.z = rep_z
#     return rep_z

# def generate_IV(data, train_dict):
#     # data.numpy()

#     train_dict['seed'] = 42
#     train_dict['emb_dim'] = 1
#     train_dict['rep_dim'] = 1
#     train_dict['coefs'] = {'coef_cx2y': 1, 'coef_zc2x': 1, 'coef_lld_zx': 1,
#                 'coef_lld_zy': 1, 'coef_lld_cx': 1,  'coef_lld_cy': 1,
#                 'coef_lld_zc': 1, 'coef_bound_zx': 1, 'coef_bound_zy': 1,
#                 'coef_bound_cx': 1, 'coef_bound_cy': 1, 'coef_bound_zc': 1, 'coef_reg': 0.001}
#     train_dict['dropout'] = 0.0
#     train_dict['rep_net_layer'] = 2
#     train_dict['x_net_layer'] = 2
#     train_dict['emb_net_layer'] = 2
#     train_dict['y_net_layer'] = 2
#     train_dict['sigma'] = 0.1
#     train_dict['lrate'] = 1e-3
#     train_dict['opt_lld_step'] = 1
#     train_dict['opt_bound_step'] = 1
#     train_dict['opt_2stage_step'] = 1
#     train_dict['epochs'] = 1000
#     train_dict['interval'] = 10

#     # tf.reset_default_graph()
#     random.seed(train_dict['seed'])
#     # tf.compat.v1.set_random_seed(train_dict['seed'])
#     np.random.seed(train_dict['seed'])
#     # os.environ['PYTHONHASHSEED'] = str(train_dict['seed'])

#     # tf.compat.v1.reset_default_graph()
#     dim_x, dim_v, dim_y = data.train.t.shape[1], data.train.x.shape[1], data.train.y.shape[1]
#     model = AutoIV(train_dict, dim_x, dim_v, dim_y)

#     """ Get trainable variables. """
#     zx_vars = get_tf_var(['zx'])
#     zy_vars = get_tf_var(['zy'])
#     cx_vars = get_tf_var(['cx'])
#     cy_vars = get_tf_var(['cy'])
#     zc_vars = get_tf_var(['zc'])
#     rep_vars = get_tf_var(['rep/rep_z', 'rep/rep_c'])
#     x_vars = get_tf_var(['x'])
#     emb_vars = get_tf_var(['emb'])
#     y_vars = get_tf_var(['y'])

#     vars_lld = zx_vars + zy_vars + cx_vars + cy_vars + zc_vars
#     vars_bound = rep_vars
#     vars_2stage = rep_vars + x_vars + emb_vars + y_vars

#     """ Set optimizer. """
#     train_opt_lld = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
#                             lrate_decay=0.95, loss=model.loss_lld, _vars=vars_lld)

#     train_opt_bound = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
#                                 lrate_decay=0.95, loss=model.loss_bound, _vars=vars_bound)

#     train_opt_2stage = get_opt(lrate=train_dict['lrate'], NUM_ITER_PER_DECAY=100,
#                                 lrate_decay=0.95, loss=model.loss_2stage, _vars=vars_2stage)

#     train_opts = [train_opt_lld, train_opt_bound, train_opt_2stage]
#     train_steps = [train_dict['opt_lld_step'], train_dict['opt_bound_step'], train_dict['opt_2stage_step']]

#     # model, train_opts, train_steps, data.train
#     # Begin Train
#     model.sess.run(tf.compat.v1.global_variables_initializer())

#     """ Training, validation, and test dict. """
#     dict_train_true = {model.v: data.train.x, model.x: data.train.t, model.y: data.train.y, model.train_flag: True}
#     dict_train = {model.v: data.train.x, model.x: data.train.t, model.x_pre: data.train.t, model.y: data.train.y, model.train_flag: False}
#     dict_valid = {model.v: data.valid.x, model.x: data.valid.t, model.x_pre: data.valid.t, model.y: data.valid.y, model.train_flag: False}
#     # dict_test = {model.v: data.test.x, model.x_pre: data.test.t, model.y: data.test.y, model.train_flag: False}

#     epochs = train_dict['epochs']
#     intt = train_dict['epochs'] // train_dict['interval']
#     for ep_th in range(epochs):
#         if (ep_th % intt == 0) or (ep_th == epochs - 1):
#             loss = model.sess.run([model.loss_cx2y,
#                                     model.loss_zc2x,
#                                     model.lld_zx,
#                                     model.lld_zy,
#                                     model.lld_cx,
#                                     model.lld_cy,
#                                     model.lld_zc,
#                                     model.bound_zx,
#                                     model.bound_zy,
#                                     model.bound_cx,
#                                     model.bound_cy,
#                                     model.bound_zc,
#                                     model.loss_reg],
#                                     feed_dict=dict_train)
#             y_pre_train = model.sess.run(model.y_pre, feed_dict=dict_train)
#             y_pre_valid = model.sess.run(model.y_pre, feed_dict=dict_valid)
#             # y_pre_test = model.sess.run(model.y_pre, feed_dict=dict_test)

#             mse_train = np.mean(np.square(y_pre_train - data.train.y))
#             mse_valid = np.mean(np.square(y_pre_valid - data.valid.y))
#             # mse_test = np.mean(np.square(y_pre_test - data.test.y))

#             print("Epoch {}: {} - {} - {}".format(ep_th, mse_train, mse_valid)) #, mse_test))
#         for i in range(len(train_opts)):  # optimizer to train
#             for j in range(train_steps[i]):  # steps of optimizer
#                 model.sess.run(train_opts[i], feed_dict=dict_train_true)

#     def get_rep_z(data):
#         dict_data = {model.v: data.x, model.x_pre: data.t, model.y: data.y, model.train_flag: False}
#         data_z = model.sess.run(model.rep_z, feed_dict=dict_data)
#         return data_z

#     rep_z = get_rep_z(data.train)
#     data.train.z = rep_z
    
#     return rep_z
