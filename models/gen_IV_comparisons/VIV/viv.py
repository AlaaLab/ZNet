#######################################################################################
# Original Source: https://github.com/xinshu-li/VIV/tree/master
# Edited by: Jenna Fields
# Script: viv.py
# Function: VIV model modified for TensorFlow 2.x
# Date: 02/06/2026
#######################################################################################

from seed_utils import set_seed

set_seed(42)

from DGP.dataset_class import ParentDataset
import os, tensorflow as tf
import tf_slim as slim
from tensorflow.keras import initializers
# import tensorflow_probability as tfp
import numpy as np
import time
import argparse
import scipy

# Enable TF 1.x compatibility for slim
tf.compat.v1.disable_eager_execution()

# tfd = tfp.distributions
#######################################################################################
def log(logfile, str, out=False):
    """ Log a string in a file """
    with open(logfile, 'a') as f:
        f.write(str + '\n')
    if out:
        print(str)

def get_FLAGS(input_args={}):
    """ Define parameter flags """
    # parser = argparse.ArgumentParser(description='Model parameters')
    
    # parser.add_argument('--earl', type=int, default=10, help='when to show bound')
    # parser.add_argument('--lrate', type=float, default=0.0001, help='Learning rate')
    # parser.add_argument('--lrate_min', type=float, default=0.001, help='Learning rate min')
    # parser.add_argument('--epochs', type=int, default=2, help='epochs')
    # parser.add_argument('--seed', type=int, default=2023, help='Seed')
    # parser.add_argument('--bs', type=int, default=256, help='Batch size')
    # parser.add_argument('--d', type=int, default=2, help='Latent dimension')
    # parser.add_argument('--rewrite_log', action='store_true', default=False, help='Whether rewrite log file')
    # parser.add_argument('--use_gpu', action='store_true', default=True, help='The use of GPU')
    # parser.add_argument('--lamba', type=float, default=0.0001, help='weight decay')
    # parser.add_argument('--nh', type=int, default=3, help='number of hidden layers')
    # parser.add_argument('--h', type=int, default=128, help='size of hidden layers')
    # parser.add_argument('--reps', type=int, default=10, help='replications')
    # parser.add_argument('--f', type=str, default='', help='kernel')
    # parser.add_argument('--activation', type=str, default='elu', help='activation function leaky_relu')
    # parser.add_argument('--loss_y', type=float, default=0.1, help='loss y')
    # parser.add_argument('--loss_t', type=float, default=0.1, help='loss t')
    # parser.add_argument('--loss_x', type=float, default=0.1, help='loss x')
    # parser.add_argument('--kl_loss', type=float, default=0.1, help='kl loss')
    # parser.add_argument('--ad_loss', type=float, default=1, help='adversarial loss')
    # parser.add_argument('--lrate_decay_num', type=int, default=10, help='NUM_ITERATIONS_PER_DECAY')
    # parser.add_argument('--lrate_decay', type=float, default=0.97, help='Decay of learning rate every 100 iterations')
    # parser.add_argument('--decay', type=float, default=0.97, help='Decay of learning rate every 100 iterations')
    # parser.add_argument('--optimizer', type=str, default='RMSProp', help='Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)')
    # parser.add_argument('--output_delay', type=int, default=100, help='Number of iterations between log/loss outputs')
    # parser.add_argument('--sparse', action='store_true', default=False, help='Use sparse matrices')
    

    FLAGS = argparse.Namespace()
    FLAGS.earl = input_args.get('earl', 10)
    FLAGS.lrate = input_args.get('lrate', 0.0001)
    FLAGS.lrate_min = input_args.get('lrate_min', 0.001)
    FLAGS.epochs = input_args.get('epochs', 2)
    FLAGS.bs = input_args.get('bs', 256)
    FLAGS.d = input_args.get('d', 2)
    FLAGS.loss_y = input_args.get('loss_y', 0.1)
    FLAGS.loss_t = input_args.get('loss_t', 0.1)
    FLAGS.loss_x = input_args.get('loss_x', 0.1)
    FLAGS.kl_loss = input_args.get('kl_loss', 0.1)
    FLAGS.ad_loss = input_args.get('ad_loss', 1.0)
    FLAGS.seed = input_args.get('seed', 42)
    FLAGS.rewrite_log = input_args.get('rewrite_log', False)
    FLAGS.use_gpu = input_args.get('use_gpu', True)
    FLAGS.lamba = input_args.get('lamba', 0.0001)
    FLAGS.nh = input_args.get('nh', 3)
    FLAGS.h = input_args.get('h', 128)
    FLAGS.reps = input_args.get('reps', 10)
    FLAGS.f = input_args.get('f', '')
    FLAGS.activation = input_args.get('activation', 'elu')
    FLAGS.lrate_decay_num = input_args.get('lrate_decay_num', 10)
    FLAGS.lrate_decay = input_args.get('lrate_decay', 0.97)
    FLAGS.decay = input_args.get('decay', 0.97)
    FLAGS.optimizer = input_args.get('optimizer', 'RMSProp')
    FLAGS.output_delay = input_args.get('output_delay', 100)
    FLAGS.sparse = input_args.get('sparse', False)

    if FLAGS.sparse:
        import scipy.sparse as sparse
    
    return FLAGS


class VIV(object):

    def __init__(self, x_cont, x_disc, y, t, args, q_labels, discrete_features, continuous_features):
        self.discrete_features = discrete_features
        self.continuous_features = continuous_features
        self.x_cont = tf.cast(x_cont, tf.float32)
        self.x_disc = tf.cast(x_disc, tf.float32)
        # print(self.x_disc.shape)
        # print(self.x_cont.shape)
        # self.x_one_hot = tf.one_hot(indices=self.x_disc, depth=len(self.discrete_features) * 2, axis=-1)
        # print(self.x_one_hot.shape)
        self.x_one_hot = self.x_disc
        # self.x_one_hot = tf.concat([tf.one_hot(self.x_disc[:, i], 2) for i in range(self.x_disc.shape[1])], axis=1)
        self.x = tf.concat([x_cont, self.x_one_hot], 1)
        self.y = y
        self.t = t
        self.q_labels = q_labels
        self.build_graph(args)

    def kl_divergence(self, mu, logvar):
        kld = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar), -1))
        return kld
    
    def wassertein_distance(self, mu, logvar):
        p1 = tf.reduce_mean(tf.reduce_sum(tf.square(mu), -1))
        p2 = tf.reduce_mean(tf.reduce_sum(tf.square(tf.sqrt(tf.exp(logvar)) - 1), -1))
        return p1 + p2
    
    def diagonal(self, M):
        new_M = tf.where(tf.abs(M) < 1e-05, M + 1e-05 * tf.abs(M), M)
        return new_M

    def reparameterize(self, mu, logvar):
        noise_ = tf.random.normal(shape=tf.shape(mu))
        output = mu + noise_ * tf.exp(0.5 * logvar)
        return output

    def permute_dims(self, u, d):
        B = tf.shape(u)[0]
        id_ = tf.range(B)
        perm_u = []
        for u_j in tf.split(u, [d, d, d, d], 1):
            id_perm = tf.random.shuffle(id_)
            perm_u_j = tf.gather(u_j, id_perm)
            perm_u.append(perm_u_j)
        return tf.concat(perm_u, 1)

    def fc_net(self, inp, layers, out_layers, scope, lamba=0.001, activation=tf.nn.relu, reuse=None, weights_initializer=initializers.GlorotNormal()):
        with slim.arg_scope([slim.fully_connected], 
                           activation_fn=activation,
                           normalizer_fn=None,
                           weights_initializer=weights_initializer,
                           reuse=reuse,
                           weights_regularizer=slim.l2_regularizer(lamba)):
            if layers:
                h = slim.stack(inp, slim.fully_connected, layers, scope=scope)
                if not out_layers:
                    return h
            else:
                h = inp
            outputs = []
            for i, (outdim, activation) in enumerate(out_layers):
                o1 = slim.fully_connected(h, outdim, activation_fn=activation, scope=scope + '_{}'.format(i + 1))
                outputs.append(o1)

            if len(outputs) > 1:
                return outputs
            else:
                return outputs[0]

    def build_graph(self, args):
        """VIV variational approximation (encoder)"""
        if args.activation == 'elu':
            activation = tf.nn.elu
        elif args.activation == 'relu':
            activation = tf.nn.relu
        elif args.activation == 'leaky_relu':
            activation = tf.nn.leaky_relu
        else:
            activation = tf.nn.relu
            
        inptz = tf.concat([self.t], 1)
        muq_z, sigmaq_z = self.fc_net(inptz, (args.nh - 1) * [args.h], [[args.d, None], [args.d, None]], 'qz_t', lamba=args.lamba, activation=activation)
        self.qz = self.reparameterize(muq_z, sigmaq_z)
        
        inptc = tf.concat([self.x, self.t, self.y], 1)
        muq_c, sigmaq_c = self.fc_net(inptc, (args.nh - 1) * [args.h], [[args.d, None], [args.d, None]], 'qc_xty', lamba=args.lamba, activation=activation)
        self.qc = self.reparameterize(muq_c, sigmaq_c)
        
        inpta = tf.concat([self.x, self.y], 1)
        muq_a, sigmaq_a = self.fc_net(inpta, (args.nh - 1) * [args.h], [[args.d, None], [args.d, None]], 'qa_xy', lamba=args.lamba, activation=activation)
        self.qa = self.reparameterize(muq_a, sigmaq_a)
        
        inptu = tf.concat([self.t, self.y], 1)
        muq_u, sigmaq_u = self.fc_net(inptu, (args.nh - 1) * [args.h], [[args.d, None], [args.d, None]], 'qu_ty', lamba=args.lamba, activation=activation)
        self.qu = self.reparameterize(muq_u, sigmaq_u)
        
        """VIV variational approximation (decoder)"""
        inpt_x = tf.concat([self.qc, self.qa], 1)
        hx = self.fc_net(inpt_x, (args.nh - 1) * [args.h], [], 'px_ca_shared', lamba=args.lamba, activation=activation)
        logits_x_disc = self.fc_net(hx, [args.h], [[len(self.discrete_features), None]], 'px_ca_bin', lamba=args.lamba, activation=activation)
        x_cont_hat = self.fc_net(hx, [args.h], [[len(self.continuous_features), None]], 'px_ca_cont', lamba=args.lamba, activation=activation)
        
        inpt_t = tf.concat([self.qz, self.qc, self.qu], 1)
        t_hat = self.fc_net(inpt_t, (args.nh - 1) * [args.h], [[1, None]], 'pt_zcu', lamba=args.lamba, activation=activation)
        
        inpt_y = tf.concat([self.qa, self.qc, self.qu, self.t], 1)
        y_hat = self.fc_net(inpt_y, (args.nh - 1) * [args.h], [[1, None]], 'py_caut', lamba=args.lamba, activation=activation)
        
        # x_disc_recon = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.x_one_hot, logits=logits_x_disc))
        x_disc_recon = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(self.x_disc, tf.float32), logits=logits_x_disc))
        x_cont_recon = tf.reduce_mean(tf.keras.losses.MSE(self.x_cont, x_cont_hat))
        y_recon = tf.reduce_mean(tf.keras.losses.MSE(self.y, y_hat))
        self.y_loss = y_recon
        t_recon = tf.reduce_mean(tf.keras.losses.MSE(self.t, t_hat))
        self.recon_loss = args.loss_y * y_recon + args.loss_x * (x_cont_recon + x_disc_recon) + args.loss_t * t_recon
        
        sigmaq_a = self.diagonal(sigmaq_a)
        sigmaq_c = self.diagonal(sigmaq_c)
        sigmaq_z = self.diagonal(sigmaq_z)
        sigmaq_u = self.diagonal(sigmaq_u)
        
        muq = tf.concat([muq_z, muq_c, muq_a, muq_u], 1)
        sigmaq = tf.concat([sigmaq_z, sigmaq_c, sigmaq_a, sigmaq_u], 1)
        
        self.kl_loss = args.kl_loss * self.wassertein_distance(muq, sigmaq)
        
        zcau = tf.concat([self.qz, self.qc, self.qa, self.qu], 1)
        zcau_perm = self.permute_dims(zcau, args.d)
        zcau_total = tf.concat([zcau, zcau_perm], 0)
        D_zcau_total = tf.squeeze(self.fc_net(zcau_total, (args.nh - 1) * [args.h], [[2, None]], 'discriminator', lamba=args.lamba, activation=activation))
        D_zcau, D_zcau_perm = tf.split(D_zcau_total, [tf.shape(zcau)[0], tf.shape(zcau)[0]], 0)
        
        zeros = tf.zeros(tf.shape(zcau)[0], tf.int32)
        ones = tf.ones(tf.shape(zcau)[0], tf.int32)
        zero_one_hot = tf.one_hot(indices=zeros, depth=2, axis=-1)
        one_one_hot = tf.one_hot(indices=ones, depth=2, axis=-1)
        
        self.discriminator_loss = 0.5 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=zero_one_hot, logits=D_zcau)) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_one_hot, logits=D_zcau_perm))
        self.vae_ad_loss = args.ad_loss * tf.reduce_mean(D_zcau[:, :1] - D_zcau[:, 1:])
        self.vae_loss = self.recon_loss + self.kl_loss + self.vae_ad_loss


def generate_IV(data: ParentDataset, resultDir, exp, input_args, iv_dir):
    
    args = get_FLAGS(input_args)

    seed = args.seed
    modelDir = os.path.join(iv_dir, 'models/')
    os.makedirs(os.path.dirname(modelDir), exist_ok=True)
    logfile = os.path.join(iv_dir, 'log_parameters.txt')
    modelfile = os.path.join(modelDir, 'm6-best')

    if args.rewrite_log:
        f = open(logfile, 'w')
        f.close()
    
    M = None

    s3 = '\nReplication {}/{}'.format(exp + 1, args.reps)
    # log(logfile, s3)
    
    train = data.train()
    val = data.val()
    test = data.test()
    
    try:
        train.to_numpy()
        val.to_numpy()
        test.to_numpy()
    except:
        pass

    # Get discrete vs continuous features
    binary_features = []
    continuous_features = []
    for i in range(data.x.shape[1]):
        unique_vals = np.unique(data.x[:, i])
        if unique_vals.shape[0] == 2 and 1 in unique_vals and 0 in unique_vals:
            binary_features.append(i)
        else:
            continuous_features.append(i)

    train_dict = {'x': train.x, 't': train.t, 'y': train.y}
    val_dict = {'x': val.x, 't': val.t, 'y': val.y}
    test_dict = {'x': test.x, 't': test.t, 'y': test.y}
    
    with tf.Graph().as_default():
        # Set seeds
        # tf.compat.v1.set_random_seed(seed)
        # np.random.seed(seed)
        
        # Create placeholders
        x_ph_disc = tf.compat.v1.placeholder(tf.int32, [M, len(binary_features)], name='x_cate')
        x_ph_cont = tf.compat.v1.placeholder(tf.float32, [M, len(continuous_features)], name='x_cont')
        t_ph = tf.compat.v1.placeholder(tf.float32, [M, 1], name='t')
        y_ph = tf.compat.v1.placeholder(tf.float32, [M, 1], name='y')
        q_labels = tf.compat.v1.placeholder(tf.bool, name='q_labels')
        
        model = VIV(x_ph_cont, x_ph_disc, y_ph, t_ph, args, q_labels, discrete_features=binary_features, continuous_features=continuous_features)
        
        with tf.compat.v1.Session() as sess:
            z, z_val, z_test = trainNet(model, train_dict, val_dict, test_dict, args, logfile, modelfile, exp, sess)
            
        full_z = np.zeros((data.x.shape[0], z.shape[1]))
        full_z[data.train_indices, :] = z
        full_z[data.val_indices, :] = z_val
        full_z[data.test_indices, :] = z_test
        
        return full_z, model


def trainNet(model: VIV, train, val, test, args, logfile, modelfile, exp, sess):
    best_vae = np.inf
    
    dict_train = {
        model.x_cont: train['x'][:, model.continuous_features], 
        model.x_disc: train['x'][:, model.discrete_features], 
        model.y: train['y'], 
        model.t: train['t'],
        model.q_labels: 0
    }
    dict_valid = {
        model.x_cont: val['x'][:, model.continuous_features], 
        model.x_disc: val['x'][:, model.discrete_features], 
        model.y: val['y'], 
        model.t: val['t'], 
        model.q_labels: 0
    }
    dict_test = {
        model.x_cont: test['x'][:, model.continuous_features], 
        model.x_disc: test['x'][:, model.discrete_features], 
        model.y: test['y'], 
        model.t: test['t'], 
        model.q_labels: 0
    }
    
    max_step = tf.compat.v1.Variable(0, trainable=False, name='max_step')
    min_step = tf.compat.v1.Variable(0, trainable=False, name='min_step')
    max_lr = tf.compat.v1.train.exponential_decay(args.lrate, max_step, args.lrate_decay_num, args.lrate_decay, staircase=True)
    min_lr = tf.compat.v1.train.exponential_decay(args.lrate_min, min_step, args.lrate_decay_num, args.lrate_decay, staircase=True)
    
    # Create optimizers
    if args.optimizer == 'Adagrad':
        max_opt = tf.compat.v1.train.AdagradOptimizer(max_lr)
        min_opt = tf.compat.v1.train.AdagradOptimizer(min_lr)
    elif args.optimizer == 'GradientDescent':
        max_opt = tf.compat.v1.train.GradientDescentOptimizer(max_lr)
        min_opt = tf.compat.v1.train.GradientDescentOptimizer(min_lr)
    elif args.optimizer == 'Adam':
        max_opt = tf.compat.v1.train.AdamOptimizer(max_lr)
        min_opt = tf.compat.v1.train.AdamOptimizer(min_lr)
    else:
        max_opt = tf.compat.v1.train.RMSPropOptimizer(max_lr, args.decay)
        min_opt = tf.compat.v1.train.RMSPropOptimizer(min_lr, args.decay)

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())
    
    variable_names = [variable.name for variable in tf.compat.v1.global_variables()]
    params_max = []
    params_min = []
    
    for var in tf.compat.v1.global_variables():
        if 'discriminator' in var.name:
            params_max.append(var)
        else:
            params_min.append(var)

    grads, global_norm = tf.clip_by_global_norm(tf.gradients(model.vae_loss, params_min), 0.1)
    train_min = min_opt.apply_gradients(zip(grads, params_min), global_step=min_step)
    train_max = max_opt.minimize(model.discriminator_loss, global_step=max_step, var_list=params_max)
    
    sess.run(tf.compat.v1.global_variables_initializer())
    
    objnan = False
    n_epoch, n_iter_per_epoch = args.epochs, 10 * int(train['x'].shape[0] / 100)
    idx = np.arange(train['x'].shape[0])
    
    for epoch in range(n_epoch):
        t0 = time.time()
        np.random.shuffle(idx)
        
        for j in range(n_iter_per_epoch):
            batch = np.random.choice(idx, args.bs)
            x_batch, y_batch, t_batch = train['x'][batch, :], train['y'][batch], train['t'][batch]
            dict_batch = {
                model.x_cont: x_batch[:, model.continuous_features], 
                model.x_disc: x_batch[:, model.discrete_features], 
                model.y: y_batch, 
                model.t: t_batch, 
                model.q_labels: 0
            }
            
            if not objnan:
                sess.run(train_max, feed_dict=dict_batch)
                sess.run(train_min, feed_dict=dict_batch)
                
                if j % args.output_delay == 0 or j == n_iter_per_epoch - 1:
                    vae_loss, recon_loss, kl_loss, ad_loss, disc_loss = sess.run(
                        [model.vae_loss, model.recon_loss, model.kl_loss, model.vae_ad_loss, model.discriminator_loss],
                        feed_dict=dict_train)
                    vae_loss_valid, recon_loss_valid, kl_loss_valid, ad_loss_valid = sess.run(
                        [model.vae_loss, model.recon_loss, model.kl_loss, model.vae_ad_loss],
                        feed_dict=dict_valid)
                    vae_loss_test, recon_loss_test, kl_loss_test, ad_loss_test = sess.run(
                        [model.vae_loss, model.recon_loss, model.kl_loss, model.vae_ad_loss],
                        feed_dict=dict_test)

                    if np.isnan(vae_loss):
                        # log(logfile, 'Experiment %d: Objective is NaN. Skipping.' % exp)
                        objnan = True
                        break
                    
                    loss_str = str(epoch) + '_' + str(j) + '_train:' + '\tVaeloss: %.3f,\trecon_loss: %.3f,\tkl_loss: %.3f,\tad_loss: %.3f,\tdiscriminator_loss%.3f' % (
                        vae_loss, recon_loss, kl_loss, ad_loss, disc_loss)
                    loss_str_valid = str(epoch) + '_' + str(j) + '_valid:' + '\tVaeloss: %.3f,\trecon_loss: %.3f,\tkl_loss: %.3f,\tad_loss: %.3f, ' % (
                        vae_loss_valid, recon_loss_valid, kl_loss_valid, ad_loss_valid)
                    loss_str_test = str(epoch) + '_' + str(j) + '_test:' + '\tVaeloss: %.3f,\trecon_loss: %.3f,\tkl_loss: %.3f,\tad_loss: %.3f ' % (
                        vae_loss_test, recon_loss_test, kl_loss_test, ad_loss_test)

                    # log(logfile, loss_str)
                    # log(logfile, loss_str_valid)
                    # log(logfile, loss_str_test)

                    vae_valid = sess.run(model.vae_loss, feed_dict=dict_valid)
                    if vae_valid <= best_vae:
                        saver.save(sess, modelfile)
                        str_loss = 'Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_vae, vae_valid)
                        best_vae = vae_valid
                        # log(logfile, str_loss)

        if objnan:
            break
            
        if epoch % args.earl == 0 or epoch == n_epoch - 1:
            vae_valid = sess.run(model.vae_loss, feed_dict=dict_valid)
            if vae_valid <= best_vae:
                saver.save(sess, modelfile)
                str_loss = 'Improved validation bound, old: {:0.3f}, new: {:0.3f}'.format(best_vae, vae_valid)
                best_vae = vae_valid
                # log(logfile, str_loss)

    saver.restore(sess, modelfile)
    z = sess.run(model.qz, feed_dict=dict_train)
    z_val = sess.run(model.qz, feed_dict=dict_valid)
    z_test = sess.run(model.qz, feed_dict=dict_test)
    
    return z, z_val, z_test


def get_IV(data, exp, iv_dir):
    load_z = np.load(iv_dir + f"z_{exp}.npz")
    data.train.z = load_z['rep_z']
