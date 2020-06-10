from time import time
import numpy as np
import tensorflow as tf
import sys
from Dataset import Dataset
import math
import heapq
import argparse
import scipy.sparse as sp
import os
import random

model_type = 'graph_pretrain'
n_layers = 2


def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")
    parser.add_argument('--user_graph', type=int, default=1,
                        help='whether use user graph(1: yes, 0: no).')
    parser.add_argument('--book_graph', type=int, default=1,
                        help='whether use book graph(1: yes, 0: no).')
    parser.add_argument('--interaction_graph', type=int, default=1,
                        help='whether use interaction graph(1: yes, 0: no).')

    parser.add_argument('--weights_path', nargs='?', default='',
                        help='Store model path.')
    parser.add_argument('--data_path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--proj_path', nargs='?', default='',
                        help='Project path.')

    parser.add_argument('--dataset', nargs='?', default='book6/',
                        help='Choose a dataset from {gowalla, yelp2018, amazon-book}')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='0: No pretrain, -1: Pretrain with the learned embeddings, 1:Pretrain with stored models.')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=300,
                        help='Number of epoch.')

    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--layer_size', nargs='?', default='[64,64,64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate.')

    parser.add_argument('--model_type', nargs='?', default='hagcn',
                        help='Specify the name of model (ngcf).')
    parser.add_argument('--adj_type', nargs='?', default='mean',
                        help='Specify the type of the adjacency (laplacian) matrix from {plain, norm, mean}.')
    parser.add_argument('--alg_type', nargs='?', default='hagcn',
                        help='Specify the type of the graph convolutional layer from {ngcf, gcn, gcmc}.')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='0 for NAIS_prod, 1 for NAIS_concat')

    parser.add_argument('--node_dropout_flag', type=int, default=0,
                        help='0: Disable node dropout, 1: Activate node dropout')
    parser.add_argument('--node_dropout', nargs='?', default='[0.1]',
                        help='Keep probability w.r.t. node dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1,0.1,0.1,0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')

    parser.add_argument('--Ks', nargs='?', default='[10, 20]',
                        help='Output sizes of every layer')

    parser.add_argument('--save_flag', type=int, default=1,
                        help='0: Disable model saver, 1: Activate model saver')

    parser.add_argument('--test_flag', nargs='?', default='full',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument('--report', type=int, default=0,
                        help='0: Disable performance report w.r.t. sparsity levels, 1: Show performance report w.r.t. sparsity levels')

    parser.add_argument('--layers', nargs='?', default='[64*4,128,32]')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    return parser.parse_args()


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i + 2)
    return 0


class NGCF_GRAPH(object):
    def __init__(self, data_config, pretrain_data):
        # argument settings
        self.model_type = 'ngcf'
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']

        self.n_fold = 20

        self.norm_adj = data_config['norm_adj']
        self.user_adj = data_config['user_adj']
        self.item_indegree_adj = data_config['item_indegree_adj']
        self.item_outdegree_adj = data_config['item_outdegree_adj']

        self.u_mat_1hop = data_config['u_mat_1hop']
        self.u_mat_2hop = data_config['u_mat_2hop']
        self.i_mat_1hop = data_config['i_mat_1hop']
        self.i_mat_2hop = data_config['i_mat_2hop']


        '''change norm'''
        self.n_nonzero_elems = self.norm_adj.count_nonzero()

        self.lr = args.lr

        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        self.weight_size = eval(args.layer_size)
        self.n_layers = n_layers

        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.decay = self.regs[0]

        self.verbose = args.verbose
        self.u_hidden_size = [128, 64]
        self.b_hidden_size = [256, 64]
        self.layers = [192, 64, 32]
        self.lamb_u = 0.00001
        self.lamb_b = 0.00001

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        self.users = tf.placeholder(tf.int32, shape=[None], name="user_id")
        self.pos_items = tf.placeholder(tf.int32, shape=[None], name="positive_book_id")
        self.neg_items = tf.placeholder(tf.int32, shape=[None], name="negative_book_id")

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************npz
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights = self._init_weights()

        if self.alg_type == 'ngcf':
            self.ua_embeddings, self.ia_embeddings = self._create_ngcf_embed()
        elif self.alg_type == 'hagcn':
            self.ua_embeddings, self.ia_embeddings = self._create_hagcn_embed()
            self.u_embeddings = self._create_u_wgcn_embed()
            self.bpr_u_embeddings = self.weights['user_embedding']
            self.i_embeddings = self._create_i_wgcn_embed()
            self.bpr_i_embeddings = self.weights['item_embedding']
            # self.i_embeddings = self.weights['item_embedding']

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddings, self.neg_items)

        self.u_embed = tf.nn.embedding_lookup(self.u_embeddings, self.users)
        self.pos_i_embed = tf.nn.embedding_lookup(self.i_embeddings, self.pos_items)
        self.neg_i_embed = tf.nn.embedding_lookup(self.i_embeddings, self.neg_items)

        self.bpr_u_embed = tf.nn.embedding_lookup(self.bpr_u_embeddings, self.users)
        self.bpr_pos_i_embed = tf.nn.embedding_lookup(self.bpr_i_embeddings, self.pos_items)
        self.bpr_neg_i_embed = tf.nn.embedding_lookup(self.bpr_i_embeddings, self.neg_items)

        '''mlp to fuse user embeddings from u_wgcn and bpr'''
        # u_vectors = tf.concat([self.u_embed, self.bpr_u_embed], axis=-1)
        # self.u_embed = tf.nn.relu(tf.matmul(u_vectors, self.weights['W_u']) + self.weights['b_u'])
        # self.u_embed_concat = (self.u_embed + self.bpr_u_embed + self.u_g_embeddings)/3
        # self.pos_i_embed_concat = (self.pos_i_embed + self.bpr_pos_i_embed + self.pos_i_g_embeddings)/3
        # self.neg_i_embed_concat (self.neg_i_embed + self.bpr_neg_i_embed + self.neg_i_g_embeddings)/3
        self.u_embed_concat = (self.u_embed + self.bpr_u_embed + self.u_g_embeddings)/3
        self.pos_i_embed_concat = (self.pos_i_embed + self.bpr_pos_i_embed + self.pos_i_g_embeddings)/3
        self.neg_i_embed_concat = (self.neg_i_embed + self.bpr_neg_i_embed + self.neg_i_g_embeddings)/3

        """
        *********************************************************
        Inference for the testing phase.
        """
        self.y_bpr = tf.reduce_sum(self.bpr_u_embed * self.bpr_pos_i_embed, axis=-1)
        self.y_u = tf.reduce_sum(self.u_embed * self.bpr_pos_i_embed, axis=-1)
        self.y_b = tf.reduce_sum(self.bpr_u_embed * self.pos_i_embed, axis=-1)
        self.y_ub = tf.reduce_sum(self.u_embed * self.pos_i_embed, axis=-1)
        self.y_hagcn = tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings, axis=-1)
        # self.batch_ratings = self.y_ub + self.y_hagcn
        self.batch_ratings = tf.reduce_sum(self.u_embed_concat * self.pos_i_embed_concat, axis=-1)

        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        #  mf_loss_ub, mf_loss_hagcn, mf_loss, loss_u, loss_b
        self.mf_loss_bpr, self.mf_loss_u, self.mf_loss_b, self.mf_loss_ub, self.mf_loss_hagcn, self.mf_loss = self.create_bpr_loss \
            (self.u_g_embeddings, self.pos_i_g_embeddings, self.neg_i_g_embeddings, self.bpr_u_embed,
             self.bpr_pos_i_embed, self.bpr_neg_i_embed, self.u_embed, \
             self.pos_i_embed, self.neg_i_embed, self.u_embed_concat, self.pos_i_embed_concat, self.neg_i_embed_concat)

        self.opt_bpr = tf.train.AdagradOptimizer(learning_rate=0.1).minimize(self.mf_loss_bpr)
        self.opt = tf.train.AdagradOptimizer(learning_rate=0.002).minimize(self.mf_loss)
        self.opt_u = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(self.mf_loss_u)
        self.opt_b = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(self.mf_loss_b)
        self.opt_hagcn = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(self.mf_loss_hagcn)
        self.opt_ub = tf.train.AdagradOptimizer(learning_rate=0.001).minimize(self.mf_loss_ub)

    def create_bpr_loss(self, u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings, bpr_u_embed, bpr_pos_i_embed,
                        bpr_neg_i_embed, u_embed, pos_i_embed, neg_i_embed, u_embed_concat, pos_i_embed_concat, neg_i_embed_concat):
        pos_scores_bpr = tf.reduce_sum(bpr_u_embed * bpr_pos_i_embed, axis=-1)
        neg_scores_bpr = tf.reduce_sum(bpr_u_embed * bpr_neg_i_embed, axis=-1)

        maxi_bpr = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores_bpr - neg_scores_bpr), 1e-100, 1e100))
        mf_loss_bpr = tf.negative(tf.reduce_sum(maxi_bpr)) + 1e-5 * (
            tf.nn.l2_loss(bpr_u_embed) + (tf.nn.l2_loss(bpr_pos_i_embed) + tf.nn.l2_loss(
                bpr_neg_i_embed)) / 2)

        pos_scores_u = tf.reduce_sum(u_embed * bpr_pos_i_embed, axis=-1)
        neg_scores_u = tf.reduce_sum(u_embed * bpr_neg_i_embed, axis=-1)

        maxi_u = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores_u - neg_scores_u), 1e-100, 1e100))
        mf_loss_u = tf.negative(tf.reduce_sum(maxi_u)) + 1e-5 * (
            tf.nn.l2_loss(u_embed) + (tf.nn.l2_loss(bpr_pos_i_embed) + tf.nn.l2_loss(
                bpr_neg_i_embed)) / 2)

        pos_scores_b = tf.reduce_sum(bpr_u_embed * pos_i_embed, axis=-1)
        neg_scores_b = tf.reduce_sum(bpr_u_embed * neg_i_embed, axis=-1)

        maxi_b = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores_b - neg_scores_b), 1e-100, 1e100))
        mf_loss_b = tf.negative(tf.reduce_sum(maxi_b)) + 1e-5 * (
            tf.nn.l2_loss(bpr_u_embed) + (tf.nn.l2_loss(pos_i_embed) + tf.nn.l2_loss(
                neg_i_embed)) / 2)

        pos_scores_ub = tf.reduce_sum(u_embed * pos_i_embed, axis=-1)
        neg_scores_ub = tf.reduce_sum(u_embed * neg_i_embed, axis=-1)

        maxi_ub = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores_ub - neg_scores_ub), 1e-100, 1e100))
        mf_loss_ub = tf.negative(tf.reduce_sum(maxi_ub)) + 1e-5 * (
            tf.nn.l2_loss(u_embed) + (tf.nn.l2_loss(pos_i_embed) + tf.nn.l2_loss(
                neg_i_embed)) / 2)

        pos_scores_hagcn = tf.reduce_sum(u_g_embeddings * pos_i_g_embeddings, axis=-1)
        neg_scores_hagcn = tf.reduce_sum(u_g_embeddings * neg_i_g_embeddings, axis=-1)

        maxi_hagcn = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores_hagcn - neg_scores_hagcn), 1e-100, 1e100))
        mf_loss_hagcn = tf.negative(tf.reduce_sum(maxi_hagcn)) + 1e-5 * (
            tf.nn.l2_loss(u_g_embeddings) + (tf.nn.l2_loss(pos_i_g_embeddings) + tf.nn.l2_loss(
                neg_i_g_embeddings)) / 2)  # /self.batch_size

        '''concate user/book embeddings to get prediction'''
        pos_scores = tf.reduce_sum(u_embed_concat * pos_i_embed_concat, axis=-1) 
        neg_scores = tf.reduce_sum(u_embed_concat * neg_i_embed_concat, axis=-1) 

        maxi = tf.log(tf.clip_by_value(tf.nn.sigmoid(pos_scores - neg_scores), 1e-100, 1e100))
        mf_loss = tf.negative(tf.reduce_sum(maxi))

        regularizer = tf.nn.l2_loss(u_embed_concat) + (tf.nn.l2_loss(pos_i_embed_concat) + tf.nn.l2_loss(
            neg_i_embed_concat))/2
        mf_loss += 0.00001 * regularizer

        return mf_loss_bpr, mf_loss_u, mf_loss_b, mf_loss_ub, mf_loss_hagcn, mf_loss

    def _init_weights(self):
        all_weights = dict()

        initializer = tf.random_normal_initializer(stddev=0.01)

        if self.pretrain_data is None:
            all_weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')

            all_weights['item_d1_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='user_embedding')
            all_weights['item_d2_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')

            print('using xavier initialization')
        else:
            all_weights['user_embedding'] = tf.Variable(initial_value=self.pretrain_data['user_embedding'], trainable=True,
                                                        name='user_embedding', dtype=tf.float32)
            all_weights['item_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embedding'], trainable=True,
                                                        name='item_embedding', dtype=tf.float32)

            all_weights['item_d1_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embedding'],
                                                        trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            all_weights['item_d2_embedding'] = tf.Variable(initial_value=self.pretrain_data['item_embedding'],
                                                        trainable=True,
                                                        name='item_embedding', dtype=tf.float32)
            print('using pretrained initialization')

        self.weight_size_list = [self.emb_dim] + self.weight_size

        for k in range(self.n_layers):
            all_weights['W_gc_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_%d' % k)

            all_weights['W_gc_u_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_u_%d' % k)
            all_weights['b_gc_u_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_u_%d' % k)

            all_weights['W_gc_i1_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_i_%d' % k)
            all_weights['b_gc_i1_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_i_%d' % k)

            all_weights['W_gc_i2_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_gc_i_%d' % k)
            all_weights['b_gc_i2_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_gc_i_%d' % k)


            all_weights['W_bi_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

        all_weights['W_att_u0'] = tf.Variable(initializer([64, 32]), name='W_att_u0')
        all_weights['b_att_u0'] = tf.Variable(initializer([1, 32]), name='b_att_u0')
        all_weights['W_att_u1'] = tf.Variable(initializer([64, 32]), name='W_att_u1')
        all_weights['b_att_u1'] = tf.Variable(initializer([1, 32]), name='b_att_u1')

        all_weights['W_att_i0'] = tf.Variable(initializer([64, 32]), name='W_att_i0')
        all_weights['b_att_i0'] = tf.Variable(initializer([1, 32]), name='b_att_i0')
        all_weights['W_att_i1'] = tf.Variable(initializer([64, 32]), name='W_att_i1')
        all_weights['b_att_i1'] = tf.Variable(initializer([1, 32]), name='b_att_i1')

        all_weights['W_gc_i0'] = tf.Variable(initializer([64, 64]), name='W_gc_i0')
        all_weights['b_gc_i0'] = tf.Variable(initializer([1, 64]), name='b_gc_i0')
        all_weights['W_gc_i1'] = tf.Variable(initializer([64, 64]), name='W_gc_i1')
        all_weights['b_gc_i1'] = tf.Variable(initializer([1, 64]), name='b_gc_i1')

        all_weights['W_gc_u0'] = tf.Variable(initializer([64, 64]), name='W_gc_u0')
        all_weights['b_gc_u0'] = tf.Variable(initializer([1, 64]), name='b_gc_u0')
        all_weights['W_gc_u1'] = tf.Variable(initializer([64, 64]), name='W_gc_u1')
        all_weights['b_gc_u1'] = tf.Variable(initializer([1, 64]), name='b_gc_u1')

        self.layers.append(1)
        for i in range(len(self.layers) - 1):
            all_weights['W_%d' % i] = tf.Variable(initializer([self.layers[i], self.layers[i + 1]]))
            all_weights['b_%d' % i] = tf.Variable(initializer([1, self.layers[i + 1]]))
        self.layers.pop()

        all_weights['W_u'] = tf.Variable(initializer([2*self.emb_dim,self.emb_dim]))
        all_weights['b_u'] = tf.Variable(initializer([1, self.emb_dim]))


        return all_weights

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        # print(tf.shape(coo.data))
        # print(tf.shape(coo.shape))
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)

    def _split_A_hat(self, X, n_entity):
        A_fold_hat = []

        fold_len = (n_entity) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_entity
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X, n_entity):
        A_fold_hat = []

        fold_len = (n_entity) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = n_entity
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat


    def _create_hagcn_embed(self):
        A_fold_hat_u1 = self._split_A_hat(self.u_mat_1hop, self.n_users)
        A_fold_hat_i1 = self._split_A_hat(self.i_mat_1hop, self.n_items)

        embeddings_u = self.weights['user_embedding']
        embeddings_v = self.weights['item_embedding']

        for k in range(self.n_layers):
            temp_embed_u, temp_embed_i = [], []
            '''user 1-hop'''
            for f in range(self.n_fold):
                indices_u1 = A_fold_hat_u1[f].indices
                emb_ui = tf.nn.embedding_lookup(embeddings_u,
                                               indices_u1[:, 0] + f * ((self.n_users) // self.n_fold))
                emb_vj = tf.nn.embedding_lookup(embeddings_v, indices_u1[:, 1])

                w_u1_i = tf.nn.tanh(tf.matmul(emb_ui, self.weights['W_att_u%d' % k]) + self.weights['b_att_u%d' % k])
                w_u1_j = tf.nn.tanh(tf.matmul(emb_vj, self.weights['W_att_u%d' % k]) + self.weights['b_att_u%d' % k])
                vals_u1 = tf.reduce_sum(tf.multiply(w_u1_i, w_u1_j), axis=-1)
                vals_u1 = tf.reshape(vals_u1, shape=[A_fold_hat_u1[f].values.get_shape()[0]])
                A_u1 = tf.sparse.softmax(tf.sparse_reorder(tf.SparseTensor(indices_u1, vals_u1, A_fold_hat_u1[f].get_shape())))
                temp_embed_u.append(tf.sparse_tensor_dense_matmul(A_u1, embeddings_v))

            '''item 1-hop'''
            for f in range(self.n_fold):
                indices_i1 = A_fold_hat_i1[f].indices

                emb_vi = tf.nn.embedding_lookup(embeddings_v,
                                                  indices_i1[:, 0] + f * ((self.n_items) // self.n_fold))
                emb_uj = tf.nn.embedding_lookup(embeddings_u, indices_i1[:, 1])

                w_vi = tf.nn.tanh(tf.matmul(emb_vi, self.weights['W_att_i%d' % k]) + self.weights['b_att_i%d' % k])
                w_uj = tf.nn.tanh(tf.matmul(emb_uj, self.weights['W_att_i%d' % k]) + self.weights['b_att_i%d' % k])
                vals_i1 = tf.reduce_sum(tf.multiply(w_vi, w_uj), axis=-1)
                vals_i1 = tf.reshape(vals_i1, shape=[A_fold_hat_i1[f].values.get_shape()[0]])
                A_i1 = tf.sparse.softmax(
                    tf.sparse_reorder(tf.SparseTensor(indices_i1, vals_i1, A_fold_hat_i1[f].get_shape())))
                temp_embed_i.append(tf.sparse_tensor_dense_matmul(A_i1, embeddings_u))

            embeddings_u = tf.concat(temp_embed_u, 0)
            embeddings_v = tf.concat(temp_embed_i, 0)

            embeddings_u = tf.nn.leaky_relu(
                tf.matmul(embeddings_u, self.weights['W_gc_u%d' % k]) + self.weights['b_gc_u%d' % k])
            embeddings_u = tf.nn.dropout(embeddings_u, 1 - self.mess_dropout[0])

            embeddings_v = tf.nn.leaky_relu(
                tf.matmul(embeddings_v, self.weights['W_gc_i%d' % k]) + self.weights['b_gc_i%d' % k])
            embeddings_v = tf.nn.dropout(embeddings_v, 1 - self.mess_dropout[0])

            u_g_embeddings = tf.math.l2_normalize(embeddings_u, axis=1)
            i_g_embeddings = tf.math.l2_normalize(embeddings_v, axis=1)

        return u_g_embeddings, i_g_embeddings

    def _create_u_wgcn_embed(self):
        if self.node_dropout_flag:
            # node dropout.
            print('node dropout:' + str(self.node_dropout[0]))
            A_fold_hat = self._split_A_hat_node_dropout(self.user_adj, self.n_users)
        else:
            print('message dropout:' + str(self.mess_dropout[0]))
            A_fold_hat = self._split_A_hat(self.user_adj, self.n_users)
        embeddings = self.weights['user_embedding']

        all_embeddings = [embeddings]
        # print(self.n_layers)
        for k in range(0, 1):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))

            embeddings = tf.concat(temp_embed, 0)

            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings, self.weights['W_gc_u_%d' % k]) + self.weights['b_gc_u_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings = [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)
        # u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return all_embeddings

    def _create_i_wgcn_embed(self):
        if self.node_dropout_flag:
            # node dropout.
            print('node dropout:' + str(self.node_dropout[0]))
            A_fold_hat_1 = self._split_A_hat_node_dropout(self.item_indegree_adj, self.n_items)
            A_fold_hat_2 = self._split_A_hat_node_dropout(self.item_outdegree_adj, self.n_items)
        else:
            print('message dropout:' + str(self.mess_dropout[0]))
            A_fold_hat_1 = self._split_A_hat(self.item_indegree_adj, self.n_items)
            A_fold_hat_2 = self._split_A_hat(self.item_outdegree_adj, self.n_items)

        embeddings_1 = self.weights['item_d1_embedding']
        embeddings_2 = self.weights['item_d2_embedding']

        for k in range(0, 1):
            temp_embed_1 = []
            temp_embed_2 = []
            for f in range(self.n_fold):
                temp_embed_1.append(tf.sparse_tensor_dense_matmul(A_fold_hat_1[f], embeddings_1))
                temp_embed_2.append(tf.sparse_tensor_dense_matmul(A_fold_hat_2[f], embeddings_2))

            embeddings_1 = tf.concat(temp_embed_1, 0)
            embeddings_2 = tf.concat(temp_embed_2, 0)
            'W_gc_i%d'
            embeddings = tf.nn.leaky_relu(
                tf.matmul(embeddings_1, self.weights['W_gc_i1_%d' % k]) + self.weights['b_gc_i1_%d' % k]+
                tf.matmul(embeddings_2, self.weights['W_gc_i2_%d' % k]) + self.weights['b_gc_i2_%d' % k])
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[k])

            norm_embeddings = tf.math.l2_normalize(embeddings, axis=1)

            all_embeddings = [norm_embeddings]

        all_embeddings = tf.concat(all_embeddings, 1)

        return all_embeddings


# get train instance for prediction model
def get_train_instances(train, num_negatives):
    user_id, pos_item, neg_item = [], [], []
    num_items = train.shape[1]
    for (u, i) in train.keys():
        for t in range(num_negatives):

            # positive instance
            user_id.append(u)
            pos_item.append(i)

            # negative instances
            j = np.random.randint(num_items)
            while (u, j) in train.keys():
                j = np.random.randint(num_items)

            neg_item.append(j)

    return user_id, pos_item, neg_item


def load_pretrained_data():
    pretrain_path = 'bpr_64.npz'
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained embeddings.')
    except Exception:
        pretrain_data = None
    return pretrain_data


def ensureDir(dir_path):
    d = os.path.dirname(dir_path)
    if not os.path.exists(d):
        os.makedirs(d)


def evaluate(sess, opt,top_5=5, top_10=10):
    test_items, test_users, test_positives = [], [], []
    batch_users = []
    adj_items, adj_users = [], []
    for idx in range(0, len(testRatings)):
        rating = testRatings[idx]
        items = testNegatives[idx]
        u1 = rating[0]
        gtItem = rating[1]
        items.append(gtItem)
        test_positives.append(gtItem)

        us = np.full(len(items), u1, dtype='int32')
        batch_users.append(u1)
        test_users.extend(us)
        test_items.extend(items)

    predictions = sess.run(opt,
                              feed_dict={model.users: test_users, model.pos_items: test_items,
                                         model.node_dropout: [0.] * len(eval(args.layer_size)),
                                         model.mess_dropout: [0.] * len(eval(args.layer_size))})

    map_item_score = []
    for i in range(len(batch_users)):
        map_item_score.append({})

    for i in range(len(predictions)):
        u = test_users[i]
        item = test_items[i]
        map_item_score[u][item] = predictions[i]

    hrs_5, ndcgs_5 = [], []
    hrs_10, ndcgs_10 = [], []
    for i in range(len(batch_users)):

        '''top 5 metric'''
        ranklist_5 = heapq.nlargest(top_5, map_item_score[i], key=map_item_score[i].get)
        hr_5 = getHitRatio(ranklist_5, test_positives[i])
        ndcg_5 = getNDCG(ranklist_5, test_positives[i])
        hrs_5.append(hr_5)
        ndcgs_5.append(ndcg_5)

        '''top 10 metric'''
        ranklist_10 = heapq.nlargest(top_10, map_item_score[i], key=map_item_score[i].get)
        hr_10 = getHitRatio(ranklist_10, test_positives[i])
        ndcg_10 = getNDCG(ranklist_10, test_positives[i])
        hrs_10.append(hr_10)
        ndcgs_10.append(ndcg_10)

    avg_hr_5 = np.array(hrs_5).mean()
    avg_ndcg_5 = np.array(ndcgs_5).mean()
    avg_hr_10 = np.array(hrs_10).mean()
    avg_ndcg_10 = np.array(ndcgs_10).mean()
    return avg_hr_5, avg_ndcg_5, avg_hr_10, avg_ndcg_10


if __name__ == '__main__':
    args = parse_args()
    layers = eval(args.layers)

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (args.dataset, args.layers, time())

    '''Loading data'''
    t1 = time()
    dataset = Dataset(args.data_path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    plain_adj, norm_adj, mean_adj, user_adj_mat, item_indegree_adj_mat, item_outdegree_adj_mat, u_mat_1hop,u_mat_2hop,i_mat_1hop,i_mat_2hop = dataset.get_adj_mat()
    config = dict()
    config['n_users'] = dataset.n_users
    config['n_items'] = dataset.n_items
    config['user_adj'] = user_adj_mat
    config['item_indegree_adj'] = item_indegree_adj_mat
    config['item_outdegree_adj'] = item_outdegree_adj_mat

    config['u_mat_1hop'] = u_mat_1hop
    config['u_mat_2hop'] = u_mat_2hop
    config['i_mat_1hop'] = i_mat_1hop
    config['i_mat_2hop'] = i_mat_2hop

    if args.adj_type == 'plain':
        config['norm_adj'] = plain_adj
        print('use the plain adjacency matrix')
    elif args.adj_type == 'norm':
        config['norm_adj'] = norm_adj
        print('use the normalized adjacency matrix')
    elif args.adj_type == 'gcmc':
        config['norm_adj'] = mean_adj
        print('use the gcmc adjacency matrix')
    else:
        config['norm_adj'] = mean_adj + sp.eye(mean_adj.shape[0])
        print('use the mean adjacency matrix')
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    if args.pretrain == -1:
        pretrain_data = load_pretrained_data()
    else:
        pretrain_data = None

    model = NGCF_GRAPH(data_config=config, pretrain_data=pretrain_data)

    '''save model weights'''
    # weights_save_path_bpr = 'weights/BPR_hamgm_%s_%s' % (str(n_layers), str(args.embed_size))
    # weights_save_path_bpr = 'weights/ub_hamgm_3_%s' % (str(args.embed_size))
    drop = eval(args.mess_dropout)
    weights_save_path_bpr = 'weights/all_add_mess_%s' %(str(drop[0]))
    weights_save_path = weights_save_path_bpr
    ensureDir(weights_save_path)
    save_saver = tf.train.Saver(max_to_keep=1)
    test_batch_size = 256
    with tf.Session()as sess_pretrain:
        saver = tf.train.Saver()
        cur_best_pre_ub = 0.
        cur_best_pre_hagcn = 0.
        best_index = 0

        if args.pretrain == 1:
            pretrain_path_from_graph = weights_save_path_bpr

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path_from_graph + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                sess_pretrain.run(tf.global_variables_initializer())
                saver.restore(sess_pretrain, ckpt.model_checkpoint_path)
                print('load the pretrained model parameters from: ', pretrain_path_from_graph)
                avg_hr_ub_5, avg_ndcg_ub_5, avg_hr_ub_10, avg_ndcg_ub_10 = evaluate(sess_pretrain, model.y_ub)
                cur_best_pre_ub = avg_hr_ub_10
                print('Init performence for ub: HR@5 = %.4f, NDCG@5 = %.4f,HR@10 = %.4f, NDCG@10 = %.4f' % (
                    avg_hr_ub_5, avg_ndcg_ub_5, avg_hr_ub_10, avg_ndcg_ub_10))

                avg_hr_hagcn_5, avg_ndcg_hagcn_5, avg_hr_hagcn_10, avg_ndcg_hagcn_10, = evaluate(sess_pretrain,
                                                                                                 model.y_hagcn)
                cur_best_pre_hagcn = avg_hr_hagcn_10
                print('Init performence for hagcn: HR@5 = %.4f, NDCG@5 = %.4f,HR@10 = %.4f, NDCG@10 = %.4f' % (
                    avg_hr_hagcn_5, avg_ndcg_hagcn_5, avg_hr_hagcn_10, avg_ndcg_hagcn_10))
            else:
                sess_pretrain.run(tf.global_variables_initializer())
                print('without pretraining_1.')
        else:
            sess_pretrain.run(tf.global_variables_initializer())
            print('without pretraining.')

  
    print("Train hybrid model")
    with tf.Session() as session:
        saver = tf.train.Saver()
        cur_best_pre_0 = 0.

        if args.pretrain == 1:
            pretrain_path = weights_save_path

            ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                session.run(tf.global_variables_initializer())
                saver.restore(session, ckpt.model_checkpoint_path)
                print('load the pretrained model parameters from: ', pretrain_path)
                avg_hr_5, avg_ndcg_5, avg_hr_10, avg_ndcg_10 = evaluate(session, model.batch_ratings)
                cur_best_pre_0 = avg_hr_10
                print('Init performence: HR@5 = %.4f, NDCG@5 = %.4f,HR@10 = %.4f, NDCG@10 = %.4f' % (avg_hr_5, avg_ndcg_5, avg_hr_10, avg_ndcg_10))
            else:
                session.run(tf.global_variables_initializer())
                print('without pretraining_1.')

        else:
            session.run(tf.global_variables_initializer())
            print('without pretraining.')

        drop = eval(args.mess_dropout)

        weights_save_path = 'weights/all_add_mess_%s' % (str(drop[0]))
        for epoch in range(200):

            # train user/book graph to get embeddings
            t1 = time()
            user_id, pos_item, neg_item = get_train_instances(train, args.num_neg)

            randnum = random.randint(0, 1000)
            random.seed(randnum)
            random.shuffle(user_id)
            random.seed(randnum)
            random.shuffle(pos_item)
            random.seed(randnum)
            random.shuffle(neg_item)
            

            j = 0
            mf_loss = 0
            u_loss = 0
            b_loss = 0
            for j in range(len(user_id) // args.batch_size):
                start = j * args.batch_size
                end = min((j + 1) * args.batch_size, len(user_id))

                users = user_id[start: end]
                pos_items = pos_item[start: end]
                neg_items = neg_item[start: end]

                _, batch_mf_loss = session.run([model.opt, model.mf_loss],
                                               feed_dict={
                                                          model.pos_items: np.array(pos_items),
                                                          model.users: np.array(users),
                                                          model.neg_items: np.array(neg_items),
                                                          model.node_dropout: eval(args.node_dropout),
                                                          model.mess_dropout: eval(args.mess_dropout),
                                                          })
                mf_loss += batch_mf_loss
                # u_loss += batch_u_loss
                # b_loss += batch_b_loss
            t2 = time()

            print("loss: ", mf_loss)

            avg_hr_5, avg_ndcg_5, avg_hr_10, avg_ndcg_10 = evaluate(session, model.batch_ratings)

            if avg_hr_10 > cur_best_pre_0:
                cur_best_pre_0 = avg_hr_10
                save_saver.save(session, weights_save_path + '/weights', global_step=epoch)
                print('save the weights in path: ', weights_save_path)

            t3 = time()
            print('Iteration %d[%.1fs %.1fs]: HR@5 = %.4f, NDCG@5 = %.4f,HR@10 = %.4f, NDCG@10 = %.4f'
                  % (epoch, t2 - t1, t3 - t2,  avg_hr_5, avg_ndcg_5, avg_hr_10, avg_ndcg_10))
