'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import scipy.sparse as sp
import numpy as np
from time import time

class Dataset(object):
    '''
    classdocs
    '''
    def __init__(self, path):
        '''
        Constructor
        '''
        self.path = path
        self.trainMatrix, self.n_users, self.n_items = self.load_rating_file_as_matrix(path + "b.train.rating")
        self.testRatings = self.load_rating_file_as_list(path + "b.test.rating")
        self.testNegatives = self.load_negative_file(path + "b.test.negative")
        assert len(self.testRatings) == len(self.testNegatives)

        # self.users = self.load_user_profile("data/book_big/user.txt")

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_user_profile(self, filename):
        # load user profile: userid, genderid, ageid, occupation
        user = []
        f = open(filename, "r")
        # jump first line
        line = f.readline()
        line = f.readline()
        while line:
            row = line.split('\t')
            user.append([eval(row[0]), eval(row[1])])
            line = f.readline()
        f.close()
        return user

    def get_adj_mat(self):
        try:
            t1 = time()
            adj_mat = sp.load_npz(self.path + '/s_adj_mat.npz')
            norm_adj_mat = sp.load_npz(self.path + '/s_norm_adj_mat.npz')
            mean_adj_mat = sp.load_npz(self.path + '/s_mean_adj_mat.npz')
            user_adj_mat = sp.load_npz(self.path + '/user_sim_mean_mat.npz')
            #user_adj_mat = (user_adj_mat + sp.eye(user_adj_mat.shape[0]))/2

            u_mat_1hop = sp.load_npz(self.path + 'u_mat_1hop.npz')
            u_mat_2hop = sp.load_npz(self.path + 'u_mat_2hop.npz')

            i_mat_1hop = sp.load_npz(self.path + 'i_mat_1hop.npz')
            i_mat_2hop = sp.load_npz(self.path + 'i_mat_2hop.npz')

            item_indegree_mat = sp.load_npz(self.path + '/item_indegree_mat_v2.npz')
            # item_indegree_mat = (item_indegree_mat + sp.eye(item_indegree_mat.shape[0]))/2

            item_outdegree_mat = sp.load_npz(self.path + '/item_outdegree_mat_v2.npz')
            # item_outdegree_mat = (item_outdegree_mat + sp.eye(item_outdegree_mat.shape[0])) / 2
            #item_adj_mat = (item_adj_mat + sp.eye(item_adj_mat.shape[0]))/2
            print('already load adj matrix', adj_mat.shape, time() - t1)

        except Exception:
            adj_mat, norm_adj_mat, mean_adj_mat = self.create_adj_mat()
            # user_adj_mat = self.create_user_adj_mat()
            # item_adj_mat = self.create_item_adj_mat()
            sp.save_npz(self.path + '/s_adj_mat.npz', adj_mat)
            sp.save_npz(self.path + '/s_norm_adj_mat.npz', norm_adj_mat)
            sp.save_npz(self.path + '/s_mean_adj_mat.npz', mean_adj_mat)
            # sp.save_npz(self.path + '/s_user_adj_mat.npz', user_adj_mat)
            # sp.save_npz(self.path + '/s_item_adj_mat.npz', item_adj_mat)
        return adj_mat, norm_adj_mat, mean_adj_mat, user_adj_mat, item_indegree_mat, item_outdegree_mat, u_mat_1hop,u_mat_2hop,i_mat_1hop,i_mat_2hop

    def create_adj_mat(self):
        t1 = time()
        adj_mat = sp.lil_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        R = self.trainMatrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T

        print('already create adjacency matrix', adj_mat.shape, time() - t1)

        t2 = time()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            # norm_adj = adj.dot(d_mat_inv)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        def check_adj_if_equal(adj):
            dense_A = np.array(adj.todense())
            degree = np.sum(dense_A, axis=1, keepdims=False)

            temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
            print('check normalized adjacency matrix whether equal to this laplacian matrix.')
            return temp

        norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
        mean_adj_mat = normalized_adj_single(adj_mat)

        print('already normalize adjacency matrix', time() - t2)
        return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dok_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat, num_users+1, num_items+1
