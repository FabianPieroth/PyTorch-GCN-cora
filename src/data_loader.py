import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
import src.utils as utils


class DataLoader(object):

    def __init__(self, **kwargs):
        self.parameters = kwargs
        # load data
        self.data_path = kwargs['project_root_dir'] + '/data/' + kwargs['dataset']
        self.raw_data = np.load(self.data_path + '.npz', allow_pickle=True)

        # get different data sets
        self.test_indices, self.val_indices, self.train_indices = self._perform_data_split()
        self.test_data, self.val_data, self.train_data = self._get_label_splits()
        # calculate feature and adjacency matrix
        self.adj_matrix = self._get_adj_matrix()
        self.feat_matrix = self._get_feature_matrix()

    def get_data(self):
        return self.adj_matrix, self.feat_matrix, self.test_data, self.val_data, self.train_data, self.test_indices, self.val_indices, self.train_indices

    def _get_feature_matrix(self):
        """
        Create sparse feature matrix and normalize rows to length 1 in L1 norm
        :return: sparse.csr_matrix
        """
        feat_matrix = self.get_sparse_matrix(matrix_name='attr_matrix')
        feat_matrix = normalize(feat_matrix, norm='l1', axis=1)

        return feat_matrix

    def _get_label_splits(self):
        test_data = self.raw_data['labels'][self.test_indices]
        val_data = self.raw_data['labels'][self.val_data]
        train_data = self.raw_data['labels'][self.train_data]

        return test_data, val_data, train_data

    def _get_adj_matrix(self):
        """
        Create sparse adjacency matrix, add self-connections and apply renormalization-trick from paper
        :return: sparse.csr_matrix
        """
        adj_matrix = self.get_sparse_matrix(matrix_name='adj_matrix')
        # include self connections
        adj_matrix = adj_matrix + sparse.identity(adj_matrix.shape[0], format='csr')
        # renormalization trick
        rowsum = np.array(adj_matrix.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_inv_sqrt = sparse.diags(d_inv_sqrt)
        # TODO: here seems the be an error!
        # adj_matrix = adj_matrix.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

        adj_matrix = d_inv_sqrt.dot(adj_matrix).dot(d_inv_sqrt)

        return adj_matrix

    def get_sparse_matrix(self, matrix_name):
        data = self.raw_data[matrix_name + '.data']
        indices = self.raw_data[matrix_name + '.indices']
        indptr = self.raw_data[matrix_name + '.indptr']
        shape = self.raw_data[matrix_name + '.shape']
        sparse_matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)

        return sparse_matrix

    def _perform_data_split(self):
        """
        Split the dataset according the same way as in the paper; all sets are distinct
        Test: indices are given
        Train: sample 20 indices per class
        Val: 500 random indices
        :return: list, list, list
        """
        test_indices = utils.read_in_indices_from_txt(self.data_path + '-test-indices.txt')

        train_indices = []
        for label in list(set(self.raw_data['labels'])):
            valid_label_indices = [int(i) for i in np.argwhere(self.raw_data['labels'] == label) if
                                   i not in test_indices]
            train_indices.extend(np.random.choice(valid_label_indices, size=self.parameters['train_samples_per_class']))

        valid_val_indices = [i for i in range(len(self.raw_data['labels'])) if i not in train_indices + test_indices]
        val_indices = list(np.random.choice(valid_val_indices, size=self.parameters['num_val_samples']))
        return test_indices, train_indices, val_indices
