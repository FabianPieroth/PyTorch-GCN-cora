import numpy as np
import scipy.sparse as sparse
from sklearn.preprocessing import normalize
import src.utils as utils
import torch


class DataLoader(object):

    def __init__(self, **kwargs):
        self.parameters = kwargs
        # load data
        self.data_path = kwargs['project_root_dir'] + '/data/' + kwargs['dataset']
        self.raw_data = np.load(self.data_path + '.npz', allow_pickle=True)

        # get different data sets
        self.test_indices, self.val_indices, self.train_indices = self._perform_data_split()
        self.labels = self._get_labels()
        # calculate feature and adjacency matrix
        self.adj_matrix = self._get_adj_matrix()
        self.feat_matrix = self._get_feature_matrix()

    def get_data(self):
        return self.adj_matrix, self.feat_matrix, self.labels, self.test_indices, self.val_indices, self.train_indices

    def _get_feature_matrix(self):
        """
        Create sparse feature matrix and normalize rows to length 1 in L1 norm
        :return: sparse.csr_matrix
        """
        feat_matrix = utils.get_sparse_matrix(raw_data=self.raw_data, matrix_name='attr_matrix')
        feat_matrix = normalize(feat_matrix, norm='l1', axis=1)

        return utils.sparse_matrix_to_tensor(feat_matrix)

    def _get_labels(self):
        return torch.LongTensor(self.raw_data['labels'])

    def _get_adj_matrix(self):
        """
        Create sparse adjacency matrix, add self-connections and apply renormalization-trick from paper
        :return: sparse.csr_matrix
        """
        adj_matrix = utils.get_sparse_matrix(raw_data=self.raw_data, matrix_name='adj_matrix')
        # include self connections
        adj_matrix = adj_matrix + sparse.identity(adj_matrix.shape[0], format='csr')
        adj_matrix = utils.renormalize_matrix(adj_matrix)

        return utils.sparse_matrix_to_tensor(adj_matrix)

    def _perform_data_split(self):
        """
        Split the dataset the same way as in the paper; all sets are distinct
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
        return torch.LongTensor(test_indices), torch.LongTensor(val_indices), torch.LongTensor(train_indices)

    def get_num_classes(self):
        return len(set(self.raw_data['labels']))

    def get_input_feat_size(self):
        return self.raw_data['attr_matrix.shape'][1]
