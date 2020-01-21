import torch
import numpy as np
import scipy.sparse as sparse


def read_in_indices_from_txt(filename):
    indices = []
    with open(filename) as f:
        for line in f:
            indices.append(int(line.rstrip('\n')))
    return indices


def calculate_accuracy(out, labels):
    predict = torch.argmax(out, dim=1)
    accuracy = np.sum([labels[i] == predict[i]
                       for i in range(len(labels))]) / len(labels)
    return accuracy


def renormalize_matrix(adj_matrix):
    # renormalization trick
    rowsum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_inv_sqrt = sparse.diags(d_inv_sqrt)
    # TODO: the adjacency matrix is not symmetric! need to fix this
    adj_matrix = adj_matrix.dot(d_inv_sqrt).transpose().dot(d_inv_sqrt)
    # adj_matrix = d_inv_sqrt.dot(adj_matrix).dot(d_inv_sqrt)
    return adj_matrix


def get_sparse_matrix(raw_data, matrix_name):
    data = raw_data[matrix_name + '.data']
    indices = raw_data[matrix_name + '.indices']
    indptr = raw_data[matrix_name + '.indptr']
    shape = raw_data[matrix_name + '.shape']
    sparse_matrix = sparse.csr_matrix((data, indices, indptr), shape=shape)

    return sparse_matrix


def sparse_matrix_to_tensor(sp_matrix):

    sp_matrix = sp_matrix.tocoo()
    values = sp_matrix.data
    indices = np.vstack((sp_matrix.row, sp_matrix.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = sp_matrix.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))
