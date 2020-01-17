import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class GCLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GCLayer, self).__init__()

        dist_bound = 1. / np.sqrt(out_features)
        self.weights = nn.Parameter(torch.FloatTensor(in_features, out_features).uniform_(-dist_bound, dist_bound))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = torch.zeros(out_features, requires_grad=False)

    def forward(self, adj_matrix, feat_matrix):

        out = torch.mm(feat_matrix, self.weights)
        out = torch.mm(adj_matrix, out) + self.bias
        #out = torch.sparse.mm(adj_matrix, out) + self.bias
        return out


class GCN(nn.Module):
    def __init__(self, input_dim ,hidden_dim, num_classes, dropout_prob, bias=False):
        super(GCN, self).__init__()

        self.layer1 = GCLayer(input_dim, hidden_dim, bias)
        self.layer2 = GCLayer(hidden_dim, num_classes, bias)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, adj_matrix, feat_matrix):
        # TODO: Fix dense multiplication!!
        adj_matrix = adj_matrix.to_dense()
        feat_matrix = feat_matrix.to_dense()
        x = F.relu(self.layer1(adj_matrix, feat_matrix))
        x = self.dropout(x)
        x = self.layer2(adj_matrix, x)

        return self.log_softmax(x)

