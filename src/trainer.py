from pathlib import Path
import numpy as np
import torch

import src.utils as utils
from src.data_loader import DataLoader
from src.models import GCN


class Trainer(object):
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.data_loader = DataLoader(**kwargs)
        self.model = GCN(input_dim=self.data_loader.get_input_feat_size(),
                         hidden_dim=kwargs['hidden_dim'],
                         num_classes=self.data_loader.get_num_classes(),
                         dropout_prob=kwargs['dropout_prob'],
                         bias=kwargs['bias'])
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=kwargs['lr'])
        self.loss = torch.nn.NLLLoss()

    def train(self):

        adj_matrix, feat_matrix, labels, _, val_indices, train_indices = self.data_loader.get_data()

        for epoch in range(self.parameters['num_epochs']):

            self.model.train()
            self.optimizer.zero_grad()
            out = self.model(adj_matrix, feat_matrix)
            train_loss = self.loss(out[train_indices], labels[train_indices])
            train_acc = utils.calculate_accuracy(out[train_indices], labels[train_indices])
            train_loss.backward()
            self.optimizer.step()

            print('\rEpoch: {}, Train Loss: {}, Train Acc: {}'.format(epoch + 1, train_loss.detach(), train_acc))

            # validation

            val_loss, val_acc = self.inference_step(adj_matrix, feat_matrix, labels, val_indices)

            print('\rEpoch: {}, Train Loss: {}, Train Acc: {}, Val Loss: {}, Val Acc: {}'.format(epoch + 1,
                                                                                                 train_loss.detach(),
                                                                                                 train_acc,
                                                                                                 val_loss,
                                                                                                 val_acc))

    def test(self):
        # inference test set
        adj_matrix, feat_matrix, labels, test_indices, val_indices, train_indices = self.data_loader.get_data()
        test_loss, test_acc = self.inference_step(adj_matrix, feat_matrix, labels, test_indices)
        print('\rTest Loss: {}, Test Acc: {}'.format(test_loss, test_acc))

        # include the logger!

    def inference_step(self, adj_matrix, feat_matrix, labels, indices):
        """
        Forward matrix and features and calculate loss and accuracy for the labels
        """
        self.model.eval()
        out = self.model(adj_matrix, feat_matrix)
        loss = self.loss(out[indices], labels[indices]).detach()
        acc = utils.calculate_accuracy(out[indices], labels[indices])
        return loss, acc


def main():
    # Get the project root directory as string
    project_root_dir = str(Path().resolve().parents[0])
    # Set random Seed
    seed = 49
    np.random.seed(seed)
    torch.manual_seed(seed)  # maybe delete later or extend to fully reproduce the results

    parameters = {
        'dataset': 'cora',
        'project_root_dir': project_root_dir,
        'train_samples_per_class': 20,
        'num_val_samples': 500,
        'hidden_dim': 32,
        'lr': 0.01,
        'dropout_prob': 0.5,
        'bias': False,
        'num_epochs': 200
    }
    trainer = Trainer(**parameters)

    trainer.train()

    trainer.test()


if __name__ == "__main__":
    main()
