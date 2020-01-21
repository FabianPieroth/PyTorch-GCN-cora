from pathlib import Path
import numpy as np
import torch

import src.utils as utils
from src.data_loader import DataLoader
from src.models import GCN
from src.logger import Logger


class Trainer(object):
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.logger = Logger(**kwargs)
        self.data_loader = DataLoader(**kwargs)
        self.model = GCN(input_dim=self.data_loader.get_input_feat_size(),
                         hidden_dim=kwargs['hidden_dim'],
                         num_classes=self.data_loader.get_num_classes(),
                         dropout_prob=kwargs['dropout_prob'],
                         bias=kwargs['bias'])
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=kwargs['lr'])
        self.cross_entropy = torch.nn.NLLLoss()

    def train(self):

        adj_matrix, feat_matrix, labels, _, val_indices, train_indices = self.data_loader.get_data()

        for epoch in range(self.parameters['num_epochs']):
            # training
            train_acc, train_loss = self.train_step(adj_matrix, feat_matrix, labels, train_indices)

            # validation
            val_loss, val_acc = self.inference_step(adj_matrix, feat_matrix, labels, val_indices)

            # logging
            self.logger.print_info(epoch + 1, train_loss.detach(), train_acc, val_loss, val_acc)
            self.logger.push_early_stopping(val_loss, self.model)

            if self.logger.early_stopping.early_stop:
                print("Early stopping")
                break

        # if the val_loss improves all epochs, we save the weights of the last model
        self.logger.early_stopping.save_checkpoint(val_loss, self.model)

    def train_step(self, adj_matrix, feat_matrix, labels, train_indices):
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(adj_matrix, feat_matrix)
        train_loss = self.loss(out[train_indices], labels[train_indices])
        train_acc = utils.calculate_accuracy(out[train_indices], labels[train_indices])
        train_loss.backward()
        self.optimizer.step()
        return train_acc, train_loss

    def test(self):
        # load the model saved at early stopping point
        state_dict_agent = torch.load(self.logger.save_folder + '/checkpoint.pt', map_location='cpu')
        self.model.load_state_dict(state_dict_agent)
        # inference test set
        adj_matrix, feat_matrix, labels, test_indices, val_indices, train_indices = self.data_loader.get_data()
        test_loss, test_acc = self.inference_step(adj_matrix, feat_matrix, labels, test_indices)
        print('\rTest Loss: {}, Test Acc: {}'.format(test_loss, test_acc))

    def inference_step(self, adj_matrix, feat_matrix, labels, indices):
        """
        Forward matrix and features and calculate loss and accuracy for the labels
        """
        self.model.eval()
        out = self.model(adj_matrix, feat_matrix)
        loss = self.loss(out[indices], labels[indices]).detach()
        acc = utils.calculate_accuracy(out[indices], labels[indices])
        return loss, acc

    def loss(self, predictions, labels):
        # calculate cross entropy loss with L2 regularization for the first layer parameters
        l2_reg = self.parameters['weight_decay'] * torch.sum(self.model.layer1.weights ** 2)
        loss = self.cross_entropy(predictions, labels) + l2_reg
        return loss


def main():
    # Get the project root directory as string
    project_root_dir = str(Path().resolve().parents[0])
    # Set random Seed
    seed = 22
    np.random.seed(seed)
    torch.manual_seed(seed)  # maybe delete later or extend to fully reproduce the results

    parameters = {
        # ## Data ## #
        'dataset': 'cora',
        'project_root_dir': project_root_dir,
        'train_samples_per_class': 20,
        'num_val_samples': 500,
        'save_folder': 'saved_models',
        # ## Model ## #
        'hidden_dim': 16,
        # ## Learning Parameters ## #
        'lr': 0.01,
        'dropout_prob': 0.5,
        'weight_decay': 5e-4,
        'bias': False,
        'num_epochs': 200,
        'patience': 10
    }
    trainer = Trainer(**parameters)

    trainer.train()

    trainer.test()


if __name__ == "__main__":
    main()
