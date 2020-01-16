from pathlib import Path
import numpy as np
import torch

from src.data_loader import DataLoader


class Trainer(object):
    def __init__(self, **kwargs):
        data_loader = DataLoader(**kwargs)


def main():
    # Get the project root directory as string
    project_root_dir = str(Path().resolve().parents[0])
    # Set random Seed
    seed = 49
    np.random.seed(seed)
    # torch.manual_seed(seed)  # maybe delete later or extend to fully reproduce the results

    parameters = {
        'dataset': 'cora',
        'project_root_dir': project_root_dir,
        'train_samples_per_class': 20,
        'num_val_samples': 500,
        'lr': 0.01,
        'dropout': 0.5
    }
    trainer = Trainer(**parameters)


if __name__ == "__main__":
    main()
