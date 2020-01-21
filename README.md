# PyTorch-GCN-corma
A reimplementation of the paper Semi-Supervised Classification with Graph Convolutional Networks by [Kipf and 
Welling 2016](https://arxiv.org/abs/1609.02907) in PyTorch.
I will solely focus on reproducing the results for the cora dataset, which I pushed into the git
repository as it is not very large.

The early stopping class is taken from the repository of [Bjarten](https://github.com/Bjarten/early-stopping-pytorch).
I only changed the process of saving the model slightly.


## Setup
Run:

```
make environment 
source activate pytorch-gcn
make requirements
```

