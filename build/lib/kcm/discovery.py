import numpy as np
from scipy.optimize import linear_sum_assignment as linear_assignment

import torch
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore')



    




def sup_con_loss(features, labels, temp=1):
    
    labels = labels.contiguous().view(-1, 1)
    
    dot_matrix = torch.matmul(features, features.T) / temp
    
    mask = ~torch.eye(len(features), dtype=bool)
    
    exp_matrix = torch.exp(dot_matrix) * mask
    
    positive_mask = (labels == labels.T) & mask
    
    log_denom = torch.log(exp_matrix.sum(dim=1, keepdim=True))
    
    log_proba = dot_matrix - log_denom
    
    losses = (positive_mask * log_proba).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    
    loss = - losses.mean()
    
    return loss





class BaselineModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 512, 256], dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim,hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.head = nn.Linear(in_dim,output_dim,bias=False)

        
    def forward(self, x):
        x = self.mlp(x)
        x = self.head(x)
        x = nn.functional.normalize(x, dim=1)
        return x, x, x






class HASHHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[512, 512, 256], dropout=0.3):
        super().__init__()

        layers = []
        in_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim,hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.mlp = nn.Sequential(*layers)

        self.hash = nn.Linear(in_dim,output_dim,bias=False)
        self.variance = nn.Linear(in_dim,output_dim,bias=False)
        self.bn_h = nn.BatchNorm1d(output_dim)
        self.bn_v = nn.BatchNorm1d(output_dim)

        
    def forward(self, x):
        x = self.mlp(x)
        h = self.hash(x)
        v = self.variance(x)

        h = self.bn_h(h)
        v = self.bn_v(v)

        # From OCD SMILE code
        v = v / (nn.Tanh()(v * 1))
        h = nn.Tanh()(h * 1) 

        x = h * v
        
        x = nn.functional.normalize(x, dim=1)
        return x, h, v






def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def split_cluster_acc_v1(y_true, y_pred, mask):

    """
    Evaluate clustering metrics on two subsets of data, as defined by the mask 'mask'
    (Mask usually corresponding to `Old' and `New' classes in GCD setting)
    :param targets: All ground truth labels
    :param preds: All predictions
    :param mask: Mask defining two subsets
    :return:
    """

    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    weight = mask.mean()

    old_acc = cluster_acc(y_true[mask], y_pred[mask])
    new_acc = cluster_acc(y_true[~mask], y_pred[~mask])
    total_acc = weight * old_acc + (1 - weight) * new_acc

    return total_acc, old_acc, new_acc

def split_cluster_acc_v2(y_true, y_pred, mask):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    First compute linear assignment on all data, then look at how good the accuracy is on subsets

    # Arguments
        mask: Which instances come from old classes (True) and which ones come from new classes (False)
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    mask = mask.astype(bool)
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    old_classes_gt = set(y_true[mask])
    new_classes_gt = set(y_true[~mask])

    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    ind_map = {j: i for i, j in ind}
    total_acc = sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

    old_acc = 0
    total_old_instances = 0
    for i in old_classes_gt:
        old_acc += w[ind_map[i], i]
        total_old_instances += sum(w[:, i])
    old_acc /= total_old_instances

    new_acc = 0
    total_new_instances = 0
    for i in new_classes_gt:
        new_acc += w[ind_map[i], i]
        total_new_instances += sum(w[:, i])
    new_acc /= total_new_instances

    return total_acc, old_acc, new_acc




def create_cluster_ids(output, y):
    
    hashes = torch.where(output > 0, 1, 0)
    unique_hashes = torch.unique(hashes, dim=0)
    hashes_bin = np.array(hashes)
    cluster_ids = np.array([int("".join(map(str, row)), 2) for row in hashes_bin])

    _, new_labels = np.unique(cluster_ids, return_inverse=True)

    return new_labels








