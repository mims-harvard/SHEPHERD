# General
import random
import numpy as np
import pandas as pd
import time
import math
from typing import NamedTuple, Optional, Tuple
import plotly.express as px

# Pytorch
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Sigmoid
from torch_geometric.data import Dataset, NeighborSampler, Data

# Sci-kit Learn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, roc_curve, precision_recall_curve

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def to_numpy(input):
        if isinstance(input, torch.sparse.FloatTensor):
            return input.to_dense().cpu().detach().numpy()
        else:
            return input.cpu().detach().numpy()


def from_numpy(np_array):
    return torch.as_tensor(np_array)


def sample_node_for_et(et, targets):
    neg_idx = torch.randperm(targets[et].shape[0])[0] # Randomly select an index into the targets for a given edge type
    node = targets[et][neg_idx] # Select the location of that edge type
    return node


class HeterogeneousEdgeIndex(NamedTuple): #adopted from NeighborSampler code in Pytorch Geometric
    edge_index: Tensor
    e_id: Optional[Tensor]
    edge_type: Optional[Tensor]
    size: Tuple[int, int]

    def to(self, *args, **kwargs):
        edge_index = self.edge_index.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        edge_type = self.edge_type.to(*args, **kwargs) if self.edge_type is not None else None

        return EdgeIndex(edge_index, e_id, edge_type, self.size)


def get_batched_data(data, all_data):
    batch_size, n_id, adjs = data
    adjs = [HeterogeneousEdgeIndex(adj.edge_index, adj.e_id, all_data.edge_attr[adj.e_id], adj.size) for adj in adjs] 
    data = Data(adjs = adjs, 
                batch_size = batch_size,
                n_id = n_id, 
                )
    return data


MAX_SIZE = 625
def get_mask(edge_index, nodes, ind):
    n_splits = math.ceil(nodes.size(0) / MAX_SIZE)
    node_mask = (edge_index[ind,:] == nodes[:MAX_SIZE].unsqueeze(-1)).nonzero()
    for i in range(1, n_splits-1):

        node_mask_mid = (edge_index[ind,:] == nodes[MAX_SIZE*i:MAX_SIZE*(i+1)].unsqueeze(-1)).nonzero()
        node_mask_mid[:,0] = node_mask_mid[:,0] + (MAX_SIZE*i)
        node_mask = torch.cat([node_mask, node_mask_mid])
    node_mask_end = (edge_index[ind,:] == nodes[MAX_SIZE*(n_splits-1):].unsqueeze(-1)).nonzero()
    node_mask_end[:,0] = node_mask_end[:,0] + (MAX_SIZE*(n_splits-1))
    node_mask = torch.cat([node_mask, node_mask_end])
    return node_mask


def get_indices_into_edge_index(edge_index, source_nodes, target_nodes):
    
    if source_nodes.size(0) > MAX_SIZE:
        source_node_mask = get_mask(edge_index, source_nodes, ind = 0)
        target_node_mask = get_mask(edge_index, target_nodes, ind = 1)
    else:
        source_node_mask = (edge_index[0,:] == source_nodes.unsqueeze(-1)).nonzero()
        target_node_mask = (edge_index[1,:] == target_nodes.unsqueeze(-1)).nonzero()
    
    vals_pos, counts_pos = torch.unique(torch.cat([source_node_mask, target_node_mask]), return_counts=True, dim=0)
    return vals_pos[counts_pos > 1][:,1], vals_pos[counts_pos > 1][:,0]


def get_edges(data, all_data, dataset_type):
    # get edge index
    edge_index = all_data.edge_index[:, all_data[f'{dataset_type}_mask']].to(data.n_id.device)
    edge_type = all_data.edge_attr[ all_data[f'{dataset_type}_mask']].to(data.n_id.device)

    # filter to edges between "seed nodes"
    source_nodes = data.n_id[:int(data.batch_size/2)]
    pos_target_nodes = data.n_id[int(data.batch_size/2):int(data.batch_size)]

    # get index into edge & node list
    ind_to_edge_index_pos, ind_to_nodes_pos = get_indices_into_edge_index(edge_index, source_nodes, pos_target_nodes)

    # get edges where both source & target are seed nodes
    data.pos_edge_indices = edge_index[:, ind_to_edge_index_pos]
    data.pos_edge_types = edge_type[ind_to_edge_index_pos]
    data.index_to_node_features_pos = ind_to_nodes_pos

    return data


def calc_metrics(pred, y, threshold=0.5):
    y[y < 0] = 0
    try: 
        roc_score = roc_auc_score(y, pred)
    except ValueError: 
        roc_score = 0.5 
    ap_score = average_precision_score(y, pred)
    acc = accuracy_score(y, pred > threshold)
    f1 = f1_score(y, pred > threshold, average = 'micro')
    return roc_score, ap_score, acc, f1


def metrics_per_rel(pred, link_labels, edge_attr_dict, total_edge_type, split, threshold=0.5, verbose=False):
    log = {}
    for attr, idx in edge_attr_dict.items():
        mask = (total_edge_type == idx)
        if mask.sum() == 0: continue
        pred_per_rel = pred[mask]
        y_per_rel = link_labels[mask]
        roc_per_rel, ap_per_rel, acc_per_rel, f1_per_rel = calc_metrics(pred_per_rel.cpu().detach().numpy(), y_per_rel.cpu().detach().numpy(), threshold)
        if verbose:
            print("ROC for edge type {}: {:.5f}".format(attr, roc_per_rel))
            print("AP for edge type {}: {:.5f}".format(attr, ap_per_rel))
            print("ACC for edge type {}: {:.5f}".format(attr, acc_per_rel))
            print("F1 for edge type {}: {:.5f}".format(attr, f1_per_rel))
        log.update({"edge_metrics/node.%s_%s_roc" % (attr, split): roc_per_rel, "edge_metrics/node.%s_%s_ap" % (attr, split): ap_per_rel, "edge_metrics/node.%s_%s_acc" % (attr, split): acc_per_rel, "edge_metrics/node.%s_%s_f1" % (attr, split): f1_per_rel})
    return log


def plot_roc_curve(pred, labels):
    fpr, tpr, thresholds = roc_curve(labels, pred)
    gmeans = np.sqrt(tpr * (1-fpr))
    max_gmean = max(gmeans)
    roc = roc_auc_score(labels, pred)
    data = {"False Positive Rate": fpr, "True Positive Rate": tpr, "Threshold": thresholds, 
            "ROC": [roc] * len(thresholds), "G-Mean": gmeans, "Max G-Mean": [max_gmean] * len(thresholds)}
    df = pd.DataFrame(data)
    fig = px.line(df, x = "False Positive Rate", y = "True Positive Rate", hover_data=list(data.keys()))
    return fig
