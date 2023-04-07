import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_cluster import random_walk
from torch_geometric.data.sampler import EdgeIndex, Adj
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch_geometric.utils import add_self_loops, add_remaining_self_loops
from torch_geometric.data import Data, DataLoader, NeighborSampler

from typing import List, Optional, Tuple, NamedTuple, Union, Callable, Dict
from collections import defaultdict
import time
import random
from collections import Counter
from operator import itemgetter
import copy
import numpy as np
from utils.pretrain_utils import get_indices_into_edge_index, HeterogeneousEdgeIndex 
from sklearn.preprocessing import label_binarize



class NeighborSampler(torch.utils.data.DataLoader):
    r"""The neighbor sampler from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper, which allows
    for mini-batch training of GNNs on large-scale graphs where full-batch
    training is not feasible.
    Given a GNN with :math:`L` layers and a specific mini-batch of nodes
    :obj:`node_idx` for which we want to compute embeddings, this module
    iteratively samples neighbors and constructs bipartite graphs that simulate
    the actual computation flow of GNNs.
    More specifically, :obj:`sizes` denotes how much neighbors we want to
    sample for each node in each layer.
    This module then takes in these :obj:`sizes` and iteratively samples
    :obj:`sizes[l]` for each node involved in layer :obj:`l`.
    In the next layer, sampling is repeated for the union of nodes that were
    already encountered.
    The actual computation graphs are then returned in reverse-mode, meaning
    that we pass messages from a larger set of nodes to a smaller one, until we
    reach the nodes for which we originally wanted to compute embeddings.
    Hence, an item returned by :class:`NeighborSampler` holds the current
    :obj:`batch_size`, the IDs :obj:`n_id` of all nodes involved in the
    computation, and a list of bipartite graph objects via the tuple
    :obj:`(edge_index, e_id, size)`, where :obj:`edge_index` represents the
    bipartite edges between source and target nodes, :obj:`e_id` denotes the
    IDs of original edges in the full graph, and :obj:`size` holds the shape
    of the bipartite graph.
    For each bipartite graph, target nodes are also included at the beginning
    of the list of source nodes so that one can easily apply skip-connections
    or add self-loops.
    .. note::
        For an example of using :obj:`NeighborSampler`, see
        `examples/reddit.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        reddit.py>`_ or
        `examples/ogbn_products_sage.py
        <https://github.com/rusty1s/pytorch_geometric/blob/master/examples/
        ogbn_products_sage.py>`_.
    Args:
        edge_index (Tensor or SparseTensor): A :obj:`torch.LongTensor` or a
            :obj:`torch_sparse.SparseTensor` that defines the underlying graph
            connectivity/message passing flow.
            :obj:`edge_index` holds the indices of a (sparse) symmetric
            adjacency matrix.
            If :obj:`edge_index` is of type :obj:`torch.LongTensor`, its shape
            must be defined as :obj:`[2, num_edges]`, where messages from nodes
            :obj:`edge_index[0]` are sent to nodes in :obj:`edge_index[1]`
            (in case :obj:`flow="source_to_target"`).
            If :obj:`edge_index` is of type :obj:`torch_sparse.SparseTensor`,
            its sparse indices :obj:`(row, col)` should relate to
            :obj:`row = edge_index[1]` and :obj:`col = edge_index[0]`.
            The major difference between both formats is that we need to input
            the *transposed* sparse adjacency matrix.
        sizes ([int]): The number of neighbors to sample for each node in each
            layer. If set to :obj:`sizes[l] = -1`, all neighbors are included
            in layer :obj:`l`.
        node_idx (LongTensor, optional): The nodes that should be considered
            for creating mini-batches. If set to :obj:`None`, all nodes will be
            considered.
        num_nodes (int, optional): The number of nodes in the graph.
            (default: :obj:`None`)
        return_e_id (bool, optional): If set to :obj:`False`, will not return
            original edge indices of sampled edges. This is only useful in case
            when operating on graphs without edge features to save memory.
            (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in
            an a sampled mini-batch and returns a transformed version.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
            :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
    """
    def __init__(self, dataset_type: str, edge_index: Union[Tensor, SparseTensor], 
                sample_edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int],
                 node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None,
                 do_filter_edges: bool = True, 
                 **kwargs):

        edge_index = edge_index.to('cpu')
        sample_edge_index = sample_edge_index.to('cpu')

        # add self loops
        sample_edge_index, _ = add_self_loops(sample_edge_index)


        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for Pytorch Lightning...
        self.dataset_type = dataset_type
        self.edge_index = edge_index #always train edge index
        self.sample_edge_index = sample_edge_index # depends on train/val/test
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None
        self.do_filter_edges = do_filter_edges

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
                sample_num_nodes = num_nodes
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
                sample_num_nodes = num_nodes
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1
                sample_num_nodes = int(sample_edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            sample_value = torch.arange(sample_edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
            self.adj_t_sample = SparseTensor(row=sample_edge_index[0], col=sample_edge_index[1],
                                      value=sample_value,
                                      sparse_sizes=(sample_num_nodes, sample_num_nodes)).t()
        else:
            adj_t = edge_index
            adj_t_sample = sample_edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
                adj_t_sample = adj_t_sample.set_value(torch.arange(adj_t_sample.nnz()), layout='coo')
            self.adj_t = adj_t
            self.adj_t_sample = adj_t_sample

        self.adj_t.storage.rowptr()
        self.adj_t_sample.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t_sample.sparse_size(0)) 
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super(NeighborSampler, self).__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    

    def filter_edges(self, edge_index, e_id, source_nodes, target_nodes):
        '''
        Filter out the edges we're trying to predict in the current batch from the edge index
        NOTE: edge_index here is re-indexed
        '''
        reindex_source_nodes = torch.arange(source_nodes.size(0))
        reindex_target_nodes = torch.arange(start = source_nodes.size(0), end = source_nodes.size(0) + target_nodes.size(0))

        # get reverse edges to filter as well
        all_source_nodes = torch.cat([reindex_source_nodes, reindex_target_nodes])
        all_target_nodes = torch.cat([reindex_target_nodes, reindex_source_nodes])
        ind_to_edge_index, ind_to_nodes = get_indices_into_edge_index(edge_index, all_source_nodes, all_target_nodes) #get index into the original edge index (this returns e_ids)
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[ind_to_edge_index] = False

        return edge_index[:, mask], e_id[mask]


    def sample(self, source_batch):
        
        #convert to tensor
        if not isinstance(source_batch, Tensor):
            source_batch = torch.tensor(source_batch)

        # sample nodes to form positive edges. we will try to predict these edges
        row, col, e_id = self.adj_t_sample.coo()    
        target_batch = random_walk(row, col, source_batch, walk_length=1, coalesced=False)[:, 1] #NOTE: only does self loops when no edges in the current partition of the dataset
        batch = torch.cat([source_batch, target_batch], dim=0) 

        batch_size: int = len(batch)
        adjs = []
        n_id = batch
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False) 
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor: #TODO: implement filter_edges if sparse tensor
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)

                if self.do_filter_edges and self.dataset_type == 'train':
                    edge_index, e_id = self.filter_edges(edge_index, e_id, source_batch, target_batch)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = adjs[0] if len(adjs) == 1 else adjs[::-1]
        out = (batch_size, n_id, adjs)
        out = self.transform(*out) if self.transform is not None else out
        return out

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)

class PatientNeighborSampler(torch.utils.data.DataLoader):
   
    def __init__(self, dataset_type: str, edge_index: Union[Tensor, SparseTensor], 
                 sample_edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], 
                 patient_dataset,
                 all_edge_attributes,
                 n_nodes: int,
                 relevant_node_idx = None,
                 do_filter_edges: Optional[bool] = False,
                 num_nodes: Optional[int] = None, 
                 return_e_id: bool = True,
                 sparse_sample: Optional[int] = 0,
                 train_phenotype_counter: Dict = None,
                 train_gene_counter: Dict = None,
                 sample_edges_from_train_patients=False,
                 upsample_cand: Optional[int] = 0,
                 n_cand_diseases=-1,
                 use_diseases=False,
                 nid_to_spl_dict = None,
                 gp_spl = None,
                 spl_indexing_dict=None,

                 hparams=None,
                 transform: Callable = None, 
                 **kwargs):

        edge_index = edge_index.to('cpu')
        sample_edge_index = sample_edge_index.to('cpu')

        # add self loops
        sample_edge_index = torch.cat((sample_edge_index, torch.stack([edge_index.unique(), edge_index.unique()])),1 )
        sample_edge_index, _ = add_remaining_self_loops(sample_edge_index)

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        # Save for Pytorch Lightning...
        self.do_filter_edges = do_filter_edges
        self.relevant_node_idx = relevant_node_idx
        self.n_nodes = n_nodes
        self.all_edge_attr = all_edge_attributes
        self.dataset_type = dataset_type
        self.sparse_sample = sparse_sample
        self.edge_index = edge_index #always train edge index
        self.sample_edge_index = sample_edge_index # depends on train/val/test
        self.patient_dataset = patient_dataset
        self.num_nodes = num_nodes
        self.train_phenotype_counter = train_phenotype_counter
        self.train_gene_counter = train_gene_counter
        self.sample_edges_from_train_patients = sample_edges_from_train_patients
        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Up-sample candidate genes
        self.upsample_cand = upsample_cand
        self.cand_gene_freq = Counter([])
        self.n_cand_diseases = n_cand_diseases
        self.use_diseases = use_diseases
        self.hparams = hparams

        # For SPL
        self.nid_to_spl_dict = nid_to_spl_dict 
        self.gp_spl = gp_spl
        self.spl_indexing_dict = spl_indexing_dict


        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1
                sample_num_nodes = int(sample_edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            sample_value = torch.arange(sample_edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
            self.adj_t_sample = SparseTensor(row=sample_edge_index[0], col=sample_edge_index[1],
                                      value=sample_value,
                                      sparse_sizes=(sample_num_nodes, sample_num_nodes)).t()
        else:
            adj_t = edge_index
            adj_t_sample = sample_edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
                adj_t_sample = adj_t_sample.set_value(torch.arange(adj_t_sample.nnz()), layout='coo')
            self.adj_t = adj_t
            self.adj_t_sample = adj_t_sample

        self.adj_t.storage.rowptr()
        self.adj_t_sample.storage.rowptr()



        super(PatientNeighborSampler, self).__init__(
            self.patient_dataset, collate_fn=self.collate, **kwargs)

    def filter_edges(self, edge_index, e_id, source_nodes, target_nodes):
        '''
        Filter out the edges we're trying to predict in the current batch from the edge index
        NOTE: edge_index here is re-indexed
        '''
        reindex_source_nodes = torch.arange(source_nodes.size(0))
        reindex_target_nodes = torch.arange(start = source_nodes.size(0), end = source_nodes.size(0) + target_nodes.size(0))

        # get reverse edges to filter as well
        all_source_nodes = torch.cat([reindex_source_nodes, reindex_target_nodes])
        all_target_nodes = torch.cat([reindex_target_nodes, reindex_source_nodes])
        ind_to_edge_index, ind_to_nodes = get_indices_into_edge_index(edge_index, all_source_nodes, all_target_nodes) #get index into the original edge index (this returns e_ids)
        mask = torch.ones(edge_index.size(1), dtype=torch.bool)
        mask[ind_to_edge_index] = False

        return edge_index[:, mask], e_id[mask]

    def get_source_nodes(self, phenotype_node_idx, candidate_gene_node_idx, correct_genes_node_idx, disease_node_idx, candidate_disease_node_idx): 
        
        # Get batch node indices based on patient phenotypes and genes
        source_batch = torch.cat(phenotype_node_idx +  candidate_gene_node_idx +  correct_genes_node_idx + disease_node_idx + candidate_disease_node_idx)

         # Randomly sample nodes in KG 
        if self.sparse_sample > 0:
            if self.relevant_node_idx == None:
                rand_idx = torch.randint(high=self.n_nodes, size=(self.sparse_sample,)) # NOTE that this can sample duplicates, but has the benefit of randomly sampling new nodes each epoch
            else:
                rand_idx = self.relevant_node_idx[torch.randint(high=self.relevant_node_idx.size(0), size=(self.sparse_sample,))]
            
            source_batch = torch.cat([source_batch, rand_idx])
            source_batch = torch.unique(source_batch)
            sparse_idx = torch.unique(rand_idx)
        else:
            source_batch = torch.unique(source_batch)
            sparse_idx = torch.Tensor([])

        return source_batch, sparse_idx

    def sample_target_nodes(self, source_batch):
        row, col, e_id = self.adj_t_sample.coo() 
        
        if self.sample_edges_from_train_patients:
            train_patient_nodes = torch.tensor(list(self.train_phenotype_counter.keys()) + list(self.train_gene_counter.keys())) 
            ind_with_train_patient_nodes = (col == train_patient_nodes.unsqueeze(-1)).nonzero(as_tuple=True)[1]
            subset_row = row[ind_with_train_patient_nodes]
            subset_col = col[ind_with_train_patient_nodes]
            try:
                # first try to find an edge that connects back to the training set patient data
                targets = random_walk(subset_row, subset_col, source_batch, walk_length=1, coalesced=False)[:, 1] #NOTE: only does self loops when no edges in the current partition of the dataset
                source_batch_1 = source_batch[~torch.eq(source_batch, targets)]
                targets_1 = targets[~torch.eq(source_batch, targets)]

                # if no edges are found, use all available edges in this split of the data
                source_batch_2 = source_batch[torch.eq(source_batch, targets)]
                targets_2 = random_walk(row, col, source_batch_2, walk_length=1, coalesced=False)[:, 1] #NOTE: only does self loops when no edges in the current partition of the dataset

                #concat the two together
                source_batch = torch.cat([source_batch_1, source_batch_2])
                targets = torch.cat([targets_1, targets_2])

            except:
                targets = random_walk(row, col, source_batch, walk_length=1, coalesced=False)[:, 1] #NOTE: only does self loops when no edges in the current partition of the dataset
        else:

            # # Add self loop to all nodes in source batch 
            # row = torch.cat([row, source_batch])
            # col = torch.cat([col, source_batch])

            targets = random_walk(row, col, source_batch, walk_length=1, coalesced=False)[:, 1] #NOTE: only does self loops when no edges in the current partition of the dataset
        return source_batch, targets

    def add_patient_information(self, patient_ids, phenotype_node_idx, candidate_gene_node_idx, correct_genes_node_idx, disease_node_idx, candidate_disease_node_idx, labels, disease_labels, patient_labels, additional_labels, adjs, batch_size, n_id, sparse_idx, target_batch): #candidate_disease_node_idx

        # Create Data Object & Add patient level information
        adjs = [HeterogeneousEdgeIndex(adj.edge_index, adj.e_id, self.all_edge_attr[adj.e_id], adj.size) for adj in adjs] 
        max_n_candidates = max([len(l) for l in candidate_gene_node_idx])
        data = Data(adjs = adjs, 
                batch_size = batch_size,
                patient_ids = patient_ids,
                n_id = n_id
                )
        if self.hparams['loss'] != 'patient_disease_NCA' and self.hparams['loss'] != 'patient_patient_NCA':
            if None in list(labels): data['one_hot_labels'] = None
            else: data['one_hot_labels'] = torch.LongTensor(label_binarize(labels, classes = list(range(max_n_candidates))))

        if self.use_diseases:
            data['disease_one_hot_labels'] = disease_labels 

        if self.hparams['loss'] == 'patient_patient_NCA':
            if patient_labels is None: data['patient_labels'] = None
            else: data['patient_labels'] = torch.stack(patient_labels)

        # Get candidate genes to phenotypes SPL
        if not self.gp_spl is None:
            if not self.spl_indexing_dict is None:
                patient_ids = np.vectorize(self.spl_indexing_dict.get)(patient_ids).astype(int)
            gene_to_phenotypes_spl = -torch.Tensor(self.gp_spl[patient_ids,:])
            # get gene idx to spl information
            cand_gene_idx_to_spl = [torch.LongTensor(np.vectorize(self.nid_to_spl_dict.get)(cand_genes)) for cand_genes in list(candidate_gene_node_idx)]
            # get SPLs for each patient's candidate genes
            batch_cand_gene_to_phenotypes_spl = [gene_spls[cand_genes] for cand_genes, gene_spls in zip(cand_gene_idx_to_spl, gene_to_phenotypes_spl)]
            # pad to same # of candidate genes
            data['batch_cand_gene_to_phenotypes_spl'] = pad_sequence(batch_cand_gene_to_phenotypes_spl, batch_first=True, padding_value=0)
            # get unique gene idx across all patients in the batch
            cand_gene_idx_flattened_unique = torch.unique(torch.cat(cand_gene_idx_to_spl)).flatten()
            # get SPLs for unique genes in the batch
            data['batch_concat_cand_gene_to_phenotypes_spl'] = gene_to_phenotypes_spl[:, cand_gene_idx_flattened_unique]
        else:
            data['batch_cand_gene_to_phenotypes_spl'] = None
            data['batch_concat_cand_gene_to_phenotypes_spl'] = None


        # Create mapping from KG node IDs to batch indices
        node2batch = {n+1: int(i+1) for i, n in enumerate(data.n_id.tolist())}
        node2batch[0] = 0

        # add phenotype / gene / disease names
        data['phenotype_names'] = [[(self.patient_dataset.node_idx_to_name(p.item()), self.patient_dataset.node_idx_to_degree(p.item())) for p in p_list] for p_list in phenotype_node_idx ]
        data['cand_gene_names'] = [[self.patient_dataset.node_idx_to_name(g.item()) for g in g_list] for g_list in candidate_gene_node_idx ]
        data['corr_gene_names'] = [[self.patient_dataset.node_idx_to_name(g.item()) for g in g_list] for g_list in correct_genes_node_idx  ]
        data['disease_names'] = [[self.patient_dataset.node_idx_to_name(d.item()) for d in d_list] for d_list in disease_node_idx ]

        if self.use_diseases:
            data['cand_disease_names'] = [[self.patient_dataset.node_idx_to_name(d.item()) for d in d_list] for d_list in candidate_disease_node_idx ]


        #reindex nodes to make room for padding
        phenotype_node_idx = [p + 1 for p in phenotype_node_idx]
        candidate_gene_node_idx = [g + 1 for g in candidate_gene_node_idx]
        correct_genes_node_idx = [g + 1 for g in correct_genes_node_idx]
        if self.use_diseases:
            disease_node_idx = [d + 1 for d in disease_node_idx]
            candidate_disease_node_idx = [d + 1 for d in candidate_disease_node_idx]

        # if there aren't any disease idx in the batch, we add filler
        if self.use_diseases:
            if all(len(t) == 0 for t in disease_node_idx):
                disease_node_idx = [torch.LongTensor([0]) for i in range(len(disease_node_idx))]
            if all(len(t) == 0 for t in candidate_disease_node_idx):
                candidate_disease_node_idx = [torch.LongTensor([0]) for i in range(len(candidate_disease_node_idx))]

        # add padding to patient phenotype and gene node idx
        data['batch_pheno_nid'] = pad_sequence(phenotype_node_idx, batch_first=True, padding_value=0) 
        if len(candidate_gene_node_idx[0]) > 0:
            data['batch_cand_gene_nid'] = pad_sequence(candidate_gene_node_idx, batch_first=True, padding_value=0) 
        data['batch_corr_gene_nid'] = pad_sequence(correct_genes_node_idx, batch_first=True, padding_value=0) 
        if self.use_diseases:
            data['batch_disease_nid'] = pad_sequence(disease_node_idx, batch_first=True, padding_value=0) 
            data['batch_cand_disease_nid'] = pad_sequence(candidate_disease_node_idx, batch_first=True, padding_value=0) 

        # Convert KG node IDs to batch IDs
        # When performing inference (i.e., predict.py), use the original node IDs because the full KG is used in forward pass of node model
        if self.dataset_type != "predict":
            data['batch_pheno_nid']  = torch.LongTensor(np.vectorize(node2batch.get)(data['batch_pheno_nid']))
            if len(candidate_gene_node_idx[0]) > 0:
                data['batch_cand_gene_nid'] = torch.LongTensor(np.vectorize(node2batch.get)(data['batch_cand_gene_nid']))
            if len(correct_genes_node_idx[0]) > 0:
                data['batch_corr_gene_nid'] = torch.LongTensor(np.vectorize(node2batch.get)(data['batch_corr_gene_nid']))
            if self.use_diseases:
                data['batch_disease_nid'] = torch.LongTensor(np.vectorize(node2batch.get)(data['batch_disease_nid']))
                data['batch_cand_disease_nid'] = torch.LongTensor(np.vectorize(node2batch.get)(data['batch_cand_disease_nid']))

        return data

    def get_candidate_diseases(self, disease_node_idx, candidate_gene_node_idx):
        cand_diseases = self.patient_dataset.get_candidate_diseases(cand_type=self.hparams['candidate_disease_type'])
        if self.n_cand_diseases != -1: cand_diseases = cand_diseases[torch.randperm(len(cand_diseases))][0:self.n_cand_diseases] 
        
        if self.hparams['only_hard_distractors']: #add candidates to every patient
            candidate_disease_node_idx = tuple(torch.unique(torch.cat([corr_dis, cand_diseases ]), sorted=False) for corr_dis in disease_node_idx)
            candidate_disease_node_idx = tuple(torch.unique(dis[torch.randperm(len(dis))], sorted=False, return_inverse=False, return_counts=False) for dis in candidate_disease_node_idx)
        else: # split candidates across all patients in the batch
            all_correct_diseases = torch.cat(disease_node_idx)
            all_diseases = torch.unique(torch.cat([all_correct_diseases, cand_diseases]))
            all_diseases = all_diseases[torch.randperm(len(all_diseases))]
            candidate_disease_node_idx = np.array_split(all_diseases, len(candidate_gene_node_idx))
            candidate_disease_node_idx = tuple(candidate_disease_node_idx)
        max_n_dis_candidates = max([len(l) for l in candidate_disease_node_idx])
        if max_n_dis_candidates == 0: 
            max_n_dis_candidates = 1
            print('WARNING: there are no disease candidates')

        disease_ind = [(dis.unsqueeze(1) == corr_dis.unsqueeze(0)).nonzero(as_tuple=True)[0] if len(corr_dis) > 0 else torch.tensor(-1) for dis, corr_dis in zip(candidate_disease_node_idx, disease_node_idx)]
        disease_labels = torch.zeros((len(candidate_disease_node_idx), max_n_dis_candidates))
        for i, ind in enumerate(disease_ind): disease_labels[i,ind[ind != -1]] = 1
        return candidate_disease_node_idx, disease_labels

    def get_candidate_patients(self, patient_ids):
        # get patients with the same disease/gene
        similar_pat_ids = [self.patient_dataset.get_similar_patients(p_id, similarity_type=self.hparams['patient_similarity_type']) for p_id in patient_ids]
        # shuffle patients & subset to n_sim_pats so we have X similar patients per patient in the batch
        similar_pat_ids = [p[:self.hparams['n_similar_patients']] for p in similar_pat_ids] #[torch.randperm(len(p))]
        # Retrieve the patients for each of the sampled patient ids if they aren't already in the batch
        patient_ids = list(patient_ids) 
        similar_pats = [self.patient_dataset[self.patient_dataset.patient_id_to_index[p_id.item()]] for p_ids in similar_pat_ids for p_id in p_ids if p_id.item() not in patient_ids]
        return similar_pats
    
    def sample(self, batch, source_batch, target_batch):
        batch_size: int = len(batch)
        adjs = []
        n_id = batch
        for size in self.sizes:

            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            e_id = adj_t.storage.value()
            size = adj_t.sparse_sizes()[::-1]
            if self.__val__ is not None:
                adj_t.set_value_(self.__val__[e_id], layout='coo')

            if self.is_sparse_tensor: #TODO: implement filter_edges if sparse tensor
                adjs.append(Adj(adj_t, e_id, size))
            else:
                row, col, _ = adj_t.coo()
                edge_index = torch.stack([col, row], dim=0)
                if self.do_filter_edges and self.dataset_type == 'train':
                    edge_index, e_id = self.filter_edges(edge_index, e_id, source_batch, target_batch)
                adjs.append(EdgeIndex(edge_index, e_id, size))

        adjs = [adjs[0]] if len(adjs) == 1 else adjs[::-1]
        return adjs, batch_size, n_id

    def collate(self, batch):
        t00 = time.time()
        phenotype_node_idx, candidate_gene_node_idx, correct_genes_node_idx, disease_node_idx, labels, additional_labels, patient_ids = zip(*batch)

        # Up-sample under-represented candidate genes
        t0 = time.time()
        if self.upsample_cand > 0:
            curr_cand_gene_freq = Counter(torch.cat(candidate_gene_node_idx).flatten().tolist())
            self.cand_gene_freq += curr_cand_gene_freq
            num_patients = len(candidate_gene_node_idx) * self.upsample_cand
            lowest_k_cand = self.cand_gene_freq.most_common()[:-num_patients-1:-1]
            lowest_k_cand = np.array_split([g[0] for g in lowest_k_cand], len(candidate_gene_node_idx))
            
            upsampled_candidate_gene_node_idx = []
            added_cand_gene = []
            for patient, cand_gene, corr_gene_idx in zip(candidate_gene_node_idx, lowest_k_cand, labels):
                
                # Remove correct genes from list of upsampled candidate genes
                corr_gene_nid = patient[corr_gene_idx]
                cand_gene = cand_gene[~np.isin(cand_gene, corr_gene_nid)].flatten()
                
                # Remove duplicates
                unique_cand_genes, new_cand_genes_freq = torch.unique(torch.tensor(patient.tolist() + list(cand_gene)), return_counts = True)
                unique_cand_genes = unique_cand_genes[new_cand_genes_freq == 1]
                cand_gene = cand_gene[np.isin(cand_gene, unique_cand_genes)]                
                
                # Add upsampled candidate genes
                added_cand_gene.extend(list(cand_gene))
                new_cand_list = torch.tensor(patient.tolist() + list(cand_gene))
                upsampled_candidate_gene_node_idx.append(new_cand_list)
            
            candidate_gene_node_idx = tuple(upsampled_candidate_gene_node_idx)
            self.cand_gene_freq += Counter(added_cand_gene)

        
        # Add similar patients to batch (for "patients like me" head)
        if self.hparams['add_similar_patients']:
            similar_pats = self.get_candidate_patients(patient_ids)
            # merge original batch with sampled patients
            phenotype_node_idx_sim, candidate_gene_node_idx_sim, correct_genes_node_idx_sim, disease_node_idx_sim, labels_sim, additional_labels_sim, patient_ids_sim = zip(*similar_pats)
            phenotype_node_idx = phenotype_node_idx + phenotype_node_idx_sim
            candidate_gene_node_idx = candidate_gene_node_idx + candidate_gene_node_idx_sim
            correct_genes_node_idx = correct_genes_node_idx + correct_genes_node_idx_sim
            disease_node_idx = disease_node_idx + disease_node_idx_sim
            labels = labels + labels_sim
            additional_labels = additional_labels + additional_labels_sim
            patient_ids = patient_ids + patient_ids_sim
        
        # get patient labels
        patient_labels = correct_genes_node_idx
        
        # Add candidate diseases to batch
        if self.hparams['add_cand_diseases']:
            candidate_disease_node_idx, disease_labels = self.get_candidate_diseases(disease_node_idx, candidate_gene_node_idx)
        else: 
            candidate_disease_node_idx = disease_node_idx
            disease_labels = torch.tensor([1] * len(candidate_disease_node_idx))

        t1 = time.time()

        # get nodes from patients + randomly sampled nodes
        source_batch, sparse_idx = self.get_source_nodes(phenotype_node_idx, candidate_gene_node_idx, correct_genes_node_idx, disease_node_idx, candidate_disease_node_idx)
       
        # sample nodes to form positive edges
        source_batch, target_batch = self.sample_target_nodes(source_batch) 
        batch = torch.cat([source_batch, target_batch], dim=0) 
        t2 = time.time()

        # get k hop adj graph
        adjs, batch_size, n_id = self.sample(batch, source_batch, target_batch)
        t3 = time.time()

        # add patient information to data object
        data = self.add_patient_information(patient_ids, phenotype_node_idx, candidate_gene_node_idx, correct_genes_node_idx, disease_node_idx, candidate_disease_node_idx, labels, disease_labels, patient_labels, additional_labels, adjs, batch_size, n_id, sparse_idx, target_batch) #candidate_disease_node_idx
        t4 = time.time()
        
        if self.hparams['time']:
            print(f'It takes {t0-t00:0.4f}s to unzip batch, {t1-t0:0.4f}s to upsample candidate gene nodes, {t2-t1:0.4f}s to sample positive nodes, {t3-t2:0.4f}s to get k-hop adjs, and {t4-t3:0.4f}s to add patient information')
        return data        

    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)



