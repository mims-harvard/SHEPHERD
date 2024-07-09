from pytorch_metric_learning.distances import LpDistance
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses import BaseMetricLossFunction

import torch, torch.nn as nn, torch.nn.functional as F, numpy as np


def unique(x, dim=None):
        """Unique elements of x and indices of those unique elements
        https://github.com/pytorch/pytorch/issues/36748#issuecomment-619514810
        e.g.
        unique(tensor([
            [1, 2, 3],
            [1, 2, 4],
            [1, 2, 3],
            [1, 2, 5]
        ]), dim=0)
        => (tensor([[1, 2, 3],
                    [1, 2, 4],
                    [1, 2, 5]]),
            tensor([0, 1, 3]))
        """
        unique, inverse = torch.unique(
            x, sorted=True, return_inverse=True, dim=dim) 
        perm = torch.arange(inverse.size(0), dtype=inverse.dtype,
                            device=inverse.device)
        inverse, perm = inverse.flip([0]), perm.flip([0])
        return unique, inverse.new_empty(unique.size(dim)).scatter_(0, inverse, perm)


def _construct_labels(candidate_embeddings, candidate_node_idx, correct_node_idx, mask):
    '''
        Format the batch to input into metric learning loss function
    '''
    batch, n_candidates, embed_dim = candidate_embeddings.shape

    # get mask
    mask_reshaped = mask.reshape(batch*n_candidates, -1)
    expanded_mask = mask_reshaped.expand(-1,embed_dim) 

    # flatten the gene node idx and gene embeddings
    candidate_node_idx_flattened = candidate_node_idx.view(batch*n_candidates, -1)
    candidate_embeddings_flattened = candidate_embeddings.view(batch*n_candidates, -1) 
    candidate_embeddings_flattened = candidate_embeddings_flattened * expanded_mask

    # get unique node idx & corresponding embeddings
    candidate_node_idx_flattened_unique, unique_ind = unique(candidate_node_idx_flattened, dim=0) 
    candidate_embeddings_flattened_unique = candidate_embeddings_flattened[unique_ind,:]
    
    # remove padding
    if candidate_node_idx_flattened_unique[0] == 0:
        candidate_embeddings_flattened_unique = candidate_embeddings_flattened_unique[1:,:] 
        candidate_node_idx_flattened_unique = candidate_node_idx_flattened_unique[1:, :]

    # create a one hot encoding of correct gene/disease in the list of all in the batch
    label_idx = torch.where(candidate_node_idx_flattened_unique.unsqueeze(1) == correct_node_idx.unsqueeze(0), 1, 0)
    label_idx = label_idx.sum(dim=-1).T
    
    return candidate_node_idx_flattened_unique, candidate_embeddings_flattened_unique, label_idx 


def _construct_disease_labels(disease_embedding, batch_disease_nid):
    if len(disease_embedding.shape) == 3:
        batch, n_candidates, embed_dim = disease_embedding.shape
        batch_disease_nid_reshaped = batch_disease_nid.view(batch*n_candidates, -1)
        disease_embedding_reshaped = disease_embedding.view(batch*n_candidates, -1) 
    else:
        batch_disease_nid_reshaped = batch_disease_nid
        disease_embedding_reshaped = disease_embedding
    
    # get unique diseases * corresponding embeddings in batch
    batch_disease_nid_unique, unique_ind = unique(batch_disease_nid_reshaped, dim=0)
    disease_embeddings_unique = disease_embedding_reshaped[unique_ind,:]

    #remove padding
    if batch_disease_nid_unique[0] == 0:
        disease_embeddings_unique = disease_embeddings_unique[1:,:] 
        batch_disease_nid_unique = batch_disease_nid_unique[1:, :]

    # create a one hot encoding of correct disease in the list of all diseases in the batch
    label_idx = torch.where(batch_disease_nid_unique.T == batch_disease_nid_reshaped, 1, 0)
    if len(disease_embedding.shape) == 3: #need to reshape the label_idx
        batch, n_candidates, embed_dim = disease_embedding.shape
        label_idx = label_idx.view(batch, n_candidates, -1)
        label_idx = torch.sum(label_idx, dim=1)
    
    return disease_embeddings_unique, label_idx


### https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch/blob/master/criteria/multisimilarity.py
class MultisimilarityCriterion(torch.nn.Module):
    def __init__(self, pos_weight, neg_weight, margin, thresh, 
                 embed_dim, only_hard_distractors=True):
        super().__init__()
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.margin     = margin
        self.thresh     = thresh
        self.only_hard_distractors = only_hard_distractors


    def forward(self, sims, mask, one_hot_labels, **kwargs):

        loss = []
        pos_terms, neg_terms = [], []
        for i in range(sims.shape[0]): 

            pos_idxs = one_hot_labels[i,:] == 1
            if self.only_hard_distractors:
                curr_mask = mask[i,:]
                neg_idxs = ((one_hot_labels[i,:] == 0) * curr_mask)
            else:
                neg_idxs = (one_hot_labels[i,:] == 0)

            if not torch.sum(pos_idxs) or not torch.sum(neg_idxs):
                print('No positive or negative examples available')
                continue

            anchor_pos_sim = sims[i][pos_idxs]
            anchor_neg_sim = sims[i][neg_idxs]

            neg_idxs = (anchor_neg_sim + self.margin) > torch.min(anchor_pos_sim)
            pos_idxs = (anchor_pos_sim - self.margin) < torch.max(anchor_neg_sim)
            if not torch.sum(neg_idxs):
                print('No negative examples available - check 2') 
            elif not torch.sum(pos_idxs):
                print('No positive examples available - check 2')
            else:
                anchor_neg_sim = anchor_neg_sim[neg_idxs]
                anchor_pos_sim = anchor_pos_sim[pos_idxs]

            pos_term = 1./self.pos_weight * torch.log(1+torch.sum(torch.exp(-self.pos_weight * (anchor_pos_sim - self.thresh))))
            neg_term = 1./self.neg_weight * torch.log(1+torch.sum(torch.exp(self.neg_weight * (anchor_neg_sim - self.thresh))))

            
            loss.append(pos_term + neg_term)
            pos_terms.append(pos_term)
            neg_terms.append(neg_term)

        if loss == []:
            loss = torch.Tensor([0]).to(sims.device)
            pos_terms = torch.Tensor([0]).to(sims.device)
            neg_terms = torch.Tensor([0]).to(sims.device)
            loss.requires_grad = True
        else:
            loss = torch.mean(torch.stack(loss))
            pos_terms = torch.mean(torch.stack(pos_terms))
            neg_terms = torch.mean(torch.stack(neg_terms))
                
        return loss


def construct_batch_labels(candidate_embeddings, candidate_node_idx, correct_node_idx, mask):
    '''
        Format the batch to input into metric learning loss function
    '''
    batch, n_candidates, embed_dim = candidate_embeddings.shape

    # get mask
    mask_reshaped = mask.reshape(batch*n_candidates, -1)
    expanded_mask = mask_reshaped.expand(-1,embed_dim) 

    # flatten the gene node idx and gene embeddings
    candidate_node_idx_flattened = candidate_node_idx.view(batch*n_candidates, -1)
    candidate_embeddings_flattened = candidate_embeddings.view(batch*n_candidates, -1) 
    candidate_embeddings_flattened = candidate_embeddings_flattened * expanded_mask

    # NOTE: assumes there are already unique values
    candidate_node_idx_flattened_unique = candidate_node_idx_flattened[candidate_node_idx_flattened.squeeze() != 0]
    candidate_embeddings_flattened_unique = candidate_embeddings_flattened[candidate_node_idx_flattened.squeeze() != 0,:]

    # create a one hot encoding of correct gene/disease in the list of all in the batch
    label_idx = torch.where(candidate_node_idx_flattened_unique.unsqueeze(1) == correct_node_idx.unsqueeze(0), 1, 0)
    label_idx = label_idx.sum(dim=-1).T
    
    return candidate_node_idx_flattened_unique, candidate_embeddings_flattened_unique, label_idx 


class NCALoss(BaseMetricLossFunction):
    def __init__(self, softmax_scale=1, only_hard_distractors=False, **kwargs):
        super().__init__(**kwargs)
        self.softmax_scale = softmax_scale
        self.only_hard_distractors = only_hard_distractors
        self.add_to_recordable_attributes(
            list_of_names=["softmax_scale"], is_stat=False
        )

    def forward(self, phenotype_embedding, disease_embedding, batch_disease_nid, batch_cand_disease_nid=None, disease_mask=None, one_hot_labels=None, indices_tuple=None, use_candidate_list=False):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        loss_dict, disease_softmax, one_hot_labels, candidate_disease_idx, candidate_disease_embeddings = self.compute_loss(phenotype_embedding, disease_embedding, batch_disease_nid, batch_cand_disease_nid, disease_mask, one_hot_labels, indices_tuple, use_candidate_list)
        self.add_embedding_regularization_to_loss_dict(loss_dict, phenotype_embedding)
        if loss_dict is None: reduction = None
        else: reduction = self.reducer(loss_dict, None, None)
        return reduction, disease_softmax, one_hot_labels, candidate_disease_idx, candidate_disease_embeddings

    # https://www.cs.toronto.edu/~hinton/absps/nca.pdf
    def compute_loss(self, phenotype_embedding, disease_embedding, batch_corr_disease_nid, batch_cand_disease_nid, disease_mask, labels, indices_tuple, use_candidate_list):

        if len(phenotype_embedding) <= 1:
            return self.zero_losses(), None, None

        if disease_embedding is None: #phenotype-phenotypes
            loss_dict, disease_softmax, labels = self.nca_computation(
                phenotype_embedding, phenotype_embedding, labels, indices_tuple, use_one_hot_labels=False
            )
            candidate_disease_idx = None
            candidate_disease_embeddings = None

        else:
            # disease-phenotypes
            if self.only_hard_distractors or use_candidate_list:
                candidate_disease_embeddings = disease_embedding
                phenotype_embedding = phenotype_embedding.unsqueeze(1)
            else:
                candidate_disease_idx, candidate_disease_embeddings, labels = construct_batch_labels(disease_embedding, batch_cand_disease_nid, batch_corr_disease_nid, disease_mask)

            loss_dict, disease_softmax, labels = self.nca_computation(
                phenotype_embedding, candidate_disease_embeddings, labels, indices_tuple, use_one_hot_labels=True
            )

        return loss_dict, disease_softmax, labels, candidate_disease_idx, candidate_disease_embeddings

    def nca_computation(
        self, query, reference, labels, indices_tuple, use_one_hot_labels
    ):
        dtype = query.dtype
        mat = self.distance(query, reference)
        if not self.distance.is_inverted:
            mat = -mat
        mat = mat.squeeze(1)

        if query is reference:
            mat.fill_diagonal_(c_f.neg_inf(dtype))
        softmax = torch.nn.functional.softmax(self.softmax_scale * mat, dim=1)

        if labels.nelement() == 0:
            loss_dict = None
        else:
            if not use_one_hot_labels:
                labels = c_f.to_dtype(
                    labels.unsqueeze(1) == labels.unsqueeze(0), dtype=dtype
                )
                labels = labels.squeeze(-1)
            exp = torch.sum(softmax * labels, dim=1) 
            non_zero = exp != 0
            loss = -torch.log(exp[non_zero])
            indices =  c_f.torch_arange_from_size(query)[non_zero]
            loss_dict = {
                "loss": {
                    "losses": loss,
                    "indices": indices,
                    "reduction_type": "element",
                }
            }
        return loss_dict, softmax, labels

    def get_default_distance(self):
        return LpDistance(power=2)


