
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from torch import nn
import torch
import torch.nn.functional as F

import numpy as np
from scipy.stats import rankdata

from allennlp.modules.attention import CosineAttention, BilinearAttention, AdditiveAttention, DotProductAttention


from utils.loss_utils import MultisimilarityCriterion, _construct_labels, unique, _construct_disease_labels
from utils.train_utils import masked_mean, masked_softmax, weighted_sum, plot_degree_vs_attention, mean_reciprocal_rank, top_k_acc

class GPAligner(pl.LightningModule):

    def __init__(self, hparams, embed_dim):
        super().__init__()
        self.hyperparameters = hparams
        print('GPAligner embedding dimension: ', embed_dim)

        # attention for collapsing set of phenotype embeddings
        self.attn_vector = nn.Parameter(torch.zeros((1, embed_dim), dtype=torch.float), requires_grad=True)   
        nn.init.xavier_uniform_(self.attn_vector)
        
        if self.hyperparameters['attention_type'] == 'bilinear':
            self.attention = BilinearAttention(embed_dim, embed_dim)
        elif self.hyperparameters['attention_type'] == 'additive':
            self.attention = AdditiveAttention(embed_dim, embed_dim)
        elif self.hyperparameters['attention_type'] == 'dotpdt':
            self.attention = DotProductAttention()
        
        if self.hyperparameters['decoder_type'] == "dotpdt": 
            self.decoder = DotProductAttention(normalize=False)
        elif self.hyperparameters['decoder_type'] == "bilinear": 
            self.decoder = BilinearAttention(embed_dim, embed_dim, activation=torch.tanh, normalize=False)
        else:
            raise NotImplementedError

        # projection layers
        self.phen_project = nn.Linear(embed_dim, embed_dim) 
        self.gene_project = nn.Linear(embed_dim, embed_dim)
        self.phen_project2 = nn.Linear(embed_dim, embed_dim)
        self.gene_project2 = nn.Linear(embed_dim, embed_dim)

        # optional disease projection layer
        if self.hyperparameters['use_diseases']:
            self.disease_project = nn.Linear(embed_dim, embed_dim)
            self.disease_project2 = nn.Linear(embed_dim, embed_dim)

        self.leaky_relu = nn.LeakyReLU(hparams['leaky_relu'])

        self.loss = MultisimilarityCriterion(hparams['pos_weight'], hparams['neg_weight'], 
                                hparams['margin'], hparams['thresh'], 
                                embed_dim, hparams['only_hard_distractors']) 



    def forward(self, phenotype_embeddings, candidate_gene_embeddings, disease_embeddings=None, phenotype_mask=None, gene_mask=None, disease_mask=None): 
        assert phenotype_mask != None
        assert gene_mask != None
        if self.hyperparameters['use_diseases']: assert disease_mask != None

        # attention weighted average of phenotype embeddings
        batched_attn = self.attn_vector.repeat(phenotype_embeddings.shape[0],1)
        attn_weights = self.attention(batched_attn, phenotype_embeddings, phenotype_mask)
        phenotype_embedding = weighted_sum(phenotype_embeddings, attn_weights)

        # project embeddings
        phenotype_embedding = self.phen_project2(self.leaky_relu(self.phen_project(phenotype_embedding)))
        candidate_gene_embeddings = self.gene_project2(self.leaky_relu(self.gene_project(candidate_gene_embeddings)))
        

        if self.hyperparameters['use_diseases']: 
            disease_embeddings = self.disease_project2(self.leaky_relu(self.disease_project(disease_embeddings)))
        else:
            disease_embeddings = None
            disease_mask = None

        return phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights 


    def _calc_similarity(self, phenotype_embeddings, candidate_gene_embeddings, disease_embeddings, batch_cand_gene_nid,  batch_corr_gene_nid, batch_disease_nid, one_hot_labels, gene_mask, phenotype_mask, disease_mask, use_candidate_list, cand_gene_to_phenotypes_spl, alpha): 
        # Normalize Embeddings (within each individual patient)
        phenotype_embeddings = F.normalize(phenotype_embeddings, p=2, dim=1) 
        batch_sz = phenotype_embeddings.shape[0]
        if disease_embeddings != None: disease_embeddings = F.normalize(disease_embeddings.squeeze(), p=2, dim=1) 
        if candidate_gene_embeddings != None:
            batch_sz, n_cand_genes, embed_dim = candidate_gene_embeddings.shape
            candidate_gene_embeddings = F.normalize(candidate_gene_embeddings.view(batch_sz*n_cand_genes,-1), p=2, dim=1).view(batch_sz, n_cand_genes, embed_dim)

        # Only use each patient's candidate genes/diseases
        if self.hyperparameters['only_hard_distractors'] or use_candidate_list:
            if disease_embeddings == None: # only use genes
                mask = gene_mask
                one_hot_labels = one_hot_labels
                raw_sims = self.decoder(phenotype_embeddings, candidate_gene_embeddings)
                if cand_gene_to_phenotypes_spl != None:
                    sims = alpha * raw_sims + (1 - alpha) * cand_gene_to_phenotypes_spl
                else: sims = raw_sims
            
            elif candidate_gene_embeddings == None: # only use diseases
                raise NotImplementedError
            
            else:
                raise NotImplementedError
        
        # Otherwise, use entire batch as candidate genes/diseases
        else:
            if disease_embeddings == None: #only use genes
                candidate_gene_idx, candidate_gene_embeddings, one_hot_labels = _construct_labels(candidate_gene_embeddings, batch_cand_gene_nid, batch_corr_gene_nid, gene_mask)
                raw_sims = self.decoder(phenotype_embeddings, candidate_gene_embeddings.unsqueeze(0).repeat(batch_sz,1,1))
                if cand_gene_to_phenotypes_spl != None:
                    sims = alpha * raw_sims + (1 - alpha) * cand_gene_to_phenotypes_spl
                else: sims = raw_sims
                mask = None
                
            elif candidate_gene_embeddings == None: #only use diseases
                candidate_embeddings, one_hot_labels = _construct_disease_labels(disease_embeddings, batch_disease_nid)
                raw_sims = self.decoder(phenotype_embeddings, candidate_embeddings.unsqueeze(0).repeat(batch_sz,1,1))
                if batch_disease_nid.shape[1] > 1:
                    raw_sims = raw_sims[batch_disease_nid[:,0].squeeze() != 0] # remove rows where the patient doesn't have 
                    one_hot_labels = one_hot_labels[batch_disease_nid[:,0].squeeze() != 0]
                else:
                    raw_sims = raw_sims[batch_disease_nid.squeeze() != 0] # remove rows where the patient doesn't have 
                    one_hot_labels = one_hot_labels[batch_disease_nid.squeeze() != 0]
                sims = raw_sims
                mask = None

            else: # use genes + diseases
                raise NotImplementedError

        return sims, raw_sims, mask, one_hot_labels


    def _rank_genes(self, phen_gene_sims, gene_mask, one_hot_labels):
        phen_gene_sims = phen_gene_sims * gene_mask
        padded_phen_gene_sims = phen_gene_sims + (~gene_mask * -100000) # we want to rank the padded values last
        gene_ranks = torch.tensor(np.apply_along_axis(lambda row: rankdata(row * -1, method='average'), axis=1, arr=padded_phen_gene_sims.detach().cpu().numpy()))
        if one_hot_labels is None: correct_gene_ranks = None
        else: 
            gene_ranks = gene_ranks.to(one_hot_labels.device)
            correct_gene_ranks = gene_ranks[one_hot_labels == 1]
        return correct_gene_ranks, padded_phen_gene_sims

    def calc_loss(self, sims, mask, one_hot_labels):
        return self.loss(sims, mask, one_hot_labels)



    
