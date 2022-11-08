
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import wandb

from torch import nn
import torch
import torch.nn.functional as F

import numpy as np
from scipy.stats import rankdata

from allennlp.modules.attention import CosineAttention, BilinearAttention, AdditiveAttention, DotProductAttention


from utils.loss_utils import NCALoss
from utils.train_utils import mean_reciprocal_rank, top_k_acc, masked_mean, masked_softmax, weighted_sum


class PatientNCA(pl.LightningModule):

    def __init__(self, hparams, embed_dim):
        super().__init__()
        self.hyperparameters = hparams

        # attention for collapsing set of phenotype embeddings
        self.attn_vector = nn.Parameter(torch.zeros((1, embed_dim), dtype=torch.float), requires_grad=True)   
        nn.init.xavier_uniform_(self.attn_vector)
        
        if self.hyperparameters['attention_type'] == 'bilinear':
            self.attention = BilinearAttention(embed_dim, embed_dim)
        elif self.hyperparameters['attention_type'] == 'additive':
            self.attention = AdditiveAttention(embed_dim, embed_dim)
        elif self.hyperparameters['attention_type'] == 'dotpdt':
            self.attention = DotProductAttention()

        # projection layers
        self.phen_project = nn.Linear(embed_dim, embed_dim)
        self.phen_project2 = nn.Linear(embed_dim, embed_dim)
        if self.hyperparameters['loss'] == 'patient_disease_NCA':
            self.disease_project = nn.Linear(embed_dim, embed_dim)
            self.disease_project2 = nn.Linear(embed_dim, embed_dim)

        self.leaky_relu = nn.LeakyReLU(hparams['leaky_relu'])

        self.loss = NCALoss(softmax_scale=self.hyperparameters['softmax_scale'], only_hard_distractors=self.hyperparameters['only_hard_distractors']) 


    def forward(self, phenotype_embeddings, disease_embeddings, phenotype_mask=None, disease_mask=None): 
        assert phenotype_mask != None  
        if self.hyperparameters['loss'] == 'patient_disease_NCA':  assert disease_mask != None

        # attention weighted average of phenotype embeddings
        batched_attn = self.attn_vector.repeat(phenotype_embeddings.shape[0],1)
        attn_weights = self.attention(batched_attn, phenotype_embeddings, phenotype_mask)
        phenotype_embedding = weighted_sum(phenotype_embeddings, attn_weights)
        
        # project embeddings
        phenotype_embedding = self.phen_project2(self.leaky_relu(self.phen_project(phenotype_embedding)))
        if self.hyperparameters['loss'] == 'patient_disease_NCA': disease_embeddings = self.disease_project2(self.leaky_relu(self.disease_project(disease_embeddings)))
        else: disease_embeddings = None

        return phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights

    def calc_loss(self, batch, phenotype_embedding, disease_embeddings, disease_mask, labels, use_candidate_list):
        if self.hyperparameters['loss'] == 'patient_disease_NCA':
            loss, softmax, labels, candidate_disease_idx, candidate_disease_embeddings = self.loss(phenotype_embedding, disease_embeddings, batch.batch_disease_nid, batch.batch_cand_disease_nid, disease_mask, labels, use_candidate_list=use_candidate_list)
        elif self.hyperparameters['loss'] == 'patient_patient_NCA':
            loss, softmax, labels, candidate_disease_idx, candidate_disease_embeddings = self.loss(phenotype_embedding, None, None, None, None, labels, use_candidate_list=False)
        else:
            raise NotImplementedError
        return loss, softmax, labels, candidate_disease_idx, candidate_disease_embeddings


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hyperparameters['lr'])
        return optimizer