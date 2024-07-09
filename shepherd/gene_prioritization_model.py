#pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# torch
from torch import nn
import torch
import torch.nn.functional as F
import numpy as np
from scipy.stats import rankdata
import pandas as pd
from pathlib import Path
import time
import wandb
import sys

sys.path.insert(0, '..') # add project_config to path

from node_embedder_model import NodeEmbeder
from task_heads.gp_aligner import GPAligner

import project_config

# import utils
from utils.pretrain_utils import get_edges, calc_metrics
from utils.loss_utils import MultisimilarityCriterion
from utils.train_utils import mean_reciprocal_rank, top_k_acc, average_rank
from utils.train_utils import fit_umap, mrr_vs_percent_overlap, plot_gene_rank_vs_x_intrain, plot_gene_rank_vs_hops, plot_degree_vs_attention, plot_nhops_to_gene_vs_attention, plot_gene_rank_vs_fraction_phenotype, plot_gene_rank_vs_numtrain, plot_gene_rank_vs_trainset
from utils.train_utils import weighted_sum

class CombinedGPAligner(pl.LightningModule):

    def __init__(self, edge_attr_dict, all_data, n_nodes=None, node_ckpt=None, hparams=None, node_hparams=None,  spl_pca=[], spl_gate=[]):
        super().__init__()
        print('Initializing Model')

        self.save_hyperparameters('hparams', ignore=["spl_pca", "spl_gate"]) # spl_pca and spl_gate never get used

        #print('Saved combined model hyperparameters: ', self.hparams)

        self.all_data = all_data

        self.all_train_nodes = {}
        self.train_patient_nodes = {}
        self.train_sparse_nodes = {}
        self.train_target_batch = {}
        self.train_corr_gene_nid = {}

        print(f"Loading Node Embedder from {node_ckpt}")

        # NOTE: loads in saved hyperparameters
        self.node_model = NodeEmbeder.load_from_checkpoint(checkpoint_path=node_ckpt, 
                                                           all_data=all_data,
                                                           edge_attr_dict=edge_attr_dict, 
                                                           num_nodes=n_nodes)
        
        self.patient_model = self.get_patient_model()
        print('End Patient Model Initialization')
        

    def get_patient_model(self):
        # NOTE: this will only work with GATv2Conv
        model = GPAligner(self.hparams.hparams, embed_dim=self.node_model.hparams.hp_dict['output']*self.node_model.hparams.hp_dict['n_heads'])
        return model


    def forward(self, batch, step_type):
        # Node Embedder
        t0 = time.time()
        print(len(batch.adjs))
        outputs, gat_attn = self.node_model.forward(batch.n_id, batch.adjs)
        pad_outputs = torch.cat([torch.zeros(1, outputs.size(1), device=outputs.device), outputs])
        t1 = time.time()

        # get masks
        phenotype_mask = (batch.batch_pheno_nid != 0)
        gene_mask = (batch.batch_cand_gene_nid != 0)

        # index into outputs using phenotype & gene batch node idx
        batch_sz, max_n_phen = batch.batch_pheno_nid.shape
        phenotype_embeddings = torch.index_select(pad_outputs, 0, batch.batch_pheno_nid.view(-1)).view(batch_sz, max_n_phen, -1)
        batch_sz, max_n_cand_genes = batch.batch_cand_gene_nid.shape
        cand_gene_embeddings = torch.index_select(pad_outputs, 0, batch.batch_cand_gene_nid.view(-1)).view(batch_sz, max_n_cand_genes, -1)

        if self.hparams.hparams['augment_genes']:            
            print("Augmenting genes...", self.hparams.hparams['aug_gene_w'])
            _, max_n_sim_cand_genes, k_sim_genes = batch.batch_sim_gene_nid.shape
            sim_gene_embeddings = torch.index_select(pad_outputs, 0, batch.batch_sim_gene_nid.view(-1)).view(batch_sz, max_n_sim_cand_genes, self.hparams.hparams['n_sim_genes'], -1)
            agg_sim_gene_embedding = weighted_sum(sim_gene_embeddings, batch.batch_sim_gene_sims)
            if self.hparams.hparams['aug_gene_by_deg']:
                print("Augmenting gene by degree...")
                aug_gene_w = self.hparams.hparams['aug_gene_w'] * torch.exp(-self.hparams.hparams['aug_gene_w'] * batch.batch_cand_gene_degs) + (1 - self.hparams.hparams['aug_gene_w'] - 0.1)
                aug_gene_w = (aug_gene_w * (torch.sum(batch.batch_sim_gene_sims, dim = -1) > 0)).unsqueeze(-1)
            else:
                aug_gene_w = (self.hparams.hparams['aug_gene_w'] * (torch.sum(batch.batch_sim_gene_sims, dim = -1) > 0)).unsqueeze(-1)
            cand_gene_embeddings = torch.mul(1 - aug_gene_w, cand_gene_embeddings) + torch.mul(aug_gene_w, agg_sim_gene_embedding)

        # Patient Embedder with or without disease information
        if self.hparams.hparams['use_diseases']: 
            disease_mask = (batch.batch_disease_nid != 0)
            batch_sz, max_n_dx = batch.batch_disease_nid.shape
            disease_embeddings = torch.index_select(pad_outputs, 0, batch.batch_disease_nid.view(-1)).view(batch_sz, max_n_dx, -1)
            t2 = time.time()
            phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights = self.patient_model.forward(phenotype_embeddings, cand_gene_embeddings, disease_embeddings, phenotype_mask, gene_mask, disease_mask)
            t3 = time.time()
        else:
            t2 = time.time()
            phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights = self.patient_model.forward(phenotype_embeddings, cand_gene_embeddings, phenotype_mask=phenotype_mask, gene_mask=gene_mask)
            t3 = time.time()

        if self.hparams.hparams['time']:
            print(f'It takes {t1-t0:0.4f}s for the node model, {t2-t1:0.4f}s for indexing into the output, and {t3-t2:0.4f}s for the patient model forward.')
        
        return outputs, gat_attn, phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights

    def _step(self, batch, step_type):
        t0 = time.time()
        if step_type != 'test':
            batch = get_edges(batch, self.all_data, step_type)
        t1 = time.time()

        # Forward pass
        node_embeddings, gat_attn, phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights = self.forward(batch, step_type)
        t2 = time.time()

        # Calculate similarities between patient phenotypes & candidate genes/diseases
        alpha = self.hparams.hparams['alpha']
        use_candidate_list = True if step_type != 'train' else False
        cand_gene_to_phenotypes_spl = batch.batch_cand_gene_to_phenotypes_spl if use_candidate_list else batch.batch_concat_cand_gene_to_phenotypes_spl
        disease_nid = batch.batch_disease_nid if self.hparams.hparams['use_diseases'] else None

        # calculate similarity between phen & genes for all genes in manual candidate list
        phen_gene_sims, raw_phen_gene_sims, phen_gene_mask, phen_gene_one_hot_labels = self.patient_model._calc_similarity(phenotype_embedding, candidate_gene_embeddings, None, batch.batch_cand_gene_nid, batch.batch_corr_gene_nid, disease_nid, batch.one_hot_labels, gene_mask, phenotype_mask, disease_mask, True, batch.batch_cand_gene_to_phenotypes_spl, alpha) 

        # calculate similarity for loss function  
        if self.hparams.hparams['loss'] == 'gene_multisimilarity' and use_candidate_list: # in this case, the similarities are the same
            sims = phen_gene_sims
            mask = phen_gene_mask
            one_hot_labels = phen_gene_one_hot_labels
        else:
            if self.hparams.hparams['loss'] == 'disease_multisimilarity': candidate_gene_embeddings = None
            elif self.hparams.hparams['loss'] == 'gene_multisimilarity': disease_embeddings = None
            sims, raw_sims, mask, one_hot_labels = self.patient_model._calc_similarity(phenotype_embedding, candidate_gene_embeddings, disease_embeddings, batch.batch_cand_gene_nid, batch.batch_corr_gene_nid, disease_nid, batch.one_hot_labels, gene_mask, phenotype_mask, disease_mask, use_candidate_list, cand_gene_to_phenotypes_spl, alpha)


        ## Rank genes
        correct_gene_ranks, phen_gene_sims = self.patient_model._rank_genes(phen_gene_sims, phen_gene_mask, phen_gene_one_hot_labels)
        t3 = time.time()

        ## Calculate patient embedding loss
        loss = self.patient_model.calc_loss(sims, mask, one_hot_labels)
        t4 = time.time()

        ## Calculate node embedding loss
        if step_type == 'test':
            node_embedder_loss = 0
            roc_score, ap_score, acc, f1 = 0,0,0,0
        else:
            # Get link predictions
            batch, raw_pred, pred = self.node_model.get_predictions(batch, node_embeddings)
            link_labels = self.node_model.get_link_labels(batch.all_edge_types)
            node_embedder_loss = self.node_model.calc_loss(pred, link_labels)

            # Calculate metrics
            metric_pred = torch.sigmoid(raw_pred)
            roc_score, ap_score, acc, f1 = calc_metrics(metric_pred.cpu().detach().numpy(), link_labels.cpu().detach().numpy())
        t5 = time.time()

        ## calc time
        if self.hparams.hparams['time']:
            print(f'It takes {t1-t0:0.4f}s to get edges, {t2-t1:0.4f}s for the forward pass,  {t3-t2:0.4f}s to rank genes, {t4-t3:0.4f}s to calc patient loss, and {t5-t4:0.4f}s to calc the node loss.')

        ## Plot gradients
        if self.hparams.hparams['plot_gradients']:
            for k, v in self.patient_model.state_dict().items():
                self.logger.experiment.log({f'gradients/{step_type}.gradients.%s' % k: wandb.Histogram(v.detach().cpu())})

        return node_embedder_loss, loss, correct_gene_ranks, roc_score, ap_score, acc, f1, node_embeddings, gat_attn, phenotype_embedding, candidate_gene_embeddings, attn_weights, phen_gene_sims, raw_phen_gene_sims, gene_mask, phenotype_mask

    def training_step(self, batch, batch_idx):
        print('training step')
        node_embedder_loss, patient_loss, correct_gene_ranks, roc_score, ap_score, acc, f1, node_embeddings, gat_attn, phenotype_embedding, candidate_gene_embeddings, attn_weights, phen_gene_sims, raw_phen_gene_sims, gene_mask, phenotype_mask = self._step(batch, 'train')

        loss = (self.hparams.hparams['lambda'] * node_embedder_loss) + ((1 - self.hparams.hparams['lambda']) *  patient_loss)
        self.log('train_loss/patient.train_overall_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_loss/patient.train_patient_loss', patient_loss, prog_bar=True, on_epoch=True)
        self.log('train_loss/patient.train_node_embedder_loss', node_embedder_loss, prog_bar=True, on_epoch=True)

        batch_sz, n_candidates, embed_dim = candidate_gene_embeddings.shape
        candidate_gene_embeddings_flattened = candidate_gene_embeddings.view(batch_sz*n_candidates, embed_dim)
        one_hot_labels_flattened = batch.one_hot_labels.view(batch_sz*n_candidates)

        return {'loss': loss, 
                'train/train_correct_gene_ranks': correct_gene_ranks, 
                "train/node.train_roc": roc_score, 
                "train/node.train_ap": ap_score, 
                "train/node.train_acc": acc, 
                "train/node.train_f1": f1, 
                'train/one_hot_labels': batch.one_hot_labels.detach().cpu(),
                'train/attention_weights': attn_weights.detach().cpu() if attn_weights != None else None,
                'train/phen_gene_sims': phen_gene_sims.detach().cpu(),
                'train/phenotype_names_degrees': batch.phenotype_names,
                }

    def validation_step(self, batch, batch_idx):
        node_embedder_loss, patient_loss, correct_gene_ranks, roc_score, ap_score, acc, f1, node_embeddings, gat_attn, phenotype_embedding, candidate_gene_embeddings, attn_weights, phen_gene_sims, raw_phen_gene_sims, gene_mask, phenotype_mask = self._step(batch, 'val')
        loss = (self.hparams.hparams['lambda'] * node_embedder_loss) + ((1 - self.hparams.hparams['lambda']) * patient_loss)
        self.log('val_loss/patient.val_overall_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_loss/patient.val_patient_loss', patient_loss, prog_bar=True)
        self.log('val_loss/patient.val_node_embedder_loss', node_embedder_loss, prog_bar=True)

        return {'loss/val_loss': loss, 
                'val/val_correct_gene_ranks': correct_gene_ranks, 
                "val/node.val_roc": roc_score, 
                "val/node.val_ap": ap_score, 
                "val/node.val_acc": acc, 
                "val/node.val_f1": f1, 
                'val/one_hot_labels': batch.one_hot_labels.detach().cpu(),
                'val/attention_weights': attn_weights.detach().cpu() if attn_weights != None else None,
                'val/phen_gene_sims': phen_gene_sims.detach().cpu(),
                'val/phenotype_names_degrees': batch.phenotype_names,
                }
    
    def write_results_to_file(self, batch_info, phen_gene_sims, gene_mask, phenotype_mask, attn_weights, correct_gene_ranks, gat_attn, node_embeddings, phenotype_embeddings, save=True, loop_type='predict'):
        # NOTE: only saves a single batch - to run at inference time, make sure batch size includes all patients

        
        # Save GAT attention weights
        #NOTE: assumes 3 layers to model
        attn_dfs = []
        layer = 0
        for edge_attn in gat_attn:
            edge_index, attn = edge_attn
            edge_index = edge_index.cpu()
            attn = attn.cpu()
            gat_attn_df = pd.DataFrame({'source': edge_index[0,:], 'target': edge_index[1,:]})
            for head in range(attn.shape[1]):
                gat_attn_df[f'attn_{head}'] =  attn[:,head]
            attn_dfs.append(gat_attn_df)
            layer += 1
        
        
        # Save scores
        all_sims, all_genes, all_patient_ids, all_labels = [], [], [], []
        for patient_id, sims, genes, g_mask in zip(batch_info["patient_ids"], phen_gene_sims, batch_info["cand_gene_names"], gene_mask):
            nonpadded_sims = sims[g_mask].tolist()
            all_sims.extend(nonpadded_sims)
            all_genes.extend(genes)
            all_patient_ids.extend([patient_id] * len(genes))
        results_df = pd.DataFrame({'patient_id': all_patient_ids, 'genes': all_genes, 'similarities': all_sims})

        # Save phenotype information
        if attn_weights is None:
            phen_df = None
        else:
            all_patient_ids, all_phens, all_attn_weights, all_degrees = [], [], [], []
            for patient_id, attn_w, phen_names, p_mask in zip(batch_info["patient_ids"], attn_weights, batch_info["phenotype_names"], phenotype_mask):
                p_names, degrees = zip(*phen_names)
                all_patient_ids.extend([patient_id] * len(phen_names))
                all_degrees.extend(degrees)
                all_phens.extend(p_names)
                all_attn_weights.extend(attn_w[p_mask].tolist())
            phen_df = pd.DataFrame({'patient_id': all_patient_ids, 'phenotypes': all_phens, 'degrees': all_degrees, 'attention':all_attn_weights})
            print(phen_df.head())

        return results_df, phen_df, attn_dfs, phenotype_embeddings.cpu(), None
  
    def test_step(self, batch, batch_idx):
        node_embedder_loss, patient_loss, correct_gene_ranks, roc_score, ap_score, acc, f1, node_embeddings, gat_attn, phenotype_embedding, candidate_gene_embeddings, attn_weights, phen_gene_sims, raw_phen_gene_sims, gene_mask, phenotype_mask = self._step(batch, 'test')
        
        return {'test/test_correct_gene_ranks': correct_gene_ranks, 
                'test/node.embed': node_embeddings.detach().cpu(), 
                'test/patient.phenotype_embed': phenotype_embedding.detach().cpu(),
                'test/one_hot_labels': batch.one_hot_labels.detach().cpu(), #one_hot_labels_flattened.detach().cpu(),
                'test/attention_weights': attn_weights.detach().cpu() if attn_weights != None else None,
                'test/phen_gene_sims': phen_gene_sims.detach().cpu(),
                'test/phenotype_names_degrees': batch.phenotype_names, # type = list
                'test/gene_mask': gene_mask.detach().cpu(),
                'test/phenotype_mask': phenotype_mask.detach().cpu(),
                "test/patient_ids": batch.patient_ids, # type = list
                "test/cand_gene_names": batch.cand_gene_names, # type = list

                'test/gat_attn': gat_attn, # type = list
                "test/n_id": batch.n_id[:batch.batch_size].detach().cpu(),
                }


    def inference(self, batch, batch_idx):
        outputs, gat_attn = self.node_model.predict(self.all_data)
        pad_outputs = torch.cat([torch.zeros(1, outputs.size(1), device=outputs.device), outputs])
        t1 = time.time()

        # get masks
        phenotype_mask = (batch.batch_pheno_nid != 0)
        gene_mask = (batch.batch_cand_gene_nid != 0)
                
        # index into outputs using phenotype & gene batch node idx
        batch_sz, max_n_phen = batch.batch_pheno_nid.shape
        phenotype_embeddings = torch.index_select(pad_outputs, 0, batch.batch_pheno_nid.view(-1)).view(batch_sz, max_n_phen, -1)
        batch_sz, max_n_cand_genes = batch.batch_cand_gene_nid.shape
        cand_gene_embeddings = torch.index_select(pad_outputs, 0, batch.batch_cand_gene_nid.view(-1)).view(batch_sz, max_n_cand_genes, -1)

        if self.hparams.hparams['augment_genes']:            
            print("Augmenting genes at inference...", self.hparams.hparams['aug_gene_w'])
            _, max_n_sim_cand_genes, k_sim_genes = batch.batch_sim_gene_nid.shape
            sim_gene_embeddings = torch.index_select(pad_outputs, 0, batch.batch_sim_gene_nid.view(-1)).view(batch_sz, max_n_sim_cand_genes, self.hparams.hparams['n_sim_genes'], -1)
            agg_sim_gene_embedding = weighted_sum(sim_gene_embeddings, batch.batch_sim_gene_sims)        
            aug_gene_w = (self.hparams.hparams['aug_gene_w'] * (torch.sum(batch.batch_sim_gene_sims, dim = -1) > 0)).unsqueeze(-1)
            cand_gene_embeddings = torch.mul(1 - aug_gene_w, cand_gene_embeddings) + torch.mul(aug_gene_w, agg_sim_gene_embedding)

        # Patient Embedder with or without disease information
        if self.hparams.hparams['use_diseases']: 
            disease_mask = (batch.batch_disease_nid != 0)
            batch_sz, max_n_dx = batch.batch_disease_nid.shape
            disease_embeddings = torch.index_select(pad_outputs, 0, batch.batch_disease_nid.view(-1)).view(batch_sz, max_n_dx, -1)
            t2 = time.time()
            phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights = self.patient_model.forward(phenotype_embeddings, cand_gene_embeddings, disease_embeddings, phenotype_mask, gene_mask, disease_mask)
            t3 = time.time()
        else:
            t2 = time.time()
            phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights = self.patient_model.forward(phenotype_embeddings, cand_gene_embeddings, phenotype_mask=phenotype_mask, gene_mask=gene_mask)
            t3 = time.time()

        if self.hparams.hparams['time']:
            print(f'It takes {t1-t0:0.4f}s for the node model, {t2-t1:0.4f}s for indexing into the output, and {t3-t2:0.4f}s for the patient model forward.')
        
        return outputs, gat_attn, phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights


    def predict_step(self, batch, batch_idx):
        node_embeddings, gat_attn, phenotype_embedding, candidate_gene_embeddings, disease_embeddings, gene_mask, phenotype_mask, disease_mask, attn_weights = self.inference(batch, batch_idx)
        
        # Calculate similarities between patient phenotypes & candidate genes/diseases
        alpha = self.hparams.hparams['alpha']
        use_candidate_list = True
        disease_nid = batch.batch_disease_nid if self.hparams.hparams['use_diseases'] else None

        # calculate similarity between phen & genes for all genes in manual candidate list
        phen_gene_sims, raw_phen_gene_sims, phen_gene_mask, phen_gene_one_hot_labels = self.patient_model._calc_similarity(phenotype_embedding, candidate_gene_embeddings, None, batch.batch_cand_gene_nid, batch.batch_corr_gene_nid, disease_nid, batch.one_hot_labels, gene_mask, phenotype_mask, disease_mask, True, batch.batch_cand_gene_to_phenotypes_spl, alpha) 

        # Rank genes
        correct_gene_ranks, phen_gene_sims = self.patient_model._rank_genes(phen_gene_sims, phen_gene_mask, phen_gene_one_hot_labels)

        results_df, phen_df, attn_dfs, phenotype_embeddings, disease_embeddings = self.write_results_to_file(batch, phen_gene_sims, gene_mask, phenotype_mask, attn_weights, correct_gene_ranks, gat_attn, node_embeddings, phenotype_embedding, save=True)
        return results_df, phen_df, *attn_dfs, phenotype_embeddings, disease_embeddings

    def _epoch_end(self, outputs, loop_type):

        correct_gene_ranks = torch.cat([x[f'{loop_type}/{loop_type}_correct_gene_ranks'] for x in outputs], dim=0)

        if loop_type == "test":
            
            batch_info = {"n_id": torch.cat([x[f'{loop_type}/n_id'] for x in outputs], dim=0),
                          "patient_ids": [pat for x in outputs for pat in x[f'{loop_type}/patient_ids']],
                          "phenotype_names": [pat for x in outputs for pat in x[f'{loop_type}/phenotype_names_degrees']],
                          "cand_gene_names": [pat for x in outputs for pat in x[f'{loop_type}/cand_gene_names']],
                          "one_hot_labels": [pat for x in outputs for pat in x[f'{loop_type}/one_hot_labels']],
                          }

            phen_gene_sims = [pat for x in outputs for pat in x[f'{loop_type}/phen_gene_sims']] 
            gene_mask = [pat for x in outputs for pat in x[f'{loop_type}/gene_mask']] 
            phenotype_mask = [pat for x in outputs for pat in x[f'{loop_type}/phenotype_mask']] 
            attn_weights = [pat for x in outputs for pat in x[f'{loop_type}/attention_weights']] 
            gat_attn = [pat for x in outputs for pat in x[f'{loop_type}/gat_attn']] 
            node_embeddings = torch.cat([x[f'{loop_type}/node.embed'] for x in outputs], dim=0)
            phenotype_embedding = torch.cat([x[f'{loop_type}/patient.phenotype_embed'] for x in outputs], dim=0)
            
            results_df, phen_df, attn_dfs, phenotype_embeddings, disease_embeddings = self.write_results_to_file(batch_info, phen_gene_sims, gene_mask, phenotype_mask, attn_weights, correct_gene_ranks, gat_attn, node_embeddings, phenotype_embedding, loop_type=loop_type)
            
            print("Writing results for test...")
            output_base = "/home/ml499/public_repos/SHEPHERD/shepherd/results/gp"
            results_df.to_csv(str(output_base) + '_scores.csv', index=False)
            print(results_df)

            phen_df.to_csv(str(output_base) + '_phenotype_attention.csv', sep = ',', index=False)
            print(phen_df)


        # Plot embeddings
        if loop_type != "train" and len(self.train_patient_nodes) > 0 and self.hparams.hparams['plot_intrain']:

            correct_gene_nid = torch.cat([x[f'{loop_type}/corr_gene_nid_orig'] for x in outputs], dim=0)
            assert correct_gene_ranks.shape[0] == correct_gene_nid.shape[0]

            # Rank of gene vs. number of train patients with causal gene
            gene_rank_corr_gene_fig, gene_rank_corr_gene_counts = plot_gene_rank_vs_numtrain(correct_gene_ranks, correct_gene_nid, self.train_corr_gene_nid)
            gene_rank_cand_gene_fig, gene_rank_cand_gene_counts = plot_gene_rank_vs_numtrain(correct_gene_ranks, correct_gene_nid, self.train_patient_nodes)
            gene_rank_sparse_fig, gene_rank_sparse_counts = plot_gene_rank_vs_numtrain(correct_gene_ranks, correct_gene_nid, self.train_sparse_nodes)
            gene_rank_target_fig, gene_rank_target_counts = plot_gene_rank_vs_numtrain(correct_gene_ranks, correct_gene_nid, self.train_target_batch)
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_num_train_corr_genes': gene_rank_corr_gene_fig})
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_num_train_cand_genes': gene_rank_cand_gene_fig})
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_num_train_sparse': gene_rank_sparse_fig})
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_num_train_target': gene_rank_target_fig})
            
            gene_nid_trainset = torch.stack([torch.tensor(gene_rank_corr_gene_counts),
                                             torch.tensor(gene_rank_cand_gene_counts),
                                             torch.tensor(gene_rank_sparse_counts),
                                             torch.tensor(gene_rank_target_counts)], dim=1)
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_trainset': plot_gene_rank_vs_trainset(correct_gene_ranks, correct_gene_nid, gene_nid_trainset)})
            
        if self.hparams.hparams['plot_PG_embed']:
            self.logger.experiment.log({f'{loop_type}/patient_embed': fit_umap(patient_emb, patient_label)})

        # plot % overlap with train patients
        if loop_type != 'train' and self.hparams.hparams['mrr_vs_percent_overlap']:
            max_percent_overlap_train = torch.cat([torch.tensor(x[f'val/max_percent_phen_overlap_train']) for x in outputs], dim=0)
            self.logger.experiment.log({f'{loop_type}/mrr_vs_percent_overlap': mrr_vs_percent_overlap(correct_gene_ranks.detach().cpu(), max_percent_overlap_train.detach().cpu())})
        
        if self.hparams.hparams['plot_frac_rank']:

            # Rank of gene vs. fraction of phenotypes to disease 
            frac_p_with_direct_edge_to_dx = [pat[0][0] for x in outputs for pat in x[f'{loop_type}/frac_p_with_direct_edge_to_dx']] # NOTE: Currently ony select first disease. 
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_frac_p_with_direct_edge_to_dx': plot_gene_rank_vs_fraction_phenotype(correct_gene_ranks.cpu(), frac_p_with_direct_edge_to_dx)})
            
            # Rank of gene vs. fraction of phenotypes to gene 
            frac_p_with_direct_edge_to_g = [pat[0][0] for x in outputs for pat in x[f'{loop_type}/frac_p_with_direct_edge_to_g']] # NOTE: Currently ony select first gene. 
            self.logger.experiment.log({f'{loop_type}/frac_p_with_direct_edge_to_g': plot_gene_rank_vs_fraction_phenotype(correct_gene_ranks.cpu(), frac_p_with_direct_edge_to_g)})
        
        if self.hparams.hparams['plot_nhops_rank']:

            # Rank of gene vs. hops from disease
            nhops_g_d = [pat[0] for x in outputs for pat in x[f'{loop_type}/n_hops_g_d']] # NOTE Currently ony select first disease.
            fig_mean, fig_min = plot_gene_rank_vs_hops(correct_gene_ranks.cpu(), nhops_g_d)
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_mean_n_hops_g_d': fig_mean})
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_min_n_hops_g_d': fig_min})

            # Rank of gene vs. mean/min hops from phenotypes
            nhops_g_p = [pat[0] for x in outputs for pat in x[f'{loop_type}/n_hops_g_p']] # NOTE Currently ony select first gene. 
            fig_mean, fig_min = plot_gene_rank_vs_hops(correct_gene_ranks.cpu(), nhops_g_p)
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_mean_n_hops_g_p': fig_mean})
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_min_n_hops_g_p': fig_min})

            # # Rank of gene vs. distance between phenotypes
            nhops_p_p = [torch.tensor(pat) for x in outputs for pat in x[f'{loop_type}/n_hops_p_p']] 
            fig_mean, fig_min = plot_gene_rank_vs_hops(correct_gene_ranks.cpu(), nhops_p_p)
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_mean_n_hops_p_p': fig_mean})
            self.logger.experiment.log({f'{loop_type}/gene_rank_vs_min_n_hops_p_p': fig_min})

        if self.hparams.hparams['plot_attn_nhops']:

            # plot phenotype attention vs n_hops to gene and degree
            attn_weights = [torch.split(x[f'{loop_type}/attention_weights'],1) for x in outputs]
            attn_weights = [w[w > 0] for batch_w in attn_weights for w in batch_w]
            phenotype_names = [pat for x in outputs for pat in x[f'{loop_type}/phenotype_names_degrees']]
            attn_weights_cpu_reshaped = torch.cat(attn_weights, dim=0)
            self.logger.experiment.log({f"{loop_type}_attn/attention weights": wandb.Histogram(attn_weights_cpu_reshaped[attn_weights_cpu_reshaped != 0])})
            self.logger.experiment.log({f"{loop_type}_attn/single patient attention weights": wandb.Histogram(attn_weights[0])})
            self.logger.experiment.log({f"{loop_type}_attn/n hops to gene vs attention weights" : plot_nhops_to_gene_vs_attention(attn_weights, phenotype_names, nhops_g_p)})
            self.logger.experiment.log({f"{loop_type}_attn/single patient n hops to gene vs attention weights" : plot_nhops_to_gene_vs_attention(attn_weights, phenotype_names, nhops_g_p, single_patient=True)})
            self.logger.experiment.log({f"{loop_type}_attn/degree vs attention weights" : plot_degree_vs_attention(attn_weights, phenotype_names)})
            self.logger.experiment.log({f"{loop_type}_attn/single patient degree vs attention weights" : plot_degree_vs_attention(attn_weights, phenotype_names, single_patient=True)})
            data = [[p_name[0], w.item(), p_name[1], n_hops_to_g] for w, p_name, n_hops_to_g in zip(attn_weights[0], phenotype_names[0], nhops_g_p[0])]
            self.logger.experiment.log({f"{loop_type}_attn/phenotypes": wandb.Table(data=data, columns=["HPO Code", "Attention Weight", "Degree", "Num Hops to Gene" ])}) 
        
        if self.hparams.hparams['plot_phen_gene_sims']:

            all_phen_gene_sims, all_raw_phen_gene_sims, all_pg_spl, all_correct_sims, all_incorrect_sims = [], [], [], [], []
            for x in outputs:
                phen_gene_sims = x[f'{loop_type}/phen_gene_sims']
                one_hot_labels = x[f'{loop_type}/one_hot_labels']
                correct_phen_squeuegene_sims = all_correct_sims.append(phen_gene_sims[one_hot_labels == 1])
                incorrect_phen_gene_sims = all_incorrect_sims.append(phen_gene_sims[one_hot_labels != 1])
                phen_gene_sims_reshaped = all_phen_gene_sims.append(phen_gene_sims.view(-1))
                
            phen_gene_sims_reshaped = torch.cat(all_phen_gene_sims)
            correct_phen_gene_sims = torch.cat(all_correct_sims)
            incorrect_phen_gene_sims = torch.cat(all_incorrect_sims)

            if len(all_pg_spl) > 0: pg_spl_reshaped = torch.cat(all_pg_spl)
            else: pg_spl_reshaped = []

            self.logger.experiment.log({f"{loop_type}_pg_similarities/phenotype-gene similarities": wandb.Histogram(phen_gene_sims_reshaped[phen_gene_sims_reshaped != -100000])})
            self.logger.experiment.log({f"{loop_type}_pg_similarities/phenotype-correct gene similarities": wandb.Histogram(correct_phen_gene_sims[correct_phen_gene_sims != -100000])})
            self.logger.experiment.log({f"{loop_type}_pg_similarities/phenotype-incorrect gene similarities": wandb.Histogram(incorrect_phen_gene_sims[incorrect_phen_gene_sims != -100000])})

            if len(pg_spl_reshaped) > 0: self.logger.experiment.log({f"{loop_type}_pg_similarities/pg spl": wandb.Histogram(pg_spl_reshaped[pg_spl_reshaped != 0])})

            phen_gene_sims_patient = outputs[0][f'{loop_type}/phen_gene_sims'][0,:]
            one_hot_labels_patient = outputs[0][f'{loop_type}/one_hot_labels'][0,:]
            correct_phen_gene_sims_patient = phen_gene_sims_patient[one_hot_labels_patient == 1]

            assert len(correct_phen_gene_sims_patient) == 1
            incorrect_phen_gene_sims_patient = phen_gene_sims_patient[one_hot_labels_patient != 1]

            self.logger.experiment.log({f"{loop_type}_pg_similarities/single patient phenotype-gene similarities": wandb.Histogram(phen_gene_sims_patient[phen_gene_sims_patient != -100000])})
            self.logger.experiment.log({f"{loop_type}_pg_similarities/single patient phenotype-correct gene similarities": wandb.Histogram(correct_phen_gene_sims_patient[correct_phen_gene_sims_patient != -100000])})
            self.logger.experiment.log({f"{loop_type}_pg_similarities/single patient phenotype-incorrect gene similarities": wandb.Histogram(incorrect_phen_gene_sims_patient[incorrect_phen_gene_sims_patient != -100000])})

            if len(pg_spl_reshaped) > 0: self.logger.experiment.log({f"{loop_type}_pg_similarities/single patient pg spl": wandb.Histogram(pg_spl_reshaped[pg_spl_reshaped != 0])})


        # top k accuracy
        top_1_acc = top_k_acc(correct_gene_ranks, k=1)
        top_3_acc = top_k_acc(correct_gene_ranks, k=3)
        top_5_acc = top_k_acc(correct_gene_ranks, k=5)
        top_10_acc = top_k_acc(correct_gene_ranks, k=10)

        #mean reciprocal rank
        mrr = mean_reciprocal_rank(correct_gene_ranks)
        avg_rank = average_rank(correct_gene_ranks)

        self.log(f'{loop_type}/gp_{loop_type}_epoch_top1_acc', top_1_acc, prog_bar=False)
        self.log(f'{loop_type}/gp_{loop_type}_epoch_top3_acc', top_3_acc, prog_bar=False)
        self.log(f'{loop_type}/gp_{loop_type}_epoch_top5_acc', top_5_acc, prog_bar=False)
        self.log(f'{loop_type}/gp_{loop_type}_epoch_top10_acc', top_10_acc, prog_bar=False)
        self.log(f'{loop_type}/gp_{loop_type}_epoch_mrr', mrr, prog_bar=False)
        self.log(f'{loop_type}/gp_{loop_type}_epoch_avg_rank', avg_rank, prog_bar=False)

        if loop_type == 'val':
            self.log(f'curr_epoch', self.current_epoch, prog_bar=False)

    def training_epoch_end(self, outputs):

        if self.hparams.hparams['plot_intrain']:
            all_train_nodes, counts = torch.unique(torch.cat([x['train/n_id'] for x in outputs], dim=0), return_counts=True)
            curr_all_train_nodes = {n.item(): c.item() if n not in self.all_train_nodes else c.item() + self.all_train_nodes[n].item() for n, c in zip(all_train_nodes, counts)}
            self.all_train_nodes.update(curr_all_train_nodes)

            train_sparse_nodes, counts = torch.unique(torch.cat([x['train/sparse_idx'] for x in outputs], dim=0), return_counts=True)
            curr_train_sparse_nodes = {n.item(): c.item() if n not in self.train_sparse_nodes else c.item() + self.train_sparse_nodes[n].item() for n, c in zip(train_sparse_nodes, counts)}
            self.train_sparse_nodes.update(curr_train_sparse_nodes)

            train_target_batch, counts = torch.unique(torch.cat([x['train/target_batch'] for x in outputs], dim=0), return_counts=True)
            curr_train_target_batch = {n.item(): c.item() if n not in self.train_target_batch else c.item() + self.train_target_batch[n].item() for n, c in zip(train_target_batch, counts)}
            self.train_target_batch.update(curr_train_target_batch)
            
            train_patient_nodes, counts = torch.unique(torch.cat([x['train/cand_gene_nid_orig'] for x in outputs], dim=0), return_counts=True)
            self.train_patient_nodes = {n.item(): c.item() for n, c in zip(train_patient_nodes, counts)}
            
            train_corr_gene_nids, counts = torch.unique(torch.cat([x['train/corr_gene_nid_orig'] for x in outputs], dim=0), return_counts=True)
            self.train_corr_gene_nid = {g.item(): c.item() for g, c in zip(train_corr_gene_nids, counts)}

        self._epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.hparams['lr'])
        return optimizer
