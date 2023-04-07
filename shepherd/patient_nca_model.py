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

import time
import wandb
import sys
import umap
from pathlib import Path

# Our code
from node_embedder_model import NodeEmbeder
from task_heads.gp_aligner import GPAligner
from shepherd.task_heads.patient_nca import PatientNCA
from utils.pretrain_utils import get_edges, calc_metrics
from utils.train_utils import mean_reciprocal_rank, top_k_acc, average_rank
from utils.train_utils import fit_umap, plot_softmax, mrr_vs_percent_overlap, plot_gene_rank_vs_x_intrain, plot_gene_rank_vs_hops, plot_degree_vs_attention, plot_nhops_to_gene_vs_attention, plot_gene_rank_vs_fraction_phenotype, plot_gene_rank_vs_numtrain, plot_gene_rank_vs_trainset

sys.path.insert(0, '..') # add project_config to path
import project_config

class CombinedPatientNCA(pl.LightningModule):

    def __init__(self, edge_attr_dict, all_data, n_nodes=None, node_ckpt=None, hparams=None):
        super().__init__()
        self.save_hyperparameters('hparams') 

        #print('Saved combined model hyperparameters: ', self.hparams)

        self.all_data = all_data

        self.all_train_nodes = []
        self.train_patient_nodes = []

        print(f"Loading Node Embedder from {node_ckpt}")
        
        # NOTE: loads in saved hyperparameters
        self.node_model = NodeEmbeder.load_from_checkpoint(checkpoint_path=node_ckpt,
                                                           all_data=all_data,
                                                           edge_attr_dict=edge_attr_dict, 
                                                           num_nodes=n_nodes)
    
        # NOTE: this will only work with GATv2Conv
        self.patient_model = PatientNCA(hparams, embed_dim=self.node_model.hparams.hp_dict['output']*self.node_model.hparams.hp_dict['n_heads'])


    def forward(self, batch):
        # Node Embedder
        t0 = time.time()
        outputs, gat_attn = self.node_model.forward(batch.n_id, batch.adjs)
        pad_outputs = torch.cat([torch.zeros(1, outputs.size(1), device=outputs.device), outputs]) 
        t1 = time.time()

        # get masks
        phenotype_mask = (batch.batch_pheno_nid != 0)
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': disease_mask = (batch.batch_cand_disease_nid != 0)
        else: disease_mask = None

        # index into outputs using phenotype & disease batch node idx
        batch_sz, max_n_phen = batch.batch_pheno_nid.shape
        phenotype_embeddings = torch.index_select(pad_outputs, 0, batch.batch_pheno_nid.view(-1)).view(batch_sz, max_n_phen, -1)
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': 
            batch_sz, max_n_dx = batch.batch_cand_disease_nid.shape
            disease_embeddings = torch.index_select(pad_outputs, 0, batch.batch_cand_disease_nid.view(-1)).view(batch_sz, max_n_dx, -1)
        else: disease_embeddings = None

        t2 = time.time()
        phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights = self.patient_model.forward(phenotype_embeddings, disease_embeddings, phenotype_mask, disease_mask)
        t3 = time.time()

        if self.hparams.hparams['time']:
            print(f'It takes {t1-t0:0.4f}s for the node model, {t2-t1:0.4f}s for indexing into the output, and {t3-t2:0.4f}s for the patient model forward.')
        
        return outputs, gat_attn, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights

    def rank_diseases(self, disease_softmax, disease_mask, labels):
        disease_mask = (disease_mask.sum(dim=1) > 0).unsqueeze(-1) # convert (batch, n_diseases) -> (batch, 1)
        disease_softmax =  disease_softmax + (~disease_mask * -100000) # we want to rank the padded values last
        disease_ranks = torch.tensor(np.apply_along_axis(lambda row: rankdata(row * -1, method='average'), axis=1, arr=disease_softmax.detach().cpu().numpy()))
        if labels is None:
            correct_disease_ranks = None
        else:
            disease_ranks = disease_ranks.to(labels.device)
            correct_disease_ranks = [ranks[lab == 1] for ranks, lab in zip(disease_ranks, labels)]

        return correct_disease_ranks

    def rank_patients(self, patient_softmax, labels):
        labels = labels * ~torch.eye(labels.shape[0], dtype=torch.bool).to(labels.device) # don't consider label positive for patients with themselves
        patient_ranks = torch.tensor(np.apply_along_axis(lambda row: rankdata(row * -1, method='average'), axis=1, arr=patient_softmax.detach().cpu().numpy()))
        if labels is None:
            correct_patient_ranks = None
        else:
            patient_ranks = patient_ranks.to(labels.device)
            correct_patient_ranks = [ranks[lab == 1] for ranks, lab in zip(patient_ranks, labels)]

        return correct_patient_ranks, labels
    
    def _step(self, batch, step_type):
        t0 = time.time()
        if step_type != 'test':
            batch = get_edges(batch, self.all_data, step_type)
        t1 = time.time()

        # forward pass
        node_embeddings, gat_attn, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights = self.forward(batch)


        # calculate patient embedding loss
        use_candidate_list = self.hparams.hparams['only_hard_distractors'] #True if step_type != 'train' else False
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': labels = batch.disease_one_hot_labels
        else: labels = batch.patient_labels
        loss, softmax, labels, candidate_disease_idx, candidate_disease_embeddings = self.patient_model.calc_loss(batch, phenotype_embedding, disease_embeddings, disease_mask, labels, use_candidate_list)
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': correct_ranks = self.rank_diseases(softmax, disease_mask, labels)
        else: correct_ranks, labels = self.rank_patients(softmax, labels)

        # calculate node embedding loss
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

        # Plot gradients
        if self.hparams.hparams['plot_gradients']:
            for k, v in self.patient_model.state_dict().items():
                self.logger.experiment.log({f'gradients/{step_type}.gradients.%s' % k: wandb.Histogram(v.detach().cpu())})

        return correct_ranks, softmax, labels, node_embedder_loss, loss, roc_score, ap_score, acc, f1, gat_attn, node_embeddings, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights, candidate_disease_idx, candidate_disease_embeddings

    def training_step(self, batch, batch_idx):
        correct_ranks, softmax, labels, node_embedder_loss, patient_loss, roc_score, ap_score, acc, f1, gat_attn, node_embeddings, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights, cand_disease_idx, cand_disease_embeddings = self._step(batch, 'train')

        loss = (self.hparams.hparams['lambda'] * node_embedder_loss) + ((1 - self.hparams.hparams['lambda']) *  patient_loss)
        self.log('train_loss/overall_loss', loss, prog_bar=True, on_epoch=True)
        self.log('train_loss/patient_loss', patient_loss, prog_bar=True, on_epoch=True)
        self.log('train_loss/node_embedder_loss', node_embedder_loss, prog_bar=True, on_epoch=True)

        batch_results = {'loss': loss, 
                "train/node.roc": roc_score, 
                "train/node.ap": ap_score, "train/node.acc": acc, "train/node.f1": f1, 
                'train/node.embed': node_embeddings.detach().cpu(), 
                'train/patient.phenotype_embed': phenotype_embedding.detach().cpu(), 
                'train/attention_weights': attn_weights.detach().cpu(),
                'train/phenotype_names_degrees': batch.phenotype_names,
                'train/correct_ranks': correct_ranks,
                'train/disease_names':  batch.disease_names,
                'train/corr_gene_names': batch.corr_gene_names,
                "train/softmax": softmax.detach().cpu(),         
                }

        if self.hparams.hparams['loss'] == 'patient_disease_NCA':
            batch_sz, n_diseases, embed_dim = disease_embeddings.shape
            batch_disease_nid_reshaped = batch.batch_disease_nid.view(-1)
            batch_results.update({
                'train/batch_disease_nid': batch_disease_nid_reshaped.detach().cpu(),
                'train/cand_disease_names': batch.cand_disease_names,
                'train/batch_cand_disease_nid': cand_disease_idx.detach().cpu(),
                'train/patient.disease_embed': cand_disease_embeddings.detach().cpu()
            })

        return batch_results

    def validation_step(self, batch, batch_idx):
        correct_ranks, softmax, labels, node_embedder_loss, patient_loss, roc_score, ap_score, acc, f1, gat_attn, node_embeddings, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights, cand_disease_idx, cand_disease_embeddings = self._step(batch, 'val')
        loss = (self.hparams.hparams['lambda'] * node_embedder_loss) + ((1 - self.hparams.hparams['lambda']) * patient_loss)
        self.log('val_loss/overall_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_loss/patient_loss', patient_loss, prog_bar=True)
        self.log('val_loss/node_embedder_loss', node_embedder_loss, prog_bar=True)

        batch_results = {"loss/val_loss": loss, 
                         "val/node.roc": roc_score, 
                         "val/node.ap": ap_score, "val/node.acc": acc, 
                         "val/node.f1": f1, 
                         'val/node.embed': node_embeddings.detach().cpu(), 
                         'val/patient.phenotype_embed': phenotype_embedding.detach().cpu(), 
                         'val/attention_weights': attn_weights.detach().cpu(),
                         'val/phenotype_names_degrees': batch.phenotype_names,
                         'val/correct_ranks': correct_ranks, 
                         'val/disease_names':  batch.disease_names,
                         'val/corr_gene_names': batch.corr_gene_names,
                         "val/softmax": softmax.detach().cpu(),
                        }

        if self.hparams.hparams['loss'] == 'patient_disease_NCA':
            batch_sz, n_diseases, embed_dim = disease_embeddings.shape
            batch_disease_nid_reshaped = batch.batch_disease_nid.view(-1)
            batch_results.update({'val/batch_disease_nid': batch_disease_nid_reshaped.detach().cpu(),
                                  'val/cand_disease_names': batch.cand_disease_names,
                                  'val/batch_cand_disease_nid': cand_disease_idx.detach().cpu(),
                                  'val/patient.disease_embed': cand_disease_embeddings.detach().cpu()
                                })
        return batch_results 

    def write_results_to_file(self, batch, softmax, correct_ranks, labels, phenotype_mask, disease_mask, attn_weights,  gat_attn, node_embeddings, phenotype_embeddings, disease_embeddings, save=True, loop_type='predict'):
        
        if save:
            if self.hparams.hparams['loss'] == 'patient_disease_NCA': task = 'disease_characterization'
            else: task = 'patients_like_me'
            run_folder = Path(project_config.PROJECT_DIR) / 'results' / task / self.hparams.hparams['run_name'] / (Path(self.predict_dataloader.dataloader.dataset.filepath).stem ) #.replce('/', '_')
            run_folder.mkdir(parents=True, exist_ok=True)
        
     
        # Save scores
        if self.hparams.hparams['loss'] == 'patient_disease_NCA':
            cand_disease_names = [d for d_list in batch['cand_disease_names'] for d in d_list]

            all_sims, all_diseases, all_patient_ids = [], [], []
            for patient_id, sims in zip(batch['patient_ids'], softmax):  #batch['cand_disease_names'], disease_mask, 
                sims = sims.tolist() 
                all_sims.extend(sims)
                all_diseases.extend(cand_disease_names)
                all_patient_ids.extend([patient_id] * len(sims))
            results_df = pd.DataFrame({'patient_id': all_patient_ids, 'diseases': all_diseases, 'similarities': all_sims})
        else:
            all_sims, all_cand_pats, all_patient_ids = [], [], []
            for patient_id, sims in zip(batch['patient_ids'], softmax):
                patient_mask = torch.Tensor([p_id != patient_id for p_id in batch['patient_ids']]).bool()
                remaining_pats = [p_id for p_id in batch['patient_ids'] if p_id != patient_id]
                all_sims.extend(sims[patient_mask].tolist())
                all_cand_pats.extend(remaining_pats)
                all_patient_ids.extend([patient_id] * len(remaining_pats))
            results_df = pd.DataFrame({'patient_id': all_patient_ids, 'candidate_patients': all_cand_pats, 'similarities': all_sims})
        print(results_df.head())
        if save:
            print('logging results to run dir: ', run_folder)
            results_df.to_csv(Path(run_folder)  /'scores.csv', sep = ',', index=False)

        # Save phenotype information
        if attn_weights is None:
            phen_df = None
        else:
            all_patient_ids, all_phens, all_attn_weights, all_degrees = [], [], [], []
            for patient_id, attn_w, phen_names, p_mask in zip(batch['patient_ids'], attn_weights, batch['phenotype_names'], phenotype_mask):
                p_names, degrees = zip(*phen_names)
                all_patient_ids.extend([patient_id] * len(phen_names))
                all_degrees.extend(degrees)
                all_phens.extend(p_names)
                all_attn_weights.extend(attn_w[p_mask].tolist())
            phen_df = pd.DataFrame({'patient_id': all_patient_ids, 'phenotypes': all_phens, 'degrees': all_degrees, 'attention':all_attn_weights})
            print(phen_df.head())
            if save:
                phen_df.to_csv(Path(run_folder) /'phenotype_attention.csv', sep = ',', index=False)
        
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
            print(f'gat_attn_df, layer={layer}', gat_attn_df.head())
            if save:
                gat_attn_df.to_csv(Path(run_folder)  / f'gat_attn_layer={layer}.csv', sep = ',', index=False) #wandb.run.dir
            layer += 1

        # Save embeddings
        if save:
            torch.save(batch["n_id"].cpu(), Path(run_folder) /'node_embeddings_idx.pth')
            torch.save(node_embeddings.cpu(), Path(run_folder) /'node_embeddings.pth')
            torch.save(phenotype_embeddings.cpu(), Path(run_folder) /'phenotype_embeddings.pth')
            if self.hparams.hparams['loss'] == 'patient_disease_NCA': torch.save(disease_embeddings.cpu(), Path(run_folder) /'disease_embeddings.pth')        
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': disease_embeddings = disease_embeddings.cpu()

        return results_df, phen_df, attn_dfs, phenotype_embeddings.cpu(), disease_embeddings

    def test_step(self, batch, batch_idx):
        correct_ranks, softmax, labels, node_embedder_loss, patient_loss, roc_score, ap_score, acc, f1, gat_attn, node_embeddings, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights, cand_disease_idx, cand_disease_embeddings = self._step(batch, 'test')
        batch_results = {'test/correct_ranks': correct_ranks, 
                         'test/node.embed': node_embeddings.detach().cpu(), 
                         'test/patient.phenotype_embed': phenotype_embedding.detach().cpu(), 
                         'test/attention_weights': attn_weights.detach().cpu(),
                         'test/phenotype_names_degrees': batch.phenotype_names,
                         'test/disease_names': batch.disease_names,
                         'test/corr_gene_names': batch.corr_gene_names,
                         'test/gat_attn': gat_attn, # type = list
                         "test/n_id": batch.n_id[:batch.batch_size].detach().cpu(),
                         "test/patient_ids": batch.patient_ids, # type = list
                         "test/softmax": softmax.detach().cpu(),
                         "test/labels": labels.detach().cpu(),
                         'test/phenotype_mask': phenotype_mask.detach().cpu(),
                         'test/disease_mask': phenotype_mask.detach().cpu(),
                        }

        if self.hparams.hparams['loss'] == 'patient_disease_NCA':
            batch_sz, n_diseases, embed_dim = disease_embeddings.shape
            batch_disease_nid_reshaped = batch.batch_disease_nid.view(-1)
            batch_results.update({
                'test/batch_disease_nid': batch_disease_nid_reshaped.detach().cpu(),
                'test/cand_disease_names': batch.cand_disease_names,
                'test/batch_cand_disease_nid': cand_disease_idx,
                'test/patient.disease_embed': cand_disease_embeddings
            })
        else:
            batch_results.update({
                'test/patient.disease_embed': None, 
                'test/batch_disease_nid': None,
                'test/cand_disease_names': None

            })
        
        return batch_results

    
    def inference(self, batch, batch_idx):
        outputs, gat_attn = self.node_model.predict(self.all_data)
        #outputs, gat_attn = self.node_model.forward(batch.n_id, batch.adjs)

        pad_outputs = torch.cat([torch.zeros(1, outputs.size(1), device=outputs.device), outputs]) 

        # get masks
        phenotype_mask = (batch.batch_pheno_nid != 0)
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': disease_mask = (batch.batch_cand_disease_nid != 0)
        else: disease_mask = None

        # index into outputs using phenotype & disease batch node idx
        batch_sz, max_n_phen = batch.batch_pheno_nid.shape
        phenotype_embeddings = torch.index_select(pad_outputs, 0, batch.batch_pheno_nid.view(-1)).view(batch_sz, max_n_phen, -1)
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': 
            batch_sz, max_n_dx = batch.batch_cand_disease_nid.shape
            disease_embeddings = torch.index_select(pad_outputs, 0, batch.batch_cand_disease_nid.view(-1)).view(batch_sz, max_n_dx, -1)
        else: disease_embeddings = None

        phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights = self.patient_model.forward(phenotype_embeddings, disease_embeddings, phenotype_mask, disease_mask)

        return outputs, gat_attn, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights


    def predict_step(self, batch, batch_idx):
        node_embeddings, gat_attn, phenotype_embedding, disease_embeddings, phenotype_mask, disease_mask, attn_weights = self.inference(batch, batch_idx)

        # calculate patient embedding loss
        use_candidate_list =  self.hparams.hparams['only_hard_distractors'] 
        if self.hparams.hparams['loss'] == 'patient_disease_NCA': labels = batch.disease_one_hot_labels
        else: labels = batch.patient_labels
        loss, softmax, labels, candidate_disease_idx, candidate_disease_embeddings = self.patient_model.calc_loss(batch, phenotype_embedding, disease_embeddings, disease_mask, labels, use_candidate_list)
        if labels.nelement() == 0:
            correct_ranks = None
        else:
            if self.hparams.hparams['loss'] == 'patient_disease_NCA': correct_ranks = self.rank_diseases(softmax, disease_mask, labels)
            else: correct_ranks, labels = self.rank_patients(softmax, labels)
        

        results_df, phen_df, attn_dfs, phenotype_embeddings, disease_embeddings = self.write_results_to_file(batch, softmax, correct_ranks, labels, phenotype_mask, disease_mask , attn_weights, gat_attn, node_embeddings, phenotype_embedding, disease_embeddings, save=True, loop_type='predict')
        return results_df, phen_df, *attn_dfs, phenotype_embeddings, disease_embeddings

    
    def _epoch_end(self, outputs, loop_type):
        correct_ranks = torch.cat([ranks  for x in outputs for ranks in x[f'{loop_type}/correct_ranks']], dim=0) #if len(ranks.shape) > 0 else ranks.unsqueeze(-1)
        correct_ranks_with_pad = [ranks if len(ranks.unsqueeze(-1)) > 0 else torch.tensor([-1]) for x in outputs for ranks in x[f'{loop_type}/correct_ranks']]

        if loop_type == "test":
            
            batch_info = {"n_id": torch.cat([x[f'{loop_type}/n_id'] for x in outputs], dim=0),
                          "patient_ids": [pat for x in outputs for pat in x[f'{loop_type}/patient_ids'] ],
                          "phenotype_names": [pat for x in outputs for pat in x[f'{loop_type}/phenotype_names_degrees']],
                          "cand_disease_names": [pat for x in outputs for pat in x[f'{loop_type}/cand_disease_names']] if outputs[0][f'{loop_type}/cand_disease_names'] is not None else None,
                          }

            softmax = [pat for x in outputs for pat in x[f'{loop_type}/softmax']] 
            labels = [pat for x in outputs for pat in x[f'{loop_type}/labels']] 
            phenotype_mask = [pat for x in outputs for pat in x[f'{loop_type}/phenotype_mask']] 
            disease_mask = [pat for x in outputs for pat in x[f'{loop_type}/disease_mask']] 
            attn_weights = [pat for x in outputs for pat in x[f'{loop_type}/attention_weights']] 
            gat_attn = [pat for x in outputs for pat in x[f'{loop_type}/gat_attn']] 
            node_embeddings = torch.cat([x[f'{loop_type}/node.embed'] for x in outputs], dim=0)
            phenotype_embedding = torch.cat([x[f'{loop_type}/patient.phenotype_embed'] for x in outputs], dim=0)
            disease_embeddings = torch.cat([x[f'{loop_type}/patient.disease_embed'] for x in outputs], dim=0) if outputs[0][f'{loop_type}/patient.disease_embed'] is not None else None
            if self.hparams.hparams['loss'] == 'patient_disease_NCA':
                cand_disease_batch_nid = torch.cat([x[f'{loop_type}/batch_cand_disease_nid'] for x in outputs], dim=0)
            else: cand_disease_batch_nid = None

            ranks_df, results_df, phen_df, attn_dfs, phenotype_embeddings, disease_embeddings = self.write_results_to_file(batch_info, softmax, correct_ranks_with_pad, labels, phenotype_mask, disease_mask, attn_weights, gat_attn, node_embeddings, phenotype_embedding, disease_embeddings, save=True, loop_type='test')

        if self.hparams.hparams['plot_patient_embed']:
            phenotype_embedding = torch.cat([x[f'{loop_type}/patient.phenotype_embed'] for x in outputs], dim=0)
            correct_gene_names = ['None' if len(li) == 0 else '  |  '.join(li) for x in outputs for li in x[f'{loop_type}/corr_gene_names'] ] 
            correct_disease_names = ['None' if len(li) == 0 else '  |  '.join(li) for x in outputs for li in x[f'{loop_type}/disease_names'] ] 

            phenotype_names = [' | '.join([item[0] for item in li][0:6])  for x in outputs for li in x[f'{loop_type}/phenotype_names_degrees'] ] #only take first few for now because they don't all fit
            patient_label = {
                        #"Id": patient_ids,
                        #"Patient Type": patient_type,
                        "Phenotypes": phenotype_names ,
                        "Node Type": correct_disease_names,
                        "Correct Gene": correct_gene_names,
                        "Correct Disease":  correct_disease_names
            }
            self.logger.experiment.log({f'{loop_type}/patient_embed': fit_umap(phenotype_embedding, patient_label)})
        
        if self.hparams.hparams['plot_disease_embed']:
            # Plot embeddings of patient aggregated phenotype & diseases              
            phenotype_embedding = torch.cat([x[f'{loop_type}/patient.phenotype_embed'] for x in outputs], dim=0)
            disease_embeddings = torch.cat([x[f'{loop_type}/patient.disease_embed'] for x in outputs], dim=0) 
            disease_batch_nid = torch.cat([x[f'{loop_type}/batch_disease_nid'] for x in outputs], dim=0)
            cand_disease_batch_nid = torch.cat([x[f'{loop_type}/batch_cand_disease_nid'] for x in outputs], dim=0)
            disease_mask = (disease_batch_nid != 0)
            cand_disease_mask = (cand_disease_batch_nid != 0)
    
            phenotype_names = [' | '.join([item[0] for item in li][0:6])  for x in outputs for li in x[f'{loop_type}/phenotype_names_degrees'] ] #only take first few for now because they don't all fit
            cand_disease_names = [item for x in outputs for li in x[f'{loop_type}/cand_disease_names'] for item in li] 
            correct_disease_names = ['None' if len(li) == 0 else '  |  '.join(li) for x in outputs for li in x[f'{loop_type}/disease_names'] ] 

            patient_emb = torch.cat([phenotype_embedding, disease_embeddings])

            patient_label = {
                        "Node Type": ["Patient Phenotype"] * phenotype_embedding.shape[0] + ['Disease'] * disease_embeddings.shape[0],
                        "Name": phenotype_names + cand_disease_names,
                        "Correct Disease": correct_disease_names + ['NA'] * disease_embeddings.shape[0]
                        }
            self.logger.experiment.log({f'{loop_type}/patient_embed': fit_umap(patient_emb, patient_label)})
     
        if 'plot_softmax' in self.hparams.hparams and self.hparams.hparams['plot_softmax']:
            softmax = [pat for x in outputs for pat in x[f'{loop_type}/softmax']] 
            softmax_diff = [s.max() - s.min() for s in softmax]
            softmax_top2_diff = [torch.topk(s, 2).values.max() - torch.topk(s, 2).values.min() for s in softmax]
            softmax_top5_diff = [torch.topk(s, 5).values.max() - torch.topk(s, 5).values.min() for s in softmax]
            self.logger.experiment.log({f'{loop_type}/softmax_top2_diff': plot_softmax(softmax_top2_diff)})
            self.logger.experiment.log({f'{loop_type}/softmax_top5_diff': plot_softmax(softmax_top5_diff)})
            self.logger.experiment.log({f'{loop_type}/softmax_diff': plot_softmax(softmax_diff)})

        if self.hparams.hparams['plot_attn_nhops']:
            # plot phenotype attention vs n_hops to gene and degree
            attn_weights = [torch.split(x[f'{loop_type}/attention_weights'],1) for x in outputs]
            attn_weights = [w[w > 0] for batch_w in attn_weights for w in batch_w]
            phenotype_names = [pat for x in outputs for pat in x[f'{loop_type}/phenotype_names_degrees']]
            attn_weights_cpu_reshaped = torch.cat(attn_weights, dim=0)
            self.logger.experiment.log({f"{loop_type}_attn/attention weights": wandb.Histogram(attn_weights_cpu_reshaped[attn_weights_cpu_reshaped != 0])})
            self.logger.experiment.log({f"{loop_type}_attn/single patient attention weights": wandb.Histogram(attn_weights[0])})
        
        if loop_type == 'val':
            self.log(f'patient.curr_epoch', self.current_epoch, prog_bar=False)

        # top k accuracy
        top_1_acc = top_k_acc(correct_ranks, k=1)
        top_3_acc = top_k_acc(correct_ranks, k=3)
        top_5_acc = top_k_acc(correct_ranks, k=5)
        top_10_acc = top_k_acc(correct_ranks, k=10)

        #mean reciprocal rank
        mrr = mean_reciprocal_rank(correct_ranks)

        self.log(f'{loop_type}/top1_acc', top_1_acc, prog_bar=False)
        self.log(f'{loop_type}/top3_acc', top_3_acc, prog_bar=False)
        self.log(f'{loop_type}/top5_acc', top_5_acc, prog_bar=False)
        self.log(f'{loop_type}/top10_acc', top_10_acc, prog_bar=False)
        self.log(f'{loop_type}/mrr', mrr, prog_bar=False)

    def training_epoch_end(self, outputs):
        self._epoch_end(outputs, 'train')

    def validation_epoch_end(self, outputs):
        self._epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        self._epoch_end(outputs, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.hparams['lr'])
        return optimizer
