# General
import numpy as np
import random
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter
import pandas as pd
import pickle
import time

sys.path.insert(0, '..') # add project_config to path

# Pytorch
import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, to_scipy_sparse_matrix
from torch_geometric.data import Data, DataLoader, NeighborSampler
from torch_geometric.utils import negative_sampling
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, SubsetRandomSampler

# Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

# W&B
import wandb

# multiprocessing
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

# Own code
import project_config
from shepherd.dataset import PatientDataset
from shepherd.gene_prioritization_model import CombinedGPAligner
from shepherd.patient_nca_model import CombinedPatientNCA 
from shepherd.samplers import PatientNeighborSampler

import preprocess
from hparams import get_pretrain_hparams, get_train_hparams


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
import faulthandler; faulthandler.enable()



def parse_args():
    parser = argparse.ArgumentParser(description="Learning node embeddings.")
    
    # Input files/parameters
    parser.add_argument("--edgelist", type=str, default=None, help="File with edge list")
    parser.add_argument("--node_map", type=str, default=None, help="File with node list")
    parser.add_argument('--saved_node_embeddings_path', type=str, default=None, help='Path within kg_embeddings folder to the saved KG embeddings')
    parser.add_argument('--patient_data', default="disease_simulated", type=str)
    parser.add_argument('--run_type', choices=["causal_gene_discovery", "disease_characterization", "patients_like_me"], type=str)

    # Tunable parameters
    parser.add_argument('--sparse_sample', default=200, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--upsample_cand', default=1, type=int)
    parser.add_argument('--neighbor_sampler_size', default=-1, type=int)
    parser.add_argument('--lmbda', type=float, default=0.5, help='Lambda')
    parser.add_argument('--alpha', type=float, default=0, help='Alpha')
    parser.add_argument('--kappa', type=float, default=0.3, help='Kappa (Only used for combined model with link prediction loss)')
    parser.add_argument('--seed', default=33, type=int)
    parser.add_argument('--batch_size', default=64, type=int) 
    
    # Resume / run inference with best checkpoint
    parser.add_argument('--resume', default="", type=str)
    parser.add_argument('--do_inference', action='store_true')
    parser.add_argument('--best_ckpt', type=str, default=None, help='Name of the best performing checkpoint')
    
    parser.add_argument('--use_wandb', type=bool, default=True)

    args = parser.parse_args()
    return args


def load_patient_datasets(hparams, inference=False):
    print('loading patient datasets')

    if inference:
        train_dataset = None
        val_dataset = None
    else:
        train_dataset = PatientDataset(project_config.PROJECT_DIR / 'patients' / hparams['train_data'],  time=hparams['time'])
        val_dataset = PatientDataset(project_config.PROJECT_DIR / 'patients' / hparams['validation_data'], time=hparams['time'])

    if inference:
        test_dataset = PatientDataset(project_config.PROJECT_DIR / 'patients' / hparams['test_data'], time=hparams['time'])
    else:
        test_dataset = None
    
    print('finished loading patient datasets')
    return train_dataset, val_dataset, test_dataset


def get_dataloaders(hparams, all_data, nid_to_spl_dict, n_nodes, gene_phen_dis_node_idx, train_dataset, val_dataset, test_dataset, inference=False):
    print('Get dataloaders', flush=True)
    shuffle = False if hparams['debug'] or inference else True
    if not hparams['sample_from_gpd']: gene_phen_dis_node_idx = None
    batch_sz = hparams['inference_batch_size'] if inference else hparams['batch_size']
    sparse_sample = 1 if inference else hparams['sparse_sample']

    #get phenotypes & genes found in train patients
    if hparams['sample_edges_from_train_patients']:
        phenotype_counter = Counter()
        gene_counter = Counter()
        for patient in train_dataset:
            phenotype_node_idx, candidate_gene_node_idx, correct_genes_node_idx, disease_node_idx, labels, additional_labels, patient_ids = patient

            phenotype_counter += Counter(list(phenotype_node_idx.numpy()))
            gene_counter += Counter(list(candidate_gene_node_idx.numpy()))
    else:
        phenotype_counter=None
        gene_counter=None

    print('Loading SPL...')
    spl = np.load(project_config.PROJECT_DIR / 'patients' / hparams['spl'])  
    if (project_config.PROJECT_DIR / 'patients' / hparams['spl_index']).exists():
        with open(str(project_config.PROJECT_DIR / 'patients' / hparams['spl_index']), "rb") as input_file:
            spl_indexing_dict = pickle.load(input_file)
    else: spl_indexing_dict=None # TODO: short term fix for simulated patients, get rid once we create this dict
    
    print('Loaded SPL information')
    

    if inference:
        train_dataloader = None
        val_dataloader = None
    else:
        print('setting up train dataloader')         
        train_dataloader = PatientNeighborSampler('train', all_data.edge_index[:,all_data.train_mask], all_data.edge_index[:,all_data.train_mask], 
                        sizes = hparams['neighbor_sampler_sizes'], patient_dataset=train_dataset, batch_size = batch_sz, 
                        sparse_sample = sparse_sample, do_filter_edges=hparams['filter_edges'], 
                        all_edge_attributes=all_data.edge_attr, n_nodes = n_nodes, relevant_node_idx=gene_phen_dis_node_idx,
                        shuffle = shuffle, train_phenotype_counter=phenotype_counter, train_gene_counter=gene_counter, sample_edges_from_train_patients=hparams['sample_edges_from_train_patients'], num_workers=hparams['num_workers'], 
                        upsample_cand=hparams['upsample_cand'], n_cand_diseases=hparams['n_cand_diseases'], use_diseases=hparams['use_diseases'], nid_to_spl_dict=nid_to_spl_dict, gp_spl=spl, spl_indexing_dict=spl_indexing_dict,
                        hparams=hparams, pin_memory=hparams['pin_memory'])
        print('finished setting up train dataloader')
        print('setting up val dataloader')
        val_dataloader = PatientNeighborSampler('val', all_data.edge_index, all_data.edge_index[:,all_data.val_mask], 
                        sizes = [-1,10,5], patient_dataset=val_dataset, batch_size = batch_sz, 
                        sparse_sample = sparse_sample, all_edge_attributes=all_data.edge_attr, n_nodes = n_nodes, 
                        relevant_node_idx=gene_phen_dis_node_idx, 
                        shuffle = False, train_phenotype_counter=phenotype_counter, train_gene_counter=gene_counter, sample_edges_from_train_patients=hparams['sample_edges_from_train_patients'], num_workers=hparams['num_workers'],
                        n_cand_diseases=hparams['n_cand_diseases'], use_diseases=hparams['use_diseases'], nid_to_spl_dict=nid_to_spl_dict, gp_spl=spl, spl_indexing_dict=spl_indexing_dict,
                        hparams=hparams,  pin_memory=hparams['pin_memory'])
        print('finished setting up val dataloader')
    
    print('setting up test dataloader')
    if inference:
        sizes = [-1,10,5]
        print('SIZES: ', sizes)
        test_dataloader = PatientNeighborSampler('test', all_data.edge_index, all_data.edge_index[:,all_data.test_mask], 
                        sizes = sizes, patient_dataset=test_dataset, batch_size = len(test_dataset), 
                        sparse_sample = sparse_sample, all_edge_attributes=all_data.edge_attr, n_nodes = n_nodes, relevant_node_idx=gene_phen_dis_node_idx,
                        shuffle = False, num_workers=hparams['num_workers'],
                        n_cand_diseases=hparams['test_n_cand_diseases'],  use_diseases=hparams['use_diseases'], nid_to_spl_dict=nid_to_spl_dict, gp_spl=spl, spl_indexing_dict=spl_indexing_dict,
                        hparams=hparams, pin_memory=hparams['pin_memory']) 
    else: test_dataloader = None
    print('finished setting up test dataloader')
    
    return train_dataloader, val_dataloader, test_dataloader


def get_model(args, hparams, node_hparams, all_data, edge_attr_dict, n_nodes, load_from_checkpoint=False):
    print("setting up model", hparams['model_type'])
    # get patient model 
    if hparams['model_type'] == 'aligner':
        if load_from_checkpoint: 
            comb_patient_model = CombinedGPAligner.load_from_checkpoint(checkpoint_path=str(Path(project_config.PROJECT_DIR /  args.best_ckpt)), 
                                    edge_attr_dict=edge_attr_dict, all_data=all_data, n_nodes=n_nodes, node_ckpt = hparams["saved_checkpoint_path"], node_hparams=node_hparams)
        else:
            comb_patient_model = CombinedGPAligner(edge_attr_dict=edge_attr_dict, all_data=all_data, n_nodes=n_nodes, hparams=hparams, node_ckpt = hparams["saved_checkpoint_path"], node_hparams=node_hparams)
    elif hparams['model_type'] == 'patient_NCA':
        if load_from_checkpoint:
            comb_patient_model = CombinedPatientNCA.load_from_checkpoint(checkpoint_path=str(Path(project_config.PROJECT_DIR) /  args.best_ckpt), 
                                    all_data=all_data, edge_attr_dict=edge_attr_dict, n_nodes=n_nodes, node_ckpt=hparams["saved_checkpoint_path"])
        else:
            comb_patient_model = CombinedPatientNCA(edge_attr_dict=edge_attr_dict, all_data=all_data, n_nodes=n_nodes, node_ckpt=hparams["saved_checkpoint_path"], hparams=hparams)
    else:
        raise NotImplementedError
    print('finished setting up model')
    return comb_patient_model


def train(args, hparams):
    print('Training Model', flush=True)

    # Hyperparameters
    node_hparams = get_pretrain_hparams(args, combined=True)
    print('Edge List: ', args.edgelist,  flush=True)
    print('Node Map: ', args.node_map, flush=True)

    # Set seed
    pl.seed_everything(hparams['seed'])

    # Read input data
    print('Read data', flush=True)
    all_data, edge_attr_dict, nodes = preprocess.preprocess_graph(args)
    n_nodes = len(nodes["node_idx"].unique())
    print(f'Number of nodes: {n_nodes}')
    gene_phen_dis_node_idx = torch.LongTensor(nodes.loc[nodes['node_type'].isin(['gene/protein', 'effect/phenotype', 'disease']), 'node_idx'].values)
    
    if args.resume != "":
        print('Resuming Run')
        # create Weights & Biases Logger
        if ":" in args.resume: # colons are not allowed in ID/resume name
            resume_id = "_".join(args.resume.split(":"))
        run_name = args.resume
        wandb_logger = WandbLogger(run_name, project=hparams['wandb_project_name'], entity='rare_disease_dx', save_dir=hparams['wandb_save_dir'], id=resume_id, resume=resume_id)
        
        #add run name to hparams dict
        hparams['run_name'] = run_name
        
        # get patient model 
        comb_patient_model = get_model(args, hparams, node_hparams, all_data, edge_attr_dict, n_nodes, load_from_checkpoint=True)
        
    else:
        print('Creating new W&B Logger')
        # create Weights & Biases Logger
        curr_time = datetime.now().strftime("%m_%d_%y:%H:%M:%S")
        lr = hparams['lr']   
        val_data = str(hparams['validation_data']).split('.txt')[0].replace('/', '.')
        run_name = "{}_lr_{}_val_{}_losstype_{}".format(curr_time, lr, val_data, hparams['loss']).replace('patients', 'pats') 
        run_name = run_name.replace('5_candidates_mapped_only', '5cand_map').replace('8.9.21_kgsolved_manual_baylor_nobgm_distractor_genes', 'manual').replace('patient_disease_NCA', 'pd_NCA').replace('_distractor', '')
        wandb_logger = WandbLogger(name=run_name, project=hparams['wandb_project_name'], entity='rare_disease_dx', save_dir=hparams['wandb_save_dir'],
                        id="_".join(run_name.split(":")), resume="allow") 
        
        #add run name to hparams dict
        print('Run name', run_name)
        hparams['run_name'] = run_name

        # get patient model 
        comb_patient_model = get_model(args, hparams, node_hparams, all_data, edge_attr_dict, n_nodes, load_from_checkpoint=False)

    # get model & dataloaders
    nid_to_spl_dict = {nid: idx for idx, nid in enumerate(nodes[nodes["node_type"] == "gene/protein"]["node_idx"].tolist())}
    train_dataset, val_dataset, test_dataset = load_patient_datasets(hparams)
    patient_train_dataloader, patient_val_dataloader, patient_test_dataloader = get_dataloaders(hparams, all_data, nid_to_spl_dict,
                                                                                                n_nodes, gene_phen_dis_node_idx, 
                                                                                                train_dataset, val_dataset, test_dataset)

    # callbacks
    print('Init callbacks')
    checkpoint_path = (project_config.PROJECT_DIR / 'checkpoints' / hparams['model_type'] / run_name) 
    hparams['checkpoint_path'] = checkpoint_path
    print('Checkpoint path: ', checkpoint_path)
    if not os.path.exists(project_config.PROJECT_DIR / 'checkpoints' / hparams['model_type']): (project_config.PROJECT_DIR / 'checkpoints' / hparams['model_type']).mkdir()
    if not os.path.exists(checkpoint_path): checkpoint_path.mkdir()
    monitor_type =  'val/mrr' if args.run_type == 'disease_characterization' or args.run_type == 'patients_like_me' else 'val/gp_val_epoch_mrr'
    fname = 'epoch={epoch:02d}-val_mrr={val/mrr:.2f}' if args.run_type == 'disease_characterization' or args.run_type == 'patients_like_me'  else 'epoch={epoch:02d}-val_mrr={val/gp_val_epoch_mrr:.2f}'
    patient_checkpoint_callback = ModelCheckpoint(
        monitor=monitor_type,
        dirpath=checkpoint_path,
        filename=fname,
        save_top_k=2,
        mode='max',
        auto_insert_metric_name = False
    )

    # log gradients with logger
    print('wandb logger watch')
    wandb_logger.watch(comb_patient_model, log='all')

    #initialize trainer
    if hparams['debug']: 
        limit_train_batches = 1
        limit_val_batches = 1 
        hparams['max_epochs'] = 6
    else: 
        limit_train_batches=1.0
        limit_val_batches=1.0

    print('initialize trainer')
    patient_trainer = pl.Trainer(gpus=hparams['n_gpus'], 
                                logger=wandb_logger, 
                                max_epochs=hparams['max_epochs'], 
                                callbacks=[patient_checkpoint_callback],
                                profiler=hparams['profiler'],
                                log_gpu_memory=hparams['log_gpu_memory'],
                                limit_train_batches=limit_train_batches, 
                                limit_val_batches=limit_val_batches,
                                weights_summary="full",
                                gradient_clip_val=hparams['gradclip'])

    #  Train
    patient_trainer.fit(comb_patient_model, patient_train_dataloader, patient_val_dataloader)

@torch.no_grad()
def inference(args, hparams):
    print('Running inference')
    # Hyperparameters
    node_hparams = get_pretrain_hparams(args, combined=True)

    hparams.update({'add_similar_patients': False})

    # Seed
    pl.seed_everything(hparams['seed'])

    # Read data
    all_data, edge_attr_dict, nodes = preprocess.preprocess_graph(args)
    n_nodes = len(nodes["node_idx"].unique())
    gene_phen_dis_node_idx = torch.LongTensor(nodes.loc[nodes['node_type'].isin(['gene/protein', 'effect/phenotype', 'disease']), 'node_idx'].values)

    # Get logger & trainer
    curr_time = datetime.now().strftime("%m_%d_%y:%H:%M:%S")
    lr = hparams['lr']   
    test_data = hparams['test_data'].split('.txt')[0].replace('/', '.')
    run_name = "{}_lr_{}_test_{}".format(curr_time, lr, test_data)
    wandb_logger = WandbLogger(run_name, project=hparams['wandb_project_name'], entity='rare_disease_dx', save_dir=hparams['wandb_save_dir'])
    print('Run name: ', run_name)
    hparams['run_name'] = run_name

    # Get datasets
    train_dataset, val_dataset, test_dataset = load_patient_datasets(hparams, inference=True)

    # Get dataloader
    nid_to_spl_dict = {nid: idx for idx, nid in enumerate(nodes[nodes["node_type"] == "gene/protein"]["node_idx"].tolist())}
    _, _, test_dataloader = get_dataloaders(hparams, all_data, nid_to_spl_dict,
                                                                        n_nodes, gene_phen_dis_node_idx, 
                                                                        train_dataset, val_dataset, test_dataset, inference=True)

    # Get patient model 
    model = get_model(args, hparams, node_hparams, all_data, edge_attr_dict, n_nodes, load_from_checkpoint=True)

    trainer = pl.Trainer(gpus=0, logger=wandb_logger)
    results = trainer.test(model, dataloaders=test_dataloader)
    print('---- RESULTS ----')
    print(results)



if __name__ == "__main__":
    
    # Get hyperparameters
    args = parse_args()
    hparams = get_train_hparams(args)

    # Run model
    if args.do_inference:
        inference(args, hparams)
    else:
        train(args, hparams)

