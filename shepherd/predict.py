# General
import numpy as np
import pickle
import random
import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import time
from collections import Counter
import pandas as pd

sys.path.insert(0, '..') # add project_config to path

# Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

# W&B
import wandb


# Own code
import project_config
from shepherd.dataset import PatientDataset
from shepherd.samplers import PatientNeighborSampler


import preprocess
from hparams import get_predict_hparams
from train import get_model, load_patient_datasets, get_dataloaders

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 


'''
Example Command:

python predict.py \
--run_type causal_gene_discovery \
--patient_data test_predict \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain.ckpt \
--best_ckpt checkpoints/causal_gene_discovery.ckpt 

python predict.py \
--run_type patients_like_me \
--patient_data test_predict \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain.ckpt \
--best_ckpt checkpoints/patients_like_me.ckpt 

python predict.py \
--run_type disease_characterization \
--patient_data test_predict \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain.ckpt \
--best_ckpt checkpoints/disease_characterization.ckpt 
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Predict using SHEPHERD")
    parser.add_argument("--edgelist", type=str, default=None, help="File with edge list")
    parser.add_argument("--node_map", type=str, default=None, help="File with node list")
    parser.add_argument('--patient_data', default="disease_simulated", type=str)
    parser.add_argument('--run_type', choices=["causal_gene_discovery", "disease_characterization", "patients_like_me"], type=str)
    parser.add_argument('--saved_node_embeddings_path', type=str, default=None, help='Path to pretrained model checkpoint')
    parser.add_argument('--best_ckpt', type=str, default=None, help='Name of the best performing checkpoint')
    args = parser.parse_args()
    return args


@torch.no_grad()
def predict(args):
    
    # Hyperparameters
    hparams = get_predict_hparams(args)

    # Seed
    pl.seed_everything(hparams['seed'])

    # Read KG
    all_data, edge_attr_dict, nodes = preprocess.preprocess_graph(args)
    n_nodes = len(nodes["node_idx"].unique())
    gene_phen_dis_node_idx = torch.LongTensor(nodes.loc[nodes['node_type'].isin(['gene/protein', 'effect/phenotype', 'disease']), 'node_idx'].values)


    # Get dataset
    print('Loading SPL...')
    spl = np.load(project_config.PROJECT_DIR / 'patients' / hparams['spl'])  
    if (project_config.PROJECT_DIR / 'patients' / hparams['spl_index']).exists():
        with open(str(project_config.PROJECT_DIR / 'patients' / hparams['spl_index']), "rb") as input_file:
            spl_indexing_dict = pickle.load(input_file)
    else: spl_indexing_dict = None 
    print('Loaded SPL information')
    
    dataset = PatientDataset(project_config.PROJECT_DIR / 'patients' / hparams['test_data'], time=hparams['time'])
    print(f'There are {len(dataset)} patients in the test dataset')
    hparams.update({'inference_batch_size': len(dataset)})
    print('batch size: ', hparams['inference_batch_size'])
    # Get dataloader
    nid_to_spl_dict = {nid: idx for idx, nid in enumerate(nodes[nodes["node_type"] == "gene/protein"]["node_idx"].tolist())}


    dataloader = PatientNeighborSampler('predict', all_data.edge_index, all_data.edge_index[:,all_data.test_mask], 
                                        sizes = [-1,10,5], patient_dataset=dataset, batch_size = hparams['inference_batch_size'], sparse_sample = 0, 
                                        all_edge_attributes=all_data.edge_attr, n_nodes = n_nodes, relevant_node_idx=gene_phen_dis_node_idx,
                                        n_cand_diseases=hparams['test_n_cand_diseases'],  use_diseases=hparams['use_diseases'], 
                                        nid_to_spl_dict=nid_to_spl_dict, gp_spl=spl, spl_indexing_dict=spl_indexing_dict,
                                        shuffle = False, num_workers=hparams['num_workers'],
                                        hparams=hparams, pin_memory=hparams['pin_memory']) # 'test'
    
    # Create Weights & Biases Logger
    run_name = 'test'
    wandb_logger = WandbLogger(name=run_name, project='rare_disease_dx_combined', entity='rare_disease_dx', save_dir=hparams['wandb_save_dir'],
                    id="_".join(run_name.split(":")), resume="allow") 

    # Get patient model 
    model = get_model(args, hparams, None, all_data, edge_attr_dict,  n_nodes,load_from_checkpoint=True)

    trainer = pl.Trainer(gpus=hparams['n_gpus'])
    
    t1 = time.time()
    results = trainer.predict(model, dataloaders=dataloader)
    t2 = time.time()
    print(f"Predicting took {t2 - t1:0.4f} seconds", len(dataset), "patients")

    scores_dfs, attn_dfs, gat_attn_df_1, gat_attn_df_2, gat_attn_df_3, phenotype_embeddings, disease_embeddings = zip(*results)
    
    print('---- RESULTS ----')
    if not os.path.exists(project_config.PROJECT_DIR / 'results'):
        os.mkdir(project_config.PROJECT_DIR / 'results')
    output_base = project_config.PROJECT_DIR / 'results' /  (str(args.best_ckpt).replace('/', '.').split('.ckpt')[0])     

    # Save scores
    scores_df = pd.concat(scores_dfs).reset_index(drop=True)
    scores_df.to_csv(str(output_base) + '_scores.csv', index=False)
    print(scores_df)

    # Save patient phenotype attention
    attn_df = pd.concat(attn_dfs).reset_index(drop=True)
    attn_df.to_csv(str(output_base) + '_phenotype_attn.csv', index=False)
    print(attn_df)

    # Save patient phenotype embeddings
    if type(phenotype_embeddings) == tuple: phenotype_embeddings = phenotype_embeddings[0]
    torch.save(phenotype_embeddings, str(output_base) + '_phenotype_embeddings.pth')
    print("Phenotype embeddings", phenotype_embeddings)

    # Save disease embeddings
    if args.run_type == "disease_characterization":
        if type(disease_embeddings) == tuple: disease_embeddings = disease_embeddings[0]
        torch.save(disease_embeddings, str(output_base) + '_disease_embeddings.pth')
        print("Disease embeddings", disease_embeddings)


if __name__ == "__main__":
    
    # Get hyperparameters
    args = parse_args()

    # perform prediction
    predict(args)
