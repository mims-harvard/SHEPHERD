# General
import numpy as np
import random
import argparse
import os
from copy import deepcopy
from pathlib import Path
import sys
from datetime import datetime

# Pytorch
import torch
import torch.nn as nn

# Pytorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Pytorch Geo
from torch_geometric.data.sampler import NeighborSampler as PyGeoNeighborSampler
from torch_geometric.data import Data, DataLoader

# W&B
import wandb

sys.path.insert(0, '..') # add project_config to path

# Own code
import preprocess
from node_embedder_model import NodeEmbeder
import project_config
from hparams import get_pretrain_hparams
from samplers import NeighborSampler


def parse_args():
    parser = argparse.ArgumentParser(description="Learn node embeddings.")
    
    # Input files/parameters
    parser.add_argument("--edgelist", type=str, default=None, help="File with edge list")
    parser.add_argument("--node_map", type=str, default=None, help="File with node list")
    parser.add_argument('--save_dir', type=str, default=None, help='Directory for saving files')
    
    # Tunable parameters
    parser.add_argument('--nfeat', type=int, default=2048, help='Dimension of embedding layer')
    parser.add_argument('--hidden', default=256, type=int)
    parser.add_argument('--output', default=128, type=int)
    parser.add_argument('--n_heads', default=4, type=int)
    parser.add_argument('--wd', default=0.0, type=float)
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout')
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--max_epochs', default=1000, type=int)
    
    # Resume with best checkpoint
    parser.add_argument('--resume', default="", type=str)
    parser.add_argument('--best_ckpt', type=str, default=None, help='Name of the best performing checkpoint')
    
    # Output
    parser.add_argument('--save_embeddings', action='store_true')

    args = parser.parse_args()
    return args


def get_dataloaders(hparams, all_data):
    print('get dataloaders')
    train_dataloader = NeighborSampler('train', all_data.edge_index[:,all_data.train_mask], all_data.edge_index[:,all_data.train_mask], sizes = hparams['neighbor_sampler_sizes'], batch_size = hparams['batch_size'], shuffle = True, num_workers=hparams['num_workers'], do_filter_edges=hparams['filter_edges'])
    val_dataloader = NeighborSampler('val', all_data.edge_index[:,all_data.train_mask], all_data.edge_index[:,all_data.val_mask], sizes = hparams['neighbor_sampler_sizes'], batch_size = hparams['batch_size'], shuffle = False, num_workers=hparams['num_workers'], do_filter_edges=hparams['filter_edges'])
    test_dataloader = NeighborSampler('test', all_data.edge_index[:,all_data.train_mask], all_data.edge_index[:,all_data.test_mask], sizes = hparams['neighbor_sampler_sizes'], batch_size = hparams['batch_size'], shuffle = False, num_workers=hparams['num_workers'], do_filter_edges=hparams['filter_edges'])
    return train_dataloader, val_dataloader, test_dataloader 


def train(args, hparams):

    # Seed
    pl.seed_everything(hparams['seed'])

    # Read input data
    all_data, edge_attr_dict, nodes = preprocess.preprocess_graph(args)

    # Set up
    if args.resume != "":
        if ":" in args.resume: # colons are not allowed in ID/resume name
            resume_id = "_".join(args.resume.split(":"))
        run_name = args.resume
        wandb_logger = WandbLogger(run_name, project='kg-train', entity='rare_disease_dx', save_dir=hparams['wandb_save_dir'], id=resume_id, resume=resume_id)
        model = NodeEmbeder.load_from_checkpoint(checkpoint_path=str(Path(args.save_dir) / 'checkpoints' /  args.best_ckpt), 
                                                 all_data=all_data, edge_attr_dict=edge_attr_dict, 
                                                 num_nodes=len(nodes["node_idx"].unique()), combined_training=False) 
    else:
        curr_time = datetime.now().strftime("%H:%M:%S")
        run_name = f"{curr_time}_run"
        wandb_logger = WandbLogger(run_name, project='kg-train', entity='rare_disease_dx', save_dir=hparams['wandb_save_dir'], id="_".join(run_name.split(":")), resume="allow")
        model = NodeEmbeder(all_data, edge_attr_dict, hp_dict=hparams, num_nodes=len(nodes["node_idx"].unique()), combined_training=False)

    checkpoint_callback = ModelCheckpoint(monitor='val/node_total_acc', dirpath=Path(args.save_dir) / 'checkpoints', filename=f'{run_name}' + '_{epoch}', save_top_k=1, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger.watch(model, log='all')

    if hparams['debug']:
        limit_train_batches = 1
        limit_val_batches = 1.0 
        hparams['max_epochs'] = 3
    else:
        limit_train_batches = 1.0
        limit_val_batches = 1.0 

    trainer = pl.Trainer(gpus=hparams['n_gpus'], logger=wandb_logger, 
                         max_epochs=hparams['max_epochs'], 
                         callbacks=[checkpoint_callback, lr_monitor], 
                         gradient_clip_val=hparams['gradclip'],
                         profiler=hparams['profiler'],
                         log_every_n_steps=hparams['log_every_n_steps'],
                         limit_train_batches=limit_train_batches, 
                         limit_val_batches=limit_val_batches,
                        ) 
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(hparams, all_data)

    # Train
    trainer.fit(model, train_dataloader, val_dataloader)
    
    # Test
    trainer.test(ckpt_path='best', test_dataloaders=test_dataloader)

@torch.no_grad()
def save_embeddings(args, hparams):
    print('Saving Embeddings')

    # Seed
    pl.seed_everything(hparams['seed'])

    # Read input data
    all_data, edge_attr_dict, nodes = preprocess.preprocess_graph(args)
    all_data.num_nodes = len(nodes["node_idx"].unique())

    model = NodeEmbeder.load_from_checkpoint(checkpoint_path=str(Path(args.save_dir) / 'checkpoints' /  args.best_ckpt), 
                                            all_data=all_data, edge_attr_dict=edge_attr_dict, 
                                            num_nodes=len(nodes["node_idx"].unique()), combined_training=False) 
   
    dataloader = DataLoader([all_data], batch_size=1)
    trainer = pl.Trainer(gpus=0, 
                        gradient_clip_val=hparams['gradclip']
                    ) 
    embeddings = trainer.predict(model, dataloaders=dataloader)  
    embed_path = Path(args.save_dir) / (str(args.best_ckpt).split('.ckpt')[0] + '.embed')
    torch.save(embeddings[0], str(embed_path))
    print(embeddings[0].shape)



if __name__ == "__main__":
    
    # Get hyperparameters
    args = parse_args()
    hparams = get_pretrain_hparams(args, combined=False) 
    
    if args.save_embeddings:
        # save node embeddings from a trained model
        save_embeddings(args, hparams)
    else:
        # Train model
        train(args, hparams)
