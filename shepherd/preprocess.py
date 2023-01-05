# General
import numpy as np
import pandas as pd
#from typing import List, Optional, Tuple, NamedTuple, Union, Callable

# Pytorch
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch import Tensor

import project_config


def preprocess_graph(args):

    # Read in nodes & edges
    nodes = pd.read_csv(project_config.KG_DIR / args.node_map, sep="\t")
    edges = pd.read_csv(project_config.KG_DIR / args.edgelist, sep="\t")

    # Initialize edge index
    edge_index = torch.LongTensor(edges[['x_idx', 'y_idx']].values.T).contiguous() 
    edge_attr = edges['full_relation']

    # Convert edge attributes to idx
    edge_attr_list = [
                      'effect/phenotype;phenotype_protein;gene/protein',
                      'gene/protein;phenotype_protein;effect/phenotype',
                      'disease;disease_phenotype_negative;effect/phenotype',
                      'effect/phenotype;disease_phenotype_negative;disease',
                      'disease;disease_phenotype_positive;effect/phenotype',
                      'effect/phenotype;disease_phenotype_positive;disease',
                      'gene/protein;protein_pathway;pathway',
                      'pathway;protein_pathway;gene/protein',
                      'disease;disease_protein;gene/protein',
                      'gene/protein;disease_protein;disease',
                      'gene/protein;protein_molfunc;molecular_function',
                      'molecular_function;protein_molfunc;gene/protein',
                      'gene/protein;protein_cellcomp;cellular_component',
                      'cellular_component;protein_cellcomp;gene/protein',
                      'gene/protein;protein_bioprocess;biological_process',
                      'biological_process;protein_bioprocess;gene/protein',
                      'biological_process;bioprocess_bioprocess;biological_process',
                      'biological_process;bioprocess_bioprocess_rev;biological_process',
                      'molecular_function;molfunc_molfunc;molecular_function',
                      'molecular_function;molfunc_molfunc_rev;molecular_function',
                      'cellular_component;cellcomp_cellcomp;cellular_component',
                      'cellular_component;cellcomp_cellcomp_rev;cellular_component',
                      'effect/phenotype;phenotype_phenotype;effect/phenotype',
                      'effect/phenotype;phenotype_phenotype_rev;effect/phenotype',
                      'gene/protein;protein_protein;gene/protein',
                      'gene/protein;protein_protein_rev;gene/protein',
                      'disease;disease_disease;disease',
                      'disease;disease_disease_rev;disease',
                      'pathway;pathway_pathway;pathway',
                      'pathway;pathway_pathway_rev;pathway'
                     ]

    edge_attr_to_idx_dict = {attr:i for i, attr in enumerate(edge_attr_list)}
    edge_attr = torch.LongTensor(np.vectorize(edge_attr_to_idx_dict.get)(edge_attr.values))

    # Get train/val/test masks
    mask = edges["mask"].values
    train_mask = torch.BoolTensor(mask == "train")
    val_mask = torch.BoolTensor(mask == "val")
    test_mask = torch.BoolTensor(mask == "test")


    # Create data object
    data = Data(edge_index = edge_index, edge_attr = edge_attr, train_mask = train_mask, val_mask = val_mask, test_mask = test_mask)
    return data, edge_attr_to_idx_dict, nodes

