import pandas as pd
import jsonlines
import networkx as nx
import snap
import obonet
import numpy as np
import re
import argparse
import random
import pickle
from tqdm import tqdm
from collections import defaultdict, Counter

from pathlib import Path
from copy import deepcopy
from itertools import combinations

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import matplotlib
matplotlib.use('Agg')

import sys
sys.path.insert(0, '..') # add config to path

import preprocess 
import project_config
from project_utils import read_simulated_patients, write_patients
pd.options.mode.chained_assignment = None

# input locations
ORPHANET_METADATA_FILE = str(project_config.PROJECT_DIR / 'preprocess' / 'orphanet' / 'orphanet_final_disease_metadata.tsv')
MONDO_MAP_FILE = str(project_config.PROJECT_DIR / 'preprocess' / 'mondo' / 'mondo_references.csv')

MONDO_OBO_FILE = str(project_config.PROJECT_DIR / 'preprocess' / 'mondo' / 'mondo.obo')
HP_TERMS = project_config.PROJECT_DIR  / 'preprocess' / 'hp_terms.csv'
MONDOTOHPO = project_config.PROJECT_DIR  /'preprocess'/ 'mondo' / 'mondo2hpo.csv'

# output locations
ORPHANET_TO_MONDO_DICT = str(project_config.PROJECT_DIR / 'preprocess' / 'orphanet' / 'orphanet_to_mondo_dict.pkl')

HPO_TO_IDX_DICT_FILE = project_config.PROJECT_DIR / 'knowledge_graph' / project_config.CURR_KG / f'hpo_to_idx_dict_{project_config.CURR_KG}.pkl'
HPO_TO_NAME_DICT_FILE = project_config.PROJECT_DIR / 'knowledge_graph'/ project_config.CURR_KG / f'hpo_to_name_dict_{project_config.CURR_KG}.pkl'
ENSEMBL_TO_IDX_DICT_FILE = project_config.PROJECT_DIR / 'knowledge_graph'/ project_config.CURR_KG / f'ensembl_to_idx_dict_{project_config.CURR_KG}.pkl'
GENE_SYMBOL_TO_IDX_DICT_FILE = project_config.PROJECT_DIR / 'knowledge_graph'/ project_config.CURR_KG / f'gene_symbol_to_idx_dict_{project_config.CURR_KG}.pkl'
MONDO_TO_NAME_DICT_FILE = project_config.PROJECT_DIR / 'knowledge_graph'/ project_config.CURR_KG  / f'mondo_to_name_dict_{project_config.CURR_KG}.pkl'
MONDO_TO_IDX_DICT_FILE = project_config.PROJECT_DIR / 'knowledge_graph'/ project_config.CURR_KG / f'mondo_to_idx_dict_{project_config.CURR_KG}.pkl'

# extracted from mondo.obo file
OBSOLETE_MONDO_DICT = {'MONDO:0008646':'MONDO:0100316', 'MONDO:0016021': 'MONDO:0100062', 
'MONDO:0017125':'MONDO:0010261', 'MONDO:0010624':'MONDO:0100213', 'MONDO:0019863':'MONDO:0011812',
'MONDO:0019523':'MONDO:0000171', 'MONDO:0018275':'MONDO:0018274', 'MONDO:0010071':'MONDO:0011939', 'MONDO:0010195':'MONDO:0008490', 
'MONDO:0008897':'MONDO:0100251', 'MONDO:0011127':'MONDO:0100344', 'MONDO:0009304':'MONDO:0012853', 
'MONDO:0008480':'MONDO:0031169', 'MONDO:0009641':'MONDO:0100294', 'MONDO:0010119':'MONDO:0031332',
'MONDO:0010766':'MONDO:0100250', 'MONDO:0008646':'MONDO:0100316', 'MONDO:0018641':'MONDO:0100245', 
'MONDO:0010272':'MONDO:0010327', 'MONDO:0012189':'MONDO:0018274', 'MONDO:0007926':'MONDO:0100280', 
'MONDO:0008032':'MONDO:0012215', 'MONDO:0009739':'MONDO:0024457', 'MONDO:0010419':'MONDO:0020721',
'MONDO:0007291':'MONDO:0031037', 'MONDO:0009245':'MONDO:0100339'} # to do starting with MONDO:0018275

def read_data(args):
    # read in KG nodes
    node_df = pd.read_csv(project_config.PROJECT_DIR / 'knowledge_graph' / project_config.CURR_KG / args.node_map, sep='\t')
    print(f'Unique node sources: {node_df["node_source"].unique()}')
    print(f'Unique node types: {node_df["node_type"].unique()}')

    node_type_dict = {idx:node_type for idx, node_type in zip(node_df['node_idx'], node_df['node_type'])}

    # read in patients
    sim_patients = read_simulated_patients(args.simulated_path)
    print(f'Number of sim patients: {len(sim_patients)}')
    
    # orphanet metadata
    orphanet_metadata = pd.read_csv(ORPHANET_METADATA_FILE, sep='\t', dtype=str)  

    # orphanet to mondo map
    mondo_map_df = pd.read_csv(MONDO_MAP_FILE, sep=',', index_col=0) #dtype=str
    obsolete_mondo_dict = {int(re.sub('MONDO:0*', '', k)):int(re.sub('MONDO:0*', '', v)) for k,v in OBSOLETE_MONDO_DICT.items()}
    mondo_map_df['mondo_id'] = mondo_map_df['mondo_id'].replace(obsolete_mondo_dict)
    mondo_map_df.to_csv(project_config.PROJECT_DIR / 'mondo_references_normalized.csv', sep=',') 

    mondo_map_df = mondo_map_df.loc[mondo_map_df['ontology'] == 'Orphanet']
    mondo_orphanet_map = {str(mondo_id):[int(v) for v in mondo_map_df.loc[mondo_map_df['mondo_id'] == mondo_id, 'ontology_id'].tolist()] for mondo_id in mondo_map_df['mondo_id'].unique().tolist() }

    mondo_obo = obonet.read_obo(MONDO_OBO_FILE) 
    mondo_to_orphanet_obo_map = {node_id:[r for r in node['xref'] if r.startswith('Orphanet')] for node_id, node in list(mondo_obo.nodes(data=True)) if 'xref' in node}
    mondo_to_orphanet_obo_map = {k.replace('MONDO:', ''): [int(v.replace('Orphanet:', '')) for v in vals] for k, vals in mondo_to_orphanet_obo_map.items() if len(vals) > 0 }
    mondo_to_orphanet_obo_map = {re.split('^0*', k)[-1]:v for k,v in mondo_to_orphanet_obo_map.items()}

    #merge two sources of mondo to orphanet mappings
    missing_keys = set(list(mondo_to_orphanet_obo_map.keys())).difference(set(list(mondo_orphanet_map.keys())))
    missing_keys2 = set(list(mondo_orphanet_map.keys())).difference(set(list(mondo_to_orphanet_obo_map.keys())))
    overlapping_keys = set(list(mondo_to_orphanet_obo_map.keys())).intersection(set(list(mondo_orphanet_map.keys())))
    print('\n ############ Retrieving mondo to orphanet maps ############')
    print(f'There are {len(missing_keys)} missing mappings from the non-obo mondo to orphanet mapping')
    print(f'There are {len(missing_keys2)} missing mappings from the obo mondo to orphanet mapping')
    disagreement_keys = [(k, mondo_orphanet_map[k],mondo_to_orphanet_obo_map[k])  for k in overlapping_keys if len(set(mondo_orphanet_map[k]).intersection(set(mondo_to_orphanet_obo_map[k]))) == 0]
    print(f'There is/are {len(disagreement_keys)} mapping(s from the two mondo dicts that don\'t agree with each other: {disagreement_keys}')
    
    merged_mondo_to_orphanet_map = {k: list(set(mondo_orphanet_map[k]).union(set(mondo_to_orphanet_obo_map[k]))) for k in overlapping_keys if k not in disagreement_keys}
    for k in missing_keys: merged_mondo_to_orphanet_map[k] = mondo_to_orphanet_obo_map[k]
    for k in missing_keys2: merged_mondo_to_orphanet_map[k] = mondo_orphanet_map[k]

    # create reverse - orphanet to mondo mapping
    orphanet_to_mondo_dict = defaultdict(list)
    for mondo, orphanet_list in merged_mondo_to_orphanet_map.items():
        for orphanet_id in orphanet_list:
            orphanet_to_mondo_dict[orphanet_id].append(mondo)
    with open(ORPHANET_TO_MONDO_DICT, 'wb') as handle:
        pickle.dump(orphanet_to_mondo_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('max number of mondo terms associated with an orphanet term: ', max([len(v) for k,v in orphanet_to_mondo_dict.items()]))

    # read in mapping from old to current phenotypes
    hp_terms = pd.read_csv(HP_TERMS)
    hp_map_dict = {'HP:' + ('0' * (7-len(str(int(hp_old))))) + str(int(hp_old)): 'HP:' + '0' * (7-len(str(int(hp_new)))) + str(int(hp_new)) for hp_old,hp_new in zip(hp_terms['id'], hp_terms['replacement_id'] ) if not pd.isnull(hp_new)}

    # read in mapping from mondo diseases to HPO phenotypes (this mapping occurs when a single entity is cross referenced by MONDO & HPO. In such cases we map to HPO)
    mondo2hpo = pd.read_csv(MONDOTOHPO)
    mondo_to_hpo_dict =  {mondo:hpo for hpo,mondo in zip(mondo2hpo['ontology_id'], mondo2hpo['mondo_id'])}

    return node_df, node_type_dict, sim_patients, orphanet_metadata, merged_mondo_to_orphanet_map, orphanet_to_mondo_dict, hp_map_dict, mondo_to_hpo_dict

def create_networkx_graph(edges):
    G = nx.MultiDiGraph()
    edge_index = list(zip(edges['x_idx'], edges['y_idx']))
    G.add_edges_from(edge_index)
    return G


###################################################################
# create maps from phenotype/gene to the idx in the KG

def create_hpo_to_node_idx_dict(node_df, hp_old_new_map):
    # get HPO nodes
    hpo_nodes = node_df.loc[node_df['node_type'] == 'effect/phenotype']
    hpo_nodes['node_id'] = hpo_nodes['node_id'].astype(str) 

    # convert HPO id to string version (e.g. 1 -> HP:0000001)
    HPO_LEN = 7
    padding_needed = HPO_LEN - hpo_nodes['node_id'].str.len()
    padded_hpo = padding_needed.apply(lambda x: 'HP:' + '0' * x)
    hpo_nodes['hpo_string'] = padded_hpo + hpo_nodes['node_id'] 

    # create dict from HPO ID to node index in graph
    hpo_to_idx_dict = {hpo:idx for hpo, idx in zip(hpo_nodes['hpo_string'].tolist(), hpo_nodes['node_idx'].tolist())}
    old_hpo_to_idx_dict = {old:hpo_to_idx_dict[new] for old, new in hp_old_new_map.items()}
    hpo_to_idx_dict = {**hpo_to_idx_dict, **old_hpo_to_idx_dict}

    hpo_to_name_dict = {hpo:name for hpo, name in zip(hpo_nodes['hpo_string'].tolist(), hpo_nodes['node_name'].tolist())}

    # save to file
    with open(HPO_TO_IDX_DICT_FILE, 'wb') as handle:
        pickle.dump(hpo_to_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(HPO_TO_NAME_DICT_FILE, 'wb') as handle:
        pickle.dump(hpo_to_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return hpo_to_idx_dict

def create_gene_to_node_idx_dict(args, node_df):
    ensembl_node_map = Path(str(args.node_map).split('.txt')[0]+ '_ensembl_ids.txt')
    if ensembl_node_map.exists():
        node_df = pd.read_csv(ensembl_node_map, sep='\t')
    else:
        print('Generating ensembl_ids for KG')
        preprocessor = preprocess.Preprocessor() #NOTE: raw data to perform preprocessing is missing from dataverse, but we provide the already processed files for our KG

        # get gene nodes & map to ensembl IDs
        gene_nodes = node_df.loc[node_df['node_type'] == 'gene/protein']
        gene_nodes = preprocessor.map_genes(gene_nodes, ['node_name'])
        gene_nodes = gene_nodes.rename(columns={'node_name_ensembl': 'node_name', 'node_name': 'gene_symbol'})

        # merge gene names with the original node df
        node_df['old_node_name'] = node_df['node_name']
        node_df.loc[node_df['node_idx'].isin(gene_nodes['node_idx']), 'node_name'] = gene_nodes['node_name']

        # save modified node df back to file
        node_df.to_csv(ensembl_node_map, sep='\t')

    # create gene to idx dict
    gene_nodes = node_df.loc[node_df['node_type'] == 'gene/protein']
    gene_symbol_to_idx_dict = {gene:idx for gene, idx in zip(gene_nodes['old_node_name'].tolist(), gene_nodes['node_idx'].tolist())}
    ensembl_to_idx_dict = {gene:idx for gene, idx in zip(gene_nodes['node_name'].tolist(), gene_nodes['node_idx'].tolist())}

    # save to file
    with open(ENSEMBL_TO_IDX_DICT_FILE, 'wb') as handle:
        pickle.dump(ensembl_to_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(GENE_SYMBOL_TO_IDX_DICT_FILE, 'wb') as handle:
        pickle.dump(gene_symbol_to_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    return node_df, gene_symbol_to_idx_dict, ensembl_to_idx_dict

def create_mondo_to_node_idx_dict(node_df, mondo_to_hpo_dict):
    '''create mondo disease to node_idx map'''

    # get disease nodes
    disease_nodes = node_df.loc[(node_df['node_type'] == 'disease')]
    disease_nodes['node_id'] = disease_nodes['node_id'].str.replace('.0', '', regex=False)
    mondo_strs = [str(mondo_str) for mondo, idx in zip(disease_nodes['node_id'].tolist(), disease_nodes['node_idx'].tolist()) for mondo_str in mondo.split('_')]
    assert len(mondo_strs) == len(list(set(mondo_strs))), 'The following dict may overwrite some mappings if because there are duplicates in the mondo ids'
    mondo_to_idx_dict = {str(mondo_str):idx for mondo, idx in zip(disease_nodes['node_id'].tolist(), disease_nodes['node_idx'].tolist()) for mondo_str in mondo.split('_')}
    
    # get mapping from phenotypes to KG idx
    phenotype_nodes = node_df.loc[node_df['node_type'] == 'effect/phenotype']
    phen_to_idx_dict = {int(phen):idx for phen, idx in zip(phenotype_nodes['node_id'].tolist(), phenotype_nodes['node_idx'].tolist()) if int(phen) in mondo_to_hpo_dict.values()}
    disease_mapped_phen_to_idx_dict = {str(mondo):phen_to_idx_dict[hpo] for mondo, hpo in mondo_to_hpo_dict.items()}

    # merge two mappings
    mondo_to_idx_dict = {**mondo_to_idx_dict, **disease_mapped_phen_to_idx_dict}

    #TODO: this dict is missing some names from phenotype diseases. This needs to be fixed.
    mondo_to_name_dict = {mondo_str:name for mondo, name in zip(disease_nodes['node_id'].tolist(), disease_nodes['node_name'].tolist()) for mondo_str in mondo.split('_')}
    #phen_to_idx_dict = {int(phen):idx for phen, idx in zip(phenotype_nodes['node_id'].tolist(), phenotype_nodes['node_name'].tolist()) if int(phen) in mondo_to_hpo_dict.values()}

    # save to file
    with open(MONDO_TO_NAME_DICT_FILE, 'wb') as handle:
        pickle.dump(mondo_to_name_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(MONDO_TO_IDX_DICT_FILE, 'wb') as handle:
        pickle.dump(mondo_to_idx_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return mondo_to_idx_dict

def map_diseases_to_orphanet(node_df, mondo_orphanet_map):
    all_orphanet_ids = []
    for node_id, node_type  in zip(node_df['node_id'], node_df['node_type']):
        if node_type == 'disease':
            mondo_ids = node_id.split('_')
            orphanet_ids = [mondo_orphanet_map[m] for m in mondo_ids if m in mondo_orphanet_map]
            orphanet_ids = [str(o) for l in orphanet_ids for o in l]
            #if len(orphanet_ids) > 0: print(mondo_ids, '_'.join(orphanet_ids))
            if len(orphanet_ids) == 0: all_orphanet_ids.append(None)
            else: all_orphanet_ids.append('_'.join(orphanet_ids)) #NOTE: some nodes that contain grouped MONDO ids are mapped to multiple orphanet ids
    node_df['orphanet_node_id'] = pd.Series(all_orphanet_ids)
    node_df.to_csv(project_config.KG_DIR / 'KG_node_map_ensembl_ids_orphanet_ids.txt', sep='\t') #TODO: check where we use this downstream

    



###################################################################
## split data into train/val/test

def filter_patients(patients, hpo_to_idx_dict, ensembl_to_idx_dict):
    '''
    Filter patients out of the dataset if their causal gene, all of their distractor genes, or all of 
    their phenotypes cannot be found in the KG.
    '''

    print(f'Number of patients pre-filtering: {len(patients)}')
    filtered_patients = [p for p in patients if len(set(p['true_genes']).intersection(set(ensembl_to_idx_dict.keys()))) > 0]
    print(f'Number of patients after filtering out those with no causal gene in the KG: {len(filtered_patients)}')

    if 'distractor_genes' in filtered_patients[0]:
        filtered_patients = [p for p in filtered_patients if len(set(p['distractor_genes']).intersection(set(ensembl_to_idx_dict.keys()))) > 0]
        print(f'Number of patients after filtering out those with no distractor genes in the KG: {len(filtered_patients)}')

    filtered_patients = [p for p in filtered_patients if len(set(p['positive_phenotypes']).intersection(set(hpo_to_idx_dict.keys()))) > 0]
    print(f'Number of patients after filtering out those with no phenotypes in the KG: {len(filtered_patients)}')

    return filtered_patients

def create_dataset_split_from_lists(filtered_patients, train_list_f, val_list_f):
    train_list = pd.read_csv(project_config.PROJECT_DIR / 'formatted_patients' / train_list_f, index_col=0)['ids'].tolist()
    val_list = pd.read_csv(project_config.PROJECT_DIR / 'formatted_patients' / val_list_f, index_col=0)['ids'].tolist()

    train_patients, val_patients, unsorted_patients = [], [], []
    for patient in filtered_patients:
        if patient['id'] in train_list: rand_train_patients.append(patient)
        elif patient['id'] in val_list: rand_val_patients.append(patient)
        else: unsorted_patients.append(patient)
    
    print(f'There are {len(train_patients)} patients in the train set and {len(val_patients)} in the val set.')
    print(f'There are {len(unsorted_patients)} unsorted patients.')
    return train_patients, val_patients

def create_disease_split_dataset(filtered_patients, frac_train=0.7, frac_val_test=0.15):
    # divide patients by disease ID into train/val/test
    diseases = list(set([p['disease_id'] for p in filtered_patients]))

    n_train = round(len(diseases) * frac_train)
    n_val_test = round(len(diseases) * frac_val_test)

    dx_train_patients = diseases[0:n_train]
    dx_val_patients = diseases[n_train:n_val_test+n_train]
    dx_test_patients = diseases[n_val_test+n_train:]

    print('Split of diseases into train/val/test: ', len(dx_train_patients), len(dx_val_patients), len(dx_test_patients))

    dx_split_train_patients = [p for p in filtered_patients if p['disease_id'] in dx_train_patients]
    dx_split_val_patients = [p for p in filtered_patients if p['disease_id'] in dx_val_patients]
    dx_split_test_patients = [p for p in filtered_patients if p['disease_id'] in dx_test_patients]

    dx_split_train_patient_ids = pd.DataFrame({'ids':[p['id'] for p in dx_split_train_patients]})
    dx_split_val_patient_ids = pd.DataFrame({'ids':[p['id'] for p in dx_split_val_patients]})
    dx_split_test_patient_ids = pd.DataFrame({'ids':[p['id'] for p in dx_split_test_patients]})
    
    #NOTE: we decided to merge the train & test sets into a single larger train set to be able to train on more diseases. We are posthoc merging to keep the code as was originally written.
    dx_split_train_patient_ids = pd.concat([]dx_split_train_patient_ids, dx_split_test_patient_ids)
    dx_split_train_patients = dx_split_train_patients + dx_split_test_patient_ids
    
    
    print(f'There are {len(dx_split_train_patients)} patients in the disease split train set and {len(dx_split_val_patients)} in the val set.')
    return dx_split_train_patients, dx_split_val_patients,  dx_split_train_patient_ids, dx_split_val_patient_ids




###################################################################
## main

'''
python preprocess_patients.py \
-split_dataset 
'''

def main():
    parser = argparse.ArgumentParser(description="Preprocessing Patients & KG.")
    parser.add_argument("-edgelist", type=str, default=f'KG_edgelist_mask.txt', help="File with edge list")
    parser.add_argument("-node_map", type=str, default=f'KG_node_map.txt', help="File with node list")
    parser.add_argument("-simulated_path", type=str, default=f'{project_config.PROJECT_DIR}/patients/simulated_patients/simulated_patients_formatted.jsonl', help="Path to simulated patients")

    parser.add_argument("-split_dataset", action='store_true', help="Split patient datasets into train/val/test.")
    parser.add_argument("-split_dataset_from_lists", action='store_true', help='Whether the train/val/test split IDs should be read from file.')
    
    args = parser.parse_args()

    ## read in data, normalize genes to ensembl ids, and create maps from genes/phenotypes to node idx
    node_df, node_type_dict, sim_patients, orphanet_metadata, mondo_orphanet_map, orphanet_mondo_map, hp_map_dict, mondo_to_hpo_dict = read_data(args)
    hpo_to_idx_dict = create_hpo_to_node_idx_dict(node_df, hp_map_dict)
    node_df, gene_symbol_to_idx_dict, ensembl_to_idx_dict = create_gene_to_node_idx_dict(args,node_df)
    mondo_to_node_idx_dict = create_mondo_to_node_idx_dict(node_df, mondo_to_hpo_dict)
    map_diseases_to_orphanet(node_df, mondo_orphanet_map)
    edges = pd.read_csv(project_config.KG_DIR / args.edgelist, sep="\t")
    graph = create_networkx_graph(edges)
    snap_graph = snap.LoadEdgeList(snap.TUNGraph, str(project_config.KG_DIR / args.edgelist), 0, 1, '\t')


    # filter patients to remove those with no causal gene, no distractor genes, or no phenotypes
    filtered_sim_patients = filter_patients(sim_patients, hpo_to_idx_dict, ensembl_to_idx_dict)
    # write patients to file
    write_patients(filtered_sim_patients, project_config.PROJECT_DIR / 'patients' / 'simulated_patients' /f'all_sim_patients_kg_{project_config.CURR_KG}.txt')
  
    if args.split_dataset:
        ## filter patients & split into train/val/test
        if args.split_dataset_from_lists:
            dx_split_train_patients, dx_split_val_patients = create_dataset_split_from_lists(filtered_sim_patients, 
                f'simulated_patients/disease_split_train_sim_patients_kg_{project_config.CURR_KG}_patient_ids.csv',
                f'simulated_patients/disease_split_val_sim_patients_kg_{project_config.CURR_KG}_patient_ids.csv'
            )
        else:
            dx_split_train_patients, dx_split_val_patients, dx_split_train_patient_ids, dx_split_val_patient_ids = create_disease_split_dataset(filtered_sim_patients)

        ## Save to file
        if not args.create_train_val_test_from_lists: 
            dx_split_train_patient_ids.to_csv(project_config.PROJECT_DIR / 'patients' / f'simulated_patients'/ f'disease_split_train_sim_patients_kg_{project_config.CURR_KG}_patient_ids.csv')
            dx_split_val_patient_ids.to_csv(project_config.PROJECT_DIR / 'patients' / f'simulated_patients'/ f'disease_split_val_sim_patients_kg_{project_config.CURR_KG}_patient_ids.csv')

        write_patients(dx_split_train_patients, project_config.PROJECT_DIR / 'patients' / 'simulated_patients'/ f'disease_split_train_sim_patients_kg_{project_config.CURR_KG}.txt')
        write_patients(dx_split_val_patients, project_config.PROJECT_DIR / 'patients' / 'simulated_patients'/ f'disease_split_val_sim_patients_kg_{project_config.CURR_KG}.txt')



if __name__ == "__main__":
    main()
