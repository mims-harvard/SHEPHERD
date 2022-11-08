import numpy as np
import pandas as pd
import sys
from tqdm import tqdm
import pickle
import argparse
sys.path.insert(0, '../..') # add config to path
sys.path.insert(0, '..')
import project_config
from project_utils import read_simulated_patients, read_dicts

'''
python add_spl_to_patients.py  \
-simulated_path disease_split_train_sim_patients_kg_8.9.21_kg.txt \
-agg_type mean
'''

# NOTE: change x_max to 9 for max, 6.5 for avg & min, or 7 for median
def normalize(x, x_min=1, x_max): # min & max determined from across simulated patients
    return 2 * (x-x_min)/(x_max-x_min) - 1

def add_spl_info(patients, spl_matrix, hpo_to_idx_dict, ensembl_to_idx_dict , nid_to_spl_dict, min_spl, max_spl, all_gene_idx, agg_type, x_max):
    max_spl = -1
    min_spl = 1000
    avg_spl_matrix = np.zeros((len(patients), len(all_gene_idx)))
    print('spl_matrix', spl_matrix.shape)
    spl_indexing = {}
    for i, patient in enumerate(tqdm(patients)):
        patient_id = patient['id']
        spl_indexing[patient_id] = i
        hpo_idx = [hpo_to_idx_dict[p] for p in patient['positive_phenotypes'] if p in hpo_to_idx_dict ]
        if agg_type == 'mean':
            avg_spl_matrix[i, :] = [np.mean([spl_matrix[g, nid_to_spl_dict[p]] for p in hpo_idx]) for g in all_gene_idx]
        elif agg_type == 'max':
            avg_spl_matrix[i, :] = np.array([np.max([spl_matrix[g, nid_to_spl_dict[p]] for p in hpo_idx]) for g in all_gene_idx])
        elif agg_type == 'min':
            avg_spl_matrix[i, :] = np.array([np.min([spl_matrix[g, nid_to_spl_dict[p]] for p in hpo_idx]) for g in all_gene_idx])
        elif agg_type == 'median':
            avg_spl_matrix[i, :] = np.array([np.median([spl_matrix[g, nid_to_spl_dict[p]] for p in hpo_idx]) for g in all_gene_idx])
        else:
            raise NotImplementedError
        if np.max(avg_spl_matrix[i, :]) > max_spl:
            max_spl = np.max(avg_spl_matrix[i,:])
        if np.min(avg_spl_matrix[i, :]) < min_spl:
            min_spl = np.min(avg_spl_matrix[i, :])
        avg_spl_matrix[i,:] = [normalize(mean, x_max=x_max) for mean in avg_spl_matrix[i,:]]

    print('Max avg SPL from gene to phenotypes:', max_spl)
    print('Min avg SPL from gene to phenotypes:', min_spl)

    return avg_spl_matrix, spl_indexing

def main():
    parser = argparse.ArgumentParser(description="Add SPL to patients.")

    parser.add_argument("-node_map", type=str, default='KG_node_map.txt', help="Path to node map")
    parser.add_argument("-agg_type", type=str, default='mean', help="Path to simulated patients")
    
    args = parser.parse_args()
    print('Aggregation type: ', args.agg_type)
    
    # change x_max to 9 for max, 6.5 for avg & min, or 7 for median
    if args.agg_type == 'mean': x_max = 6.5
    elif args.agg_type == 'max': x_max = 9
    elif args.agg_type == 'min': x_max = 6.5
    elif args.agg_type == 'median': x_max = 7
    else: raise NotImplementedError

    node_df = pd.read_csv(project_config.KG_DIR / args.node_map, sep='\t')
    all_gene_idx = node_df.loc[node_df['node_type'] == 'gene/protein', 'node_idx'].tolist()

    # read in SPL matrix from all nodes to phenotypes
    #map from overall node id to idx within SPL matrix
    nid_to_spl_dict = {nid: idx for idx, nid in enumerate(node_df[node_df["node_type"] == "effect/phenotype"]["node_idx"].tolist())}
    spl_matrix = np.load(project_config.KG_DIR / 'KG_shortest_path_matrix_onlyphenotypes.npy') 
    print('spl_matrix shape: ', spl_matrix.shape)

    # get SPL from genes to phenotypes
    spl_genes_to_phens = spl_matrix[all_gene_idx,:]
    min_spl = np.min(spl_genes_to_phens)
    max_spl = np.max(spl_genes_to_phens)
    print(f'min SPL: {min_spl}, max SPL: {max_spl} from genes to phenotypes')

    # read in mapping dictionaries      
    hpo_to_idx_dict, ensembl_to_idx_dict, _, _ = read_dicts()

    # add SPL info to simulated patients
    train_sim_patients = read_simulated_patients(project_config.PROJECT_DIR / 'patients' / 'simulated_patients' / f'disease_split_train_sim_patients_kg_{project_config.CURR_KG}.txt' ) 
    val_sim_patients = read_simulated_patients(project_config.PROJECT_DIR / 'patients' / 'simulated_patients' / f'disease_split_val_sim_patients_kg_{project_config.CURR_KG}.txt' ) 
    sim_patients = train_sim_patients + val_sim_patients 
    sim_patients_spl_matrix, sim_spl_indexing = add_spl_info(sim_patients, spl_matrix, hpo_to_idx_dict, ensembl_to_idx_dict, nid_to_spl_dict, min_spl, max_spl , all_gene_idx, args.agg_type, x_max)
    with open(str(project_config.PROJECT_DIR / 'patients' / f'simulated_patients/disease_split_all_sim_patients_kg_{project_config.CURR_KG}_spl_index_dict.pkl'), 'wb') as handle:
        pickle.dump(sim_spl_indexing, handle, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(str(project_config.PROJECT_DIR / 'patients' / f'simulated_patients/disease_split_all_sim_patients_kg_{project_config.CURR_KG}_agg={args.agg_type}_spl_matrix.npy'), sim_patients_spl_matrix)


if __name__ == "__main__":
    main()

