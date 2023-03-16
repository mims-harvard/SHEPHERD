

import random
import numpy as np
import pandas as pd
import pickle
import sys
import torch
import time
import re
from torch.utils.data import Dataset
from collections import defaultdict

from project_utils import read_patients
import project_config


class PatientDataset(Dataset):

    def __init__(self, filepath, gp_spl=None, raw_data=False, mondo_map_file=str(project_config.PROJECT_DIR / 'mondo_references.csv'), needs_disease_mapping=False, time=False): 
        self.filepath = filepath
        self.patients = read_patients(filepath)
        print('Dataset filepath: ', filepath)
        print('Number of patients: ', len(self.patients))
        
        # add placeholder for true genes/diseases if they don't exist
        for patient in self.patients:
            if 'true_genes' not in patient: patient['true_genes'] = []
            if 'true_diseases' not in patient: patient['true_diseases'] = []

        self.raw_data = raw_data
        self.needs_disease_mapping = needs_disease_mapping
        self.time = time

        # create HPO to node_idx map
        with open(project_config.KG_DIR / f'hpo_to_idx_dict_{project_config.CURR_KG}.pkl', 'rb') as handle:
            self.hpo_to_idx_dict = pickle.load(handle)
        with open(project_config.KG_DIR / f'hpo_to_name_dict_{project_config.CURR_KG}.pkl', 'rb') as handle:
            self.hpo_to_name_dict = pickle.load(handle)
        self.idx_to_hpo_dict = {v:self.hpo_to_name_dict[k] if k in self.hpo_to_name_dict else k for k, v in self.hpo_to_idx_dict.items()}


        # create ensembl to node_idx map
        # NOTE: assumes conversion from gene symbols to ensembl IDs has already occurred
        with open(str(project_config.KG_DIR  / f'ensembl_to_idx_dict_{project_config.CURR_KG}.pkl'), 'rb') as handle:
            self.ensembl_to_idx_dict = pickle.load(handle)
        self.idx_to_ensembl_dict = {v:k for k, v in self.ensembl_to_idx_dict.items()}

        # orphanet to mondo disease map
        with open(str(project_config.PROJECT_DIR / 'preprocess' / 'orphanet' / 'orphanet_to_mondo_dict.pkl'), 'rb') as handle:
            self.orpha_mondo_map = pickle.load(handle)

        with open(project_config.KG_DIR / f'mondo_to_idx_dict_{project_config.CURR_KG}.pkl', 'rb') as handle:
            self.disease_to_idx_dict = pickle.load(handle)
        with open(project_config.KG_DIR / f'mondo_to_name_dict_{project_config.CURR_KG}.pkl', 'rb') as handle:
            self.disease_to_name_dict = pickle.load(handle)
        self.idx_to_disease_dict = {v:self.disease_to_name_dict[k] if k in self.disease_to_name_dict else k for k, v in self.disease_to_idx_dict.items()}


        # degree dict from idx to degree - used for debugging
        #NOTE: may need to subtract 1 to index into this dict
        with open(str(project_config.KG_DIR / f'degree_dict_{project_config.CURR_KG}.pkl'), 'rb') as handle:
            self.degree_dict = pickle.load(handle)

        # missing_patients = [patient for patient in self.patients  if len(set(patient['true_genes']).difference(set(list(self.ensembl_to_idx_dict)))) > 0 ]
        # print(f'There are {len(missing_patients)} patients of {len(self.patients)} whose correct gene is not in the KG')
        # print('patients with missing causal genes in KG: ', [p['id'] for p in missing_patients])
        # print('len patients before filtering out patients with causal genes: ', len(self.patients))
        # self.patients = [patient for patient in self.patients if len(set(patient['true_genes']).difference(set(list(self.ensembl_to_idx_dict)))) == 0 ]
        # print('len patients after filtering out patients with causal genes: ', len(self.patients))

        # get patients with similar genes
        if all(['true_genes' in patient for patient in self.patients]): # first check to make sure all patients have true genes
            genes_to_patients = defaultdict(list)
            for patient in self.patients:
                for g in patient['true_genes']:
                    genes_to_patients[g].append(patient['id'])
            self.patients_with_same_gene = defaultdict(list)
            for patients in genes_to_patients.values():
                for p in patients:
                    self.patients_with_same_gene[p].extend([pat for pat in patients if pat != p])

        # get patients with similar diseases
        if all(['true_diseases' in patient for patient in self.patients]): # first check to make sure all patients have true diseases
            dis_to_patients = defaultdict(list)
            for patient in self.patients:
                patient_diseases = patient['true_diseases']
                for d in patient_diseases: 
                    dis_to_patients[d].append(patient['id'])
            self.patients_with_same_disease = defaultdict(list)
            for patients in dis_to_patients.values():
                for p in patients:
                    self.patients_with_same_disease[p].extend([pat for pat in patients if pat != p])

        # map from patient id to index in dataset
        self.patient_id_to_index = {p['id']:i for i, p in enumerate(self.patients)}

        print('Finished initalizing dataset')


    def __len__(self):
        '''
        Returns the length of the dataset
        '''
        return len(self.patients)

    def __getitem__(self, idx):
        '''
        Returns a single example from the dataset
        '''
        t0 = time.time()
        patient = self.patients[idx]

        additional_labels_dict = {}
        if 'additional_labels' in patient:
            for label, values in patient['additional_labels'].items():
                if label == "n_hops_cand_g_p": continue
                if values == None: values = [[-1]]
                if type(values) != list: values = [[values]] #  wrap in list if needed
                if type(values) == list and type(values[0]) != list: values = [values]     
                additional_labels_dict[label] = values 
            if 'max_percent_phen_overlap_train' not in patient['additional_labels']: additional_labels_dict['max_percent_phen_overlap_train'] = [[-1]] 
            if 'max_phen_overlap_train' not in patient['additional_labels']: additional_labels_dict['max_phen_overlap_train'] = [[-1]]
       
        phenotype_node_idx = [self.hpo_to_idx_dict[p] for p in patient['positive_phenotypes'] if p in self.hpo_to_idx_dict ]
        correct_genes_node_idx = [self.ensembl_to_idx_dict[g] for g in patient['true_genes'] if g in self.ensembl_to_idx_dict ]
        if 'all_candidate_genes' in patient:
            candidate_gene_node_idx = [self.ensembl_to_idx_dict[g] for g in patient['all_candidate_genes'] if g in self.ensembl_to_idx_dict ]
        else: candidate_gene_node_idx = []

        if 'true_diseases' in patient:
            if self.needs_disease_mapping:
                orpha_diseases = [ int(d) if len(re.match("^[0-9]*", d)[0]) > 0 else d for d in patient['true_diseases']]
                mondo_diseases = [mondo_d for orpha_d in set(orpha_diseases).intersection(set(self.orpha_mondo_map.keys())) for mondo_d in self.orpha_mondo_map[orpha_d]]
            else:
                mondo_diseases = [str(d) for d in patient['true_diseases']]
            disease_node_idx = [self.disease_to_idx_dict[d] for d in mondo_diseases if d in self.disease_to_idx_dict]
        else: disease_node_idx = None
        
        if not self.raw_data:
            phenotype_node_idx = torch.LongTensor(phenotype_node_idx)
            correct_genes_node_idx = torch.LongTensor(correct_genes_node_idx)
            candidate_gene_node_idx = torch.LongTensor(candidate_gene_node_idx)
            if 'true_diseases' in patient: disease_node_idx = torch.LongTensor(disease_node_idx)
            
        
        assert len(phenotype_node_idx) >= 1, f'There are no phenotypes for patient: {patient}'
        
        #NOTE: assumes that patient has a single causal/correct gene (the model still outputs a score for each candidate gene)
        if not self.raw_data:
            if len(correct_genes_node_idx) > 1:
                #print('NOTE: The patient has multiple correct genes, but we\'re only selecting the first.')
                correct_genes_node_idx = correct_genes_node_idx[0].unsqueeze(-1)

        # get index of correct gene
        if len(correct_genes_node_idx) == 0: # no correct genes available for the patient
            label_idx = None
        else:
            if self.raw_data:
                label_idx = [candidate_gene_node_idx.index(g) for g in correct_genes_node_idx]
            else:
                label_idx = (candidate_gene_node_idx == correct_genes_node_idx[0]).nonzero(as_tuple=True)[0]

        if self.time:
            t1 = time.time()
            print(f'It takes {t1-t0:0.4f}s to get an item from the dataset')
            
        return (phenotype_node_idx, candidate_gene_node_idx, correct_genes_node_idx, disease_node_idx, label_idx, additional_labels_dict, patient['id'])

    def node_idx_to_degree(self, idx):
        if idx in self.degree_dict:
            return self.degree_dict[idx]
        else:
            return -1
    
    def node_idx_to_name(self, idx):
        if idx in self.idx_to_hpo_dict:
            return self.idx_to_hpo_dict[idx]
        elif idx in self.idx_to_ensembl_dict:
            return self.idx_to_ensembl_dict[idx]
        elif idx in self.idx_to_disease_dict:
            return self.idx_to_disease_dict[idx]
        elif idx == 0:
            return 'padding'
        else:
            print(idx)
            raise Exception

    def get_similar_patients(self, patient_id, similarity_type='gene'):
        if similarity_type == 'gene':
            sim_pats = np.array(self.patients_with_same_gene[patient_id])
            np.random.shuffle(sim_pats)
            return sim_pats
        elif similarity_type == 'disease':
            sim_pats = np.array(self.patients_with_same_disease[patient_id])
            np.random.shuffle(sim_pats)
            return sim_pats
        else:
            raise NotImplementedError

    def get_candidate_diseases(self, cand_type='all_kg_nodes'):
        if cand_type == 'all_kg_nodes':
            all_kg_diseases_idx = np.unique(list(self.disease_to_idx_dict.values())) # get idx of all diseases in KG
            return torch.LongTensor(all_kg_diseases_idx)
        elif cand_type == 'orphanet':
            orpha_mondo_diseases = [d[0] for d in list(self.orpha_mondo_map.values())]
            orpha_mondo_idx = np.unique([self.disease_to_idx_dict[d] for d in orpha_mondo_diseases if d in self.disease_to_idx_dict])
            return torch.LongTensor(orpha_mondo_idx)
        else:
            raise NotImplementedError

