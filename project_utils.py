import json
import jsonlines
            
import sys
import pickle
import project_config


##############################################
# Read in/write patients

def read_patients(filename):
    patients = []
    with jsonlines.open(filename) as reader:
        for patient in reader:
            patients.append(patient)
    return patients


def write_patients(patients, filename):
    with open(filename, "w") as output_file:
        for patient_dict in patients:
            json.dump(patient_dict, output_file)
            output_file.write('\n')
            

def read_dicts():
    with open(project_config.KG_DIR / f'hpo_to_idx_dict_{project_config.CURR_KG}.pkl', 'rb') as handle:
        hpo_to_idx_dict = pickle.load(handle)

    with open(str(project_config.KG_DIR / f'ensembl_to_idx_dict_{project_config.CURR_KG}.pkl'), 'rb') as handle:
        ensembl_to_idx_dict = pickle.load(handle)

    with open(project_config.KG_DIR /  f'mondo_to_idx_dict_{project_config.CURR_KG}.pkl', 'rb') as handle:
        disease_to_idx_dict = pickle.load(handle)

    with open(str(project_config.PROJECT_DIR / 'preprocess' / 'orphanet' / 'orphanet_to_mondo_dict.pkl'), 'rb') as handle:
        orpha_mondo_map = pickle.load(handle)

    return hpo_to_idx_dict, ensembl_to_idx_dict, disease_to_idx_dict, orpha_mondo_map
