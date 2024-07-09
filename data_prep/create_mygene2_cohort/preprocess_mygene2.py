from pathlib import Path
import sys
import pandas as pd
import numpy as np
import pickle
from collections import defaultdict
sys.path.insert(0, '../') # add project_config to path
sys.path.insert(0, '../../../udn_knowledge_graph/') # add config to path

import project_config
import preprocess 
from project_utils import write_patients

MYGENE2_FOLDER = project_config.PROJECT_DIR / 'patients' / 'mygene2_patients'


def concat_patients():
    # combine all of the mygene2 patients
    if not (MYGENE2_FOLDER / 'mygene2_patients.csv').exists():
        dfs = []
        for f in MYGENE2_FOLDER.iterdir():
            if not str(f).endswith('.sh') and not str(f).endswith('.csv'):
                df = pd.read_csv(f)
                dfs.append(df)
        all_df = pd.concat(dfs)
        all_df.to_csv(MYGENE2_FOLDER / 'mygene2_patients.csv')
    else:
        all_df = pd.read_csv(MYGENE2_FOLDER / 'mygene2_patients.csv', index_col=0)
    return all_df

def normalize_patients():
    # Map genes to ensembl IDs
    if not (MYGENE2_FOLDER / 'mygene2_patients_genes_normalized.csv').exists():
        preprocessor = preprocess.Preprocessor(hpo_source_year='2019') #TODO: uncomment if need to use

        all_df = preprocessor.map_genes(all_df, ['Gene Name']) \
                .rename(columns={'Gene Name_ensembl':'ensembl_ids', 'Gene Name_mapping_status': 'mapping_status'})
        gene_to_ensembl = {'SP3A': 'ENSG00000172845', 'GNA01':'ENSG00000087258', 'MUNC13-1':'ENSG00000130477', 'METAZOA_SRP': 'ENSG00000274848'}
        all_df = all_df.replace({"ensembl_ids": gene_to_ensembl})

        all_df.to_csv(MYGENE2_FOLDER / 'mygene2_patients_genes_normalized.csv')
    else:
        all_df = pd.read_csv(MYGENE2_FOLDER / 'mygene2_patients_genes_normalized.csv', index_col=0)

    # Normalize Phenotypes
    if not (MYGENE2_FOLDER / 'mygene2_patients_phenotypes_normalized.csv').exists():
        phenotypes_df = all_df[['MyGene2 profile', 'Phenotypes']]
        split_phenotypes = phenotypes_df['Phenotypes'].str.split(',', expand=True)
        stacked = pd.concat([phenotypes_df['MyGene2 profile'], split_phenotypes],axis=1).set_index('MyGene2 profile').stack()
        stacked = stacked.reset_index().drop(columns=['level_1'])
        stacked.columns = ['MyGene2 profile', 'Phenotypes']
        
        preprocessor = preprocess.Preprocessor(hpo_source_year='2019') #TODO: uncomment if need to use
        stacked = preprocessor.map_phenotypes(stacked, col_name = 'Phenotypes').drop(columns=['Phenotypes']).rename(columns={'standardized_hpo_id':'Phenotypes'})
        stacked.to_csv(MYGENE2_FOLDER / 'mygene2_patients_phenotypes_normalized.csv')
    else:
        phenotypes_df = pd.read_csv(MYGENE2_FOLDER / 'mygene2_patients_phenotypes_normalized.csv', index_col=0)

    return all_df, phenotypes_df



def convert_to_jsonl(phenotypes_df, all_df, omim_to_mondo_map, omim_to_orpha_match, orphanet_to_category_map):
    n_with_multiple_genes = 0
    n_with_single_gene_and_omim = 0
    patients = []

    profile_ids = []
    omim_ids = []
    for profile_id in phenotypes_df['MyGene2 profile'].unique():
        phenotypes = phenotypes_df.loc[phenotypes_df['MyGene2 profile'] == profile_id, 'Phenotypes'].tolist()
        genes = all_df.loc[all_df['MyGene2 profile'] == profile_id, 'ensembl_ids'].tolist()
        omim = all_df.loc[all_df['MyGene2 profile'] == profile_id, 'OMIM'].tolist()
        omim = [o for o in omim if not pd.isnull(o)]
        omim = np.unique(omim).tolist()
        omim  = [str(o).split('.')[0] for o in omim]
        mondo = [m for o in omim for m in omim_to_mondo_map[o]]
        condition = all_df.loc[all_df['MyGene2 profile'] == profile_id, 'Condition'].tolist()
        condition = [o for o in condition if not pd.isnull(o)]
        condition = np.unique(condition).tolist()
        pubmed_id = all_df.loc[all_df['MyGene2 profile'] == profile_id, 'Pubmed ID'].tolist()
        pubmed_id = [o for o in pubmed_id if not pd.isnull(o)]
        pubmed_id = np.unique(pubmed_id).tolist()
        assert len(pubmed_id) <= 1
        assert len(omim) <= 1
        assert len(condition) <= 1

        if len(genes) > 1: 
            n_with_multiple_genes += 1
        if len(genes) == 1 and len(omim) > 0:
            n_with_single_gene_and_omim += 1
            profile_ids.append(profile_id)
            omim_ids.append(omim[0])
            patient = { 
                        'id': f'mygene2_{profile_id}', 
                        "positive_phenotypes": phenotypes, 
                        "all_candidate_genes": genes, 
                        "true_genes":genes,
                        "true_diseases": mondo,
                        "omim": omim[0] if len(omim) > 0 else None,
                        "disease_name": condition[0] if len(condition) > 0 else None,
                        "pubmed_id": pubmed_id[0] if len(pubmed_id) > 0 else None,
                        "orpha_id": omim_to_orpha_match[int(omim[0])].tolist() if len(omim) > 0 and int(omim[0]) in omim_to_orpha_match else None,
                        }
            if not patient['orpha_id'] is None:
                patient['orpha_category'] = np.unique([orphanet_to_category_map[str(orpha)] for orpha in patient['orpha_id'] if str(orpha) in orphanet_to_category_map]).tolist() 

            patients.append(patient)

    df = pd.DataFrame({'Profile ID': profile_ids, 'OMIM': omim_ids})
    print(df.head(), len(df), len(df['OMIM'].unique()))
    df.to_csv(project_config.PROJECT_DIR / 'patients' / 'mygene2_patients' / 'mygene2_5.7.22_profileid_to_omim.txt')

    print(f'Of the {len(phenotypes_df["MyGene2 profile"].unique())} total patients, {n_with_multiple_genes} have multiple genes listed')
    print(f'Of the {len(phenotypes_df["MyGene2 profile"].unique())} total patients, {n_with_single_gene_and_omim} have a single gene + OMIM listed')
    return patients

#########
# don't map to orpha ID
# [609943:1465, 209850:null,  603855:null, 164230:null,  606631:null ]

# map to more than one orpha ID
# [115200., 146300., 158600., 211530., 253010., 266150., 300100.,
#       300672., 312750., 314580., 604841., 610253., 614970., 615066.,
#       615290., 615369., 616266., 616708.]

def main():
    with open(str(project_config.PROJECT_DIR / 'omim_to_mondo_dict.pkl'), 'rb') as handle:
        omim_to_mondo_map = pickle.load(handle)

    

    # OMIM to ORPHA
    orpha_to_omim_df = pd.read_csv(project_config.PROJECT_DIR / 'data' / 'orphanet'/ 'orphanet_to_omim_mapping_df.csv')
    orpha_to_omim_df_validated = orpha_to_omim_df.loc[orpha_to_omim_df['External_Mapping_Status'] == 'Validated']
    omim_to_orpha_match = defaultdict(list)
    for omim, orpha in zip(orpha_to_omim_df_validated['External_ID'], orpha_to_omim_df_validated['OrphaNumber']): omim_to_orpha_match[omim].append(orpha)
    omim_to_orpha_match = {k:np.unique(v) for k,v in omim_to_orpha_match.items()}
    omim_to_orpha_match[609943] = 1465 # outdated code so had to manually add mapping

    #ORPHA to DX CATEGORY
    ORPHANET_METADATA_FILE_OLD = project_config.UDN_DATA / 'orphanet/flat_files/orphanet_final_disease_metadata.tsv'
    ORPHANET_METADATA_FILE = project_config.PROJECT_DIR / 'data/orphanet/categorization_of_orphanet_diseases.csv' 
    orphanet_metadata = pd.read_csv(ORPHANET_METADATA_FILE, sep=',', dtype=str)  
    orphanet_metadata_old = pd.read_csv(ORPHANET_METADATA_FILE_OLD, sep='\t', dtype=str)  
    orphanet_to_category_map = {orphanumber:category for orphanumber, category in zip(orphanet_metadata['OrphaNumber'], orphanet_metadata['Category'])}
    orphanet_to_category_old_map = {orphanumber:category for orphanumber, category in zip(orphanet_metadata_old['OrphaNumber'], orphanet_metadata_old['Category'])}
    orphanet_to_category_map = {**orphanet_to_category_map, **orphanet_to_category_old_map}
    orphanet_to_category_map['438274'] = 'Rare neoplastic disease' #https://www.orpha.net/consor/cgi-bin/Disease_Search.php?lng=EN&data_id=1285&Disease_Disease_Search_diseaseGroup=966&Disease_Disease_Search_diseaseType=ORPHA&Disease(s)/group%20of%20diseases=Hypertrichosis-acromegaloid-facial-appearance-syndrome&title=Hypertrichosis-acromegaloid%20facial%20appearance%20syndrome&search=Disease_Search_Simple
    orphanet_to_category_map['254704'] = 'Unknown'
    orphanet_to_category_map['397715'] = 'Rare developmental defect during embryogenesis'
    orphanet_to_category_map['402075'] = 'Rare cardiac disease'
    orphanet_to_category_map['52688'] = 'Rare hematologic disease'


    all_df = concat_patients()
    all_df, phenotypes_df = normalize_patients()
    patients = convert_to_jsonl(phenotypes_df, all_df, omim_to_mondo_map, omim_to_orpha_match, orphanet_to_category_map)
    write_patients(patients, project_config.PROJECT_DIR / 'patients' / 'mygene2_patients' / 'mygene2_5.7.22.txt')
    

    
if __name__ == "__main__":
    main()
