# Running SHEPHERD on your own data
*Assuming that you are using the SHEPHERD models that we provide*

## Steps
1. [Create conda environment](https://github.com/mims-harvard/SHEPHERD#two-set-up-environment)
2. Download data from [Harvard Dataverse](https://github.com/mims-harvard/SHEPHERD#three-download-datasets)
3. Set up [configuration file](https://github.com/mims-harvard/SHEPHERD#four-set-configuration-file)
4. Download [model checkpoint](https://figshare.com/articles/software/SHEPHERD/21444873)
5. Preprocess [your own patient data](https://github.com/mims-harvard/SHEPHERD/blob/main/data_prep/README.md)
    - *(Optional for patients-like-me identification)* If you would like to compare your patient cohort to an external cohort of patients (e.g., simulated patients), combine the jsonlines files of your own patient cohort and the external patient cohort.
7. Update `MY_TEST_DATA` in [`project_config.py`](https://github.com/mims-harvard/SHEPHERD/blob/main/project_config.py)
8. Generate [shortest paths calculations](https://github.com/mims-harvard/SHEPHERD/blob/main/data_prep/shortest_paths/add_spl_to_patients.py) using the flag `--only_test_data`
9. Update `MY_SPL_DATA` and `MY_SPL_INDEX_DATA` in [`project_config.py`](https://github.com/mims-harvard/SHEPHERD/blob/main/project_config.py)
10. Run `predict.py` to [generate predictions for your patients](https://github.com/mims-harvard/SHEPHERD#generate-predictions-for-patients)
    - Make sure that the run type and checkpoints are aligned (i.e., use `--run_type causal_gene_discovery` with `--best_ckpt checkpoints/causal_gene_discovery`)
    - Make sure that the patient data flag is set to your own dataset (i.e., `--patient_data my_data`)

## Results
The output of `predict.py` consists of:
- Dataframe of scores for each patient (`scores.csv`)
    - **For causal gene discovery:** Each patient's list of candidate genes are scored. The columns of the table are: patient ID, identifier of the candidate gene, similarity score, and binary correct label.
    - **For patients-like-me identification:** All patients in the input jsonlines file are scored. The columns of the table are: patient ID, identifier of the candidate patient, similarity score, and binary correct label. *Note that if you would like to compare only a subset of the patients, you can subset the scores of those patients and re-normalize.*
    - **For novel disease characterization:** Either all diseases in the knowledge graph or all Orphanet diseases are scored. The columns of the table are: patient ID, identifier of candidate disease (MONDO or Orphanet name), similarity score, and binary correct label.
- Phenotype attention (`phenotype_attn.csv`)
- Patient embeddings (`phenotype_embeddings.pth`)
- *(Only for novel disease characterization)* Disease embeddings (`disease_embeddings.pth`)
