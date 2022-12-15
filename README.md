# Deep learning for diagnosing patients with rare genetic diseases

**Authors**:
- [Emily Alsentzer](https://emilyalsentzer.com) (Equal contribution)
- [Michelle M. Li](http://michellemli.com) (Equal contribution)
- [Shilpa N. Kobren](http://shilpakobren.com)
- [Undiagnosed Diseases Network](https://undiagnosed.hms.harvard.edu)
- [Isaac S. Kohane](http://zaklab.org)
- [Marinka Zitnik](http://zitniklab.hms.harvard.edu)


## Overview of SHEPHERD

There are over 7,000 unique rare diseases, some of which affecting 3,500 or fewer patients in the US. Due to clinicians' limited experience with such diseases and the considerable heterogeneity of their clinical presentations, many patients with rare genetic diseases remain undiagnosed. While artificial intelligence has demonstrated success in assisting diagnosis, its success is usually contingent on the availability of large annotated datasets. Here, we present SHEPHERD, a deep learning approach for multi-faceted rare disease diagnosis. To overcome the limitations of supervised learning, SHEPHERD performs label-efficient training by (1) training exclusively on simulated rare disease patients without the use of any real labeled data and (2) incorporating external knowledge of known phenotype, gene and disease associations via knowledge-guided deep learning.

### The Rare Disease Diagnosis Pipeline

After years of failed diagnostic attempts, once a patient is accepted to the UDN, they receive a thorough clinical workup and genetic sequencing, and their case is analyzed in an iterative process to identify the candidate genes likely to explain the patient's symptoms. SHEPHERD can be utilized throughout the pipeline to accelerate the diagnosis process: after the clinical workup to find similar patients, after the sequencing analysis to identify strong candidate genes, and after case review to further prioritize candidate genes, characterize the patient's disease, and/or validate candidate genes by finding phenotype and genotype-matched patients.

<p align="center">
<img src="img/rare_diseases_pipeline.png?raw=true" width="600" >
</p>

### The SHEPHERD Algorithm

SHEPHERD is guided by existing knowledge of diseases, phenotypes, and genes to learn novel connections between a patient's clinico-genomic information and phenotype and gene relationships. SHEPHERD takes in as input the patient’s set of phenotypes as well a list of either candidates genes, patients, or diseases and leverages an external rare disease knowledge graph to perform multi-faceted rare disease diagnosis: causal gene discovery, patients-like-me identification, and novel disease characterization.

<p align="center">
<img src="img/shepherd_overview.png?raw=true" width="250" >
</p>


## Installation and Setup

### :one: Download the Repo

First, clone the GitHub repository:

```
git clone https://github.com/mims-harvard/SHEPHERD
cd SHEPHERD
```

### :two: Set Up Environment

This codebase leverages Python, Pytorch, Pytorch Geometric, etc. To create an environment with all of the required packages, please ensure that [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is installed and then execute the commands:

```
conda env create -f environment.yml
conda activate shepherd
bash install_pyg.sh
```

### :three: Download Datasets

The data is hosted on [Harvard Dataverse](https://doi.org/10.7910/DVN/TZTPFL). To maintain the directory structure while downloading the files, make sure to select all files and download in the original format.

We provide the following datasets for training SHEPHERD:
- Rare disease knowledge graph
- Disease-split train and validation sets for simulated patients
- MyGene2 patients

More details about the simulated rare disease patients can be found [here](https://github.com/EmilyAlsentzer/rare-disease-simulation). We are unfortunately unable to provide the UDN patients due to patient privacy concerns.

The rare disease knowledge graph and patient datasets are provided in the appropriate format for SHEPHERD. If you would like to add your own set of patients, please adhere to the format used in the MyGene2 and simulated rare disease patients' files.


### :four: Set Configuration File

Go to `project_config.py` and set the project directory (`PROJECT_DIR`) to be the path to the data folder downloaded in the previous step.

If you would like to use your own data, be sure to
1. Modify the data variables in `project_config.py` in lines 10-16.
2. Generate the required shortest path length data files for your patients using the code and instructions in `data_prep/shortest_paths`


### :five: (Optional) Download Model Checkpoints
We also provide checkpoints for SHEPHERD after pretraining and after training on the rare disease diagnosis tasks. The checkpoints for SHEPHERD can be found [here](https://figshare.com/articles/software/SHEPHERD/21444873). You'll need to move them to the directory specified by `project_config.PROJECT_DIR / 'checkpoints'` (see above step). Make sure all downloaded files are unzipped. You can use these checkpoints directly with the `predict.py` scripts below instead of training the models yourself.


## Usage

### Pretrain on Rare Disease KG

```
cd shepherd
python pretrain.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --save_dir checkpoints/
```

To see and/or modify the default hyperparameters, please see the `get_pretrain_hparams()` function in `shepherd/hparams.py`.

### Train SHEPHERD

:sparkles: To run causal gene discovery:

```
cd shepherd
python train.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --patient_data disease_simulated \
        --run_type causal_gene_discovery \
        --saved_node_embeddings_path checkpoints/<BEST_PRETRAIN_CHECKPOINT>.ckpt
```

:sparkles: To run patients-like-me identification:

```
cd shepherd
python train.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --patient_data disease_simulated \
        --run_type patients_like_me \
        --saved_node_embeddings_path checkpoints/<BEST_PRETRAIN_CHECKPOINT>.ckpt
```

:sparkles: To run novel disease characterization:

```
cd shepherd
python train.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --patient_data disease_simulated \
        --run_type disease_characterization \
        --saved_node_embeddings_path checkpoints/<BEST_PRETRAIN_CHECKPOINT>.ckpt
```

To see and/or modify the default hyperparameters, please see the `get_train_hparams()` function in `shepherd/hparams.py`.

### Report SHEPHERD Performance Metrics on Test Patient Dataset

After training SHEPHERD, you can calculate SHEPHERD's performance on a test patient dataset. Simply run the same command used to train the model with the additional flags: `--do_inference` and `--best_ckpt <PATH/TO/BEST_MODEL_CHECKPOINT.ckpt>`.

### Generate Predictions for Patients

After training SHEPHERD, you can generate predictions for patients (without performance metrics).

The results of the `predict.py` script are found in 
```
project_config.PROJECT_RESULTS/<TASK>/<RUN_NAME>/<DATASET_NAME>
```
where
- `<TASK>` is `causal_gene_discovery`, `patients_like_me`, or `disease_characterization`
- `<RUN_NAME>` is the name of the run created during training
- `<DATASET_NAME>` is the name of your patient cohort

:sparkles: To run causal gene discovery:

```
cd shepherd
python predict.py \
        --run_type causal_gene_discovery \
        --patient_data disease_simulated \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --saved_node_embeddings_path checkpoints/<BEST_PRETRAIN_CHECKPOINT>.ckpt \
        --best_ckpt PATH/TO/BEST_MODEL_CHECKPOINT.ckpt 
```

:sparkles: To run patients-like-me identification:

```
cd shepherd
python predict.py \
        --run_type patients_like_me \
        --patient_data disease_simulated \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --saved_node_embeddings_path checkpoints/<BEST_PRETRAIN_CHECKPOINT>.ckpt \
        --best_ckpt PATH/TO/BEST_MODEL_CHECKPOINT.ckpt 
```

:sparkles: To run novel disease characterization:

```
cd shepherd
python predict.py \
        --run_type disease_characterization \
        --patient_data disease_simulated \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --saved_node_embeddings_path checkpoints/<BEST_PRETRAIN_CHECKPOINT>.ckpt \
        --best_ckpt PATH/TO/BEST_MODEL_CHECKPOINT.ckpt 
```

To see and/or modify the default hyperparameters, please see the `get_predict_hparams()` function in `shepherd/hparams.py`.

## Additional Resources

- [Paper]()
- [Project Website](https://zitniklab.hms.harvard.edu/projects/SHEPHERD/)

```
@article{shepherd,
  title={Deep learning for diagnosing patients with rare genetic diseases},
  author={Alsentzer, Emily and Li, Michelle M. and Kobren, Shilpa and Undiagnosed Diseases Network and Kohane, Isaac S. and Zitnik, Marinka},
  journal={bioRxiv},
  year={2022}
}
```

## Questions

Please leave a Github issue or contact Emily Alsentzer at ealsentzer@bwh.harvard.edu and Michelle Li at michelleli@g.harvard.edu.
