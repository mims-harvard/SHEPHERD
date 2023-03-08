# Prepare your own patient dataset

Your preprocessing script must do the following:
1. Map genes to Ensembl IDs
2. Map phenotypes to the 2019 version of HPO
3. Output a jsonlines file where each json (i.e., line in the file) contains information for a single patient

Please refer to the `create_mygene2_cohort/preprocess_mygene2.py` for an example preprocessing script. An example patient from MyGene2 dataset:

```
{"id": "mygene2_52",
 "positive_phenotypes": ["HP:0002987", "HP:0003273", "HP:0040083", "HP:0001239", "HP:0001371", "HP:0005830"],
 "all_candidate_genes": ["ENSG00000155657"], "true_genes": ["ENSG00000155657"],
 "true_diseases": [16675],
 "omim": "187370",
 "disease_name": "Distal arthrogryposis type 10",
 "orpha_id": [251515],
 "orpha_category": ["Rare developmental defect during embryogenesis"]
}
```

The minimal information needed for each patient are:
- Patient ID ("id")
- List of phenotypes present in the patient as HPO terms ("positive_phenotypes")
- List of causal genes as Ensembl IDs ("true_genes")

To run causal gene discovery, the json must also include:
- List of all candidate genes as Ensembl IDs ("all_candidate_genes")

To run patients-like-me identification, the json does not need any additional information.

To run novel disease characterization, the json must also include:
- List of true disease names as MONDO IDs ("true_diseases")

Optional information:
- Omim ID
- Disease name
- Orphanet ID
- Orphanet category
