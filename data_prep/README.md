# Prepare your own patient dataset

## Preprocessing steps
Your preprocessing script must do the following:
1. Map genes to Ensembl IDs
2. Map phenotypes to the 2019 version of HPO
3. Output a jsonlines file where each json (i.e., line in the file) contains information for a single patient

Please refer to the `create_mygene2_cohort/preprocess_mygene2.py` for an example preprocessing script.

## Patient information

An example patient from the simulated patients dataset:

```
{
 "id": 9,
 "positive_phenotypes": ["HP:0000221", "HP:0000232", "HP:0001155", "HP:0005692", "HP:0012471", "HP:0100540", "HP:0001999", "HP:0001249", "HP:0010285", "HP:0000924", "HP:0004459"],
 "all_candidate_genes": ["ENSG00000196277", "ENSG00000104899", "ENSG00000143156", "ENSG00000088451", "ENSG00000157557", "ENSG00000165125", "ENSG00000157766", "ENSG00000108821", "ENSG00000142655", "ENSG00000184470", "ENSG00000157119", "ENSG00000069431", "ENSG00000131828", "ENSG00000179111", "ENSG00000168646"],
 "true_genes": ["ENSG00000069431"],
 "true_diseases": ["966"]
}
```

### Required

The minimal information required for each patient are:
- Patient ID ("id")
- List of phenotypes present in the patient as HPO terms ("positive_phenotypes")

To run causal gene discovery, the json must also include:
- List of all candidate genes as Ensembl IDs ("all_candidate_genes")

To run patients-like-me identification or novel disease characterization, the json does not require any additional information.

### Optional
- Causal genes ("true_genes"). *If available, please provide causal genes as Ensembl IDs.*
- Disease names ("true_diseases"). *If available, please provide true disease names as MONDO IDs.*
- Omim ID
- Orphanet ID
- Orphanet category
