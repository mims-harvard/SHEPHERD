# Steps to process the Rare Disease KG

1) **Create a base knowledge graph for rare diseases.** We provide example scripts for building a base graph (i.e. PrimeKG) and adding any missing Orphanet edges. Please refer to `build_graph.ipynb` and `add_orphanet_data_to_kg.py`. We also include the output of the `add_orphanet_data_to_kg.py` for your reference, but none of the input data for `build_graph.ipynb` or `add_orphanet_data_to_kg.py`.

2) **Process and finalize knowledge graph for rare diseases.** The script `prepare_graph.py` adds triadic closures between phenotypes, diseases, and genes, extracts the largest connected component of the knowledge graph, splits the knowledge graph edges into train, validation, and test sets, and produces the required knowledge graph inputs for SHEPHERD.