
import pickle
import pandas as pd
pd.options.mode.chained_assignment = None
import sys
import obonet
import re

sys.path.insert(0, '..') # add config to path
sys.path.insert(0, '../../udn_knowledge_graph/') # add config to path

import project_config

from simulate_patients.utils import preprocess


'''
There are some relationships found in Orphanet that are missing from the KG. We add them here. 
'''

gene_symbol_alias_dict = {'RARS':'RARS1', 'C11ORF80':'C11orf80', 'KIF1BP':'KIFBP', 'ICK':'CILK1', 'AARS':'AARS1', 'SPG23':'DSTYK', 
        'C9ORF72':'C9orf72', 'C8ORF37':'C8orf37', 'HARS':'HARS1', 'GARS':'GARS1', 'C19ORF12':'C19orf12', 
        'C12ORF57':'C12orf57', 'ADSSL1':'ADSS1', 'C12ORF65':'C12orf65', 'MARS':'MARS1', 'CXORF56':'STEEP1', 
        'SARS':'SARS1', 'C12ORF4':'C12orf4', 'MUT':'MMUT', 'LOR':'LORICRIN'}



def find_missing_genes(kg_nodes, orpha_genes, use_ensembl=False):
    gene_nodes = kg_nodes.loc[kg_nodes['node_type'] == 'gene/protein']

    if not use_ensembl:
        # manually curated mapping from aliases to gene symbol name using genecards.org

        orpha_genes = set([gene_symbol_alias_dict[g] if g in gene_symbol_alias_dict else g for g in orpha_genes])
    genes_in_kg = set(gene_nodes['node_name'].tolist())
    orphanet_genes_missing_in_kg = set(orpha_genes).difference(genes_in_kg)
    ensembl_str = 'ensembl' if use_ensembl else ''
    print(f'There are {len(orphanet_genes_missing_in_kg)} orphanet {ensembl_str} genes missing in the KG of {len(set(orpha_genes))} genes')
    return orphanet_genes_missing_in_kg, genes_in_kg

def find_missing_phenotypes(kg_nodes, orpha_phenotypes):
    # phenotypes
    phenotype_nodes = kg_nodes.loc[kg_nodes['node_type'] == 'effect/phenotype']
    HPO_LEN = 7
    padding_needed = HPO_LEN - phenotype_nodes['node_id'].str.len()
    padded_hpo = padding_needed.apply(lambda x: 'HP:' + '0' * x)
    phenotype_nodes['hpo_string'] = padded_hpo + phenotype_nodes['node_id'] 
    orphanet_phenotypes_missing_in_kg = set(orpha_phenotypes).difference(set(phenotype_nodes['hpo_string'].tolist()))
    print(f'There are {len(orphanet_phenotypes_missing_in_kg)} orphanet phenotypes missing in the KG of {len(set(orpha_phenotypes))} phenotypes')
    return orphanet_phenotypes_missing_in_kg, phenotype_nodes

def find_missing_diseases(kg_nodes, orpha_diseases, orphanet_to_mondo_dict, mondo_definitions_obo_map, mondo_to_hpo_dict, phenotype_nodes):

    # get mapping from MONDO disease to KG idx
    disease_nodes = kg_nodes.loc[kg_nodes['node_type'] == 'disease']
    disease_nodes['node_id'] = disease_nodes['node_id'].str.replace('.0', '', regex=False)
    disease_to_idx_dict = {str(mondo_str):idx + 1 for mondo, idx in zip(disease_nodes['node_id'].tolist(), disease_nodes['node_index'].tolist()) for mondo_str in mondo.split('_')}

    # get mapping from phenotypes to KG idx
    phenotype_nodes = kg_nodes.loc[kg_nodes['node_type'] == 'effect/phenotype']
    phen_to_idx_dict = {int(phen):idx + 1 for phen, idx in zip(phenotype_nodes['node_id'].tolist(), phenotype_nodes['node_index'].tolist()) if int(phen) in mondo_to_hpo_dict.values()}
    disease_mapped_phen_to_idx_dict = {str(mondo):phen_to_idx_dict[hpo] for mondo,hpo in mondo_to_hpo_dict.items()}

    # merge two mappings
    disease_to_idx_dict = {**disease_to_idx_dict, **disease_mapped_phen_to_idx_dict}

    # get orphanet diseases that can't be mapped to MONDO
    orphanet_diseases_not_mapped_to_mondo = [orpha_d for orpha_d in orpha_diseases if int(orpha_d) not in orphanet_to_mondo_dict]
    print(f'There are {len(orphanet_diseases_not_mapped_to_mondo)} orphanet diseases that cannot be mapped to MONDO. ')
    
    # get mondo diseases not found in KG
    mondo_diseases = [mondo_d for orpha_d in set(orpha_diseases).difference(set(orphanet_diseases_not_mapped_to_mondo)) for mondo_d in orphanet_to_mondo_dict[int(orpha_d)]]
    diseases_in_kg = set(disease_to_idx_dict.keys())
    diseases_missing_in_kg = list(set(mondo_diseases).difference(diseases_in_kg))
    print(f'There are {len(diseases_missing_in_kg)} orphanet -> mondo diseases missing in the KG of {len(set(mondo_diseases))} diseases')
    
    disease_names = [mondo_definitions_obo_map[d] if d in mondo_definitions_obo_map else None for d in diseases_missing_in_kg]
    missing_disease_df = pd.DataFrame({'node_id': diseases_missing_in_kg, 'node_type': ['disease'] * len(diseases_missing_in_kg), 'node_name': disease_names, 'node_source': ['Orphanet'] * len(diseases_missing_in_kg)})
    return diseases_missing_in_kg, orphanet_diseases_not_mapped_to_mondo, missing_disease_df, diseases_in_kg

def map_to_ensembl(kg_nodes):
    if (project_config.KG_DIR / 'raw' / 'nodes_with_ensembl.csv').exists():
        kg_nodes = pd.read_csv(project_config.KG_DIR / 'raw' / 'nodes_with_ensembl.csv')
        kg_nodes.columns = ['node_index','node_id','node_type','node_name','node_source','old_node_name']
    else:
        preprocessor = preprocess.Preprocessor()

        # get gene nodes & map to ensembl IDs
        gene_nodes = kg_nodes.loc[kg_nodes['node_type'] == 'gene/protein']
        gene_nodes = preprocessor.map_genes(gene_nodes, ['node_name'])
        print('Done mapping genes')
        gene_nodes = gene_nodes.rename(columns={'node_name_ensembl': 'node_name', 'node_name': 'gene_symbol'})

        # merge gene names with the original node df
        kg_nodes['old_node_name'] = kg_nodes['node_name']
        kg_nodes.loc[kg_nodes['node_index'].isin(gene_nodes['node_index']), 'node_name'] = gene_nodes['node_name']

        # save modified node df back to file
        kg_nodes.to_csv(project_config.KG_DIR / 'raw' / 'nodes_with_ensembl.csv', sep=',', index=False)
    return kg_nodes



def find_missing_disease_gene_edges(orpha_disease_gene_edges, kg, kg_nodes, orphanet_to_mondo_dict, orphanet_genes_missing_in_kg, diseases_missing_in_kg, mondo_to_hpo_dict ):
    ## get edges from orphanet
    # map gene names to their alias
    orpha_disease_gene_edges = [(d, gene_symbol_alias_dict[g]) if g in gene_symbol_alias_dict else (d,g) for (d,g) in list(orpha_disease_gene_edges)]
    # convert orphanet orpha to gene edges into mondo to gene edges
    orpha_disease_gene_edges = [(orphanet_to_mondo_dict[int(d)], g)  for (d,g) in orpha_disease_gene_edges if int(d) in orphanet_to_mondo_dict]
    orpha_disease_gene_edges = [(d, g) for d_list, g in orpha_disease_gene_edges for d in d_list]

    ## get edges from KG
    gene_disease_edges = kg.loc[(kg['x_type'] == 'gene/protein') & (kg['y_type'] == 'disease')]
    disease_gene_edges = kg.loc[(kg['x_type'] == 'disease') & (kg['y_type'] == 'gene/protein')] # reverse relations
    #all_gene_disease_edges = pd.concat([gene_disease_edges, disease_gene_edges])
    assert len(gene_disease_edges.index) == 0
    #print(len(gene_disease_edges.index), len(disease_gene_edges.index))
    kg_gene_disease_edges_tup = list(zip(disease_gene_edges['y_id'], disease_gene_edges['x_name']))
    kg_gene_disease_edges_tup = [(d,g) for d_list, g in kg_gene_disease_edges_tup for d in d_list.split('_')]

    missing_edges = set(orpha_disease_gene_edges).difference(set(kg_gene_disease_edges_tup))
    edges_where_gene_not_in_kg = [(d,g) for (d,g) in missing_edges if g in orphanet_genes_missing_in_kg]
    edges_where_disease_not_in_kg = [(d,g) for (d,g) in missing_edges if d in diseases_missing_in_kg]

    print(f'There are {len(missing_edges)} edges of {len(set(orpha_disease_gene_edges))} found in orphanet but not the KG.')
    print(f'There are {len(edges_where_gene_not_in_kg)} edges where the gene node is not in the KG: {edges_where_gene_not_in_kg}.')
    print(f'There are {len(edges_where_disease_not_in_kg)} edges where the disease node is not in the KG.\n')

    # construct edges df
    filtered_missing_edges = [(d,g) for (d,g) in missing_edges if (g not in orphanet_genes_missing_in_kg) and (d not in diseases_missing_in_kg)]
    disease_id = [d for (d,g) in filtered_missing_edges]
    gene_name = [g for (d,g) in filtered_missing_edges]
    disease_nodes = kg_nodes.query('node_type == "disease"')
    phenotype_nodes = kg_nodes.query('node_type == "effect/phenotype"')
    disease_nodes['node_id_list'] = disease_nodes['node_id'].str.split('_')
    gene_nodes = kg_nodes.query('node_name in @gene_name and node_type == "gene/protein"')

    disease_index = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_index'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_index'].tolist()[0] for d in disease_id]
    gene_index = [gene_nodes.loc[gene_nodes['node_name'] == g, 'node_index'].tolist()[0]  for g in gene_name]
    disease_name = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_name'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_name'].tolist()[0] for d in disease_id]

    gene_id = [gene_nodes.loc[gene_nodes['node_name'] == g, 'node_id'].tolist()[0] for g in gene_name]
    grouped_disease_id = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_id'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_id'].tolist()[0] for d in disease_id]

    disease_type = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_type'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_type'].tolist()[0] for d in disease_id]
    disease_source = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_source'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_source'].tolist()[0] for d in disease_id]
    
    #
    relations = ['phenotype_protein' if int(d) in mondo_to_hpo_dict else 'disease_protein' for d in disease_id]

    missing_edge_df = pd.DataFrame({'relation': relations, 'display_relation': ['associated with'] * len(filtered_missing_edges),
        'x_index': disease_index,'x_id': grouped_disease_id,'x_type': disease_type,'x_name': disease_name, 'x_source':disease_source ,
        'y_index': gene_index,'y_id': gene_id, 'y_type':(['gene/protein'] * len(filtered_missing_edges)) ,
            'y_name': gene_name, 'y_source': (['NCBI'] * len(filtered_missing_edges)) })
    return missing_edge_df

def find_missing_disease_phenotype_edges(orpha_disease_phenotype_edges, kg, kg_nodes, orphanet_to_mondo_dict, orphanet_phenotypes_missing_in_kg, diseases_missing_in_kg, mondo_to_hpo_dict):
    # disease phenotype edges from orphanet
    orpha_disease_phenotype_edges = [(orphanet_to_mondo_dict[int(d)], p)  for (d,p) in orpha_disease_phenotype_edges if int(d) in orphanet_to_mondo_dict]
    orpha_disease_phenotype_edges = [(d, re.sub('HP:0*', '', p)) for d_list,p in orpha_disease_phenotype_edges for d in d_list]

    # disease phenotype edges from the KG
    phen_disease_edges = kg.loc[(kg['x_type'] == 'effect/phenotype') & (kg['y_type'] == 'disease')]
    disease_phen_edges = kg.loc[(kg['x_type'] == 'disease') & (kg['y_type'] == 'effect/phenotype')] # reverse relations

    assert len(phen_disease_edges.index) == 0
    kg_disease_phen_edges_tup = list(zip(disease_phen_edges['x_id'], disease_phen_edges['y_id']))
    kg_disease_phen_edges_tup = [(d,p) for d_list, p in kg_disease_phen_edges_tup for d in d_list.split('_')]
    
    # find missing edges
    missing_edges = set(orpha_disease_phenotype_edges).difference(set(kg_disease_phen_edges_tup))
    orphanet_phenotypes_missing_in_kg = [re.sub('HP:0*', '', p) for p in orphanet_phenotypes_missing_in_kg]
    edges_where_phenotype_not_in_kg = [(d,p) for (d,p) in missing_edges if p in orphanet_phenotypes_missing_in_kg]
    edges_where_disease_not_in_kg = [(d,p) for (d,p) in missing_edges if d in diseases_missing_in_kg]
    print(f'There are {len(missing_edges)} edges of {len(set(orpha_disease_phenotype_edges))} found in orphanet but not the KG.')
    print(f'There are {len(edges_where_phenotype_not_in_kg)} edges where the phenotype node is not in the KG.')
    print(f'There are {len(edges_where_disease_not_in_kg)} edges where the disease node is not in the KG.\n')

    # construct edges df
    filtered_missing_edges = [(d,p) for (d,p) in missing_edges if (p not in orphanet_phenotypes_missing_in_kg) and (d not in diseases_missing_in_kg)]
    disease_id = [d for (d,p) in filtered_missing_edges]
    disease_nodes = kg_nodes.query('node_type == "disease"')
    disease_nodes['node_id_list'] = disease_nodes['node_id'].str.split('_')

    phenotype_id = [p for (d,p) in filtered_missing_edges]
    phenotype_nodes = kg_nodes.query('node_type == "effect/phenotype"')

    disease_index = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_index'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_index'].tolist()[0] for d in disease_id]
    phenotype_index = [phenotype_nodes.loc[phenotype_nodes['node_id'] == p, 'node_index'].tolist()[0]  for p in phenotype_id]
    disease_name = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_name'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_name'].tolist()[0] for d in disease_id]
    phenotype_name = [phenotype_nodes.loc[phenotype_nodes['node_id'] == p, 'node_name'].tolist()[0] for p in phenotype_id]
    grouped_disease_id = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_id'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_id'].tolist()[0] for d in disease_id]

    disease_type = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_type'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_type'].tolist()[0] for d in disease_id]
    disease_source = [phenotype_nodes.loc[phenotype_nodes['node_id'] == str(mondo_to_hpo_dict[int(d)]), 'node_source'].tolist()[0] if int(d) in mondo_to_hpo_dict else disease_nodes.loc[disease_nodes['node_id'] == d, 'node_source'].tolist()[0] for d in disease_id]
    
    relations = ['phenotype_phenotype' if int(d) in mondo_to_hpo_dict else 'disease_phenotype_positive' for d in disease_id]
    display_relations = ['parent-child' if int(d) in mondo_to_hpo_dict else 'phenotype present' for d in disease_id]


    missing_edge_df = pd.DataFrame({'relation': relations, 'display_relation': display_relations,
        'x_index': disease_index,'x_id': grouped_disease_id,'x_type':  disease_type ,'x_name': disease_name, 'x_source':disease_source ,
        'y_index': phenotype_index,'y_id': phenotype_id, 'y_type':(['effect/phenotype'] * len(filtered_missing_edges))  ,'y_name': phenotype_name , 'y_source': (['HPO'] * len(filtered_missing_edges))})
    return missing_edge_df

def clean_edges(df): 
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.query('not ((x_id == y_id) and (x_type == y_type) and (x_source == y_source) and (x_name == y_name))')
    return df

def main():
    # Read in Orphanet Data
    orphanet_phenotypes = pd.read_csv(project_config.UDN_DATA / 'orphanet' / 'flat_files_2019-10-29' / 'orphanet_final_disease_hpo_normalized.tsv', sep='\t', dtype=str)
    orphanet_genes = pd.read_csv(project_config.UDN_DATA / 'orphanet' / 'flat_files_2019-10-29' / 'orphanet_final_disease_genes.tsv', sep='\t', dtype=str)
    orphanet_ensembl_genes = pd.read_csv(project_config.UDN_DATA / 'orphanet' / 'flat_files_2019-10-29' / 'orphanet_final_disease_genes_normalized.tsv', sep='\t', dtype=str)

    # read in mapping from old to current phenotypes
    hp_terms = pd.read_csv(project_config.KG_DIR / 'raw' / 'sources' / 'hpo' / 'hp_terms.csv')
    hp_map_dict = {'HP:' + ('0' * (7-len(str(int(hp_old))))) + str(int(hp_old)): 'HP:' + '0' * (7-len(str(int(hp_new)))) + str(int(hp_new)) for hp_old,hp_new in zip(hp_terms['id'], hp_terms['replacement_id'] ) if not pd.isnull(hp_new)}
    orphanet_phenotypes.replace({"HPO_ID": hp_map_dict}, inplace=True)

    orpha_phenotypes = orphanet_phenotypes['HPO_ID'].unique().tolist()
    orpha_diseases = orphanet_phenotypes['OrphaNumber'].unique().tolist() + orphanet_genes['OrphaNumber'].unique().tolist()
    orpha_gene_symbols = orphanet_genes['Gene_Symbol'].unique().tolist()
    orpha_ensembl_ids = orphanet_ensembl_genes['Ensembl_ID'].unique().tolist()

    orpha_disease_phenotype_edges = list(zip(orphanet_phenotypes['OrphaNumber'], orphanet_phenotypes['HPO_ID']))
    orpha_disease_gene_edges = list(zip(orphanet_genes['OrphaNumber'], orphanet_genes['Gene_Symbol']))

    # Read in Current KG    
    kg_raw = pd.read_csv(project_config.KG_DIR /'our_kg'/ 'auxillary' / 'kg_raw.csv', dtype=str,  low_memory=False) #no grouping of diseases or taking LCC
    
    # get all nodes from source & targets
    kg_raw_nodes_x = kg_raw[['x_id','x_type','x_name','x_source']]
    kg_raw_nodes_y = kg_raw[['y_id','y_type','y_name','y_source']]
    kg_raw_nodes_x.columns = [ 'node_id', 'node_type', 'node_name', 'node_source']
    kg_raw_nodes_y.columns = ['node_id', 'node_type', 'node_name', 'node_source']
    kg_raw_nodes = pd.concat([kg_raw_nodes_x, kg_raw_nodes_y]).drop_duplicates()

    # Manually add genes we know are missing from the KG: 
    # 'FMR3' is no longer an HGNC gene, even though it's associated with a disease still in orphanet: https://www.orpha.net/consor/cgi-bin/OC_Exp.php?Expert=100973&lng=EN
    # 'SCA32' is not actually a gene
    missing_genes = ['USH1E', 'SLC7A2-IT1', 'USH1K', 'USH1H', ] #'FMR3', 'SCA32'
    missing_gene_ids = ['7396', 'SLC7A2-IT1', '101180907', '100271837']# TODO: how to handle missing gene ids?
    assert len(set(missing_genes).intersection(set(kg_raw_nodes.loc[kg_raw_nodes['node_type'] == 'gene/protein', 'node_name']))) == 0
    missing_genes_df = pd.DataFrame({'node_id': missing_gene_ids, 'node_type': ['gene/protein'] * len(missing_gene_ids), 'node_name':missing_genes, 'node_source': ['Orphanet'] * len(missing_gene_ids)})
    # Manually add diseases we know are missing from the KG: 
    missing_diseases = ['Late-onset nephronophthisis']
    missing_disease_ids = ['19742']
    assert len(set(missing_disease_ids).intersection(set(kg_raw_nodes.loc[kg_raw_nodes['node_type'] == 'gene/protein', 'node_id']))) == 0
    missing_diseases_df = pd.DataFrame({'node_id': missing_disease_ids, 'node_type': ['disease'] * len(missing_disease_ids), 'node_name': missing_diseases, 'node_source': ['Orphanet'] * len(missing_disease_ids)})
    kg_raw_nodes = pd.concat([kg_raw_nodes, missing_genes_df, missing_diseases_df], ignore_index=True)

    # set node index
    kg_raw_nodes['node_index'] = list(range(len(kg_raw_nodes.index)))

    # read mondo definitions
    mondo_obo = obonet.read_obo('/home/ema30/zaklab/udn_data/mondo/mondo.obo') 
    mondo_definitions_obo_map = {node_id:node['name'] for node_id, node in list(mondo_obo.nodes(data=True)) if 'name' in node}
    mondo_definitions_obo_map = {re.sub('MONDO:0*', '', node_id):name for node_id, name in mondo_definitions_obo_map.items()}

    # read in orpha to mondo dict
    with open(str(project_config.PROJECT_DIR / 'orphanet_to_mondo_dict.pkl'), 'rb') as handle:
        orphanet_to_mondo_dict = pickle.load(handle)

    # read in mapping from mondo diseases to HPO phenotypes (this mapping occurs when a single entity is cross referenced by MONDO & HPO. In such cases we map to HPO)
    mondo2hpo = pd.read_csv(project_config.KG_DIR /'our_kg'/ 'auxillary' / 'mondo2hpo.csv')
    mondo_to_hpo_dict =  {mondo:hpo for hpo,mondo in zip(mondo2hpo['ontology_id'], mondo2hpo['mondo_id'])}

    # get last node idx
    print('Last node index: ', kg_raw_nodes['node_index'].iloc[-1])

    # identify which orphanet genes, phenotypes, diseases are missing from the KG
    orphanet_genes_missing_in_kg , genes_in_kg = find_missing_genes(kg_raw_nodes, orpha_gene_symbols)
    #kg_raw_nodes_ensembl = map_to_ensembl(kg_raw_nodes)     #map KG genes to ensembl IDs
    #orphanet_genes_missing_in_kg_ensembl = find_missing_genes(kg_raw_nodes_ensembl, orpha_ensembl_ids, use_ensembl=True)
    orphanet_phenotypes_missing_in_kg, phenotype_nodes = find_missing_phenotypes(kg_raw_nodes, orpha_phenotypes)
    diseases_missing_in_kg, orphanet_diseases_not_mapped_to_mondo, missing_disease_df, diseases_in_kg = find_missing_diseases(kg_raw_nodes, orpha_diseases, orphanet_to_mondo_dict, mondo_definitions_obo_map, mondo_to_hpo_dict, phenotype_nodes)
    
    print('\nmissing genes: ', orphanet_genes_missing_in_kg)
    #print('missing ensembl genes: ', orphanet_genes_missing_in_kg_ensembl)
    print('missing diseases', diseases_missing_in_kg)
    print('diseases that don\'t map to MONDO: ', orphanet_diseases_not_mapped_to_mondo, '\n')
    print('missing phenotypes', orphanet_phenotypes_missing_in_kg, '\n')

    # find missing edges #NOTE currently only gets edges if both nodes are in KG
    missing_dg_edge_df = find_missing_disease_gene_edges(orpha_disease_gene_edges, kg_raw, kg_raw_nodes,  orphanet_to_mondo_dict, orphanet_genes_missing_in_kg, diseases_missing_in_kg, mondo_to_hpo_dict)
    missing_dp_edge_df = find_missing_disease_phenotype_edges(orpha_disease_phenotype_edges, kg_raw, kg_raw_nodes, orphanet_to_mondo_dict, orphanet_phenotypes_missing_in_kg, diseases_missing_in_kg, mondo_to_hpo_dict)

    kg_added = pd.concat([kg_raw, missing_dg_edge_df, missing_dp_edge_df])
    kg_added = kg_added[['relation','display_relation','x_id','x_type','x_name','x_source','y_id','y_type','y_name','y_source']]
    kg_added = clean_edges(kg_added)

    print(f'There were {len(kg_raw.index)} edges in the original KG and {len(kg_added.index)} edges in the KG with orphanet edges.')
    kg_added.to_csv(project_config.KG_DIR /'our_kg'/ 'auxillary'/ 'kg_raw_orphanet.csv', index=False)


if __name__ == "__main__":
    main()
