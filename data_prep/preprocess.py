
import sys
import obonet
import networkx
import re
import time
import numpy as np
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
sys.path.insert(0, '..') # add project_config to path
import project_config as config
import logging
import pickle
logging.basicConfig(level=logging.INFO)
#############
# SOURCES
# HGNC (DOWNLOAD) - https://www.genenames.org/download/custom/
# ENSEMBL (BIOMART) - http://useast.ensembl.org/biomart/martview/b11081b5e1087424087a575e2792953e
    # biomart_ensembl_ncbi_map
# NCBI (FTP) - https://ftp.ncbi.nih.gov/gene/DATA/
    # gene info
    # gene history
    # gene2refseq
    # gene2ensembl



class Preprocessor():

    def write_pkl(self, d, filename):
        file_path = Path(config.PREPROCESS_PATH / filename)
        with open(str(file_path), 'wb') as f: 
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pkl(self, filename):
        file_path = Path(config.PREPROCESS_PATH / filename)
        with open(str(file_path), 'rb') as f: 
            return pickle.load(f)


    def read_hgnc_mappings(self):
        # Read in hgnc data
        hgnc = pd.read_csv(config.HGNC_PATH / "hgnc_complete_set-12-31-19.tsv", sep='\t', dtype=str)
        # hgnc = hgnc.query("symbol != 'LINC00856'")
        hgnc = hgnc[hgnc.symbol.notnull()]
        hgnc.loc[hgnc.alias_symbol == 'WISP-3', 'alias_symbol'] = 'WISP3' #other sources don't have a dash in the gene name
        self.hgnc = hgnc

        self.hgnc_biomart = pd.read_csv(config.HGNC_PATH / "hgnc_biomart_hgnc_ensembl_ncbi.txt", sep='\t', dtype=str).drop(columns=['Previous name']).drop_duplicates()


        if Path(config.PREPROCESS_PATH / 'prev_to_curr_dict.pkl').exists():
            self.prev_to_curr_dict = self.load_pkl('prev_to_curr_dict.pkl')
            self.curr_to_prev_dict = self.load_pkl('curr_to_prev_dict.pkl')
            self.alias_to_curr_dict = self.load_pkl('alias_to_curr_dict.pkl')
            self.curr_to_alias_dict = self.load_pkl('curr_to_alias_dict.pkl')
        else:
            alias_genes = hgnc['alias_symbol'].str.split('|').tolist()
            hgnc_no_null_prev = hgnc.loc[~pd.isnull(hgnc.prev_symbol)]
            # map from previous gene symbol to current gene symbol
            self.prev_to_curr_dict = {p:symbol for symbol, prev_symbols in zip(hgnc_no_null_prev.symbol, hgnc_no_null_prev.prev_symbol) for p in prev_symbols.split('|')}
            self.write_pkl(self.prev_to_curr_dict, 'prev_to_curr_dict.pkl')
            # map from current gene symbol to previous gene symbol
            self.curr_to_prev_dict = {symbol:prev_symbols.split('|') for symbol, prev_symbols in zip(hgnc_no_null_prev.symbol, hgnc_no_null_prev.prev_symbol) }
            self.write_pkl(self.curr_to_prev_dict, 'curr_to_prev_dict.pkl')

            hgnc_no_null_alias = hgnc.loc[~pd.isnull(hgnc.alias_symbol)]
            # map from alias gene symbol to current gene symbol
            self.alias_to_curr_dict = {a:symbol for symbol, alias_symbols in zip(hgnc_no_null_alias.symbol, hgnc_no_null_alias.alias_symbol) for a in alias_symbols.split('|')}
            self.write_pkl(self.alias_to_curr_dict, 'alias_to_curr_dict.pkl')
            # map from current gene symbol to alias gene symbol
            self.curr_to_alias_dict = {symbol:alias_symbols.split('|') for symbol, alias_symbols in zip(hgnc_no_null_alias.symbol, hgnc_no_null_alias.alias_symbol) }
            self.write_pkl(self.curr_to_alias_dict, 'curr_to_alias_dict.pkl')
     
    def read_shilpa_mappings(self):
        self.ensembl_to_hgnc_shilpa = pd.read_csv(config.HGNC_PATH / "GRCh38_ensembl_gene_list_from_shilpa.tsv", sep='\t', skiprows=4, dtype=str)
        if Path(config.PREPROCESS_PATH / 'synonym_to_ensembl_dict.pkl').exists():
            self.synonym_to_ensembl_dict = self.load_pkl('synonym_to_ensembl_dict.pkl')
        else:            
            # map gene synonyms to ensembl ids using Shilpa's curated mapping
            synonym_genes = self.ensembl_to_hgnc_shilpa['gene_synonyms'].str.split(',').tolist()
            ensembl_ids = self.ensembl_to_hgnc_shilpa['ensembl_gene_id'].tolist() 
            self.synonym_to_ensembl_dict =  {s.split('.')[0]:ensembl for synonyms, ensembl in zip(synonym_genes, ensembl_ids) for s in synonyms}
            self.write_pkl(self.synonym_to_ensembl_dict, 'synonym_to_ensembl_dict.pkl')

    def read_biomart_mappings(self):
        #hgnc-ensembl-ncbi mappings from Ensembl Biomart
        if Path(config.BIOMART_PATH / 'combined_ensembl_biomart.txt').exists():
            self.ensembl_biomart_mapping = pd.read_csv(config.BIOMART_PATH / 'combined_ensembl_biomart.txt', sep='\t', dtype=str)
            #self.entrez_to_ensembl_dict = self.load_pkl('entrez_to_ensembl_dict.pkl')

        else:
            ensembl_to_hgnc = pd.read_csv(config.HGNC_PATH / "ensembl_to_hgnc_map.txt", sep='\t').drop(columns=['HGNC ID', 'EntrezGene ID']) #unknown origin
            ensembl_to_hgnc['NCBI gene ID'] = None
            biomart_mapping = pd.read_csv(config.BIOMART_PATH / 'biomart_ensembl_genesymb_entrez.txt', delimiter = '\t', dtype=str)
            biomart_mapping = biomart_mapping.drop(columns=['Transcript stable ID', 'Transcript stable ID version']).drop_duplicates()
            
            # combine two mappings
            all_biomart_mapping = pd.concat([ensembl_to_hgnc, biomart_mapping], sort=False).drop(columns=['Gene stable ID version'])
            all_biomart_mapping = all_biomart_mapping.drop_duplicates()
            self.ensembl_biomart_mapping = all_biomart_mapping.sort_values(['NCBI gene ID'], na_position='last').drop_duplicates(subset=['Gene stable ID', 'Gene name', 'HGNC symbol'], keep='first')
            self.ensembl_biomart_mapping['NCBI gene ID'] = self.ensembl_biomart_mapping['NCBI gene ID'].astype(str)
            self.ensembl_biomart_mapping.to_csv(config.BIOMART_PATH / 'combined_ensembl_biomart.txt', sep='\t', index=False)

            # entrez -> ensembl mapping from biomart
            # biomart_mapping_no_null = self.ensembl_biomart_mapping.loc[~pd.isnull(self.ensembl_biomart_mapping['NCBI gene ID'])]
            # self.entrez_to_ensembl_dict = {entrez:ensembl for entrez, ensembl in  zip(biomart_mapping_no_null['NCBI gene ID'], biomart_mapping_no_null['Gene stable ID'])}   
            # self.write_pkl(self.entrez_to_ensembl_dict, 'entrez_to_ensembl_dict.pkl')

    def read_ncbi_mappings(self):
        self.gene2ensembl = pd.read_csv(config.NCBI_PATH / 'gene2ensembl', sep='\t', dtype='str')


        if Path(config.PREPROCESS_PATH / 'disc_symbol_to_geneid.pkl').exists():
            self.disc_symbol_to_geneid = self.load_pkl('disc_symbol_to_geneid.pkl')

            self.disc_geneid_to_geneid = self.load_pkl('disc_geneid_to_geneid.pkl')

            self.synonym_to_entrez_symbol_dict = self.load_pkl('synonym_to_entrez_symbol_dict.pkl')
            self.geneid_to_entrez_symbol_dict = self.load_pkl('geneid_to_entrez_symbol_dict.pkl')
            self.entrez_symbol_to_geneid_dict = self.load_pkl('entrez_symbol_to_geneid_dict.pkl')
            self.refseq_geneid_to_symbol_map = self.load_pkl('refseq_geneid_to_symbol_map.pkl')
            self.refseq_symbol_to_geneid_map = self.load_pkl('refseq_symbol_to_geneid_map.pkl')

        else:
            # NCBI GENEID HISTORY
            gene_history = pd.read_csv(config.NCBI_PATH / 'gene_history', sep='\t', dtype='str')
            # map from discontinued symbol to current NCBI gene ID
            self.disc_symbol_to_geneid = {disc_symbol:geneid for disc_symbol, geneid in zip(gene_history['Discontinued_Symbol'],gene_history['GeneID']) if geneid != '-'}
            self.write_pkl(self.disc_symbol_to_geneid, 'disc_symbol_to_geneid.pkl')
            # map from discontinued NCBI gene ID to current NCBI gene ID
            self.disc_geneid_to_geneid = {disc_geneid:geneid for disc_geneid, geneid in zip(gene_history['Discontinued_GeneID'],gene_history['GeneID'] ) if geneid != '-'}
            self.write_pkl(self.disc_geneid_to_geneid, 'disc_geneid_to_geneid.pkl')


            # GENE INFO
            # map from synonym to gene symbol
            self.gene_info = pd.read_csv(config.NCBI_PATH / 'gene_info', sep='\t', dtype = 'str')
            gene_info_with_synonyms = self.gene_info.loc[~pd.isnull(self.gene_info.Synonyms)]
            self.synonym_to_entrez_symbol_dict = {s:symbol for symbol, synonyms in zip(gene_info_with_synonyms.Symbol, gene_info_with_synonyms.Synonyms) for s in synonyms.split('|')}
            self.write_pkl(self.synonym_to_entrez_symbol_dict, 'synonym_to_entrez_symbol_dict.pkl')

            # map from NCBI gene ID to Gene Symbol
            self.geneid_to_entrez_symbol_dict = {geneid:symbol for geneid, symbol in zip(self.gene_info.GeneID, self.gene_info.Symbol) }
            self.write_pkl(self.geneid_to_entrez_symbol_dict, 'geneid_to_entrez_symbol_dict.pkl')

            self.entrez_symbol_to_geneid_dict = {symbol:geneid for geneid, symbol in zip(gene_info.GeneID, gene_info.Symbol) }
            self.write_pkl(self.entrez_symbol_to_geneid_dict, 'entrez_symbol_to_geneid_dict.pkl')

            # # GENE2 REFSEQ
            gene2refseq = pd.read_csv(config.NCBI_PATH / 'gene2refseq', sep='\t', dtype='str')
            # map from NCBI gene id to symbol
            self.refseq_geneid_to_symbol_map = {gene_id:symbol for gene_id, symbol in zip(gene2refseq['GeneID'], gene2refseq['Symbol'])}
            self.write_pkl(self.refseq_geneid_to_symbol_map, 'refseq_geneid_to_symbol_map.pkl')

            # map from symbol to NCBI gene ID
            self.refseq_symbol_to_geneid_map = {symbol:gene_id for gene_id, symbol in zip(gene2refseq['GeneID'], gene2refseq['Symbol'])}
            self.write_pkl(self.refseq_symbol_to_geneid_map, 'refseq_symbol_to_geneid_map.pkl')
          
    def get_unique_ensembl_ids(self):
        ensembl_ids = self.hgnc['ensembl_gene_id'].tolist() + self.hgnc_biomart['Ensembl gene ID'].tolist() \
            + self.ensembl_biomart_mapping['Gene stable ID'].tolist() \
            + self.ensembl_to_hgnc_shilpa['ensembl_gene_id'].tolist()
        ensembl_ids = set(ensembl_ids)
        return [e for e in ensembl_ids if not pd.isnull(e)]

    def get_gene_symbols(self):
        symbols = self.hgnc['symbol'].unique().tolist() + self.hgnc_biomart['Approved symbol'].tolist()
        symbols = set(symbols)
        return [s for s in symbols if not pd.isnull(s)]

    def get_ncbi_ids(self):
        #TODO: make sure this works
        if Path(config.NCBI_PATH / 'all_ncbi_ids.txt').exists():
            ncbi = pd.read_csv(config.NCBI_PATH / 'all_ncbi_ids.txt', sep='\t', dtype=str)
            
        else:
            ncbi = self.hgnc_biomart['NCBI gene ID'].tolist() + self.ensembl_biomart_mapping['NCBI gene ID'].tolist() + self.gene_info['GeneID'].tolist()
            ncbi = set(ncbi)
            ncbi = pd.DataFrame({'ids':[s for s in ncbi if not pd.isnull(s)]})
            ncbi['ids'] = ncbi['ids'].astype(str)
            ncbi.to_csv(config.NCBI_PATH / 'all_ncbi_ids.txt', sep='\t', index=False)
        return ncbi['ids'].tolist()


    def __init__(self, process_phenotypes=True, process_genes=True, hpo_source_year='2019'):
        if process_phenotypes:
            # read in hpo hierarchy
            HPO_2015_DIR = config.UDN_DATA / 'hpo' / 'Jan_2015' 
            HPO_2019_DIR = config.UDN_DATA / 'hpo' / '2019' 
            HPO_2020_DIR = config.UDN_DATA / 'hpo' / '8_2020' 
            HPO_2021_DIR = config.UDN_DATA / 'hpo' / '2021' 

            self.hpo_2019 = obonet.read_obo(HPO_2019_DIR / 'hp.obo') 
            self.hpo_2015 = obonet.read_obo(HPO_2015_DIR / 'hp.obo') 
            self.hpo_2020 = obonet.read_obo(HPO_2020_DIR / 'hp.obo') 
            self.hpo_2021 = obonet.read_obo(HPO_2021_DIR / 'hp.obo') 

            self.hpo_source_year = hpo_source_year
            if hpo_source_year == '2019': self.hpo_ontology = self.hpo_2019
            elif hpo_source_year == '2020': self.hpo_ontology = self.hpo_2020
            elif hpo_source_year == '2021': self.hpo_ontology = self.hpo_2021
            else: raise NotImplementedError

            # create dictionary mapping alternate HPO IDs to current HPO IDs
            alt_to_curr_hpo_path = Path(config.PREPROCESS_PATH / f'alt_to_curr_hpo_dict_{hpo_source_year}.pkl')
            if alt_to_curr_hpo_path.exists():
                with open(str(alt_to_curr_hpo_path), 'rb') as f:
                    self.alt_to_curr_hpo_dict = pickle.load(f)
            else:
                self.alt_to_curr_hpo_dict = {}
                for node_id in self.hpo_ontology.nodes():
                    node_dict = self.hpo_ontology.nodes[node_id]
                    if 'alt_id' in node_dict:
                        for alt_id in node_dict['alt_id']:
                            self.alt_to_curr_hpo_dict[alt_id] = node_id
                with open(str(alt_to_curr_hpo_path), 'wb') as f:
                    pickle.dump(self.alt_to_curr_hpo_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                    


            #map from outdated HPO codes to codes in 2019 hierarchy
            # Not in the 2015 or 2019 HPO
            # HP:0500014 (Abnormal test result) -> None, throw out
            # HP:0040290 (Abnormality of skeletal muscles) -> 'HP:0003011'
            # HP:0030963 (obsolete Abnormal aortic morphology) -> 'HP:0001679'
            # HP:0030971 (obsolete Abnormal vena cava morphology) -> 'HP:0005345'

            # in 2015 HPO, but not 2019. Parent is also not in 2019
            # HP:0005111 (Dilatation of the ascending aorta) -> 'HP:0005128'
            # HP:0001146 (Pigmentary retinal degeneration) -> 'HP:0000580'
            # 'HP:0007757' (choroid hypoplasia) -> 'HP:0000610' (Abnormal choroid morphology)
            # HP:0009620 (obsolete Radial deviation of the thumb) -> HP:0040021 (Radial deviation of the thumb)
            # HP:0009448 (obsolete Aplasia of the phalanges of the 3rd finger) -> HP:0009447 (Aplasia/Hypoplasia of the phalanges of the 3rd finger)
            # HP:0004110 (obsolete Radially deviated index finger phalanges) -> HP:0009467 (Radial deviation of the 2nd finger)
            # HP:3000026 obsolete Abnormality of common carotid artery plus branches -> HP:0430021 Abnormal common carotid artery morphology 
            self.OLD_TO_2019_HPO_DICT = {'HP:0040290':'HP:0003011', 'HP:0030963':'HP:0001679', 
                        'HP:0030971':'HP:0005345', 'HP:0005111':'HP:0004970',
                        'HP:0001146':'HP:0000580', 'HP:0100637':'HP:0012720', 
                        'HP:0007757':'HP:0000610', 'HP:0009620': 'HP:0040021',
                        'HP:0009448':'HP:0009447', 'HP:0004110': 'HP:0009467',
                        'HP:3000026': 'HP:0430021'}

        if process_genes:
            logging.info('Initializing Gene Mappings....')

            # manually created from gene cards
            self.genecards_alias_dict = {'RARS':'RARS1', 'C11ORF80':'C11orf80', 'KIF1BP':'KIFBP', 'ICK':'CILK1', 'AARS':'AARS1', 'SPG23':'DSTYK', 
                            'C9ORF72':'C9orf72', 'C8ORF37':'C8orf37', 'HARS':'HARS1', 'GARS':'GARS1', 'C19ORF12':'C19orf12', 
                            'C12ORF57':'C12orf57', 'ADSSL1':'ADSS1', 'C12ORF65':'C12orf65', 'MARS':'MARS1', 'CXORF56':'STEEP1', 
                            'SARS':'SARS1', 'C12ORF4':'C12orf4', 'MUT':'MMUT', 'LOR':'LORICRIN'}

            self.miscapitalized_gene_dict = {'C10ORF10':'C10orf10','C21ORF59':'C21orf59', 'C4ORF26':'C4orf26', 'C9ORF72':'C9orf72', 'C8ORF37':'C8orf37', 'CXORF56':'CXorf56', 'C12ORF4':'C12orf4', 'C2ORF43':'C2orf43',
                            'C11ORF80':'C11orf80', 'C19ORF12': 'C19orf12', 'C12ORF57': 'C12orf57', 'C17ORF96':'C17orf96', 'C19ORF68':'C19orf68', 'C19ORF10':'C19orf10', 'C9ORF3':'C9orf3', 'C11ORF73':'C11orf73',
                            'C12ORF65':'C12orf65', 'C5ORF42': 'C5orf42', 'C2ORF71': 'C2orf71', 'C10ORF2': 'C10orf2', 'mTOR': 'MTOR', 'C12ORF55':'C12orf55', 'C19ORF18':'C19orf18', 'C12ORF66':'C12orf66', 'C5ORF63':'C5orf63',
                            'C11ORF95': 'C11orf95', 'C15ORF41': 'C15orf41', 'C16ORF57': 'C16orf57', 'C20orff78': 'C20orf78', 'trnM(cau)': 'trnM-CAU', 'C10ORF82':'C10orf82', 'CXORF27':'CXorf27'}
            self.misc_gene_to_ensembl_map = {'RP13-996F3.4':'ENSG00000259243', 'NRXN1':'ENSG00000179915', 'LOC401010':'ENSG00000217950',
                            'TXNB': 'ENSG00000168477', 'TREF1': 'ENSG00000124496', 'DLCRE1C': 'ENSG00000152457',
                            'CDC37L1C':'ENSG00000106993', 'UNQ2560':'ENSG00000145476', 'COLA11A2':'ENSG00000204248',
                            'SL12A2':'ENSG00000064651', 'TNFRS1B': 'ENSG00000028137', 'HUWE': 'ENSG00000086758', 
                            'PARRC2C': 'ENSG00000117523', 'AK055785':'ENSG00000153443', 'TRNC18':'ENSG00000182095',
                            'ZYFVE26':'ENSG00000072121', 'DPMI':'ENSG00000000419', 'SRD53A':'ENSG00000128039',
                            'VLDR':'ENSG00000147852', 'MENT1':'ENSG00000112759', 'DRPL-A':'ENSG00000111676',
                            'FRDA1':'ENSG00000165060', 'A1640G':'ENSG00000163554', 'GS1-541M1.1':'ENSG00000220216',
                            'MT-TT':'ENSG00000210195', '4576': 'ENSG00000210195', 'ZMYDN12': 'ENSG00000066185', 'NRNX1':'ENSG00000179915',
                            'NPNP4': 'ENSG00000131697', 'STEEP1': 'ENSG00000018610', 'ENSG00000288642':'ENSG00000288642'
                            } 
            logging.info('Reading hgnc mappings....')
            self.read_hgnc_mappings()
            logging.info('Reading biomart mappings....')
            self.read_biomart_mappings()
            logging.info('Reading shilpa mappings....')
            self.read_shilpa_mappings()
            logging.info('Reading ncbi mappings....')
            self.read_ncbi_mappings()


            # # get list of withdrawn hgnc symbols
            self.withdrawn_hgnc = pd.read_csv(config.HGNC_PATH / "withdrawn-12-31-19.tsv", sep='\t')

            logging.info('Retrieving unique lists of genes....')

            self.all_unique_ensembl_ids = self.get_unique_ensembl_ids()
            self.all_gene_symbols = self.get_gene_symbols()
            self.all_ncbi_ids = self.get_ncbi_ids()
          
            # remove entries that map to NULL
            self.hgnc = self.hgnc.loc[~pd.isnull(self.hgnc['ensembl_gene_id'])]
            self.ensembl_biomart_mapping = self.ensembl_biomart_mapping.loc[~pd.isnull(self.ensembl_biomart_mapping['Gene stable ID'])]
            self.ensembl_to_hgnc_shilpa = self.ensembl_to_hgnc_shilpa.loc[~pd.isnull(self.ensembl_to_hgnc_shilpa['ensembl_gene_id'])]
            self.hgnc_biomart = self.hgnc_biomart.loc[~pd.isnull(self.hgnc_biomart['Ensembl gene ID'])]
            self.gene2ensembl = self.gene2ensembl.loc[~pd.isnull(self.gene2ensembl['Ensembl_gene_identifier'])]

    def map_phenotype_to_hpo(self, hpo_id) -> str:
        '''
        HP:0500014 : abnormal test result -> can't map to anything

        '''
        # if the hpo is already in 2019 HPO, return it
        if hpo_id in self.hpo_ontology.nodes():
            return hpo_id
        # else map alternate hpo ids to the current version
        elif hpo_id in self.alt_to_curr_hpo_dict:
            return self.alt_to_curr_hpo_dict[hpo_id]
        elif hpo_id in self.OLD_TO_2019_HPO_DICT: #TODO: generalize beyond 2019 hpo dict
            return self.OLD_TO_2019_HPO_DICT[hpo_id]
        
        # if all else fails, map the hpo to its parents & return that
        else:
            for ontology in [self.hpo_2015, self.hpo_2019, self.hpo_2020, self.hpo_2021]:
                if hpo_id in ontology.nodes():
                    ancestors  = list(ontology.successors(hpo_id))
                    for ancestor in ancestors:
                        if ancestor in self.hpo_ontology.nodes():
                            return ancestor
                    logging.error(f'{hpo_id} can not be converted to its equivalent in the 2019 HPO hierarchy')
                    return hpo_id

            logging.error(f'{hpo_id} can not be converted to its equivalent in the 2019 HPO hierarchy')
            return hpo_id

    def convert_gene_to_ensembl_helper(self, g, log=False):
        #convert gene symbols to ensembl IDs



        if g in self.all_unique_ensembl_ids:
            return g

        # HGNC
        if g in self.hgnc['symbol'].tolist():
            return self.hgnc.loc[self.hgnc['symbol'] == g, 'ensembl_gene_id'].iloc[0]
        if g in self.hgnc['entrez_id'].tolist():
            return self.hgnc.loc[self.hgnc['entrez_id'] == g, 'ensembl_gene_id'].iloc[0]

        # ENSEMBL BIOMART
        if g in self.ensembl_biomart_mapping['HGNC symbol'].tolist():
            return self.ensembl_biomart_mapping.loc[self.ensembl_biomart_mapping['HGNC symbol'] == g, 'Gene stable ID'].iloc[0]
        if g in self.ensembl_biomart_mapping['Gene name'].tolist():
            return self.ensembl_biomart_mapping.loc[self.ensembl_biomart_mapping['Gene name'] == g, 'Gene stable ID'].iloc[0]
        if g in self.ensembl_biomart_mapping['NCBI gene ID'].tolist():
            return self.ensembl_biomart_mapping.loc[self.ensembl_biomart_mapping['NCBI gene ID'] == g, 'Gene stable ID'].iloc[0]   
        
        # SHILPA
        if g in self.ensembl_to_hgnc_shilpa['primary_gene_names'].tolist():
            return self.ensembl_to_hgnc_shilpa.loc[self.ensembl_to_hgnc_shilpa['primary_gene_names'] == g, 'ensembl_gene_id'].iloc[0]

        # HGNC BIOMART
        if g in self.hgnc_biomart['Approved symbol'].tolist():
            return self.hgnc_biomart.loc[self.hgnc_biomart['Approved symbol'] == g, 'Ensembl gene ID'].iloc[0]
        if g in self.hgnc_biomart['Alias symbol'].tolist():
            return self.hgnc_biomart.loc[self.hgnc_biomart['Alias symbol'] == g, 'Ensembl gene ID'].iloc[0]
        if g in self.hgnc_biomart['Previous symbol'].tolist():
            return self.hgnc_biomart.loc[self.hgnc_biomart['Previous symbol'] == g, 'Ensembl gene ID'].iloc[0]
        if g in self.hgnc_biomart['NCBI gene ID'].tolist():
            return self.hgnc_biomart.loc[self.hgnc_biomart['NCBI gene ID'] == g, 'Ensembl gene ID'].iloc[0]

        # Shilpa's mapping from synonyms to Ensembl IDs
        if g in self.synonym_to_ensembl_dict:
            return self.synonym_to_ensembl_dict[g]

        if g in self.gene2ensembl['GeneID'].tolist(): #NCBI gene ID to Ensembl ID
            return self.gene2ensembl.loc[self.gene2ensembl['GeneID'] == g,'Ensembl_gene_identifier'].iloc[0]

        if g in self.misc_gene_to_ensembl_map:
            return self.misc_gene_to_ensembl_map[g]
        else:
            return None

    def map_gene_to_ensembl_id(self, g, log=False):
        # check if null or empty string
        if pd.isnull(g) or g == '':
            return None, 'did not map' 

        # special case where two ensembl IDs are synonymous
        if g == 'ENSG00000262919':
            return 'ENSG00000147382', 'mapped to ensembl'

        # TODO: how do we know the ensembl IDs themselves aren't expired / out of date
        # check if already is an ensembl ID
        if g in self.all_unique_ensembl_ids:
            return g, 'mapped to ensembl'

        if g in self.misc_gene_to_ensembl_map:
            return self.misc_gene_to_ensembl_map[g], 'mapped to ensembl'

        # convert to hgnc alias
        if g in self.genecards_alias_dict:
            g = self.genecards_alias_dict[g]

        # fix miscapitalization issues
        if g in self.miscapitalized_gene_dict:
            g = self.miscapitalized_gene_dict[g]

        

        g = re.sub(r'([0-9]|X)(ORF)([0-9])', r'\1orf\3', g)

        
        # first look if the original gene name has a mapping to ensembl IDs
        new_g = self.convert_gene_to_ensembl_helper(g, log)
        
        # if not, convert previous/alias symbols to their current ones & check if that has mapping to ensembl ID
        if g in self.prev_to_curr_dict and not new_g:
            new_g =  self.convert_gene_to_ensembl_helper(self.prev_to_curr_dict[g], log)
        if g in self.alias_to_curr_dict and not new_g:
            new_g =  self.convert_gene_to_ensembl_helper(self.alias_to_curr_dict[g], log)

        # then look if any alias has a mapping to an Ensembl ID
        if g in self.curr_to_alias_dict and not new_g:
            for alias in self.curr_to_alias_dict[g]:
                new_g = self.convert_gene_to_ensembl_helper(alias, log)
                if new_g: break
        
        # finally look if a previous symbol has a mapping to an Ensembl ID
        if g in self.curr_to_prev_dict and not new_g:
            for prev in self.curr_to_prev_dict[g]:
                new_g = self.convert_gene_to_ensembl_helper(prev, log)
                if new_g: break

        # from NCBI: "Symbols beginning with LOC. When a published symbol 
        # is not available, and orthologs have not yet been determined, 
        # Gene will provide a symbol that is constructed as 'LOC' + the GeneID. 
        # This is not retained when a replacement symbol has been identified, 
        # although queries by the LOC term are still supported. In other words, 
        # a record with the symbol LOC12345 is equivalent to GeneID = 12345. 
        # So if the symbol changes, the record can still be retrieved on the web 
        # using LOC12345 as a query, or from any file using GeneID = 12345."
        if g.startswith('LOC') and not new_g:
            new_g = self.convert_gene_to_ensembl_helper(g.replace('LOC', ''), log)

        # map discontinued entrez symbols to ensembl
        if g in self.disc_symbol_to_geneid and not new_g: 
            new_g =  self.convert_gene_to_ensembl_helper(self.disc_symbol_to_geneid[g], log)

        # map discontinued entrez geneIDs to ensembl
        if g in self.disc_geneid_to_geneid and not new_g: 
            new_g =  self.convert_gene_to_ensembl_helper(self.disc_geneid_to_geneid[g], log)
    
        # # convert entrez gene id to symbol & see if there's a mapping to ensembl ID
        if g in self.geneid_to_entrez_symbol_dict and not new_g:
            new_g =  self.convert_gene_to_ensembl_helper(self.geneid_to_entrez_symbol_dict[g], log)
        if g in self.refseq_geneid_to_symbol_map and not new_g:
            new_g =  self.convert_gene_to_ensembl_helper(self.refseq_geneid_to_symbol_map[g], log)
        # # convert entrez symbol to entrez gene id & see if there's a mapping to ensembl ID
        if g in self.entrez_symbol_to_geneid_dict and not new_g:
            new_g =  self.convert_gene_to_ensembl_helper(self.entrez_symbol_to_geneid_dict[g], log)
        if g in self.refseq_symbol_to_geneid_map and not new_g:
            new_g =  self.convert_gene_to_ensembl_helper(self.refseq_symbol_to_geneid_map[g], log)


        # convert gene synonym to most common name (according to entrez gene_info)
        if g in self.synonym_to_entrez_symbol_dict and not new_g:
            new_g =  self.convert_gene_to_ensembl_helper(self.synonym_to_entrez_symbol_dict[g], log)

        if new_g:
            g = new_g

        if g == 'ENSG00000262919':
            return 'ENSG00000147382', 'mapped to ensembl'


        if g in self.all_unique_ensembl_ids:
            return g, 'mapped to ensembl'
        
        else: # we return the original gene (mapped to a current symbol) if we're not able to convert it

            if not pd.isnull(g) and g.startswith('LOC'): g = g.replace('LOC', '')
            if g in self.prev_to_curr_dict: g = self.prev_to_curr_dict[g]
            if g in self.alias_to_curr_dict: g =  self.alias_to_curr_dict[g]

            if g in self.disc_symbol_to_geneid: g = self.disc_symbol_to_geneid[g]
            if g in self.disc_geneid_to_geneid: g = self.disc_geneid_to_geneid[g]
            if g in self.entrez_symbol_to_geneid_dict: g = self.entrez_symbol_to_geneid_dict[g]

            if g in self.synonym_to_entrez_symbol_dict: g = self.synonym_to_entrez_symbol_dict[g]
            if not pd.isnull(g) and g.startswith('LOC'): g = g.replace('LOC', '')

        
            if g in self.all_gene_symbols:
                # if log:
                #     logging.error(f'There is likely a gene symbol, but no Ensembl ID for gene: {g}')
                return g, 'mapped to symbol/ncbi'
            # elif g in self.withdrawn_hgnc['WITHDRAWN_SYMBOL'].tolist():
            #     logging.error(f'The following gene symbol has been withdrawn: {g}')
            elif g in self.all_ncbi_ids:
                # if log:
                #     logging.error(f'There is likely a NCBI gene ID, but no Ensembl ID for gene: {g}')
                return g, 'mapped to symbol/ncbi'
            else:
                if log:
                    logging.error(f'The following gene can not be converted to an Ensembl ID: {g}')
                return g, 'did not map'

    def map_phenotypes(self, df, col_name = 'HPO_ID'):
        hpo_map = {p:self.map_phenotype_to_hpo(p) for p in df[col_name].unique()}
        df['standardized_hpo_id'] = df[col_name].replace(hpo_map) #TODO: generalize this so it's not always 2019
        assert len(df.loc[pd.isnull(df['standardized_hpo_id'])].index) == 0
        return df

    def map_genes(self, df, col_names, log=True, n_processes=4):
        genes_to_map = list(set([g for col_name in col_names for g in df[col_name].tolist()]))
        #genes_to_map = genes_to_map[0:100] #TODO: delete - just for testing
        #t0 = time.time()
        gene_to_ensembl_map = {}
        gene_to_status_map = {}
        for g in genes_to_map:
            ensembl_id, mapped_status = self.map_gene_to_ensembl_id(g, log=log)
            gene_to_ensembl_map[g] = ensembl_id
            gene_to_status_map[g] = mapped_status
        #gene_to_ensembl_map = {g:self.map_gene_to_ensembl_id(g, log=log)[0] for g in genes_to_map}
        #t1 = time.time()
        # with Pool(processes=n_processes) as pool: 
        #     ensembl_id, mapped_status = pool.map(self.map_gene_to_ensembl_id, genes_to_map)
        #vectorized_map_to_ensembl_id = np.vectorize(self.map_gene_to_ensembl_id)
        #ensembl_id, mapped_status = vectorized_map_to_ensembl_id(genes_to_map)
        #t2 = time.time()
        #print(f'It took {t1-t0:04f}s to do a for loop and {t2-t1:0.4f}s to vectorize.')
        
        for col_name in col_names:
            df[(col_name + '_ensembl')] = df[col_name].replace(gene_to_ensembl_map)
            df[(col_name + '_mapping_status')] = df[col_name].replace(gene_to_status_map)
        return df



if __name__ == "__main__":
    pass