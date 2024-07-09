import xml.etree.ElementTree as ET
from collections import OrderedDict
import pandas as pd
import sys

sys.path.insert(0, '..') # add project_config to path

import project_config

tree = ET.parse(str(project_config.PROJECT_DIR / 'data' / 'orphanet'/ 'raw'/ 'en_product1_5.8.2022.xml'))

root = tree.getroot()
disorderlist_cross = root[1]


rel_list = []
for disorder in disorderlist_cross:
    disorder_orphan = disorder.find("OrphaCode").text
    disorder_name = disorder.find("Name").text
    disorder_synonyms = "|".join([x.text
                                  for x in disorder.find("SynonymList")])
    for child in disorder.find("ExternalReferenceList"):
        external_source = child.find("Source").text
        external_id = child.find("Reference").text
        external_mapping_rel = child.find("DisorderMappingRelation").find("Name").text
        external_mapping_status = child.find("DisorderMappingValidationStatus").find("Name").text
        rel_list.append(OrderedDict([
            ("OrphaNumber", disorder_orphan),
            ("Disorder_Name", disorder_name),
            ("External_Source", external_source),
            ("External_ID", external_id),
            ("External_Mapping_Rel", external_mapping_rel),
            ("External_Mapping_Status", external_mapping_status),
            ("Disorder_Synonyms", disorder_synonyms)
        ]))
rel_df = pd.DataFrame(rel_list)
print(rel_df.columns)


orpha_to_omim_df = rel_df.loc[rel_df['External_Source'] == 'OMIM']
orpha_to_omim_df.to_csv(project_config.PROJECT_DIR / 'data' / 'orphanet'/ 'orphanet_to_omim_mapping_df.csv', index=False)


print(orpha_to_omim_df['External_Mapping_Status'].unique())
print(orpha_to_omim_df['External_Mapping_Rel'].unique())
orpha_to_omim_df_validated = orpha_to_omim_df.loc[orpha_to_omim_df['External_Mapping_Status'] == 'Validated']
orpha_to_omim_df_validated_exact = orpha_to_omim_df_validated.loc[orpha_to_omim_df_validated['External_Mapping_Rel'] == 'E (Exact mapping: the two concepts are equivalent)']

print(len(orpha_to_omim_df), len(orpha_to_omim_df_validated), len(orpha_to_omim_df_validated_exact))
#print(orpha_to_omim_df.loc[orpha_to_omim_df['External_Mapping_Status'] == 'Not yet validated', ['OrphaNumber', 'Disorder_Name', 'External_ID', 'External_Mapping_Rel']])
# print(orpha_to_omim_df.loc[['OrphaNumber', 'Disorder_Name', 'External_Source', 'External_ID',
#        'External_Mapping_Rel', 'External_Mapping_Status']])

