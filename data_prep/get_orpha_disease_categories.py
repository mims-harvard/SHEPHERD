
import pandas as pd
import xml.etree.ElementTree as ET
from pathlib import Path
import pickle as pkl
import sys
from collections import OrderedDict
sys.path.insert(0, '..') # add config to path
import project_config


##############################
# Get Orphanet Disease Categories Linearization

ORPHANET_RAW_PATH = project_config.PROJECT_DIR /'data'/ 'orphanet' / 'raw'

def get_linearized_categories():
    tree = ET.parse(str(ORPHANET_RAW_PATH / 'en_product7_5.8.2022.xml'))

    root = tree.getroot()
    disorderlist_linear = root[1]


    linear_list = []
    linear_list_cat = []
    debug_disorder = None
    for disorder in disorderlist_linear:
        print('\n-----')
        #print('disorder', disorder.tag, disorder.attrib, disorder.tail, disorder.keys(), disorder.items(), disorder.text)
        print(list(disorder))
        disorder_orphan = disorder.find("OrphaCode").text
        print('disorder_orphan', disorder_orphan)
        disorder_name = disorder.find("Name").text
        print('disorder_name', disorder_orphan)

        disorder_link = disorder.find("ExpertLink").text
        print('disorder_link', disorder_link)

        for child in disorder.find("DisorderDisorderAssociationList"):
            disorder1 = child.find("TargetDisorder")
            if "cycle" in disorder1.attrib:
                disorder1_orphan = disorder_orphan
                disorder1_name = disorder_name
            else:
                disorder1_orphan = disorder1.find("OrphaCode").text

                disorder1_name = disorder1.find("Name").text
            association_type = child.find("DisorderDisorderAssociationType").find("Name").text
            linear_list_cat.append(OrderedDict([
                ("OrphaNumber", disorder_orphan),
                ("Disorder_Name", disorder_name),
                ("Category", disorder1_name),
                ]))
            

    linear_cat_df = pd.DataFrame(linear_list_cat).drop_duplicates()
    linear_cat_df.to_csv(project_config.PROJECT_DIR / 'preprocess' / 'orphanet' / 'categorization_of_orphanet_diseases.csv')

def get_annotations(node, all_children):
    if node.find('ClassificationNodeChildList').attrib['count'] == 0:
        return all_children
    else:
        for node2 in node.find('ClassificationNodeChildList'):
            #print('Node2', list(node2))
            disorder = node2.find('Disorder').find('Name').text
            disorder_id = node2.find('Disorder').find('OrphaCode').text #node2.find('Disorder').attrib['id']
            disorder_type = node2.find('Disorder').find('DisorderType').find('Name').text
            #print(f'Disorder {disorder} with ID {disorder_id} of Type {disorder_type}'  )
            all_children.add((disorder, disorder_id, disorder_type))

            children = get_annotations(node2, all_children)
            all_children = all_children.union(children)
        return all_children


def get_all_categories():
    classifications_dir =  ORPHANET_RAW_PATH / 'rare_dx_classifications'
    orpha_id_to_classes = defaultdict(list)
    for path in classifications_dir.glob("*.xml"):
        print('Processing ', path)
        tree = ET.parse(str(path))
        root = tree.getroot()
        classificationlist = list(root[1])
        for d in classificationlist:
            print(list(d))
            classification = d.find('Name').text
            print('Classification: ', classification)
            classification_node = d.find('ClassificationNodeRootList')[0]
            category = classification_node.find('Disorder').find('Name').text
            print('Category: ', category)

            all_children = set()
            all_children = get_annotations(classification_node, all_children)
            print('all_children', len(all_children))
            for disorder,disorder_id, disorder_type in all_children:
                orpha_id_to_classes[disorder_id].append(category)
                
    print(orpha_id_to_classes)
    with open(project_config.PROJECT_DIR / 'preprocess' / 'orphanet' /'orphanet_id_to_classes_dict.pkl', 'wb') as handle:
        pkl.dump(orpha_id_to_classes, handle, protocol=pkl.HIGHEST_PROTOCOL)



def main():
    get_all_categories()
    get_linearized_categories()


if __name__ == "__main__":
    main()

