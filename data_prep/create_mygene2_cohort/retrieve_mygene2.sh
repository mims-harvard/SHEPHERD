#!/bin/bash

#NOTE: The following script downloads genes from mygene2, one per file. These will be downloaded by default in whatever directory that this script is run in. You may need ot move the resulting files afterwards into project_config.PROJECT_ROOT/patients/mygene2_patients


#TODO change input path to the path to genes.csv output by retrieve_mygene2.py
INPUT=project_config.PROJECT_DIR/patients/mygene2_patients/genes.csv

[ ! -f $INPUT ] && { echo "$INPUT file not found"; exit 99; }
while IFS=, read -r gene 
do
    echo "$gene"
    wget https://www.mygene2.org/MyGene2/api/data/export/$gene
done < $INPUT
