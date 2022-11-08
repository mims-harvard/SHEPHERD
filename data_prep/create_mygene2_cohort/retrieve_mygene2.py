
import requests
from bs4 import BeautifulSoup
import codecs
import re
import pandas as pd

import sys
sys.path.insert(0, '../') # add project_config to path
import project_config
import urllib.request

'''
Get list of genes found at https://www.mygene2.org/MyGene2/genes

extracts genes in the format api/data/export/SIN3A
'''

f=codecs.open("mygene2.html", 'r')
html_content = f.read()


soup = BeautifulSoup(html_content)

genes = []
for link in soup.find_all("a"):
    href = link.get("href")
    if href:
        matches = re.findall(r'api/data/export/(.*)', href)
        for match in matches:
            genes.append(match)

print(f'There were {len(genes)} genes extracted.')

genes = pd.DataFrame({'genes': genes})
genes.to_csv(project_config.PROJECT_DIR / 'patients' / 'mygene2_patients' / 'genes.csv', index=False, header=False)
    
