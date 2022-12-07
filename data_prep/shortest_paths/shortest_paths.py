import multiprocessing
import numpy as np
import sys
import time
import snap
import pandas as pd

import sys
sys.path.insert(0, '..') # add config to path
import project_config

# INPUT KG FILE
snap_graph = snap.LoadEdgeList(snap.PUNGraph, str(project_config.KG_DIR / 'KG_edgelist_mask.txt'), 0, 1)

t0 = time.time()

node_ids = np.sort([node.GetId() for node in snap_graph.Nodes()])
n_nodes = len(list(snap_graph.Nodes()))
print(f'There are {n_nodes} nodes in the graph')

def get_shortest_path(node_id):
    NIdToDistH = snap.TIntH()
    path_len = snap.GetShortPath(snap_graph, int(node_id), NIdToDistH)
    paths = np.zeros((n_nodes))
    for dest_node in NIdToDistH: 
        paths[dest_node] = NIdToDistH[dest_node]
    return paths

with multiprocessing.Pool(processes=20) as pool:
    shortest_paths = pool.map(get_shortest_path, node_ids)

all_shortest_paths = np.stack(shortest_paths)
print(all_shortest_paths.shape)
t1 = time.time()
print(f'It took {t1-t0:0.4f}s to calculate the shortest paths')

# save all shortest paths
np.save(project_config.KG_DIR / 'KG_shortest_path_matrix.npy', all_shortest_paths) 

# subset to shortest paths from all nodes to phenotypes
node_map = pd.read_csv(project_config.KG_DIR / "KG_node_map.txt", sep="\t")
desired_idx = node_map[node_map["node_type"] == "effect/phenotype"]["node_idx"].tolist()
all_shortest_paths_to_phens = all_shortest_paths[:, desired_idx]
with open(project_config.KG_DIR / "KG_shortest_path_matrix_onlyphenotypes.npy", "wb") as f:
    np.save(f, all_shortest_paths_to_phens)


