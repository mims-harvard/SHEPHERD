import multiprocessing
import numpy as np
import sys
import time
import snap
import pandas as pd

import sys
#sys.path.insert(0, '..') # add config to path
sys.path.insert(0, '../..') # add config to path
import project_config

# Filenames
suffix = "_noGO" # ""
edgelist_f = "KG_edgelist_mask%s.txt" % suffix
nodemap_f = "KG_node_map%s.txt" % suffix
spl_mat_all_f = "KG_shortest_path_matrix%s.npy" % suffix
spl_mat_onlyphenotypes_f = "KG_shortest_path_matrix_onlyphenotypes%s.npy" % suffix

print("Filenames:")
print(edgelist_f)
print(nodemap_f)
print(spl_mat_all_f)
print(spl_mat_onlyphenotypes_f)

print("Starting to calculate shortest paths...")

# INPUT KG FILE
node_map = pd.read_csv(project_config.KG_DIR / nodemap_f, sep="\t")
snap_graph = snap.LoadEdgeList(snap.PUNGraph, str(project_config.KG_DIR / edgelist_f), 0, 1)

t0 = time.time()

node_ids = np.sort([node.GetId() for node in snap_graph.Nodes()])
n_nodes = len(node_map)
print(n_nodes, len(list(snap_graph.Nodes())), len(node_ids))
print(f'There are {n_nodes} nodes in the graph')
assert max(node_ids) == n_nodes - 1
if "noGO" not in edgelist_f: assert len(node_map) == len(node_ids)

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

# save all shortest paths (requires no missing nodes)
if "noGO" not in spl_mat_all_f:
    np.save(project_config.KG_DIR / spl_mat_all_f, all_shortest_paths)

# subset to shortest paths from all nodes to phenotypes
desired_idx = node_map[node_map["node_type"] == "effect/phenotype"]["node_idx"].tolist()
all_shortest_paths_to_phens = all_shortest_paths[:, desired_idx]
with open(project_config.KG_DIR / spl_mat_onlyphenotypes_f, "wb") as f:
    np.save(f, all_shortest_paths_to_phens)


