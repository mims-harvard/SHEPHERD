# Bash script to run SHEPHERD to generate predictions


# Command to run this bash script:
# bash predict.sh

# Command to run the predict script for each task. Comment out tasks you don't want to generate predictions for. 
python predict.py \
--run_type causal_gene_discovery \
--patient_data test_predict \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain.ckpt \
--best_ckpt checkpoints/causal_gene_discovery.ckpt

python predict.py \
--run_type patients_like_me \
--patient_data test_predict \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain.ckpt \
--best_ckpt checkpoints/patients_like_me.ckpt

python predict.py \
--run_type disease_characterization \
--patient_data test_predict \
--edgelist KG_edgelist_mask.txt \
--node_map KG_node_map.txt \
--saved_node_embeddings_path checkpoints/pretrain.ckpt \
--best_ckpt checkpoints/disease_characterization.ckpt
