# Bash script to train SHEPHERD for causal gene discovery

# Command to run this bash script:
# bash run_causal_gene_discovery.sh

# Command to run with the best hyperparameters from the paper
python train.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --patient_data disease_simulated \
        --run_type causal_gene_discovery \
        --saved_node_embeddings_path checkpoints/pretrain.ckpt \
        --sparse_sample 100 \
        --lr 1e-05 \
        --upsample_cand 2 \
        --neighbor_sampler_size -1 \
        --lmbda 0.4 \
        --alpha 0.5 \
        --kappa 0.18

