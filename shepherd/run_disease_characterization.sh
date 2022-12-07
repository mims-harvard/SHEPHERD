# Bash script to train SHEPHERD for novel disease characterization

# Command to run this bash script:
# bash run_disease_characterization.sh

# Command to run with the best hyperparameters from the paper
python train.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --patient_data disease_simulated \
        --run_type disease_characterization \
        --saved_node_embeddings_path checkpoints/pretrain.ckpt \
        --sparse_sample 300 \
        --lr 1e-05 \
        --upsample_cand 3 \
        --neighbor_sampler_size 15 \
        --lmbda 0.9 \
        --kappa 0.029999999999999992