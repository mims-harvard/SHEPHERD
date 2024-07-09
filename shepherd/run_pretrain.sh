# Bash script to pretrain SHEPHERD

# Command to run this bash script:
# bash run_pretrain.sh

# Command to run with the best hyperparameters from the paper
python pretrain.py \
        --edgelist KG_edgelist_mask.txt \
        --node_map KG_node_map.txt \
        --save_dir checkpoints/ \
        --nfeat 2048 \
        --hidden 256 \
        --output 128 \
        --n_heads 4 \
        --wd 0.0 \
        --dropout 0.3 \
        --lr 0.0001