import project_config

####################################################################
#
# NODE EMBEDDER MODEL HYPERPARAMETERS
#
####################################################################

def get_pretrain_hparams(args, combined=False):    
    print('node embedder args: ', args)

    # Default
    hparams = {
               # Tunable parameters
               'nfeat': args.nfeat if not combined else 4096,
               'hidden': args.hidden if not combined else 256,
               'output': args.output if not combined else 128,
               'n_heads': args.n_heads if not combined else 2,
               'wd': args.wd if not combined else 5e-4,
               'dropout': args.dropout if not combined else 0.2,
               'lr': args.lr if not combined else 0.0001,

               # Fixed parameters
               'decoder_type': 'bilinear',
               'norm_method': "batch_layer",
               'loss': 'max-margin',
               'pred_threshold': 0.5,
               'negative_sampler_approach': 'by_edge_type',
               'filter_edges': True,
               'n_gpus': 1,
               'num_workers': 4,
               'batch_size': 512,
               'inference_batch_size': 64,
               'neighbor_sampler_sizes': [15, 10, 5],
               'max_epochs': 200,
               'gradclip': 1.0,
               'lr_factor': 0.01,
               'lr_patience': 1000,
               'lr_threshold': 1e-4,
               'lr_threshold_mode': 'rel',
               'lr_cooldown': 0,
               'min_lr': 0,
               'eps': 1e-8,
               'seed': 1,
               'profiler': None,
               'wandb_save_dir': project_config.PROJECT_DIR / 'wandb' / 'preprocess',
               'log_every_n_steps': 10,
               'time': False,
               'debug': False
        }
    
    print('Pretrain hparams: ', hparams)
    
    return hparams



####################################################################
#
# TRAIN MODEL HYPERPARAMETERS
#
####################################################################


def get_train_hparams(args):
    print('Train model args: ', args)

    # Default
    hparams = {
               # Tunable parameters
               'sparse_sample': args.sparse_sample, # Randomly sample N nodes from KG
               'lr': args.lr,
               'upsample_cand': args.upsample_cand, 
               'neighbor_sampler_sizes': [args.neighbor_sampler_size, 10, 5],
               'lambda': args.lmbda, # Contribution of two loss functions
               'alpha': args.alpha, # Contribution of GP gate. NOTE: This is not used for patients-like-me or novel disease characterization
               'kappa': (1 - args.lmbda) * args.kappa,
               'seed': args.seed,
               'batch_size': args.batch_size,
               
               'augment_genes': True if args.aug_gene_w > 0 else False,
               'n_sim_genes': args.n_sim_genes,
               'aug_gene_w': args.aug_gene_w,
               'aug_gene_by_deg': args.aug_gene_by_deg,

               'n_transformer_layers': args.n_transformer_layers,
               'n_transformer_heads': args.n_transformer_heads,
               
               # Fixed parameters
               'pos_weight': 1,
               'neg_weight': 20,
               'margin': 0.4,
               'thresh': 1,
               'filter_edges': False,
               'softmax_scale': 1,
               'leaky_relu': 0.1,
               'decoder_type': 'bilinear',
               'combined_training': True,
               'sample_from_gpd': True,
               'attention_type': 'bilinear',
               'n_cand_diseases': 1000,
               'test_n_cand_diseases': -1, 
               'candidate_disease_type': 'all_kg_nodes',
               'patient_similarity_type': 'gene', # How we determine labels for similar patients in "Patients Like Me"
               'n_similar_patients': 2, # Number of patients with the same gene/disease that we add to the batch
               'only_hard_distractors': False, # Flag when true only uses the curated hard distractors at train time
               'sample_edges_from_train_patients': False, # Preferentially sample edges connected to training patients
               'gradclip': 1.0,
               'inference_batch_size': 64,
               'max_epochs': 100, 
               'n_gpus': 1, 
               'num_workers': 4,
               'wandb_save_dir' : project_config.PROJECT_DIR / 'wandb',
               'precision': 16, 
               'reload_dataloaders_every_n_epochs': 0,
               'profiler': 'simple',
               'pin_memory': False,
               'time': False,
               'log_gpu_memory': True,
               'debug': False, 
               'plot_softmax': False,
               'plot_intrain': False, # Flag to plot gene rank vs. in train sets
               'plot_PG_embed': False, # Flag to plot embeddings with phenotype and gene labels
               'plot_disease_embed': False, # Flag to plot embeddings with disease labels
               'plot_patient_embed': False, # Flag to plot embeddings for patients
               'plot_degree_rank': False, # Flag to plot degree vs. gene rank
               'plot_nhops_rank': False, # Flag to plot nhops vs. gene rank
               'plot_frac_rank': False, # Flag to plot fraction of ___ vs. gene rank
               'plot_gradients': False, # Flag to plot gradients
               'plot_attn_nhops': False, # Flag to plot attn weights vs. nhops
               'plot_phen_gene_sims': False, # Flag to plot phenotype-gene similarities
               'mrr_vs_percent_overlap': False, # Flag to plot MRR vs. percent overlap of phenotypes
               'saved_checkpoint_path': project_config.PROJECT_DIR  / f'{args.saved_node_embeddings_path}', 
    }

    # Get hyperparameters based on run type arguments
    hparams = get_run_type_args(args, hparams)

    # Get hyperparameters based on patient data arguments
    hparams = get_patient_data_args(args, hparams)

    print('Train hparams: ', hparams)

    return hparams


def get_run_type_args(args, hparams):
    if args.run_type == 'causal_gene_discovery':
        hparams.update({
                        'model_type': 'aligner', 
                        'loss': 'gene_multisimilarity', 
                        'use_diseases': False,
                        'add_cand_diseases': False,
                        'add_similar_patients': False,
                        'wandb_project_name': 'causal-gene-discovery'
                       })
    elif args.run_type == 'disease_characterization':
        hparams.update({
                        'model_type': 'patient_NCA',
                        'loss': 'patient_disease_NCA',
                        'use_diseases': True,
                        'add_cand_diseases': True ,
                        'add_similar_patients': False,
                        'wandb_project_name': 'disease-heterogeneity',
                       })
    elif args.run_type == 'patients_like_me':
        hparams.update({
                        'model_type': 'patient_NCA',
                        'loss': 'patient_patient_NCA',
                        'use_diseases': False,
                        'add_cand_diseases': False,
                        'add_similar_patients': True,
                        'wandb_project_name': 'patients-like-me',
                       })
    else:
        raise Exception('You must specify run type.')
    return hparams


def get_patient_data_args(args, hparams):
    if args.patient_data == "disease_simulated":
        hparams.update({'train_data': f'simulated_patients/disease_split_train_sim_patients_{project_config.CURR_KG}.txt',
                        'validation_data': f'simulated_patients/disease_split_val_sim_patients_{project_config.CURR_KG}.txt', 
                        'test_data': f'simulated_patients/disease_split_all_sim_patients_{project_config.CURR_KG}.txt',
                        'spl': f'simulated_patients/disease_split_all_sim_patients_{project_config.CURR_KG}_spl_matrix.npy',
                        'spl_index': f'simulated_patients/disease_split_all_sim_patients_{project_config.CURR_KG}_spl_index_dict.pkl'
                        })
    elif args.patient_data == "my_data":
        hparams.update({'train_data': project_config.MY_TRAIN_DATA,
                        'validation_data': project_config.MY_VAL_DATA,
                        'test_data': project_config.MY_TEST_DATA,
                        'spl': project_config.MY_SPL_DATA, # Result of add_spl_to_patients.py (suffix: _spl_matrix.npy)
                        'spl_index': project_config.MY_SPL_INDEX_DATA, # Result of add_spl_to_patients.py (suffix: _spl_index_dict.pkl)
                        })
    else:
        raise Exception('You must specify patient data.')
    return hparams



####################################################################
#
# PREDICTION HYPERPARAMETERS
#
####################################################################


def get_predict_hparams(args):
    hparams = {
               'seed': 33,
               'n_gpus': 0, # NOTE: currently predict scripts only work with CPU
               'num_workers': 4, 
               'profiler': 'simple',
               'pin_memory': False,
               'time': False,
               'log_gpu_memory': False,
               'debug': False,

               'augment_genes': True,
               'n_sim_genes': 3,
               'aug_gene_w': 0.5,

               'wandb_save_dir' : project_config.PROJECT_DIR / 'wandb',
               'saved_checkpoint_path': project_config.PROJECT_DIR  / f'{args.saved_node_embeddings_path}',
               'test_n_cand_diseases': -1, 
               'candidate_disease_type': 'all_kg_nodes', 
               'only_hard_distractors': False, # Flag when true only uses the curated hard distractors at train time
               'patient_similarity_type': 'gene', # How we determine labels for similar patients in "Patients Like Me"
               'n_similar_patients': 2, # (Patients Like Me only) Number of patients with the same gene/disease that we add to the batch
    }

    # Get hyperparameters based on run type arguments
    hparams = get_run_type_args(args, hparams)    
    hparams.update({'add_similar_patients' : False})
    hparams = get_patient_data_args(args, hparams)

    print('Predict hparams: ', hparams)

    return hparams
