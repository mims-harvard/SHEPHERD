# Incorporating network distance in SHEPHERD

Steps for computing and adding network distance (namely, shortest path length) for each patient:
1) If you are using your own KG, run `shortest_paths.py`. Alternatively, the outputs of this script are available in the data downloaded from Harvard Dataverse.
2) Run `add_spl_to_patients.py`
    - If you want to run an already-trained SHEPHERD model on your own patient dataset, use the flag `--only_test_data`. **Before** running this script, make sure to check that `MY_TEST_DATA` in `project_config.py` is set to your own patient dataset. **After** running the script, make sure that `MY_SPL_DATA` and `MY_SPL_INDEX_DATA` in `project_config.py` is set to the output files from this script.
