# Network Prebunking Problem
This repository contains the source code for "Network Prebunking Problem: Optimizing Prebunking Targets to Suppress the Spread of Misinformation in Social Networks"

## Code 
To reproduce our main results (Fig. 2), please type the following command:

```
python main_real.py
```

When the code is executed, a directory `results_real/{graph_name}/` is created, and under that directory, a list of intervention targets selected by each algorithm (`{alg_name}.npy`) and simulation results of misinformation spread when intervening on the top k nodes of that list (`{alg_name}_sim_results.npy`) are recorded.

The results can be viewed in `plot_results.ipynb`.

## Citation
If you find this repository useful, please cite the following paper:

```
TBA
```
