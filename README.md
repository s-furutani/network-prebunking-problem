# Network Prebunking Problem
This repository contains datasets and source codes for "Network Prebunking Problem: Optimizing Prebunking Targets to Suppress the Spread of Misinformation in Social Networks"

## Code 
This repository provides a unified interface for running different types of experiments through `main.py`. The code supports three types of experiments:

### Real Graph Experiments
To reproduce our main results (Fig. 2) with real social network data:

```bash
python main.py --type real --graph politifact
# or
python main.py --type real --graph gossipcop
```

### Synthetic Graph Experiments
To run experiments with synthetic networks:

```bash
python main.py --type synthetic --graph ca_HepTh
# Available synthetic graphs: ca_HepTh, Facebook, WikiVote, LastFM, Deezer, Enron, Epinions, Twitter
```

### Uncertainty Analysis Experiments
To run experiments with uncertainty in node susceptibility:

```bash
python main.py --type uncertain --graph politifact
# or
python main.py --type uncertain --graph gossipcop
```

### Additional Parameters
You can customize various parameters:

```bash
python main.py --type synthetic --graph ca_HepTh --kmax 100 --theta 0.01 --mu_eps 0.5 --sigma_eps 0.1
```

Available parameters:
- `--kmax`: Maximum number of intervention nodes (default: 200)
- `--theta`: Threshold for algorithm convergence (default: 0.001)
- `--mu_eps`: Mean of epsilon distribution (default: 0.5)
- `--sigma_eps`: Standard deviation of epsilon distribution (default: 0.1)
- `--mu_q`: Mean of q distribution (synthetic only, default: 0.7)
- `--sigma_q`: Standard deviation of q distribution (synthetic only, default: 0.3)
- `--num_seeds`: Number of seed nodes (synthetic only, default: 5)

### Output
When the code is executed, a directory `results_{type}/{graph_name}/` is created, and under that directory, a list of intervention targets selected by each algorithm (`{alg_name}.npy`) and simulation results of misinformation spread when intervening on the top k nodes of that list (`{alg_name}_sim_results.npy`) are recorded.

The results can be viewed in `plot_results.ipynb`.

## Algorithms
The following algorithms are implemented and compared:
- **Random**: Random selection baseline
- **Degree**: Degree-based selection
- **Distance**: Distance-based selection
- **Gullible**: Susceptibility-based selection
- **MIA-NPP**: Our proposed algorithm
- **CMIA-O**: Approximation algorithm for the Influence Blocking Maximization problem
- **AdvancedGreedy**: Approximation algorithm for the Influence Minimization problem

## Citation
If you find this repository useful, please cite the following paper:

```
TBA
```
