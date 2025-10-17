# Network Prebunking Problem
This repository contains datasets and source codes for "Network Prebunking Problem: Optimizing Prebunking Targets to Suppress the Spread of Misinformation in Social Networks"

## Code 
This repository provides a unified interface for running different types of experiments through `main.py`. The code supports three types of experiments:

### Experiments on real social networks with real parameters
To reproduce our main results (Fig. 2) with PolitiFact & GossipCop networks:

```bash
python main.py --type real --graph politifact
# or
python main.py --type real --graph gossipcop
```

### Experiments on real social networks with synthetic parameters
To run experiments with other social networks:

```bash
python main.py --type synthetic --graph ca_HepTh
# Available synthetic graphs: ca_HepTh, Facebook, WikiVote, LastFM, Deezer, Enron, Epinions, Twitter, Congress, Reed98
```

### Experiments under uncertainty
To run experiments with uncertainty in model parameters:

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
- `--num_mc_samples`: Number of Monte Carlo samples for CELF (default: 5000)

### Output
When the code is executed, a directory `results_{type}/{graph_name}/` is created, and under that directory, a list of intervention targets selected by each algorithm (`{alg_name}.npy`) and simulation results of misinformation spread when intervening on the top k nodes of that list (`{alg_name}_sim_results.npy`) are recorded.

The results can be viewed in `plot_results.ipynb`.

## Algorithms
The following algorithms are implemented and compared:
- **Random**: Random selection
- **Degree**: Degree-based selection
- **Distance**: Distance-based selection
- **Gullible**: Susceptibility-based selection
- **CELF**: Greedy selection based on the CELF implementation
- **CMIA-O**: Approximation algorithm for the IBM problem based on the MIA framework
- **AdvancedGreedy**: Approximation alogorithm for the IMIN problem based on the dominator tree and graph sampling
- **MIA-NPP**: Our proposed algorithm

## Citation
If you find this repository useful, please cite the following paper:

```
TBA
```
