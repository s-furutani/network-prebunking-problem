# Network Prebunking Problem

This repository contains datasets and source codes for the paper:

> **Network Prebunking Problem: Optimizing Prebunking Targets to Suppress the Spread of Misinformation in Social Networks**

## Overview

Prebunking is a preventive intervention that aims to strengthen individuals' cognitive resistance to misinformation by presenting weakened doses of misinformation or teaching common manipulation techniques before they encounter actual misinformation. While prebunking has been shown to be effective at the individual level, an important question remains: **How can we identify the optimal targets for prebunking interventions to mitigate the spread of misinformation in a social network?**

This repository provides algorithms to solve the **Network Prebunking Problem**, a combinatorial optimization problem that aims to select an optimal set of prebunking targets to minimize the spread of misinformation under limited intervention budgets.

## Problem Formulation

### IC-N Model (Information Diffusion Model)

We model the competitive diffusion of misinformation and corrective information using the IC-N model, which extends the classic Independent Cascade (IC) model by incorporating negative opinions.

In the IC-N model:
- Each node v has a **misinformation susceptibility** `q_v ∈ [0, 1]`
- When an inactive node v is activated by a positive (misinformation-sharing) neighbor:
  - It becomes positive with probability `q_v`
  - It becomes negative (sharing corrective information) with probability `1 - q_v`
- When activated by a negative neighbor, it always becomes negative
- **Prebunking** on node v reduces its susceptibility: `q'_v = (1 - ε_v) * q_v`, where `ε_v` is the intervention effect

### Edge Propagation Probability Models

This implementation supports two models for assigning edge propagation probabilities:

- **Weighted Cascade (WC)**: `p_uv = 1 / d_in[v]` (inversely proportional to in-degree of target node)
- **Trivalency (TR)**: `p_uv` sampled uniformly from `{0.001, 0.01, 0.1}`

## Algorithms

The following algorithms are implemented:

| Algorithm | Description |
|-----------|-------------|
| **MIA-NPP** | Our proposed algorithm based on the Maximum Influence Arborescence (MIA) framework |
| **CELF** | Greedy selection with Cost-Effective Lazy Forward optimization |
| **CMIA-O** | Approximation algorithm for the Influence Blocking Maximization (IBM) problem |
| **AdvancedGreedy** | Approximation algorithm for the Influence Minimization (IMIN) problem |
| **Degree** | Baseline: select nodes with highest out-degree |
| **Distance** | Baseline: select nodes closest to seed nodes |
| **Gullible** | Baseline: select nodes with highest susceptibility `q_v` |
| **Random** | Baseline: random selection |

## Installation

```bash
pip install networkx numpy scipy matplotlib tqdm python-igraph
```

## Usage

This repository provides a unified interface through `main.py` supporting three experiment types.

### Experiments on Real Social Networks (UPFD Dataset)

```bash
# PolitiFact network
python main.py --type fakenewsnet --graph politifact

# GossipCop network
python main.py --type fakenewsnet --graph gossipcop
```

### Benchmark Experiments (real graphs with benchmark parameters)

```bash
# With Weighted Cascade (WC) model (default)
python main.py --type benchmark --graph Reed98

# With Trivalency (TR) model
python main.py --type benchmark --graph Reed98 --edge_prob_model TR
```

Available graphs: `ca_HepTh`, `Facebook`, `WikiVote`, `LastFM`, `Deezer`, `Enron`, `Epinions`, `Twitter`, `Congress`, `Reed98`

### Experiments under Parameter Uncertainty

```bash
python main.py --type uncertain --graph politifact
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--type` | Experiment type: `fakenewsnet`, `benchmark`, or `uncertain` | (required) |
| `--graph` | Graph name | `politifact` |
| `--kmax` | Maximum number of intervention nodes | `200` |
| `--theta` | Influence threshold for MIA construction | `0.001` |
| `--mu_eps` | Mean of intervention effect distribution | `0.5` |
| `--sigma_eps` | Std of intervention effect distribution | `0.1` |
| `--mu_q` | Mean of susceptibility distribution (benchmark only) | `0.7` |
| `--sigma_q` | Std of susceptibility distribution (benchmark only) | `0.3` |
| `--num_seeds` | Number of seed nodes (benchmark only) | `5` |
| `--num_mc_samples` | Number of MC samples for CELF | `5000` |
| `--edge_prob_model` | Edge probability model: `WC` or `TR` (benchmark only) | `WC` |
| `--no_cache` | Disable MIIA caching | `False` |

### MIIA Caching

The MIIA (Maximum Influence In-Arborescence) structures are cached to disk to speed up repeated experiments with different parameters (e.g., different `theta` values or uncertainty evaluations). Cache files are automatically stored under the results directory:

```
results/
├── fakenewsnet/{graph_name}/cache/
├── benchmark/{graph_name}_{WC|TR}/cache/
└── uncertain/{graph_name}_sig_eps_XX/cache/
```

To disable caching:
```bash
python main.py --type benchmark --graph Reed98 --no_cache
```

## Output

Results are saved under `results/` directory:
- `results/fakenewsnet/{graph_name}/` - FakeNewsNet experiments
- `results/benchmark/{graph_name}_{WC|TR}/` - Benchmark experiments (real graphs, benchmark params)
- `results/uncertain/{graph_name}_sig_eps_XX/` - Uncertainty experiments

Each directory contains:
- `{algorithm}.npy`: List of selected intervention targets
- `{algorithm}_sim_results.npy`: Simulation results of misinformation spread

Visualize results using `plot_results.ipynb`.

## Datasets

### Real Networks (UPFD)
- **PolitiFact**: 30,813 nodes, 33,488 edges
- **GossipCop**: 75,915 nodes, 85,308 edges

### Social Networks
| Network | Nodes | Edges | Type |
|---------|-------|-------|------|
| Facebook | 4,039 | 88,234 | Undirected |
| WikiVote | 7,115 | 103,689 | Directed |
| LastFM | 7,624 | 27,806 | Undirected |
| ca-HepTh | 8,638 | 49,633 | Undirected |
| Deezer | 28,281 | 92,752 | Undirected |
| Enron | 36,692 | 183,831 | Undirected |
| Epinions | 75,879 | 508,837 | Directed |
| Twitter | 81,306 | 1,768,149 | Directed |

## Citation

If you find this repository useful, please cite:

```bibtex
@article{furutani2025network,
  title={Network Prebunking Problem: Optimizing Prebunking Targets to Suppress the Spread of Misinformation in Social Networks},
  author={Furutani, Satoshi and Shibahara, Toshiki and Akiyama, Mitsuaki and Aida, Masaki},
  journal={arXiv preprint},
  year={2025}
}
```

## License

This project is for research purposes.
