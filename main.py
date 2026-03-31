import os
import sys
import csv
import time
import random
import argparse

import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import truncnorm

import load_real_graph
import load_graph
import algorithm
import simulation

def embed_model_parameters(graph, graph_name, q, epsilon, edge_prob_model='WC', is_benchmark=False, random_seed=42):
    """
    Embed model parameters into the graph.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Target graph
    graph_name : str
        Name of the graph
    q : dict
        Misinformation susceptibility q_v in [0, 1] for each node v
    epsilon : dict
        Individual intervention effect epsilon_v in [0, 1] for each node v
    edge_prob_model : str
        Edge propagation probability model ('WC' or 'TR')
        - 'WC' (Weighted Cascade): p_uv = 1 / d_in[v] (inversely proportional to in-degree)
        - 'TR' (Trivalency): p_uv sampled uniformly from {0.001, 0.01, 0.1}
    is_benchmark : bool
        Whether to use artificial model parameters (if True, propagation probabilities are artificially assigned)
    random_seed : int
        Random seed for reproducibility (used only for TR model)
    
    Returns
    -------
    graph : nx.DiGraph
        Graph with embedded parameters
    """
    if is_benchmark:
        if graph_name == 'Congress':
            # Congress graph already has weight attributes
            for u, v in graph.edges():
                pe = graph[u][v]['weight']
                graph[u][v]['p_e'] = pe
                graph[u][v]['-logp'] = - np.log(pe)
        else:
            graph = graph.to_directed()
            
            if edge_prob_model == 'WC':
                # Weighted Cascade (WC) model: p_uv = 1 / d_in[v]
                d_in = graph.in_degree()
                for u, v in graph.edges():
                    pe = 1.0 / d_in[v]
                    graph[u][v]['p_e'] = pe
                    graph[u][v]['-logp'] = - np.log(pe)
            elif edge_prob_model == 'TR':
                # Trivalency (TR) model: p_uv in {0.001, 0.01, 0.1}
                # Set seed for reproducibility of edge probability assignment
                random.seed(random_seed)
                tr_probs = [0.001, 0.01, 0.1]
                for u, v in graph.edges():
                    pe = random.choice(tr_probs)
                    graph[u][v]['p_e'] = pe
                    graph[u][v]['-logp'] = - np.log(pe)
            else:
                raise ValueError(f"Unknown edge_prob_model: {edge_prob_model}. Use 'WC' or 'TR'.")
    
    # Set misinformation susceptibility and intervention effect for each node
    for v in graph.nodes():
        graph.nodes[v]['q'] = q[v]
        graph.nodes[v]['epsilon'] = epsilon[v]
    return graph

def get_random_high_degree_nodes(graph, num_nodes):
    """Select random nodes from high degree nodes"""
    random.seed(42)
    out_degrees = dict(graph.out_degree())
    top_50_nodes = sorted(out_degrees, key=out_degrees.get, reverse=True)[:50]
    S = list(random.sample(top_50_nodes, num_nodes))
    return S

def get_largest_degree_node(graph):
    """Get the node with maximum degree"""
    out_degrees = dict(graph.out_degree())
    largest_deg_node = max(out_degrees, key=out_degrees.get)
    S = [largest_deg_node]
    return S

def write_data(directory, filename, data):
    """Save data to file"""
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    np.save(filepath, data)

def load_algorithm_timings(directory):
    """
    実験ディレクトリの algorithm_timings.csv を読み、
    alg_name をキーにした辞書（行ごとの辞書）を返す。無ければ {}。
    """
    path = os.path.join(directory, 'algorithm_timings.csv')
    if not os.path.isfile(path):
        return {}
    timings = {}
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg_name = row.get('alg_name', '').strip()
            if not alg_name:
                continue
            timings[alg_name] = {
                'alg_name': alg_name,
                'theta': row.get('theta', '').strip(),
                'num_mc_samples': row.get('num_mc_samples', '').strip(),
                'num_graph_samples': row.get('num_graph_samples', '').strip(),
                'time_sec': row.get('time_sec', '').strip(),
            }
    return timings

def save_algorithm_timings(directory, timings_dict):
    """
    タイミング辞書を algorithm_timings.csv に書き出す。
    列: alg_name, theta, num_mc_samples, num_graph_samples, time_sec
    """
    if not timings_dict:
        return
    if not os.path.exists(directory):
        os.makedirs(directory)
    path = os.path.join(directory, 'algorithm_timings.csv')
    columns = ['alg_name', 'theta', 'num_mc_samples', 'num_graph_samples', 'time_sec']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for alg_name in sorted(timings_dict.keys()):
            row = timings_dict[alg_name]
            writer.writerow({c: row.get(c, '') for c in columns})
    print(f'[main] Algorithm timings saved to {path}')

def get_truncated_normal(mu, sigma, lower=0, upper=1):
    """Sample from truncated normal distribution"""
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def run_fakenewsnet_experiment(graph_name, kmax=200, theta=0.01, mu_eps=0.5, sigma_eps=0.1, num_mc_samples=5000, use_cache=True, with_std=False):
    """Run experiment with FakeNewsNet graph data"""
    print(f'Running FakeNewsNet experiment with graph: {graph_name}')
    
    graph = load_real_graph.FakeNewsNet_interaction_network(graph_name)
    S = ['root']
    q = {node: graph.nodes[node]['susceptibility'] for node in graph.nodes}
    q['root'] = 1.0

    random.seed(42)
    np.random.seed(42)

    epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
    graph = embed_model_parameters(graph, graph_name, q, epsilon, is_benchmark=False)

    print(f'graph: {graph_name}, seed node: {S}, epsilon ~ truncN({mu_eps}, {sigma_eps})')
    
    directory = f'results/fakenewsnet/{graph_name}/'
    cache_dir = f'{directory}cache/' if use_cache else None
    run_algorithms(graph, S, kmax, theta, directory, num_mc_samples=num_mc_samples, cache_dir=cache_dir)
    run_simulation(graph, S, kmax, directory, theta=theta, num_mc_samples=num_mc_samples, with_std=with_std)

def run_benchmark_experiment(graph_name, kmax=200, theta=0.01, mu_q=0.7, sigma_q=0.3, mu_eps=0.5, sigma_eps=0.1, num_seeds=5, num_mc_samples=5000, edge_prob_model='WC', use_cache=True, with_std=False):
    """
    Run experiment with artificial model parameters on benchmark social network graphs.
    
    Parameters
    ----------
    graph_name : str
        Name of the graph to use
    kmax : int
        Maximum number of intervention target nodes
    theta : float
        Influence threshold for MIA construction
    mu_q, sigma_q : float
        Distribution parameters for susceptibility q (truncated normal)
    mu_eps, sigma_eps : float
        Distribution parameters for intervention effect epsilon (truncated normal)
    num_seeds : int
        Number of seed nodes
    num_mc_samples : int
        Number of Monte Carlo samples for CELF
    use_cache : bool
        Whether to use MIIA caching (cache stored under results directory)
    edge_prob_model : str
        Edge propagation probability model ('WC' or 'TR')
    """
    print(f'Running experiment with benchmark social network: {graph_name}, edge_prob_model: {edge_prob_model}')
    
    # Load graph
    if graph_name == 'ca_HepTh':
        graph, graph_name = load_graph.ca_HepTh_graph()
    elif graph_name == 'Facebook':
        graph, graph_name = load_graph.Facebook_graph()
    elif graph_name == 'WikiVote':
        graph, graph_name = load_graph.WikiVote_graph()
    elif graph_name == 'LastFM':
        graph, graph_name = load_graph.LastFM_graph()
    elif graph_name == 'Deezer':
        graph, graph_name = load_graph.Deezer_graph()
    elif graph_name == 'Enron':
        graph, graph_name = load_graph.Enron_graph()
    elif graph_name == 'Epinions':
        graph, graph_name = load_graph.Epinions_graph()
    elif graph_name == 'Twitter':
        graph, graph_name = load_graph.Twitter_graph()
    elif graph_name == 'Congress':
        graph, graph_name = load_graph.Congress_network()
    elif graph_name == 'Reed98':
        graph, graph_name = load_graph.Reed98_network()
    elif graph_name == 'ER_test':
        random.seed(0)
        np.random.seed(0)
        graph = nx.erdos_renyi_graph(500, 0.05)
        graph = graph.to_directed()
        print(len(graph.nodes()), len(graph.edges()))
    elif graph_name == 'BA_test':
        random.seed(0)
        np.random.seed(0)
        graph = nx.barabasi_albert_graph(500, 10)
        graph = graph.to_directed()
        print(len(graph.nodes()), len(graph.edges()))
    else:
        raise ValueError(f"Unknown graph name: {graph_name}")

    # Select seed nodes (uses random.seed(42) internally for reproducibility)
    S = get_random_high_degree_nodes(graph, num_seeds)

    # Set random seed for reproducibility of q and epsilon generation
    random.seed(42)
    np.random.seed(42)

    # Generate susceptibility q and intervention effect epsilon
    q = {node: get_truncated_normal(mu_q, sigma_q) for node in graph.nodes()}
    epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
    # q = {node: 1 if random.random() < 0.7 else 0 for node in graph.nodes}
    # epsilon = {node: 1 for node in graph.nodes}
    for s in S:
        q[s] = 1.0
        epsilon[s] = 0.0

    # Embed parameters into graph (TR model uses random_seed=123 for edge probabilities)
    graph = embed_model_parameters(graph, graph_name, q, epsilon, edge_prob_model=edge_prob_model, is_benchmark=True, random_seed=123)

    print(f'graph: {graph_name}, n={len(graph.nodes())}, m={len(graph.edges())}, seed node: {S}')
    print(f'  edge_prob_model: {edge_prob_model}, q ~ truncN({mu_q}, {sigma_q}), epsilon ~ truncN({mu_eps}, {sigma_eps})')
    
    # Include edge_prob_model in directory path to distinguish WC and TR results
    directory = f'results/benchmark/{graph_name}_{edge_prob_model}/'
    cache_dir = f'{directory}cache/' if use_cache else None
    run_algorithms(graph, S, kmax, theta, directory, num_mc_samples=num_mc_samples, cache_dir=cache_dir)
    run_simulation(graph, S, kmax, directory, theta=theta, num_mc_samples=num_mc_samples, with_std=with_std)

def run_uncertain_experiment(graph_name, kmax=200, theta=0.01, mu_eps=0.5, sigma_eps=0.1, use_cache=True, with_std=False):
    """Run experiment with uncertainty in node susceptibility and intervention effect"""
    print(f'Running uncertain experiment with graph: {graph_name}')
    
    graph = load_real_graph.FakeNewsNet_interaction_network(graph_name)
    S = ['root']
    q = {node: graph.nodes[node]['susceptibility'] for node in graph.nodes}
    q['root'] = 1.0

    random.seed(42)
    np.random.seed(42)

    epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
    epsilon_ate = {node: mu_eps for node in graph.nodes}

    graph_copy = graph.copy()
    graph_truth = embed_model_parameters(graph_copy, graph_name, q, epsilon, is_benchmark=False)

    print(f'graph: {graph_name}, seed node: {S}, epsilon ~ truncN({mu_eps}, {sigma_eps})')
    
    sigma_delta = [0, 0.1, 0.2, 0.5]  # σ_δ ∈ {0.0, 0.1, 0.2, 0.5}
    directory = f'results/fakenewsnet/{graph_name}/'
    cache_dir = f'{directory}cache/' if use_cache else None

    def sigdelta_fmt(s):
        """σ_δ をファイル名用の文字列に (0 -> '0', 0.1 -> '1e-01', 1.0 -> '1e+00' など)。"""
        return '0' if s == 0 else f'{s:.0e}'

    theta_fmt = '1e-02' if theta == 0.01 else f'{theta:.0e}'
    print(f'computing prebunking node set X by MIA-NPP with σ_δ ∈ {sigma_delta}')

    for sig in tqdm(sigma_delta):
        alg_name = f'MIA-NPP_theta={theta_fmt}_sigdelta={sigdelta_fmt(sig)}'
        # if os.path.exists(os.path.join(directory, alg_name + '.npy')):
        #     continue
        q_noise = dict()
        for node in q.keys():
            q_observed = q[node] + np.random.normal(loc=0, scale=np.sqrt(sig))
            q_noise[node] = np.clip(q_observed, 0, 1)
        q_noise['root'] = 1.0

        graph_copy = graph.copy()
        graph_noise = embed_model_parameters(graph_copy, graph_name, q_noise, epsilon_ate, is_benchmark=False)

        # Note: MIIA cache is effective here because the graph structure and edge probabilities
        # are the same across iterations (only node attributes q differ).
        X = algorithm.MIA_NPP(graph_noise, S, kmax, theta, cache_dir=cache_dir)
        write_data(directory, alg_name, X)

    print('conducting simulation')
    for sig in tqdm(sigma_delta):
        alg_name = f'MIA-NPP_theta={theta_fmt}_sigdelta={sigdelta_fmt(sig)}'
        path = os.path.join(directory, alg_name + '.npy')
        if not os.path.exists(path):
            continue
        X = np.load(path, allow_pickle=True)
        if with_std:
            out_npz = os.path.join(directory, alg_name + '_sim_results_with_std.npz')
            if os.path.exists(out_npz):
                continue
            mean_spread, std_spread = simulation.run_ICN_simulation(
                graph_truth, S, X, kmax, return_std=True
            )
            np.savez_compressed(
                os.path.join(directory, alg_name + '_sim_results_with_std'),
                mean=np.asarray(mean_spread, dtype=float),
                std=np.asarray(std_spread, dtype=float),
            )
        else:
            results = simulation.run_ICN_simulation(graph_truth, S, X, kmax)
            write_data(directory, alg_name + '_sim_results', results)

def run_algorithms(graph, S, kmax, theta, directory, num_mc_samples=5000, cache_dir=None):
    """
    Run all algorithms for prebunking target selection.
    Skips computation if result file already exists.
    各アルゴリズムの介入集合 X 計算時間を algorithm_timings.csv に記録する。
    
    Parameters
    ----------
    graph : nx.DiGraph
        Input graph
    S : list
        Seed node set
    kmax : int
        Maximum number of intervention nodes
    theta : float
        Influence threshold for MIA-based algorithms
    directory : str
        Output directory for results
    num_mc_samples : int
        Number of Monte Carlo samples for CELF
    cache_dir : str or None
        Directory to store MIIA cache files. If None, caching is disabled.
    """
    timings = load_algorithm_timings(directory)

    def run_if_not_exists(alg_name, alg_func, params=None):
        """Result が無いときだけ実行し、実行時間を timings に記録する。params は theta / num_mc_samples / num_graph_samples の辞書。"""
        params = params or {}
        path = os.path.join(directory, alg_name + '.npy')
        if os.path.exists(path):
            print(f'{alg_name}: [SKIP] {path} already exists')
            return
        print(f'{alg_name}:')
        t0 = time.perf_counter()
        X = alg_func()
        elapsed = time.perf_counter() - t0
        write_data(directory, alg_name, X)
        # パラメータ条件と実行時間を記録（CSV 用に空でない値は str で）
        timings[alg_name] = {
            'alg_name': alg_name,
            'theta': str(params['theta']) if params.get('theta') is not None else '',
            'num_mc_samples': str(params['num_mc_samples']) if params.get('num_mc_samples') is not None else '',
            'num_graph_samples': str(params['num_graph_samples']) if params.get('num_graph_samples') is not None else '',
            'time_sec': elapsed,
        }
        print(f'  -> {elapsed:.2f} sec')
        save_algorithm_timings(directory, timings)

    print('computing prebunking node set X')
    
    # Random
    # run_if_not_exists('Random', lambda: algorithm.BaselineRandom(graph, S, kmax))
    
    # Degree
    # run_if_not_exists('Degree', lambda: algorithm.BaselineDegree(graph, S, kmax))
    
    # Distance
    # run_if_not_exists('Distance', lambda: algorithm.BaselineDistance(graph, S, kmax))
    
    # Gullible
    # run_if_not_exists('Gullible', lambda: algorithm.BaselineGullible(graph, S, kmax))

    # MIA-NPP (with MIIA caching)
    # run_if_not_exists(f'MIA-NPP_theta={theta:.0e}', lambda: algorithm.MIA_NPP(graph, S, kmax, theta, cache_dir=cache_dir), {'theta': theta})

    # CMIA-O (with MIIA caching)
    # run_if_not_exists(f'CMIA-O_theta={theta:.0e}', lambda: algorithm.CMIA_O(graph, S, kmax, theta, cache_dir=cache_dir), {'theta': theta})

    # AdvancedGreedy
    num_graph_samples = 1000
    # run_if_not_exists(f'AdvancedGreedy_rho={num_graph_samples}', lambda: algorithm.AdvancedGreedy(graph, S, kmax, num_graph_samples), {'num_graph_samples': num_graph_samples})

    # CELF
    # run_if_not_exists(f'CELF_rho={num_mc_samples}', lambda: algorithm.CELF(graph, S, kmax, num_mc_samples, use_crn=True), {'num_mc_samples': num_mc_samples})

def run_simulation(graph, S, kmax, directory, theta=0.01, num_mc_samples=5000, with_std=False):
    """Run simulation for all algorithms (MIA-NPP/CMIA-O use theta for filename)."""
    print('conducting simulation')

    alg_names = [
        # 'MIA-NPP_theta=5e-02',
        'MIA-NPP_theta=1e-02',
        # 'CMIA-O_theta=1e-02',
        # 'AdvancedGreedy_rho=1000',
        'CELF_rho=1000',
        # 'CELF_rho=2000',
        # 'Distance',
        # 'Degree',
        # 'Gullible',
        # 'Random'
    ]
    for alg_name in tqdm(alg_names):
        path = directory + alg_name + '.npy'
        if with_std:
            sim_path = os.path.join(directory, alg_name + '_sim_results_with_std.npz')
        else:
            sim_path = directory + alg_name + '_sim_results.npy'
        print(f'{alg_name}: {path}, {sim_path}')
        if not os.path.exists(path):
            continue  # no algorithm result, skip
        if os.path.exists(sim_path):
            continue  # simulation result already exists, skip
        X = np.load(path, allow_pickle=True)
        if with_std:
            mean_spread, std_spread = simulation.run_ICN_simulation(
                graph, S, X, kmax, return_std=True
            )
            np.savez_compressed(
                os.path.join(directory, alg_name + '_sim_results_with_std'),
                mean=np.asarray(mean_spread, dtype=float),
                std=np.asarray(std_spread, dtype=float),
            )
        else:
            results = simulation.run_ICN_simulation(graph, S, X, kmax)
            write_data(directory, alg_name + '_sim_results', results)

def main():
    parser = argparse.ArgumentParser(description='Network Prebunking Problem Experiments')
    parser.add_argument('--type', choices=['fakenewsnet', 'benchmark', 'uncertain'], required=True,
                       help='Experiment type')
    parser.add_argument('--graph', type=str, default='politifact',
                       help='Graph name (fakenewsnet: politifact/gossipcop, benchmark: Reed98/Facebook/WikiVote/LastFM/Deezer/Enron/Epinions/Twitter)')
    parser.add_argument('--kmax', type=int, default=200, help='Maximum number of nodes')
    parser.add_argument('--theta', type=float, default=0.01, help='Threshold')
    parser.add_argument('--mu_eps', type=float, default=0.5, help='Mean of epsilon distribution')
    parser.add_argument('--sigma_eps', type=float, default=0.1, help='Standard deviation of epsilon distribution')
    parser.add_argument('--mu_q', type=float, default=0.7, help='Mean of q distribution (benchmark network only)')
    parser.add_argument('--sigma_q', type=float, default=0.3, help='Standard deviation of q distribution (benchmark network only)')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seed nodes (benchmark network only)')
    parser.add_argument('--num_mc_samples', type=int, default=1000, help='Number of MC samples for CELF')
    parser.add_argument('--edge_prob_model', choices=['WC', 'TR'], default='WC',
                       help='Edge propagation probability model (benchmark network only): WC (Weighted Cascade, p=1/d_in) or TR (Trivalency, p in {0.001, 0.01, 0.1})')
    parser.add_argument('--no_cache', action='store_true',
                       help='Disable MIIA caching (cache is stored under results directory by default)')
    parser.add_argument('--with_std', action='store_true',
                       help='ICN simulation で平均に加え標準偏差も計算し、*_sim_results_with_std.npz に保存する')
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    
    print('****' * 10)
    print('')
    
    if args.type == 'fakenewsnet':
        run_fakenewsnet_experiment(
            args.graph, args.kmax, args.theta, args.mu_eps, args.sigma_eps,
            args.num_mc_samples, use_cache, with_std=args.with_std,
        )
    elif args.type == 'benchmark':
        run_benchmark_experiment(
            args.graph, args.kmax, args.theta, args.mu_q, args.sigma_q,
            args.mu_eps, args.sigma_eps, args.num_seeds, args.num_mc_samples,
            args.edge_prob_model, use_cache, with_std=args.with_std,
        )
    elif args.type == 'uncertain':
        run_uncertain_experiment(
            args.graph, args.kmax, args.theta, args.mu_eps, args.sigma_eps,
            use_cache, with_std=args.with_std,
        )
    
    print('')
    print('****' * 10)

if __name__ == "__main__":
    main()
