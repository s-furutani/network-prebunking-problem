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

def embed_model_parameters(graph, q, epsilon, is_synthetic=False):
    """Embed model parameters into the graph"""
    if is_synthetic:
        graph = graph.to_directed()
        d_in = graph.in_degree()
        for u, v in graph.edges():
            pe = 1./d_in[v]
            graph[u][v]['p_e'] = pe
            graph[u][v]['-logp'] = - np.log(pe)
    
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

def get_truncated_normal(mu, sigma, lower=0, upper=1):
    """Sample from truncated normal distribution"""
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

def run_real_experiment(graph_name, kmax=200, theta=0.001, mu_eps=0.5, sigma_eps=0.1):
    """Run experiment with real graph data"""
    print(f'Running real experiment with graph: {graph_name}')
    
    graph = load_real_graph.FakeNewsNet_interaction_network(graph_name)
    S = ['root']
    q = {node: graph.nodes[node]['susceptibility'] for node in graph.nodes}
    q['root'] = 1.0

    random.seed(42)
    np.random.seed(42)

    epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
    graph = embed_model_parameters(graph, q, epsilon, is_synthetic=False)

    print(f'graph: {graph_name}, seed node: {S}, epsilon ~ truncN({mu_eps}, {sigma_eps})')
    
    directory = f'results_real/{graph_name}/'
    run_algorithms(graph, S, kmax, theta, directory)
    run_simulation(graph, S, kmax, directory)

def run_synthetic_experiment(graph_name, kmax=200, theta=0.001, mu_q=0.7, sigma_q=0.3, mu_eps=0.5, sigma_eps=0.1, num_seeds=5):
    """Run experiment with synthetic graph data"""
    print(f'Running synthetic experiment with graph: {graph_name}')
    
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
    else:
        raise ValueError(f"Unknown synthetic graph: {graph_name}")

    S = get_random_high_degree_nodes(graph, num_seeds)

    random.seed(42)
    np.random.seed(42)

    q = {node: get_truncated_normal(mu_q, sigma_q) for node in graph.nodes()}
    epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
    for s in S:
        q[s] = 1.0

    graph = embed_model_parameters(graph, q, epsilon, is_synthetic=True)

    print(f'graph: {graph_name}, seed node: {S}, q ~ truncN({mu_q}, {sigma_q}), epsilon ~ truncN({mu_eps}, {sigma_eps})')
    
    directory = f'results_synthetic/{graph_name}/'
    run_algorithms(graph, S, kmax, theta, directory)
    run_simulation(graph, S, kmax, directory)

def run_uncertain_experiment(graph_name, kmax=200, theta=0.001, mu_eps=0.5, sigma_eps=0.1):
    """Run experiment with uncertainty in node susceptibility"""
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
    graph_truth = embed_model_parameters(graph_copy, q, epsilon, is_synthetic=False)

    print(f'graph: {graph_name}, seed node: {S}, epsilon ~ truncN({mu_eps}, {sigma_eps})')
    
    sigma_delta = [0, 0.1, 0.5, 1]  # σ_δ ∈ {0.0, 0.1, 0.5, 1.0}
    directory = f'results_uncertain/{graph_name} (sig_eps={int(sigma_eps * 10):02d})/'

    print('computing prebunking node set X by MIA-NPP with σ_δ ∈ {0.0, 0.1, 0.5, 1.0}')
    
    for sig in tqdm(sigma_delta):
        q_noise = dict()
        for node in q.keys():
            q_observed = q[node] + np.random.normal(loc=0, scale=np.sqrt(sig))
            q_noise[node] = np.clip(q_observed, 0, 1)
        q_noise['root'] = 1.0

        graph_copy = graph.copy()
        graph_noise = embed_model_parameters(graph_copy, q_noise, epsilon_ate, is_synthetic=False)

        X = algorithm.MIA_NPP(graph_noise, S, kmax, theta)
        write_data(directory, f'MIA-NPP (sig_delta={int(sig * 10):02d})', X)

    print('conducting simulation')
    for sig in tqdm(sigma_delta):
        alg_name = f'MIA-NPP (sig_delta={int(sig * 10):02d})'
        path = directory + alg_name + '.npy'
        if os.path.exists(path):
            X = np.load(path, allow_pickle=True)
            results = simulation.run_simulation(graph_truth, S, X, kmax)
            filename = alg_name + '_sim_results'
            write_data(directory, filename, results)

def run_algorithms(graph, S, kmax, theta, directory):
    """Run all algorithms"""
    print('computing prebunking node set X')
    
    # Random
    print('Random:')
    X = algorithm.BaselineRandom(graph, S, kmax)
    write_data(directory, 'Random', X)
    
    # Degree
    print('Degree:')
    X = algorithm.BaselineDegree(graph, S, kmax)
    write_data(directory, 'Degree', X)
    
    # Distance
    print('Distance:')
    X = algorithm.BaselineDistance(graph, S, kmax)
    write_data(directory, 'Distance', X)
    
    # Gullible
    print('Gullible:')
    X = algorithm.BaselineGullible(graph, S, kmax)
    write_data(directory, 'Gullible', X)
    
    # MIA-NPP
    print('MIA-NPP:')
    X = algorithm.MIA_NPP(graph, S, kmax, theta)
    write_data(directory, 'MIA-NPP', X)
    
    # CMIA-O
    print('CMIA-O:')
    X = algorithm.CMIA_O(graph, S, kmax, theta)
    write_data(directory, 'CMIA-O', X)
    
    # AdvancedGreedy
    print('AdvancedGreedy:')
    num_samples = 100
    X = algorithm.AdvancedGreedy(graph, S, kmax, num_samples)
    write_data(directory, 'AdvancedGreedy', X)

def run_simulation(graph, S, kmax, directory):
    """Run simulation for all algorithms"""
    print('conducting simulation')
    
    alg_names = ['MIA-NPP', 'CMIA-O', 'AdvancedGreedy', 'Distance', 'Degree', 'Gullible', 'Random']
    for alg_name in tqdm(alg_names):
        path = directory + alg_name + '.npy'
        if os.path.exists(path):
            X = np.load(path, allow_pickle=True)
            results = simulation.run_simulation(graph, S, X, kmax)
            filename = alg_name + '_sim_results'
            write_data(directory, filename, results)

def main():
    parser = argparse.ArgumentParser(description='Network Prebunking Problem Experiments')
    parser.add_argument('--type', choices=['real', 'synthetic', 'uncertain'], required=True,
                       help='Experiment type')
    parser.add_argument('--graph', type=str, default='politifact',
                       help='Graph name (real: politifact/gossipcop, synthetic: ca_HepTh/Facebook/WikiVote/LastFM/Deezer/Enron/Epinions/Twitter)')
    parser.add_argument('--kmax', type=int, default=200, help='Maximum number of nodes')
    parser.add_argument('--theta', type=float, default=0.001, help='Threshold')
    parser.add_argument('--mu_eps', type=float, default=0.5, help='Mean of epsilon distribution')
    parser.add_argument('--sigma_eps', type=float, default=0.1, help='Standard deviation of epsilon distribution')
    parser.add_argument('--mu_q', type=float, default=0.7, help='Mean of q distribution (synthetic only)')
    parser.add_argument('--sigma_q', type=float, default=0.3, help='Standard deviation of q distribution (synthetic only)')
    parser.add_argument('--num_seeds', type=int, default=5, help='Number of seed nodes (synthetic only)')
    
    args = parser.parse_args()
    
    print('****' * 10)
    print('')
    
    if args.type == 'real':
        run_real_experiment(args.graph, args.kmax, args.theta, args.mu_eps, args.sigma_eps)
    elif args.type == 'synthetic':
        run_synthetic_experiment(args.graph, args.kmax, args.theta, args.mu_q, args.sigma_q, args.mu_eps, args.sigma_eps, args.num_seeds)
    elif args.type == 'uncertain':
        run_uncertain_experiment(args.graph, args.kmax, args.theta, args.mu_eps, args.sigma_eps)
    
    print('')
    print('****' * 10)

if __name__ == "__main__":
    main()
