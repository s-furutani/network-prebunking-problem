import os
import sys
import csv
import time
import random

import networkx as nx
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import truncnorm

import load_real_graph
import algorithm
import simulation

# import importlib
# importlib.reload(util)
# importlib.reload(netw)
# importlib.reload(alg)

def embed_model_parameters(graph, q, epsilon):
    for v in graph.nodes():
        graph.nodes[v]['q'] = q[v]
        graph.nodes[v]['epsilon'] = epsilon[v]
    return graph

def write_data(directory, filename, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    np.save(filepath, data)

def get_truncated_normal(mu, sigma, lower=0, upper=1):
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

########################################################################

# graph_name = 'politifact'
graph_name = 'gossipcop'

graph = load_real_graph.FakeNewsNet_interaction_network(graph_name)
S = ['root']
q = {node: graph.nodes[node]['susceptibility'] for node in graph.nodes}
q['root'] = 1.0

random.seed(42)
np.random.seed(42)

mu_eps=0.5
sigma_eps=0.1 # σ_ε ∈ {0.1, 0.5}

epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
epsilon_ate = {node: mu_eps for node in graph.nodes}

graph_copy = graph.copy()
graph_truth = embed_model_parameters(graph_copy, q, epsilon)

kmax = 200
theta = 0.001

print('****' * 10)
print('')
print(f'graph: {graph_name}, seed node: {S}, epsilon ~ truncN({mu_eps}, {sigma_eps})')
print('')

sigma_delta = [0, 0.1, 0.5, 1]  # σ_δ ∈ {0.0, 0.1, 0.5, 1.0}

### computing prebunking node set X ###
print('computing prebunking node set X by MIA-NPP with σ_δ ∈ {0.0, 0.1, 0.5, 1.0}')

directory = f'results_uncertain/{graph_name} (sig_eps={int(sigma_eps * 10):02d})/'

for sig in tqdm(sigma_delta):
    q_noise = dict()
    for node in q.keys():
        q_observed = q[node] + np.random.normal(loc=0, scale=np.sqrt(sig))  # q_observed = q_truth + delta (~ N(0, sig))
        q_noise[node] = np.clip(q_observed, 0, 1)
    q_noise['root'] = 1.0

    graph_copy = graph.copy()
    graph_noise = embed_model_parameters(graph_copy, q_noise, epsilon_ate)

    X = algorithm.MIA_NPP(graph_noise, S, kmax, theta)
    write_data(directory, f'MIA-NPP (sig_delta={int(sig * 10):02d})', X)

### conducting simulation ###
print('conducting simulation')

for sig in tqdm(sigma_delta):
    alg_name = f'MIA-NPP (sig_delta={int(sig * 10):02d})'
    path = directory + alg_name + '.npy'
    path_sim = directory + alg_name + '_sim_results.npy'
    # if os.path.exists(path) and not os.path.exists(path_sim):
    if os.path.exists(path):
        X = np.load(path, allow_pickle=True)
        results = simulation.run_simulation(graph_truth, S, X, kmax)
        filename = alg_name + '_sim_results'
        write_data(directory, filename, results)
