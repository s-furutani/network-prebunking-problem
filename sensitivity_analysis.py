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
sigma_eps=0.1

epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
graph = embed_model_parameters(graph, q, epsilon)

kmax = 200

print('****' * 10)
print('')
print(f'graph: {graph_name}, seed node: {S}, epsilon ~ truncN({mu_eps}, {sigma_eps})')
print('')

### computing prebunking node set X ###
print('computing prebunking node set X')

directory = f'results_sensitivity/{graph_name}/'

thetas = [0.1, 0.01, 0.001, 0.0001]
for theta in tqdm(thetas):
    X = algorithm.MIA_NPP(graph, S, kmax, theta)
    write_data(directory, f'theta={int(theta * 10000):05d}', X)

### conducting simulation ###
print('conducting simulation')

for theta in tqdm(thetas):
    name = f'theta={int(theta * 10000):05d}'
    path = directory + name +'.npy'
    if os.path.exists(path):
        X = np.load(path, allow_pickle=True)
        results = simulation.run_simulation(graph, S, X, kmax)
        filename = name + '_sim_results'
        write_data(directory, filename, results)
