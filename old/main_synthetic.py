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

import load_graph
import algorithm
import simulation

def embed_model_parameters(graph, q, epsilon):
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
    random.seed(42)
    out_degrees = dict(graph.out_degree())
    top_50_nodes = sorted(out_degrees, key=out_degrees.get, reverse=True)[:50]
    S = list(random.sample(top_50_nodes, num_nodes))
    return S

def get_largest_degree_node(graph):
    out_degrees = dict(graph.out_degree())
    largest_deg_node = max(out_degrees, key=out_degrees.get)
    S = [largest_deg_node]
    return S

def write_data(directory, filename, data):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, filename)
    np.save(filepath, data)

def get_truncated_normal(mu, sigma, lower=0, upper=1):
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)

########################################################################

# graph, graph_name = load_graph.Facebook_graph()
# graph, graph_name = load_graph.WikiVote_graph()
# graph, graph_name = load_graph.LastFM_graph()
graph, graph_name = load_graph.ca_HepTh_graph()
# graph, graph_name = load_graph.Deezer_graph()
# graph, graph_name = load_graph.Enron_graph()
# graph, graph_name = load_graph.Epinions_graph()
# graph, graph_name = load_graph.Twitter_graph()

### initial settings ###

S = get_random_high_degree_nodes(graph, 5)

random.seed(42)
np.random.seed(42)

mu_q = 0.7
sigma_q = 0.3
mu_eps=0.5
sigma_eps=0.1

q = {node: get_truncated_normal(mu_q, sigma_q) for node in graph.nodes()}
epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in graph.nodes}
for s in S:
    q[s] = 1.0

graph = embed_model_parameters(graph, q, epsilon)

kmax = 200
theta = 0.001

print('****' * 10)
print('')
print(f'graph: {graph_name}, seed node: {S}, q ~ truncN({mu_q}, {sigma_q}), epsilon ~ truncN({mu_eps}, {sigma_eps})')
print('')



### computing prebunking node set X ###
print('computing prebunking node set X')

directory = f'results_synthetic/{graph_name}/'
### Random
print('Random:')
X = algorithm.BaselineRandom(graph, S, kmax)
write_data(directory, 'Random', X)
### Degree
print('Degree:')
X = algorithm.BaselineDegree(graph, S, kmax)
write_data(directory, 'Degree', X)
### Distance
print('Distance:')
X = algorithm.BaselineDistance(graph, S, kmax)
write_data(directory, 'Distance', X)
### Gullible
print('Gullible:')
X = algorithm.BaselineGullible(graph, S, kmax)
write_data(directory, 'Gullible', X)
### MIA-NPP
print('MIA-NPP:')
X = algorithm.MIA_NPP(graph, S, kmax, theta)
write_data(directory, 'MIA-NPP', X)
### CMIA-O
print('CMIA-O:')
X = algorithm.CMIA_O(graph, S, kmax, theta)
write_data(directory, 'CMIA-O', X)
## AdvancedGreedy
print('AdvancedGreedy:')
num_samples = 100
X = algorithm.AdvancedGreedy(graph, S, kmax, num_samples)
write_data(directory, 'AdvancedGreedy', X)



### conducting simulation ###
print('conducting simulation')

alg_names = ['MIA-NPP', 'CMIA-O', 'AdvancedGreedy', 'Distance', 'Degree', 'Gullible', 'Random']
for alg_name in tqdm(alg_names):
    path = directory + alg_name + '.npy'
    if os.path.exists(path):
        X = np.load(path, allow_pickle=True)
        results = simulation.run_simulation(graph, S, X, kmax)
        filename = f'{alg_name}_sim_results'
        write_data(directory, filename, results)
