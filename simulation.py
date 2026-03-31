import numpy as np
import random
from collections import deque
from tqdm import tqdm

def run_ICN(graph, S, q, rng=None):
    if rng is None:
        seed = np.random.SeedSequence().entropy
        rng = np.random.default_rng(int(seed))
    node_status = {node: 0 for node in graph.nodes()}  # node activation status (0: inactive, 1: positive, 2: negative) 
    for s in S:
        node_status[s] = 1
    current_activated = set(S)  # current activated nodes
    while len(current_activated) > 0:
        current_positive = set()
        current_negative = set()
        for u in current_activated:
            if node_status[u] == 1:
                current_positive.add(u)
            else:
                current_negative.add(u)
        new_activated = set()
        for u in current_negative:
            for v in graph.successors(u):
                if node_status[v] == 0 and rng.random() < graph[u][v]['p_e']:
                    # if node u is negative, node v always become negative
                    node_status[v] = 2
                    new_activated.add(v)
        for u in current_positive:
            for v in graph.successors(u):
                if node_status[v] == 0 and rng.random() < graph[u][v]['p_e']:
                    # if node u is positive, node v becomes positive with prob. q_v and negative with prob. (1 - q_v)
                    if rng.random() < q[v]:
                        node_status[v] = 1
                    else:
                        node_status[v] = 2
                    new_activated.add(v)
        current_activated = new_activated
    fin_posi = {node for node, stat in node_status.items() if stat == 1} # finally positively activated nodes; fin_posi = {v ∈ V | node_status[v] = 1} 
    fin_nega = {node for node, stat in node_status.items() if stat == 2} # finally negatively activated nodes; fin_nega = {v ∈ V | node_status[v] = 2} 
    return fin_posi, fin_nega

def run_IC(graph, S):
    node_status = {node: 0 for node in graph.nodes()}  # node activation status (0: inactive, 1: active) 
    for s in S:
        node_status[s] = 1
    current_active = set(S)  # current activated nodes
    while len(current_active) > 0:
        new_activated = set()
        for u in current_active:
            for v in graph.successors(u):
                if node_status[v] == 0 and random.random() < graph[u][v]['p_e']:
                    node_status[v] = 1
                    new_activated.add(v)
        current_active = new_activated
    fin_active = {node for node, stat in node_status.items() if stat == 1} # finally positively activated nodes; fin_posi = {v ∈ V | node_status[v] = 1} 
    return fin_active

def run_COICM(graph, S_M, S_T):
    S_M = set(S_M)
    S_T = set(S_T)
    node_status = {node: 0 for node in graph.nodes()}  # node activation status (0: inactive, 1: positive, 2: negative) 
    for s in S_M:
        node_status[s] = 1
    for s in S_T:
        node_status[s] = 2
    current_activated = S_M | S_T  # current activated nodes
    while len(current_activated) > 0:
        current_positive = set()
        current_negative = set()
        for u in current_activated:
            if node_status[u] == 1:
                current_positive.add(u)
            else:
                current_negative.add(u)
        new_activated = set()
        for u in current_negative:
            for v in graph.successors(u):
                if node_status[v] == 0 and random.random() < graph[u][v]['p_e']:
                    # if node u is negative, node v always become negative
                    node_status[v] = 2
                    new_activated.add(v)
        for u in current_positive:
            for v in graph.successors(u):
                if node_status[v] == 0 and random.random() < graph[u][v]['p_e']:
                    node_status[v] = 1
                    new_activated.add(v)
        current_activated = new_activated
    fin_posi = {node for node, stat in node_status.items() if stat == 1} # finally positively activated nodes; fin_posi = {v ∈ V | node_status[v] = 1} 
    fin_nega = {node for node, stat in node_status.items() if stat == 2} # finally negatively activated nodes; fin_nega = {v ∈ V | node_status[v] = 2} 
    return fin_posi, fin_nega

def run_ICN_simulation(graph, S, X, kmax, num_simulation=2000, num_step=10, return_std=False):
    """
    各 k について ICN を num_simulation 回実行し、最終的な正感染ノード数 len(fin_p) の平均を返す。
    return_std=True のときは、同じ指標について標本標準偏差（ddof=1）も返す。
    """
    step = kmax//num_step
    xaxis = np.arange(0, kmax + step, step)
    print('xaxis:', xaxis)
    ave_misinfo_spread = []
    std_misinfo_spread = [] if return_std else None
    for k in tqdm(xaxis):
        X_k = X[:int(k)]
        q_X = {node: get_qX(node, X_k, graph) for node in graph.nodes()}
        spreads = []
        for _ in range(num_simulation):
            fin_p, fin_n = run_ICN(graph, S, q_X)
            spreads.append(len(fin_p))
        ave_misinfo_spread.append(float(np.mean(spreads)))
        if return_std:
            if num_simulation > 1:
                std_misinfo_spread.append(float(np.std(spreads, ddof=1)))
            else:
                std_misinfo_spread.append(0.0)
    if return_std:
        return ave_misinfo_spread, std_misinfo_spread
    return ave_misinfo_spread


def run_IC_simulation(graph, S, B, kmax):
    step = kmax//10
    xaxis = np.arange(0, kmax + step, step)
    num_simulation = 1000
    Y_ave = []
    for k in xaxis:
        B_k = B[:int(k)]
        blocked_graph = graph.copy()
        blocked_graph.remove_nodes_from(B_k)
        num_posi = 0
        for _ in range(num_simulation):
            fin_active = run_IC(blocked_graph, S)
            num_posi += len(fin_active)
        Y_ave.append(num_posi / num_simulation)
    return Y_ave

def run_COICM_simulation(graph, S_M, S_T, kmax):
    step = kmax//10
    xaxis = np.arange(0, kmax + step, step)
    num_simulation = 1000
    Y_ave = []
    for k in xaxis:
        ST_k = S_T[:int(k)]
        num_posi = 0
        for _ in range(num_simulation):
            fin_p, fin_n = run_COICM(graph, S_M, ST_k)
            num_posi += len(fin_p)
        Y_ave.append(num_posi / num_simulation)
    return Y_ave

def get_qX(v, X, graph):
    if v in X:
        qvX = (1 - graph.nodes[v]['epsilon']) * graph.nodes[v]['q']
    else:
        qvX = graph.nodes[v]['q']
    return qvX