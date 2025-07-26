import time
import random
import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import wraps
import warnings


#################################################################
#                          utility
#################################################################

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        t = np.round(end_time - start_time, 2)
        print(f"⏱ {func.__name__} took {t} sec\n")
        return result
    return wrapper

def get_qX(v, X, graph):
    if v in X:
        qvX = (1 - graph.nodes[v]['epsilon']) * graph.nodes[v]['q']
    else:
        qvX = graph.nodes[v]['q']
    return qvX

def get_pp(graph, path):
    """Returns path propagation probability"""
    pp = 1
    edges = list(nx.utils.pairwise(path))
    for edge in edges:
        pp = pp * graph[edge[0]][edge[1]]['p_e']
    return pp

def get_seed_aggregated_graph(graph, S):
    out_neighbors_of_S = {v for s in S for v in graph.successors(s)}
    new_node = 's'
    new_graph = graph.copy()
    new_edges = []
    for v in out_neighbors_of_S:
        p_tmp = 1.0
        for u in S:
            if graph.has_edge(u, v):
                p_uv = graph[u][v]['p_e']
                p_tmp *= (1 - p_uv)
        p_sv = 1 - p_tmp
        new_edges.append((new_node, v, {'p_e': p_sv}))
    new_graph.add_node(new_node, q=1.0, epsilon=0.0)
    new_graph.add_edges_from(new_edges)
    new_graph.remove_nodes_from(S)
    return new_graph, new_node

def get_MIOA(graph, root_node, theta):
    mioa = nx.DiGraph()
    lengths, paths = nx.single_source_dijkstra(graph, source=root_node, weight='-logp')
    for node, path in paths.items():
        if node == root_node:  
            continue
        pp = get_pp(graph, path)
        if pp > theta:
            mioa.add_edges_from(nx.utils.pairwise(path))
    for u, w in mioa.edges():
        mioa[u][w]['p_e'] = graph[u][w]['p_e']
    for u in mioa.nodes():
        mioa.nodes[u]['q'] = graph.nodes[u]['q']
        mioa.nodes[u]['epsilon'] = graph.nodes[u]['epsilon']
    return mioa

def get_MIOA_with_igraph(graph, root_node, theta):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Create mapping from node to index and back
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}

    # Create igraph from NetworkX
    edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
    weights = [-np.log(graph[u][v]['p_e']) for u, v in graph.edges()]

    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(len(mapping))
    ig_graph.add_edges(edges)
    ig_graph.es['weight'] = weights

    root_idx = mapping[root_node]
    paths = ig_graph.get_shortest_paths(root_idx, weights='weight', mode='OUT', output='vpath')

    mioa = nx.DiGraph()
    for target_idx, path in enumerate(paths):
        if not path or target_idx == root_idx:
            continue
        node_path = [reverse_mapping[i] for i in path]
        pp = get_pp(graph, node_path)
        if pp > theta:
            mioa.add_edges_from(nx.utils.pairwise(node_path))

    for u, w in mioa.edges():
        mioa[u][w]['p_e'] = graph[u][w]['p_e']
    for u in mioa.nodes():
        mioa.nodes[u]['q'] = graph.nodes[u]['q']
        mioa.nodes[u]['epsilon'] = graph.nodes[u]['epsilon']
    return mioa

def get_MIIA(graph, root_node, theta):
    miia = nx.DiGraph()
    reversed_graph = graph.reverse(copy=False)
    lengths, paths = nx.single_source_dijkstra(reversed_graph, source=root_node, weight='-logp')
    for node, path in paths.items():
        if node == root_node:  
            continue
        r_path = path[::-1]
        pp = get_pp(graph, r_path)
        if pp > theta:
            miia.add_edges_from(nx.utils.pairwise(r_path))
    for u, w in miia.edges():
        miia[u][w]['p_e'] = graph[u][w]['p_e']
    for u in miia.nodes():
        miia.nodes[u]['q'] = graph.nodes[u]['q']
        miia.nodes[u]['epsilon'] = graph.nodes[u]['epsilon']
    return miia

def get_MIIA_with_igraph(graph, root_node, theta):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Create mapping from node to index and back
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}

    # Create igraph from NetworkX
    edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
    weights = [-np.log(graph[u][v]['p_e']) for u, v in graph.edges()]

    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(len(mapping))
    ig_graph.add_edges(edges)
    ig_graph.es['weight'] = weights

    root_idx = mapping[root_node]
    paths = ig_graph.get_shortest_paths(root_idx, weights='weight', mode='IN', output='vpath')

    miia = nx.DiGraph()
    for target_idx, path in enumerate(paths):
        if not path or target_idx == root_idx:
            continue
        node_path = [reverse_mapping[i] for i in path]
        r_path = node_path[::-1]  # reverse for correct edge direction
        pp = get_pp(graph, r_path)
        if pp > theta:
            miia.add_edges_from(nx.utils.pairwise(r_path))
    return miia


def remove_unreachable_nodes_from_S(miia, S):
    # remove unreachable nodes from S
    reachable_set = set()
    for s in S:
        if s not in miia.nodes():
            continue
        else:
            reachable_set |= nx.descendants(miia, s)
            reachable_set.add(s)
    unreachable_nodes = set(miia.nodes()) - reachable_set
    miia.remove_nodes_from(unreachable_nodes) 
    return miia

def remove_edges_to_S(miia, S):
    S = set(S)
    edges_to_S = [(u, v) for u, v in miia.edges() if v in S]
    miia.remove_edges_from(edges_to_S)
    return miia

def remove_cc_without_root(graph, root_node):
    ### Remove weakly connected components that do not include the root node.
    for cc in list(nx.weakly_connected_components(graph)):
        if root_node not in cc:
            graph.remove_nodes_from(cc)
    return graph

def get_SMIIA(graph, S, root_node, theta):
    # construct MIIA of a root node
    smiia = get_MIIA(graph, root_node, theta)
    smiia_nodes = set(smiia.nodes())

    # add edges from seed node s to each of its out-neighbors v, if both s and v are in SMIIA
    for s in S:
        if s not in smiia_nodes:
            continue
        for _, node, data in graph.out_edges(s, data=True):
            if node in smiia_nodes:
                smiia.add_edge(s, node, **data)

    smiia = remove_edges_to_S(smiia, S)  # remove directed edges to S
    smiia = remove_unreachable_nodes_from_S(smiia, S)  # remove unreachable nodes from S
    smiia = remove_cc_without_root(smiia, root_node)  # remove isolated seed nodes
    
    return smiia


#################################################################
#         get all MIIAs/PMIIAs/SMIIAs of nodes in U
#################################################################

# def get_all_MIIAs_in_U(graph, S, U, theta):
#     reversed_graph = graph.reverse(copy=False)
#     miia = {}
#     for v in tqdm(U):
#         miia_v = nx.DiGraph()
#         lengths, paths = nx.single_source_dijkstra(reversed_graph, source=v, weight='-logp')
#         for node, path in paths.items():
#             if node == v:  
#                 continue
#             r_path = path[::-1]
#             pp = get_pp(graph, r_path)
#             if pp > theta:
#                 miia_v.add_edges_from(nx.utils.pairwise(r_path))
#         for u, w in miia_v.edges():
#             miia_v[u][w]['p_e'] = graph[u][w]['p_e']
#         for u in miia_v.nodes():
#             miia_v.nodes[u]['q'] = graph.nodes[u]['q']
#             miia_v.nodes[u]['epsilon'] = graph.nodes[u]['epsilon']
#         miia[v] = miia_v
#     return miia

def get_all_MIIAs_in_U(graph, S, U, theta):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Create igraph from NetworkX
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    
    edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
    weights = [-np.log(graph[u][v]['p_e']) for u, v in graph.edges()]
    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(len(mapping))
    ig_graph.add_edges(edges)
    ig_graph.es['weight'] = weights

    miia = {}
    for v in tqdm(U):
        vid = mapping[v]
        paths = ig_graph.get_shortest_paths(vid, weights='weight', mode='IN', output='vpath')
        miia_v = nx.DiGraph()
        for target_id, path in enumerate(paths):
            if not path or target_id == vid:
                continue
            path_nodes = [reverse_mapping[i] for i in path]
            pp = get_pp(graph, path_nodes[::-1])  # Reverse path to match NetworkX edge order
            if pp > theta:
                miia_v.add_edges_from(nx.utils.pairwise(path_nodes[::-1]))
        for u, w in miia_v.edges():
            miia_v[u][w]['p_e'] = graph[u][w]['p_e']
        for u in miia_v.nodes():
            miia_v.nodes[u]['q'] = graph.nodes[u]['q']
            miia_v.nodes[u]['epsilon'] = graph.nodes[u]['epsilon']
        miia_v = remove_unreachable_nodes_from_S(miia_v, S)
        miia[v] = miia_v
    return miia

def get_all_PMIIAs_in_U(graph, S, X, U, theta):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Create igraph from NetworkX
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
    
    edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
    weights = [-np.log(graph[u][v]['p_e']) for u, v in graph.edges()]
    ig_graph = ig.Graph(directed=True)
    ig_graph.add_vertices(len(mapping))
    ig_graph.add_edges(edges)
    ig_graph.es['weight'] = weights

    miia = {}
    for v in tqdm(U):
        vid = mapping[v]
        paths = ig_graph.get_shortest_paths(vid, weights='weight', mode='IN', output='vpath')
        miia_v = nx.DiGraph()
        for target_id, path in enumerate(paths):
            if not path or target_id == vid:
                continue
            path_nodes = [reverse_mapping[i] for i in path]
            pp = get_pp(graph, path_nodes[::-1])  # Reverse path to match NetworkX edge order
            if pp > theta:
                miia_v.add_edges_from(nx.utils.pairwise(path_nodes[::-1]))
        for u, w in miia_v.edges():
            miia_v[u][w]['p_e'] = graph[u][w]['p_e']
        for u in miia_v.nodes():
            miia_v.nodes[u]['q'] = graph.nodes[u]['q']
            miia_v.nodes[u]['epsilon'] = graph.nodes[u]['epsilon']
        SX = set(S) | set(X)
        miia_v = remove_edges_to_S(miia_v, SX)
        miia[v] = miia_v
    return miia

def get_all_SMIIAs_in_U(graph, S, U, theta):
    smiia = get_all_MIIAs_in_U(graph, S, U, theta)

    # add edges from seed node s to each of its out-neighbors v, if both s and v are in SMIIA
    for v in U:
        for s in S:
            smiia_nodes = smiia[v].nodes()
            if s not in smiia_nodes:
                continue
            for _, node, data in graph.out_edges(s, data=True):
                if node in smiia_nodes:
                    smiia[v].add_edge(s, node, **data)
        
        smiia[v] = remove_edges_to_S(smiia[v], S)  # remove directed edges to S
        smiia[v] = remove_unreachable_nodes_from_S(smiia[v], S)  # remove unreachable nodes from S
        smiia[v] = remove_cc_without_root(smiia[v], root_node=v)  # remove isolated seed nodes
        
    return smiia

#################################################################
#                          MIA-NPP
#################################################################

@measure_time
def MIA_NPP(graph, S, k, theta):

    pap_cache = {}
    cache_hits = 0
    cache_misses = 0
    def get_pap_cached(v, S, X, miia_v):
        nonlocal cache_hits, cache_misses
        key = (v, tuple(sorted(X)))
        if key in pap_cache:
            cache_hits += 1
            pap_c = pap_cache[key]
        else:
            cache_misses += 1
            pap_c = get_pap_in_MIIA(v, S, X, miia_v)
            pap_cache[key] = pap_c
        return pap_c

    def get_pap_in_MIIA(u, S, X, MIIA):
        S = set(S)
        V_miia = set(MIIA.nodes())
        Z_0 = S & V_miia
        if u in S:
            return 1
        elif not bool(Z_0):  #  return 0 if there are no seed nodes in MIIA
            return 0
        else:
            ### initialization ###
            ppi_t = {v:0 for v in V_miia}  #  $\pi_t^+(v)$
            npi_t = {v:0 for v in V_miia}  #  $\pi_t^-(v)$
            pap_t = {v:0 for v in V_miia}  #  $ap_t^+(v)$
            nap_t = {v:0 for v in V_miia}  #  $ap_t^-(v)$
            for s in Z_0:
                ppi_t[s] = 1
                pap_t[s] = 1
            
            ### main loop ###
            Z_t = Z_0
            while Z_t:
                Z_tmp = set()
                beta_tmp = {v: 1 for v in V_miia}
                n_beta_tmp = {v: 1 for v in V_miia}
                for w in Z_t:
                    child_w = list(MIIA.successors(w))
                    if len(child_w) == 1:
                        v = child_w[0]
                        beta_tmp[v] *= (1 -  (ppi_t[w] + npi_t[w]) * MIIA[w][v]['p_e'])
                        n_beta_tmp[v] *= (1 - npi_t[w] * MIIA[w][v]['p_e'])
                        Z_tmp.add(v)
                    elif len(child_w) == 0:
                        continue
                    else:
                        raise ValueError(f"non-seed node {w} should have at most one child in MIIA, but found: {child_w}")
                for v in V_miia:
                    ppi_t[v] = 0
                    npi_t[v] = 0
                for v in Z_tmp:
                    qvX = get_qX(v, X, MIIA)
                    ppi_t[v] = qvX * ((1 - beta_tmp[v]) - (1 - n_beta_tmp[v])) * (1 - pap_t[v]) * (1 - nap_t[v])
                    npi_t[v] = (qvX * (1 - n_beta_tmp[v]) + (1 - qvX) * (1 - beta_tmp[v])) * (1 - pap_t[v]) * (1 - nap_t[v])
                    pap_t[v] = pap_t[v] + ppi_t[v]
                    nap_t[v] = nap_t[v] + npi_t[v]
                Z_t = Z_tmp
            
            return pap_t[u]

    S = set(S)
    V = set(graph.nodes())
    X = []
    U = set()
    Delta = {v:0 for v in V}
    for s in S:
        mioa_s = get_MIOA(graph, s, theta)
        U = U | set(mioa_s.nodes())
    U = U - S

    print("constructing all MIIAs in U")
    miia = get_all_MIIAs_in_U(graph, S, U, theta)

    print('first loop')
    for u in tqdm(U):
        pap_u_X = get_pap_in_MIIA(u, S, X, miia[u])
        for v in miia[u].nodes():
            pap_u_Xv = get_pap_in_MIIA(u, S, X + [v], miia[u])
            Delta_vu = (pap_u_X - pap_u_Xv)
            Delta[v] = Delta[v] + Delta_vu
    print('main loop')
    for i in tqdm(range(k)):
        u = max(U, key=lambda v: Delta[v])
        # mioa_u = get_MIOA(graph, u, theta)
        mioa_u = get_MIOA_with_igraph(graph, u, theta)
        V_mioa_u = set(mioa_u.nodes())
        for v in (V_mioa_u & U):
            pap_v_X = get_pap_cached(v, S, X, miia[v])
            for w in miia[v].nodes():
                if w != u:
                    pap_v_Xw = get_pap_cached(v, S, X + [w], miia[v])
                    Delta_wv = (pap_v_X - pap_v_Xw)
                    Delta[w] = Delta[w] - Delta_wv
        X.append(u)
        U = U - {u}
        for v in (V_mioa_u & U):
            pap_v_X = get_pap_cached(v, S, X, miia[v])
            for w in miia[v].nodes():
                if w != u:
                    pap_v_Xw = get_pap_cached(v, S, X + [w], miia[v])
                    Delta_wv = (pap_v_X - pap_v_Xw)
                    Delta[w] = Delta[w] + Delta_wv
    print(f"[Cache] hits: {cache_hits}, misses: {cache_misses}, hit rate: {cache_hits / (cache_hits + cache_misses + 1e-9):.2%}")
    return X


#################################################################
#                          SMIA-NPP
#################################################################

@measure_time
def SMIA_NPP(graph, S, k, theta):
    def get_pap_in_SMIIA(u, S, X, SMIIA):
        S = set(S)
        V_smiia = set(SMIIA.nodes())
        if u in S:
            return 1
        elif not bool(S & V_smiia):  #  return 0 if there are no seed nodes in SMIIA
            return 0
        else:
            Z_0 = S & V_smiia
            Z_1 = set()
            for v in Z_0:
                Z_1.update(SMIIA.successors(v))
            ### initialization ###
            ppi_t = {v:0 for v in V_smiia}  #  $\pi_t^+(v)$
            npi_t = {v:0 for v in V_smiia}  #  $\pi_t^-(v)$
            pap_t = {v:0 for v in V_smiia}  #  $ap_t^+(v)$
            nap_t = {v:0 for v in V_smiia}  #  $ap_t^-(v)$
            for s in Z_0:
                ppi_t[s] = 1
                pap_t[s] = 1
            ### t = 1 ###
            for v in V_smiia:
                ppi_t[v] = 0
                npi_t[v] = 0
            for v in Z_1:
                alpha = 1
                for s in set(SMIIA.predecessors(v)) & Z_0:
                    alpha *= (1 - SMIIA[s][v]['p_e'])
                alpha_tmp = 1 - alpha
                qvX = get_qX(v, X, SMIIA)
                ppi_t[v] = qvX * alpha_tmp
                npi_t[v] = (1 - qvX) * alpha_tmp
                pap_t[v] = pap_t[v] + ppi_t[v]
                nap_t[v] = nap_t[v] + npi_t[v]
            Z_t = Z_1
            ### main loop ###
            while Z_t:
                Z_tmp = set()
                beta_tmp = {v: 1 for v in V_smiia - S}
                n_beta_tmp = {v: 1 for v in V_smiia - S}
                for w in Z_t:
                    child_w = list(SMIIA.successors(w))
                    if len(child_w) == 1:
                        v = child_w[0]
                        beta_tmp[v] *= (1 -  (ppi_t[w] + npi_t[w]) * SMIIA[w][v]['p_e'])
                        n_beta_tmp[v] *= (1 - npi_t[w] * SMIIA[w][v]['p_e'])
                        Z_tmp.add(v)
                    elif len(child_w) == 0:
                        continue
                    else:
                        raise ValueError(f"non-seed node {w} should have at most one child in SMIIA, but found: {child_w}")
                for v in V_smiia:
                    ppi_t[v] = 0
                    npi_t[v] = 0
                for v in Z_tmp:
                    qvX = get_qX(v, X, SMIIA)
                    ppi_t[v] = qvX * ((1 - beta_tmp[v]) - (1 - n_beta_tmp[v])) * (1 - pap_t[v]) * (1 - nap_t[v])
                    npi_t[v] = (qvX * (1 - n_beta_tmp[v]) + (1 - qvX) * (1 - beta_tmp[v])) * (1 - pap_t[v]) * (1 - nap_t[v])
                    pap_t[v] = pap_t[v] + ppi_t[v]
                    nap_t[v] = nap_t[v] + npi_t[v]
                Z_t = Z_tmp
            
            return pap_t[u]

    S = set(S)
    V = set(graph.nodes())
    X = []
    U = set()
    Delta = {v:0 for v in V}
    for s in S:
        mioa_s = get_MIOA(graph, s, theta)
        U = U | set(mioa_s.nodes())
    U = U - S
    pap = {}
    
    pap_cache = {}
    cache_hits = 0
    cache_misses = 0
    def get_pap_cached(v, S, X, smiia_v):
        nonlocal cache_hits, cache_misses
        key = (v, tuple(sorted(X)))
        if key in pap_cache:
            cache_hits += 1
            pap_c = pap_cache[key]
        else:
            cache_misses += 1
            pap_c = get_pap_in_SMIIA(v, S, X, smiia_v)
            pap_cache[key] = pap_c
        return pap_c

    print("constructing all SMIIAs in U")
    smiia = get_all_SMIIAs_in_U(graph, S, U, theta)

    print('first loop')
    for u in tqdm(U):
        pap_u_X = get_pap_in_SMIIA(u, S, X, smiia[u])
        for v in smiia[u].nodes():
            pap_u_Xv = get_pap_in_SMIIA(u, S, X + [v], smiia[u])
            Delta_vu = (pap_u_X - pap_u_Xv)
            Delta[v] = Delta[v] + Delta_vu
    print('main loop')
    for i in tqdm(range(k)):
        u = max(U, key=lambda v: Delta[v])
        # mioa_u = get_MIOA(graph, u, theta)
        mioa_u = get_MIOA_with_igraph(graph, u, theta)
        V_mioa_u = set(mioa_u.nodes())
        for v in (V_mioa_u & U):
            pap_v_X = get_pap_cached(v, S, X, smiia[v])
            for w in smiia[v].nodes():
                if w != u:
                    pap_v_Xw = get_pap_cached(v, S, X + [w], smiia[v])
                    Delta_wv = (pap_v_X - pap_v_Xw)
                    Delta[w] = Delta[w] - Delta_wv
        X.append(u)
        U = U - {u}
        for v in (V_mioa_u & U):
            pap_v_X = get_pap_cached(v, S, X, smiia[v])
            for w in smiia[v].nodes():
                if w != u:
                    pap_v_Xw = get_pap_cached(v, S, X + [w], smiia[v])
                    Delta_wv = (pap_v_X - pap_v_Xw)
                    Delta[w] = Delta[w] + Delta_wv
    print(f"[Cache] hits: {cache_hits}, misses: {cache_misses}, hit rate: {cache_hits / (cache_hits + cache_misses + 1e-9):.2%}")
    return X

#################################################################
#                      GreedyMIOA-NPP 
#################################################################

@measure_time
def GreedyMIOA_NPP(graph, S, k, theta):
    def get_pap_in_MIOA(v, s, X, mioa_s, shortest_paths):
        # mip_sv = nx.shortest_path(mioa_s, source=s, target=v)  # by definition, MIOA must have an unique path from s to v
        mip_sv = shortest_paths[v]
        mip_edges = list(nx.utils.pairwise(mip_sv))
        pp_sv = 1
        qq_sv = 1
        for u, w in mip_edges:
            pp_sv *= mioa_s[u][w]['p_e']
            qq_sv *= get_qX(w, X, mioa_s)
        pap_v = pp_sv * qq_sv
        return pap_v

    def get_psigma_in_MIOA(s, X, mioa_s, shortest_paths):
        psigma = 1  # ap^+(s) = 1
        V = set(mioa_s.nodes()) - {s}
        for v in V:
            psigma += get_pap_in_MIOA(v, s, X, mioa_s, shortest_paths)
        return psigma

    X = []
    agg_G, agg_s = get_seed_aggregated_graph(graph, S)
    mioa_s = get_MIOA(agg_G, agg_s, theta)
    shortest_paths = dict(nx.single_source_shortest_path(mioa_s, source=agg_s))
    U = set(mioa_s.nodes()) - {agg_s}
    delta = {v:0 for v in U}
    psigma_X = get_psigma_in_MIOA(agg_s, X, mioa_s, shortest_paths)
    print('first loop')
    for v in tqdm(U):
        psigma_Xv = get_psigma_in_MIOA(agg_s, X + [v], mioa_s, shortest_paths)
        delta[v] = psigma_X - psigma_Xv
    print('main loop')
    for i in tqdm(range(k)):
        u = max(U, key=lambda v: delta[v])
        X.append(u)
        U = U - {u}
        psigma_X = get_psigma_in_MIOA(agg_s, X, mioa_s, shortest_paths)
        for v in nx.descendants(mioa_s, u):
            psigma_Xv = get_psigma_in_MIOA(agg_s, X + [v], mioa_s, shortest_paths)
            delta[v] = psigma_X - psigma_Xv
    return X


#################################################################
#                          CMIA-O
#################################################################

@measure_time
def CMIA_O(graph, S, k, theta, tau=0):

    def get_tau_hop_reachable_nodes(g, S, tau):
        result = set()
        for s in S:
            lengths = nx.single_source_shortest_path_length(g, s, cutoff=tau)
            result.update(lengths.keys())
        return result

    nap_cache = {}
    cache_hits = 0
    cache_misses = 0
    def get_nap_cached(v, S, X, miia_v):
        nonlocal cache_hits, cache_misses
        key = (v, tuple(sorted(X)))
        if key in nap_cache:
            cache_hits += 1
            nap_c = nap_cache[key]
        else:
            cache_misses += 1
            nap_c = get_nap_in_MIIA(u, S, X, miia_v)
            if len(nap_cache) > 50000:
                nap_cache.clear()  # CMIA-O requires cache clearing to avoid OOM errors on large graphs
            nap_cache[key] = nap_c
        return nap_c

    def get_nap_in_MIIA(u, Sn, Sp, MIIA):
        Sn = set(Sn)  # corresp. S
        Sp = set(Sp)  # corresp. X
        V_miia = set(MIIA.nodes())
        Z_0p = Sp & V_miia
        Z_0n = Sn & V_miia
        if u in Sn:
            return 1
        elif u in Sp:
            return 0
        elif not bool(Z_0p | Z_0n):  # return 0 if there are no seed nodes in MIIA
            return 0
        else:
            ### initialization ###
            ppi_t = {v:0 for v in V_miia}  #  $\pi_t^+(v)$
            npi_t = {v:0 for v in V_miia}  #  $\pi_t^-(v)$
            pap_t = {v:0 for v in V_miia}  #  $ap_t^+(v)$
            nap_t = {v:0 for v in V_miia}  #  $ap_t^-(v)$
            for s in Z_0p:
                ppi_t[s] = 1
                pap_t[s] = 1
            for s in Z_0n:
                npi_t[s] = 1
                nap_t[s] = 1
            ### main loop ###
            Z_tn = Z_0n
            Z_tp = Z_0p
            while Z_tn:
                Z_tmp_p = set()
                Z_tmp_n = set()
                p_beta_tmp = {v: 1 for v in V_miia}
                n_beta_tmp = {v: 1 for v in V_miia}

                ### positive loop ###
                for w in Z_tp:
                    child_w = list(MIIA.successors(w))
                    if len(child_w) == 1:
                        v = child_w[0]
                        p_beta_tmp[v] *= (1 -  ppi_t[w] * MIIA[w][v]['p_e'])
                        Z_tmp_p.add(v)
                    elif len(child_w) == 0:
                        continue
                    else:
                        raise ValueError(f"non-seed node {w} should have at most one child in MIIA, but found: {child_w}")
                for v in Z_tmp_p:
                    ppi_t[v] = (1 - p_beta_tmp[v]) * (1 - pap_t[v]) * (1 - nap_t[v])
                    pap_t[v] = pap_t[v] + ppi_t[v]
                
                ### negative loop ###
                for w in Z_tn:
                    child_w = list(MIIA.successors(w))
                    if len(child_w) == 1:
                        v = child_w[0]
                        n_beta_tmp[v] *= (1 -  npi_t[w] * MIIA[w][v]['p_e'])
                        Z_tmp_n.add(v)
                    elif len(child_w) == 0:
                        continue
                    else:
                        raise ValueError(f"non-seed node {w} should have at most one child in MIIA, but found: {child_w}")
                for v in Z_tmp_n:
                    npi_t[v] = p_beta_tmp[v] * (1 - n_beta_tmp[v]) * (1 - pap_t[v]) * (1 - nap_t[v])
                    nap_t[v] = nap_t[v] + npi_t[v]
                Z_tp = Z_tmp_p
                Z_tn = Z_tmp_n
            return nap_t.get(u, 0.0)

    S = set(S)
    V = set(graph.nodes())
    X = []
    U = set()
    DecInf = {v:0 for v in V}
    for s in S:
        mioa_s = get_MIOA(graph, s, theta)
        U = U | set(mioa_s.nodes())

    if tau == 0:
        U = U - S
    elif tau > 0:
        OOT = get_tau_hop_reachable_nodes(graph, S, tau)  # tau-hop reachable nodes are out-of-targets
        U = U - OOT

    print("constructing all MIIAs in U")
    miia = get_all_PMIIAs_in_U(graph, S, X, U, theta)

    print('first loop')
    for u in tqdm(U):
        nap_u_X = get_nap_in_MIIA(u, S, X, miia[u])
        for v in miia[u].nodes():
            nap_u_Xv = get_nap_in_MIIA(u, S, X + [v], miia[u])
            DecInf[v] += nap_u_X - nap_u_Xv
    print('main loop')
    for i in tqdm(range(k)):
        u = max(U, key=lambda v: DecInf[v])
        mioa_u = get_MIOA_with_igraph(graph, u, theta)
        V_mioa_u = set(mioa_u.nodes())
        for v in (V_mioa_u & U):
            nap_v_X = get_nap_cached(v, S, X, miia[v])
            for w in miia[v].nodes():
                if w != u:
                    nap_v_Xw = get_nap_cached(v, S, X + [w], miia[v])
                    DecInf[w] -= nap_v_X - nap_v_Xw
        X.append(u)
        U = U - {u}
        for v in (V_mioa_u & U):
            nap_v_X = get_nap_cached(v, S, X, miia[v])
            for w in miia[v].nodes():
                if w != u:
                    nap_v_Xw = get_nap_cached(v, S, X + [w], miia[v])
                    DecInf[w] += nap_v_X - nap_v_Xw
    print(f"[Cache] hits: {cache_hits}, misses: {cache_misses}, hit rate: {cache_hits / (cache_hits + cache_misses + 1e-9):.2%}")
    return X


#################################################################
#                          AdvancedGreedy
#################################################################

def generate_sampled_graph(graph):
    edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if random.random() <= data.get("p_e", 1.0)
    ]
    return nx.DiGraph(edges)

def dominator_tree(graph, source):
    '''
    constructing the dominator tree of a graph with a root (source) node
    '''
    if not nx.is_directed(graph):
        graph = graph.to_directed()
    if source not in graph:
        DT = nx.DiGraph()
        DT.add_node(source)
        return DT
    else:
        DT_edges_r = nx.immediate_dominators(graph, source).items()  # {(u, idom(u))}
        DT_edges = [(y, x) for x, y in DT_edges_r]  # {(idom(u), u)}
        DT = nx.DiGraph(DT_edges)
        DT.remove_edges_from(list(nx.selfloop_edges(DT)))
        return DT

def compute_subtree_sizes(tree, root_node):
    """
    Compute subtree sizes for all nodes in one DFS pass.
    Returns a dict: {node: size}
    """
    size = {}

    def dfs(v):
        size[v] = 1  # count itself
        for child in tree.successors(v):  # trace child nodes
            size[v] += dfs(child)  # add child's subtree size
        return size[v]
    try:
        dfs(root_node)
    except RecursionError:
        size[root_node] = 1
    return size     

@measure_time
def AdvancedGreedy(graph, S, k, num_samples):
    """
    calculating the blocker set by AdvancedGreedy    
    
    Input: 
    - graph: a graph with a single seed node
    - source: the seed node
    - k: a budget
    - num_samples: # of sampled graphs

    Output:
    - B: the blocker set
    """
    def compute_ESD(graph, source, num_samples):
        ESD = {node: 0 for node in graph.nodes()}
        for _ in range(num_samples):
            g = generate_sampled_graph(graph)  # generate a sampled graph
            DT = dominator_tree(g, source)  # construct the domniator tree of g
            st_size = compute_subtree_sizes(DT, source)       
            # compute Expected Spread Decrease (ESD) for each node u ∈ V-{s}    
            for u in graph.nodes():
                ESD[u] += st_size.get(u, 0) / num_samples
        ESD[source] = -9999999  # process for avoiding source node selection
        return ESD

    agg_G, agg_s = get_seed_aggregated_graph(graph, S)
    V = set(agg_G.nodes())
    B = []
    for _ in tqdm(range(k)):
        NB = list(V - set(B))
        graph_NB = agg_G.subgraph(NB).copy()
        ESD = compute_ESD(graph_NB, agg_s, num_samples)
        x = agg_s  # Note: ESD[source] = -9999999
        for u in graph_NB.nodes():
            if ESD[u] > ESD[x]:
                x = u
        B.append(x)
    return B


#################################################################
#                         Baseline
#################################################################

@measure_time
def BaselineRandom(graph, S, k):
    V = set(graph.nodes())
    S = set(S)
    random.seed(42)
    X = list(random.sample(list(V - S), k))
    return X

@measure_time
def BaselineDegree(graph, S, k):
    out_degrees = graph.out_degree()
    candidates = [
        node for node, _ in sorted(out_degrees, key=lambda x: x[1], reverse=True)
        if node not in S
    ]
    X = candidates[:k]
    return X

@measure_time
def BaselineDistance(graph, S, k):
    min_dist = {v: float('inf') for v in graph.nodes()}
    for s in S:
        lengths = nx.single_source_dijkstra_path_length(graph, source=s, weight='-logp')
        for node, dist in lengths.items():
            min_dist[node] = min(min_dist[node], dist)
    candidates = [
        node for node in sorted(graph.nodes(), key=lambda node: min_dist[node])
        if node not in S
    ]
    X = candidates[:k]
    return X

@measure_time
def BaselineGullible(graph, S, k):
    V = set(graph.nodes())
    S = set(S)
    X = sorted(list(V - S), key=lambda v: graph.nodes[v]['q'], reverse=True)[:k]
    return X