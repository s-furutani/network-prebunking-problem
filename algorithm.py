import os
import time
import random
import pickle
import hashlib
import networkx as nx
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functools import wraps
import heapq
import warnings
import simulation

#################################################################
#                          utility
#################################################################

#################################################################
#                       MIIA Cache
#################################################################

# Base theta for MIIA cache: theta > BASE_THETA can be derived by filtering theta=BASE_THETA MIIA
BASE_THETA = 0.01

def get_graph_hash(graph):
    """
    Generate a hash string for the graph based on its structure and edge weights.
    Used to identify cached MIIA files.
    """
    # Create a deterministic string representation of graph
    edges_str = str(sorted([(u, v, graph[u][v].get('p_e', 1.0)) for u, v in graph.edges()]))
    return hashlib.md5(edges_str.encode()).hexdigest()[:12]

def get_miia_cache_path(cache_dir, graph_hash, theta):
    """Get the file path for MIIA cache."""
    return os.path.join(cache_dir, f"miia_{graph_hash}_theta{theta}.pkl")

def save_miia_cache(miia_dict, cache_path):
    """
    Save MIIA dictionary to file.
    
    Parameters
    ----------
    miia_dict : dict
        Dictionary mapping node -> MIIA (NetworkX DiGraph)
    cache_path : str
        Path to save the cache file
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(miia_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[Cache] Saved MIIA cache to {cache_path}")

def load_miia_cache(cache_path):
    """
    Load MIIA dictionary from file.
    
    Parameters
    ----------
    cache_path : str
        Path to the cache file
    
    Returns
    -------
    miia_dict : dict or None
        Dictionary mapping node -> MIIA, or None if cache doesn't exist
    """
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            miia_dict = pickle.load(f)
        print(f"[Cache] Loaded MIIA cache from {cache_path}")
        return miia_dict
    return None

def filter_miia_by_theta(miia_v, S, v, theta):
    """
    Filter MIIA (built for a smaller theta) to keep only edges on paths with pp >= theta.
    Used when loading theta=BASE_THETA cache and deriving MIIA for larger theta.
    
    Parameters
    ----------
    miia_v : nx.DiGraph
        MIIA for node v (e.g. built with theta=BASE_THETA)
    S : set
        Seed node set
    v : node
        Root node of this MIIA
    theta : float
        New influence threshold (theta > theta used to build miia_v)
    
    Returns
    -------
    miia_filtered : nx.DiGraph
        MIIA containing only edges on paths from S to v with path probability > theta
    """
    Z_0 = set(S) & set(miia_v.nodes())
    if not Z_0 or v not in miia_v.nodes():
        return nx.DiGraph()
    
    edges_to_keep = set()
    for s in Z_0:
        try:
            for path in nx.all_simple_paths(miia_v, s, v):
                pp = get_pp(miia_v, path)
                if pp >= theta:
                    for u, w in nx.utils.pairwise(path):
                        edges_to_keep.add((u, w))
        except (nx.NodeNotFound, nx.NetworkXNoPath):
            continue
    
    if not edges_to_keep:
        return nx.DiGraph()
    
    miia_filtered = nx.DiGraph()
    miia_filtered.add_edges_from((u, w, miia_v[u][w]) for u, w in edges_to_keep)
    for node in miia_filtered.nodes():
        if node in miia_v.nodes():
            miia_filtered.nodes[node].update(miia_v.nodes[node])
    
    miia_filtered = remove_unreachable_nodes_from_S(miia_filtered, S)
    return miia_filtered

def get_or_build_miia_cache(graph, S, U, theta, cache_dir=None, miia_type='MIIA', X=None):
    """
    Get MIIA from cache or build and cache it.
    When theta > BASE_THETA, tries to load theta=BASE_THETA cache and filter by theta
    instead of rebuilding from scratch.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Input graph
    S : set
        Seed node set
    U : set
        Candidate node set
    theta : float
        Influence threshold
    cache_dir : str or None
        Directory to store cache files. If None, caching is disabled.
    miia_type : str
        Type of MIIA to build: 'MIIA', or 'PMIIA'
    X : list or None
        Intervention set (only used for PMIIA)
    
    Returns
    -------
    miia_dict : dict
        Dictionary mapping node -> MIIA
    """
    S = set(S)
    graph_hash = get_graph_hash(graph) if cache_dir else None
    cache_path = os.path.join(cache_dir, f"{miia_type.lower()}_{graph_hash}_theta{theta}.pkl") if cache_dir else None
    
    if cache_dir is not None:
        # Try to load cache for requested theta
        miia_dict = load_miia_cache(cache_path)
        if miia_dict is not None:
            if set(miia_dict.keys()) >= set(U):
                return {u: miia_dict[u] for u in U}
            print(f"[Cache] Cache exists but doesn't cover all nodes in U, rebuilding...")
        
        # If theta > BASE_THETA and no cache for theta, try loading BASE_THETA and filter
        if theta > BASE_THETA:
            base_cache_path = os.path.join(cache_dir, f"{miia_type.lower()}_{graph_hash}_theta{BASE_THETA}.pkl")
            base_miia = load_miia_cache(base_cache_path)
            if base_miia is not None and set(base_miia.keys()) >= set(U):
                print(f"[Cache] Filtering theta={BASE_THETA} MIIA by theta={theta}...")
                miia_dict = {}
                for u in U:
                    miia_dict[u] = filter_miia_by_theta(base_miia[u], S, u, theta)
                if cache_dir is not None:
                    save_miia_cache(miia_dict, cache_path)
                return miia_dict
    
    # Build MIIA from scratch
    print(f"[Cache] Building {miia_type}s for {len(U)} nodes...")
    if miia_type == 'MIIA':
        miia_dict = get_all_MIIAs_in_U(graph, S, U, theta)
    elif miia_type == 'PMIIA':
        if X is None:
            X = []
        miia_dict = get_all_PMIIAs_in_U(graph, S, X, U, theta)
    else:
        raise ValueError(f"Unknown miia_type: {miia_type}")
    
    if cache_dir is not None:
        if miia_type != 'PMIIA' or (X is not None and len(X) == 0):
            save_miia_cache(miia_dict, cache_path)
    
    return miia_dict

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

def get_MIOA(graph, root_node, theta, use_igraph=False):
    """
    root_node を根とする MIOA を返す。
    use_igraph=True のとき igraph で最短経路を計算（大規模グラフで高速）、False のとき NetworkX。
    """
    if use_igraph:
        return _get_MIOA_igraph(graph, root_node, theta)
    mioa = nx.DiGraph()
    lengths, paths = nx.single_source_dijkstra(graph, source=root_node, weight='-logp')
    for node, path in paths.items():
        if node == root_node:
            continue
        pp = get_pp(graph, path)
        if pp >= theta:
            mioa.add_edges_from(nx.utils.pairwise(path))
    for u, w in mioa.edges():
        mioa[u][w]['p_e'] = graph[u][w]['p_e']
    for u in mioa.nodes():
        mioa.nodes[u]['q'] = graph.nodes[u]['q']
        mioa.nodes[u]['epsilon'] = graph.nodes[u]['epsilon']
    return mioa


def _get_MIOA_igraph(graph, root_node, theta):
    """igraph を用いた MIOA 計算（get_MIOA(..., use_igraph=True) から呼ばれる）。"""
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mapping = {node: i for i, node in enumerate(graph.nodes())}
    reverse_mapping = {i: node for node, i in mapping.items()}
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
        if pp >= theta:
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
        if pp >= theta:
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
        if pp >= theta:
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


#################################################################
#         get all MIIAs/PMIIAs/SMIIAs of nodes in U
#################################################################


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
            if pp >= theta:
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
            if pp >= theta:
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

#################################################################
#                          MIA-NPP
#################################################################

@measure_time
def MIA_NPP(graph, S, k, theta, cache_dir=None):
    """
    MIA-NPP algorithm for selecting prebunking targets.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Input graph with edge attribute 'p_e' and node attributes 'q', 'epsilon'
    S : list or set
        Seed node set
    k : int
        Number of nodes to select
    theta : float
        Influence threshold for MIIA construction
    cache_dir : str or None
        Directory to store/load MIIA cache files. If None, caching is disabled.
    
    Returns
    -------
    X : list
        Selected prebunking target nodes
    """
    pap_cache = {}
    cache_hits = 0
    cache_misses = 0
    def get_pap_cached(v, S, X, miia_v, graph_for_q):
        nonlocal cache_hits, cache_misses
        key = (v, tuple(sorted(X)))
        if key in pap_cache:
            cache_hits += 1
            pap_c = pap_cache[key]
        else:
            cache_misses += 1
            pap_c = get_pap_in_MIIA(v, S, X, miia_v, graph_for_q)
            pap_cache[key] = pap_c
        return pap_c

    def get_pap_in_MIIA(u, S, X, MIIA, graph_for_q):
        """PAP 計算。MIIA は構造のみ使い、q/epsilon は graph_for_q から読む（キャッシュ MIIA の古い q を避ける）。"""
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
                    qvX = get_qX(v, X, graph_for_q)
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
        mioa_s = get_MIOA(graph, s, theta, use_igraph=True)
        U = U | set(mioa_s.nodes())
    print(f'U={len(U)}, S={len(S)}')

    # Build or load MIIA from cache
    miia = get_or_build_miia_cache(graph, S, U, theta, cache_dir, miia_type='MIIA')

    print('first loop')
    for u in tqdm(U):
        pap_u_X = get_pap_in_MIIA(u, S, X, miia[u], graph)
        for v in miia[u].nodes():
            pap_u_Xv = get_pap_in_MIIA(u, S, X + [v], miia[u], graph)
            Delta_vu = (pap_u_X - pap_u_Xv)
            Delta[v] = Delta[v] + Delta_vu
    print('main loop')
    for i in tqdm(range(k)):
        if not U:
            break  # theta large: candidate set exhausted before k selections
        u = max(U, key=lambda v: Delta[v])
        mioa_u = get_MIOA(graph, u, theta, use_igraph=True)
        V_mioa_u = set(mioa_u.nodes())
        for v in (V_mioa_u & U):
            pap_v_X = get_pap_cached(v, S, X, miia[v], graph)
            for w in miia[v].nodes():
                if w != u:
                    pap_v_Xw = get_pap_cached(v, S, X + [w], miia[v], graph)
                    Delta_wv = (pap_v_X - pap_v_Xw)
                    Delta[w] = Delta[w] - Delta_wv
        X.append(u)
        U = U - {u}
        for v in (V_mioa_u & U):
            pap_v_X = get_pap_cached(v, S, X, miia[v], graph)
            for w in miia[v].nodes():
                if w != u:
                    pap_v_Xw = get_pap_cached(v, S, X + [w], miia[v], graph)
                    Delta_wv = (pap_v_X - pap_v_Xw)
                    Delta[w] = Delta[w] + Delta_wv
    print(f"[Cache] hits: {cache_hits}, misses: {cache_misses}, hit rate: {cache_hits / (cache_hits + cache_misses + 1e-9):.2%}")
    return X



#################################################################
#                          CMIA-O
#################################################################

@measure_time
def CMIA_O(graph, S, k, theta, tau=0, cache_dir=None):
    """
    CMIA-O algorithm for influence blocking maximization.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Input graph with edge attribute 'p_e' and node attributes 'q', 'epsilon'
    S : list or set
        Seed node set (misinformation seeds)
    k : int
        Number of nodes to select
    theta : float
        Influence threshold for MIIA construction
    tau : int
        Delay parameter (nodes within tau hops of S are excluded)
    cache_dir : str or None
        Directory to store/load MIIA cache files. If None, caching is disabled.
    
    Returns
    -------
    X : list
        Selected clarification seed nodes
    """
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
            nap_c = get_nap_in_MIIA(v, S, X, miia_v)
            # if len(nap_cache) > 50000:
            #     nap_cache.clear()  # CMIA-O requires cache clearing to avoid OOM errors on large graphs
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
        mioa_s = get_MIOA(graph, s, theta, use_igraph=True)
        U = U | set(mioa_s.nodes())

    if tau == 0:
        U = U - S
    elif tau > 0:
        OOT = get_tau_hop_reachable_nodes(graph, S, tau)  # tau-hop reachable nodes are out-of-targets
        U = U - OOT

    # Build or load PMIIA from cache (X=[] at this point, so caching is valid)
    miia = get_or_build_miia_cache(graph, S, U, theta, cache_dir, miia_type='PMIIA', X=X)

    print('first loop')
    for u in tqdm(U):
        nap_u_X = get_nap_in_MIIA(u, S, X, miia[u])
        for v in miia[u].nodes():
            nap_u_Xv = get_nap_in_MIIA(u, S, X + [v], miia[u])
            DecInf[v] += nap_u_X - nap_u_Xv
    print('main loop')
    for i in tqdm(range(k)):
        if not U:
            break  # theta large: candidate set exhausted before k selections
        u = max(U, key=lambda v: DecInf[v])
        mioa_u = get_MIOA(graph, u, theta, use_igraph=True)
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
    """Original (slow) version - kept for reference"""
    edges = [
        (u, v, data)
        for u, v, data in graph.edges(data=True)
        if random.random() <= data.get("p_e", 1.0)
    ]
    return nx.DiGraph(edges)

def generate_sampled_edges_fast(edges_array, probs_array):
    """
    Fast edge sampling using NumPy vectorization.
    
    Parameters
    ----------
    edges_array : np.ndarray
        Shape (m, 2) array of edge indices
    probs_array : np.ndarray
        Shape (m,) array of propagation probabilities
    
    Returns
    -------
    sampled_edges : np.ndarray
        Sampled edges as array of shape (k, 2)
    """
    mask = np.random.random(len(probs_array)) <= probs_array
    return edges_array[mask]

def dominator_tree(graph, source):
    """Original (slow) version using NetworkX - kept for reference"""
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

def compute_dominators_and_subtree_sizes_igraph(sampled_edges, n_nodes, source_idx):
    """
    Compute dominator tree and subtree sizes using igraph (fast C implementation).
    
    Parameters
    ----------
    sampled_edges : np.ndarray
        Shape (k, 2) array of sampled edge indices
    n_nodes : int
        Number of nodes in the graph
    source_idx : int
        Index of the source node
    
    Returns
    -------
    subtree_sizes : np.ndarray
        Array of subtree sizes for each node (0 if unreachable)
    """
    # Create igraph from sampled edges
    g = ig.Graph(n=n_nodes, edges=sampled_edges.tolist(), directed=True)
    
    # Compute dominators using igraph (returns list where dom[v] = immediate dominator of v)
    # API: python-igraph uses g.dominator(root, mode='out'), not dominators
    dom = list(g.dominator(source_idx, mode='out'))
    if len(dom) != n_nodes:
        raise ValueError(f"dominator returned len {len(dom)}, expected n_nodes={n_nodes}")

    # Build dominator tree as adjacency list (parent -> children)
    children = [[] for _ in range(n_nodes)]
    for v in range(n_nodes):
        raw = dom[v]
        if raw != raw or raw < 0:  # NaN or -1 (unreachable)
            continue
        idom = int(raw)
        if idom != v:  # not self-loop
            children[idom].append(v)
    
    # Compute subtree sizes using iterative DFS (avoids RecursionError)
    subtree_sizes = np.zeros(n_nodes, dtype=np.int32)
    
    # Root has no dominator (often -1). Source must not be NaN (unreachable/error)
    raw_src = dom[source_idx]
    if raw_src != raw_src:
        raise ValueError(f"source_idx={source_idx} has NaN dominator (unreachable or invalid graph)")
    
    stack = [(source_idx, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            subtree_sizes[node] = 1 + sum(subtree_sizes[c] for c in children[node])
        else:
            stack.append((node, True))
            for child in children[node]:
                stack.append((child, False))
    
    return subtree_sizes

def compute_subtree_sizes(tree, root_node):
    """
    Original version - Compute subtree sizes for all nodes in one DFS pass.
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
    Calculate the blocker set using AdvancedGreedy algorithm.
    Optimized version using NumPy vectorization and igraph.
    
    Parameters
    ----------
    graph : nx.DiGraph
        Input graph with edge attribute 'p_e' for propagation probability
    S : list
        Seed node set
    k : int
        Budget (number of nodes to select)
    num_samples : int
        Number of sampled graphs for Monte Carlo estimation

    Returns
    -------
    B : list
        Selected blocker node set
    """
    def compute_ESD_fast(node_list, node_to_idx, edges_array, probs_array, source_idx, num_samples):
        """
        Fast computation of Expected Spread Decrease using NumPy and igraph.
        """
        n_nodes = len(node_list)
        ESD_array = np.zeros(n_nodes, dtype=np.float64)
        
        for _ in range(num_samples):
            # Fast edge sampling using NumPy vectorization
            sampled_edges = generate_sampled_edges_fast(edges_array, probs_array)
            
            # Fast dominator tree and subtree size computation using igraph
            subtree_sizes = compute_dominators_and_subtree_sizes_igraph(
                sampled_edges, n_nodes, source_idx
            )
            
            ESD_array += subtree_sizes
        
        ESD_array /= num_samples
        ESD_array[source_idx] = -9999999  # Avoid selecting source node
        
        # Convert to dict for compatibility
        return {node_list[i]: ESD_array[i] for i in range(n_nodes)}

    # Aggregate seed nodes into a single virtual node
    agg_G, agg_s = get_seed_aggregated_graph(graph, S)
    V = set(agg_G.nodes())
    B = []
    
    for _ in tqdm(range(k)):
        NB = list(V - set(B))
        graph_NB = agg_G.subgraph(NB).copy()
        
        # Preprocessing: convert graph to NumPy arrays for fast computation
        node_list = list(graph_NB.nodes())
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        edges = list(graph_NB.edges())
        if len(edges) == 0:
            break
            
        edges_array = np.array([[node_to_idx[u], node_to_idx[v]] for u, v in edges], dtype=np.int32)
        probs_array = np.array([graph_NB[u][v].get('p_e', 1.0) for u, v in edges], dtype=np.float64)
        source_idx = node_to_idx[agg_s]
        
        # Compute ESD using fast version
        ESD = compute_ESD_fast(node_list, node_to_idx, edges_array, probs_array, source_idx, num_samples)
        
        # Select node with maximum ESD
        x = agg_s  # Note: ESD[source] = -9999999
        for u in graph_NB.nodes():
            if ESD[u] > ESD[x]:
                x = u
        B.append(x)
    
    return B


#################################################################
#                         Baseline
#################################################################


def MC_simulation(graph, S, X, num_mc_samples=1000, q_X_base=None, intervention_node=None):
    """
    Monte Carlo で ICN の負のスプレッド平均を返す。
    q_X_base と intervention_node を渡すと、q_X を base から 1 ノードだけ更新して使い、
    辞書の全ノード分 get_qX を呼ぶのを避ける（X+[v] 評価の高速化用）。
    """
    if q_X_base is not None and intervention_node is not None:
        q_X = q_X_base.copy()
        v = intervention_node
        q_X[v] = (1 - graph.nodes[v]['epsilon']) * graph.nodes[v]['q']
    else:
        q_X = {node: get_qX(node, X, graph) for node in graph.nodes()}
    ave_correct_spread = 0
    for seed in range(num_mc_samples):
        rng = np.random.default_rng(seed)
        fin_p, fin_n = simulation.run_ICN(graph, S, q_X, rng)
        ave_correct_spread += len(fin_n)
    ave_correct_spread = ave_correct_spread / num_mc_samples
    return ave_correct_spread


# -----------------------------
# CRN: 乱数場を固定（edge-live packbits + node coin を trial/node で決定的に）
# -----------------------------
def build_fixed_edge_index(graph):
    """
    ノード順・隣接順を固定（sorted）。(u,v) → edge_id。edge_id 順に p_e 配列。
    """
    nodes_sorted = sorted(graph.nodes())
    succ = {u: sorted(graph.successors(u)) for u in nodes_sorted}
    edge_id = {}
    p_e = []
    eid = 0
    for u in nodes_sorted:
        for v in succ[u]:
            edge_id[(u, v)] = eid
            pe = graph[u][v].get("p_e", 0.01)
            p_e.append(float(pe))
            eid += 1
    return nodes_sorted, succ, edge_id, np.asarray(p_e, dtype=np.float32)


def precompute_edge_live_packed(num_mc_samples, p_e):
    """
    edge_live[r, eid] を packbits で (R, ceil(m/8)) の uint8 にして返す。
    """
    m = p_e.shape[0]
    m_bytes = (m + 7) // 8
    edge_live_packed = np.empty((num_mc_samples, m_bytes), dtype=np.uint8)
    for r in range(num_mc_samples):
        rng = np.random.default_rng(r)
        live = (rng.random(m) < p_e)
        edge_live_packed[r] = np.packbits(live, bitorder="little")
    return edge_live_packed


def edge_is_live(edge_live_packed_r, eid):
    """packbits(bitorder="little") に合わせたビット参照。"""
    byte = edge_live_packed_r[eid >> 3]
    bit = (byte >> (eid & 7)) & 1
    return bit == 1


def splitmix64(x):
    """splitmix64 系で trial/node から U(0,1) を決定的に生成。"""
    x = (x + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    z = z ^ (z >> 31)
    return z & 0xFFFFFFFFFFFFFFFF


def hash_u01(trial, node):
    """(trial, node) → U(0,1) を決定的に返す。node が int でない場合は hash(node) を使用。"""
    n = node if isinstance(node, (int, np.integer)) else (hash(node) & 0xFFFFFFFFFFFFFFFF)
    x = (np.uint64(trial) << np.uint64(32)) ^ np.uint64(n) ^ np.uint64(0xD1B54A32D192ED03)
    h = splitmix64(int(x))
    return ((h >> 11) & ((1 << 53) - 1)) / float(1 << 53)


def run_ICN_crn_packed_hash(graph, S, q, nodes_sorted, succ, edge_id, edge_live_packed_r, trial):
    """
    CRN 版 ICN: 辺の成否は edge_live_packed_r、正/負コインは hash_u01(trial, v)。
    順序固定（sorted）で乱数消費ズレを防ぐ。
    """
    node_status = {node: 0 for node in nodes_sorted}
    for s in S:
        node_status[s] = 1
    current_activated = sorted(S)
    while current_activated:
        current_positive = [u for u in current_activated if node_status[u] == 1]
        current_negative = [u for u in current_activated if node_status[u] == 2]
        new_activated = []
        for u in current_negative:
            for v in succ[u]:
                if node_status[v] != 0:
                    continue
                eid = edge_id[(u, v)]
                if edge_is_live(edge_live_packed_r, eid):
                    node_status[v] = 2
                    new_activated.append(v)
        for u in current_positive:
            for v in succ[u]:
                if node_status[v] != 0:
                    continue
                eid = edge_id[(u, v)]
                if edge_is_live(edge_live_packed_r, eid):
                    if hash_u01(trial, v) < q[v]:
                        node_status[v] = 1
                    else:
                        node_status[v] = 2
                    new_activated.append(v)
        current_activated = sorted(set(new_activated))
    fin_posi = {node for node, stat in node_status.items() if stat == 1}
    fin_nega = {node for node, stat in node_status.items() if stat == 2}
    return fin_posi, fin_nega


def MC_simulation_crn(graph, S, X, num_mc_samples=1000, q_X_base=None, intervention_node=None, crn_cache=None):
    """
    CRN 版 MC: crn_cache 必須。q_X_base / intervention_node の最適化を維持。
    """
    if crn_cache is None:
        raise ValueError("crn_cache is required.")
    nodes_sorted, succ, edge_id, edge_live_packed = crn_cache
    if q_X_base is not None and intervention_node is not None:
        q_X = q_X_base.copy()
        v = intervention_node
        q_X[v] = (1 - graph.nodes[v]['epsilon']) * graph.nodes[v]['q']
    else:
        q_X = {node: get_qX(node, X, graph) for node in graph.nodes()}
    ave_neg = 0.0
    for r in range(num_mc_samples):
        fin_p, fin_n = run_ICN_crn_packed_hash(
            graph, S, q_X, nodes_sorted, succ, edge_id, edge_live_packed[r], trial=r
        )
        ave_neg += len(fin_n)
    return ave_neg / num_mc_samples


def build_crn_cache(graph, num_mc_samples):
    """CRN 用の事前計算: 固定辺インデックス + edge_live packbits。"""
    nodes_sorted, succ, edge_id, p_e = build_fixed_edge_index(graph)
    edge_live_packed = precompute_edge_live_packed(num_mc_samples, p_e)
    return (nodes_sorted, succ, edge_id, edge_live_packed)


@measure_time
def Greedy(graph, S, k, num_mc_samples):
    V = set(graph.nodes())
    S = set(S)
    X = []
    for _ in tqdm(range(k), desc='Greedy'):
        best_gain = -9999999
        best_v = None
        f_SX = MC_simulation(graph, S, X, num_mc_samples)
        for v in V - set(X):
            f_SXv = MC_simulation(graph, S, X + [v], num_mc_samples)
            gain_v = f_SXv - f_SX
            if gain_v > best_gain:
                best_gain = gain_v
                best_v = v
        X.append(best_v)
        # print(f'best_v: {best_v}, best_gain: {best_gain}')    
    return X

@measure_time
def CELF_naive(graph, S, k, num_mc_samples):
    """
    元の CELF 実装（後で戻せるように残す）。毎回 full sort、q_X は毎回全ノード構築。
    """
    V = set(graph.nodes())
    S = set(S)
    X = []
    gains = []
    nodes = []

    f_SX = MC_simulation(graph, S, X, num_mc_samples)
    for v in tqdm(V, desc='initial calculation'):
        f_SXv = MC_simulation(graph, S, X + [v], num_mc_samples)
        gain_v = f_SXv - f_SX
        gains.append(gain_v)
        nodes.append(v)
    Q = sorted(zip(nodes, gains), key=lambda x: x[1], reverse=True)
    best_v = Q[0][0]
    best_gain = Q[0][1]
    X.append(best_v)
    Q = Q[1:]

    for _ in tqdm(range(k - 1), desc='CELF'):
        check = False
        f_SX = f_SX + best_gain
        while not check:
            current_v = Q[0][0]
            f_SXv = MC_simulation(graph, S, X + [current_v], num_mc_samples)
            gain_v = f_SXv - f_SX
            Q[0] = (current_v, gain_v)
            Q = sorted(Q, key=lambda x: x[1], reverse=True)
            check = (Q[0][0] == current_v)
        best_v = Q[0][0]
        best_gain = Q[0][1]
        X.append(best_v)
        Q = Q[1:]
    return X


@measure_time
def CELF(graph, S, k, num_mc_samples, use_crn=False):
    # Note:
    # CELF can be slow on large graphs, as it repeatedly evaluates marginal gains using Monte Carlo simulations.
    # 高速化: (1) q_X を round ごとに 1 回だけ構築し、X+[v] 評価は q_X_base + intervention_node で渡す。
    #        (2) 優先度キュー（max-heap）でトップだけ再評価し、毎回の full sort を避ける。
    #        (3) use_crn=True のとき CRN（edge-live packbits + hash_u01）で分散低減・高速化。
    V = set(graph.nodes())
    S = set(S)
    X = []

    if use_crn:
        crn_cache = build_crn_cache(graph, num_mc_samples)
        def mc(*args, **kwargs):
            kwargs["crn_cache"] = crn_cache
            return MC_simulation_crn(*args, **kwargs)
    else:
        def mc(*args, **kwargs):
            return MC_simulation(*args, **kwargs)

    # 初期ラウンド: f_SX と全ノードの marginal gain を計算
    f_SX = mc(graph, S, X, num_mc_samples)
    base_q = {node: graph.nodes[node]['q'] for node in graph.nodes()}
    best_gain = {}  # v -> 現在の gain（heap の stale 判定用）
    # max-heap: (-gain, v) で最小ヒープとして最大 gain がトップに
    heap = []
    for v in tqdm(V, desc='CELF initial'):
        f_SXv = mc(graph, S, X, num_mc_samples, q_X_base=base_q, intervention_node=v)
        gain_v = f_SXv - f_SX
        best_gain[v] = gain_v
        heapq.heappush(heap, (-gain_v, v))
    best_v = heapq.heappop(heap)[1]
    best_gain_val = best_gain[best_v]
    X.append(best_v)

    for _ in tqdm(range(k - 1), desc='CELF'):
        f_SX = f_SX + best_gain_val
        q_X_base = {node: get_qX(node, X, graph) for node in graph.nodes()}
        check = False
        while not check:
            neg_g, current_v = heapq.heappop(heap)
            if best_gain.get(current_v) is None or best_gain[current_v] != -neg_g:
                continue  # stale、次を pop
            f_SXv = mc(graph, S, X, num_mc_samples, q_X_base=q_X_base, intervention_node=current_v)
            gain_v = f_SXv - f_SX
            best_gain[current_v] = gain_v
            heapq.heappush(heap, (-gain_v, current_v))
            # トップが再評価したノードと同じなら終了
            neg_g2, top_v = heap[0]
            check = (top_v == current_v and best_gain[current_v] == -neg_g2)
        best_v = heap[0][1]
        best_gain_val = best_gain[best_v]
        X.append(best_v)
        heapq.heappop(heap)  # 選んだノードを heap から除去
    return X


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