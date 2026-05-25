import networkx as nx
import numpy as np

### Undirected graph ###

def Facebook_graph():
    edges = []
    with open('data/facebook_combined.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split(' '))
            edges.append(data)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    lwcc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    graph = graph.to_directed()
    return graph, 'Facebook'

def LastFM_graph():
    edges = []
    with open('data/lastfm_asia_edges.csv', 'r') as file:
        for line in file:
            data = tuple(line.strip().split(','))
            edges.append(data)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    lwcc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    graph = graph.to_directed()
    return graph, 'LastFM'

def ca_HepTh_graph():
    edges = []
    with open('data/ca-HepTh.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split('\t'))
            edges.append(data)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    lwcc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    graph = graph.to_directed()
    return graph, 'ca-HepTh'

def cit_HepTh_graph():
    edges = []
    with open('data/cit-HepTh.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split('\t'))
            edges.append(data)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'cit-HepTh'

def Deezer_graph():
    edges = []
    with open('data/deezer_europe_edges.csv', 'r') as file:
        for line in file:
            data = tuple(line.strip().split(','))
            edges.append(data)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    lwcc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    graph = graph.to_directed()
    return graph, 'Deezer'

def Enron_graph():
    edges = []
    with open('data/email-Enron.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split('\t'))
            edges.append(data)
    graph = nx.Graph()
    graph.add_edges_from(edges)
    lwcc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    graph = graph.to_directed()
    return graph, 'Enron'

### Directed graph ###

def WikiVote_graph():
    edges = []
    with open('data/wiki-Vote.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split('\t'))
            edges.append(data)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'WikiVote'

def Epinions_graph():
    edges = []
    with open('data/soc-Epinions.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split('\t'))
            edges.append(data)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'Epinions'

def Twitter_graph():
    edges = []
    with open('data/twitter_combined.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split(' '))
            edges.append(data)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'Twitter'

def Stanford_Web_graph():
    edges = []
    with open('data/web-Stanford.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split('\t'))
            edges.append(data)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'Stanford_Web'

def Pokec_social_network():
    edges = []
    with open('data/soc-pokec-relationships.txt', 'r') as file:
        for line in file:
            data = tuple(line.strip().split('\t'))
            edges.append(data)
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'Pokec'

def Twitter_Higgs_graph():
    edges = []
    with open('data/soc-twitter-higgs.edges', 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'Twitter_Higgs'

def Congress_network():
    graph = nx.read_edgelist('data/congress_network/congress.edgelist', create_using=nx.DiGraph(), data=True)
    lwcc = max(nx.weakly_connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    return graph, 'Congress'

def Reed98_network():
    edges = []
    with open('data/socfb-Reed98.mtx', 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('%') or not line:
                continue
            parts = line.split()
            if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
                continue
            if len(parts) >= 2:
                edges.append((parts[0], parts[1]))
    
    graph = nx.Graph()
    graph.add_edges_from(edges)
    lwcc = max(nx.connected_components(graph), key=len)
    graph = graph.subgraph(lwcc)
    graph = graph.to_directed()
    return graph, 'Reed98'

# for i in range(9):
#     g = graphs[i]
#     print('***'*10)
#     print(graph_names[i])
#     dd = list(dict(g.out_degree()).values())
#     N = len(g.nodes())
#     M = len(g.edges())
#     print('|V| =', N, ', |E| =', M, ', d_max =', np.max(dd), ', d_min =', np.min(dd), ', d_ave =', np.round(np.average(dd), 2))

# ******************************
# Facebook
# |V| = 4039 , |E| = 176468 , d_max = 1045 , d_min = 1 , d_ave = 43.69
# ******************************
# WikiVote
# |V| = 7066 , |E| = 103663 , d_max = 893 , d_min = 0 , d_ave = 14.67
# ******************************
# LastFM
# |V| = 7624 , |E| = 55612 , d_max = 216 , d_min = 1 , d_ave = 7.29
# ******************************
# HepTh
# |V| = 8638 , |E| = 49633 , d_max = 65 , d_min = 1 , d_ave = 5.75
# ******************************
# HepPh
# |V| = 11204 , |E| = 235268 , d_max = 491 , d_min = 1 , d_ave = 21.0
# ******************************
# Deezer
# |V| = 28281 , |E| = 185504 , d_max = 172 , d_min = 1 , d_ave = 6.56
# ******************************
# Enron
# |V| = 33696 , |E| = 361622 , d_max = 1383 , d_min = 1 , d_ave = 10.73
# ******************************
# Epinions
# |V| = 75877 , |E| = 508836 , d_max = 1801 , d_min = 0 , d_ave = 6.71
# ******************************
# Twitter
# |V| = 81306 , |E| = 1768149 , d_max = 1205 , d_min = 0 , d_ave = 21.75

# graph, graph_name = Twitter_Higgs_graph()
# print(graph_name)
# print(len(graph.nodes()), len(graph.edges()))
# dd = list(dict(graph.out_degree()).values())
# print('d_max =', np.max(dd), ', d_min =', np.min(dd), ', d_ave =', np.round(np.average(dd), 2))