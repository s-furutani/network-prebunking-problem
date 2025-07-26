import os
import csv
import json
import pickle
import pandas as pd
import numpy as np
import networkx as nx

def Nikolov_susceptibility_graph():
    path = './data/Right_and_Left/'
    
    measures_file = path+"measures.tab"
    friends_file = path+"anonymized-friends.json"
    
    df = pd.read_csv(measures_file, sep="\t")
    G = nx.DiGraph() 
    
    for _, row in df.iterrows():
        node_id = int(row["ID"])  # IDを整数に変換
        partisanship = row["Partisanship"]
        misinformation = row["Misinformation"]
        G.add_node(node_id, partisanship=partisanship, misinformation=misinformation)
    
    with open(friends_file, "r") as f:
        friend_data = json.load(f)
    
    for node, friends in friend_data.items():
        node = int(node)  # ノードIDを整数に変換
        friends = [int(f) for f in friends]  # 友達リストも整数に変換
    
        for friend in friends:
            G.add_edge(node, friend)  # 有向エッジを追加
    
    valid_nodes = [node for node in G.nodes if 'misinformation' in G.nodes[node]]
    
    subG = G.subgraph(valid_nodes).copy()
    weakly_connected_components = list(nx.weakly_connected_components(subG))
    lcc = max(weakly_connected_components, key=len)
    subG = subG.subgraph(lcc).copy()
    
    print(f"num_nodes: {subG.number_of_nodes()}, num_edges: {subG.number_of_edges()}")
    return subG


def convert_diffusion_networks_to_interaction_network(dataset_name):
    path = f'data/{dataset_name}/'
    
    # read mapping dictionary
    if dataset_name == 'politifact':
        with open(path + 'pol_id_twitter_mapping.pkl', 'rb') as f:
            node_to_twitter_id = pickle.load(f)
    elif dataset_name == 'gossipcop':
        with open(path + 'gos_id_twitter_mapping.pkl', 'rb') as f:
            node_to_twitter_id = pickle.load(f)
    
    # convert diffusion networks (node label: node number) to the interection network (node label: twitter id )
    combined_graph_edges = []
    with open(path + 'A.txt', 'r') as file:
        for line in file:
            node1, node2 = map(int, line.strip().split(', '))
            twitter_id1 = node_to_twitter_id[node1]
            twitter_id2 = node_to_twitter_id[node2]
            if dataset_name in twitter_id1:
                combined_graph_edges.append(('root', twitter_id2))
            else:
                combined_graph_edges.append((twitter_id1, twitter_id2))
                    
    # save edgelist of interaction network
    with open(path + 'combined_graph_edges_with_twitter_ids.txt', 'w') as f:
        for edge in combined_graph_edges:
            f.write(f"{edge[0]},{edge[1]}\n")

def compute_susceptibility(dataset_name):
    path = f'data/{dataset_name}/'

    ### read raw data
    data = np.load(path + 'node_graph_id.npy')
    node_graph_id = {idx: v for idx, v in enumerate(data)}
    
    data = np.load(path + 'graph_labels.npy')
    graph_labels = {}
    for idx, v in node_graph_id.items():
        graph_labels[idx] = data[v]

    if dataset_name == 'politifact':
        with open(path + 'pol_id_twitter_mapping.pkl', 'rb') as file:
            node_twitter_id = pickle.load(file)
        
        with open(path + 'pol_id_time_mapping.pkl', 'rb') as file:
            node_tweet_time = pickle.load(file)
    elif dataset_name == 'gossipcop':
        with open(path + 'gos_id_twitter_mapping.pkl', 'rb') as file:
            node_twitter_id = pickle.load(file)
        
        with open(path + 'gos_id_time_mapping.pkl', 'rb') as file:
            node_tweet_time = pickle.load(file)
    
    ### write node table
    with open(path + "node_info.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["node", "graph_id", "graph_label", "twitter_id", "tweet_utime"])
        for i in range(len(node_graph_id)):
            writer.writerow([i, node_graph_id[i], graph_labels[i], node_twitter_id[i], node_tweet_time[i]])
    
    ### compute susceptibility
    alpha = 1 # smoothing parameter
    data = pd.read_csv(path + 'node_info.csv')
    data['twitter_id'] = data['twitter_id'].apply(lambda x: 'root' if dataset_name in x else x)
    
    result = data.groupby('twitter_id').agg(
        num_share = ('twitter_id', 'size'), 
        num_fake_share = ('graph_label', lambda x: (x == 1).sum())
    ).reset_index()
    result['susceptibility'] = result.apply(
        lambda row: 1 if row['twitter_id'] == 'root' else (row['num_fake_share'] + alpha) / (row['num_share'] + 2 * alpha),  # Laplace additive smoothing
        axis=1
    )
    result.to_csv(path + 'twitter_id_susceptibility.csv', index=False)
    # print(result.head())

def FakeNewsNet_interaction_network(dataset_name):
    path = f'data/{dataset_name}/'
    graph_path = path + "combined_graph_edges_with_twitter_ids.txt"
    suscep_path = path + "twitter_id_susceptibility.csv"
    if not os.path.exists(graph_path):
        convert_diffusion_networks_to_interaction_network(dataset_name)
    if not os.path.exists(suscep_path):
        compute_susceptibility(dataset_name)

    G = nx.DiGraph()
    with open(graph_path, "r") as f:
        for line in f:
            source, target = line.strip().split(",")  
            G.add_edge(source, target)
    
    df = pd.read_csv(suscep_path)
    df["twitter_id"] = df["twitter_id"].astype(str).str.split(".").str[0]
    susceptibility_dict = dict(zip(df["twitter_id"], df["susceptibility"]))
    num_share_dict = dict(zip(df["twitter_id"], df["num_share"]))
    nx.set_node_attributes(G, susceptibility_dict, "susceptibility")

    num_all_share = num_share_dict['root']
    c = 30  
    # Based on preliminary diffusion simulations, we set the propagation scaling factor c=30, which results in approximately 10% of the network being activated on average.
    #   - Politifact (N = 30,813 nodes; average number of positive and negative active nodes):
    #       c = 1:  (41.74, 36.24)
    #       c = 5:  (210.13, 181.31)
    #       c = 10: (430.75, 369.26)
    #       c = 20: (895.89, 758.09)
    #       c = 30: (1377.73, 1156.77)
    #   - Gossipcop (N = 75,915 nodes; average number of positive and negative active nodes):
    #       c = 1:  (231.03, 309.70)
    #       c = 5:  (1070.69, 957.86)
    #       c = 10: (1999.65, 1598.51)
    #       c = 20: (3675.98, 2740.40)
    #       c = 30: (5198.59, 3734.72)
    for u, v in G.edges():
        v_share = num_share_dict.get(v, 0)
        p_uv = min(v_share / num_all_share * c, 1.0)
        G[u][v]['p_e'] = p_uv
        G[u][v]['-logp'] = - np.log(p_uv)
    
    return G
