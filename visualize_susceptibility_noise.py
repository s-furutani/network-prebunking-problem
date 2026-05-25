# %%
"""
Reed98 上の susceptibility 観測ノイズ q_obs をグラフ信号として可視化する。

セル構成:
  1. 設定・ヘルパー
  2. グラフ読み込み / q・q_obs 生成・npy 保存（固有ベクトル計算はここだけ）
  3. グラフ信号プロット（各パネルに Spearman ρ を記載、描画パラメータは trial-and-error 用）
"""

import os

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

import load_graph

# ---- 設定 ----
GRAPH_LOADER = load_graph.Reed98_network  # 差し替え用
SIGMA_DELTAS = [0, 0.1, 0.2, 0.5]
Q_MIN = 0.1
Q_MAX = 0.9
RANDOM_SEED = 42
CACHE_DIR = 'results/viz_susceptibility_noise/Reed98'
REGENERATE = False  # True で q / q_obs を再計算して上書き保存

# プロット調整用（セル3でいじる）
OUTPUT_PATH = 'figures/susceptibility_noise_reed98.png'
LAYOUT_SEED = 42
# 各サブプロット内の Spearman ρ 注記
SPEARMAN_TEXT_X = 0.05
SPEARMAN_TEXT_Y = 0.95
SPEARMAN_FONTSIZE = 11
NODE_SIZE = 20
EDGE_ALPHA = 0.08
FIGSIZE = (20, 5)
DPI = 150
# カラーバー用の右マージン（サブプロットと重ならないよう余白を確保）
SUBPLOT_RIGHT = 0.86   # tight_layout の右端
CBAR_LEFT = 0.88       # カラーバー axes の左端
CBAR_WIDTH = 0.02
CBAR_FONTSIZE = 22     # カラーバーのラベル・目盛り


def generate_q_homophilic(graph, q_min=Q_MIN, q_max=Q_MAX):
    """
    フィードラー向量のランキングと q のランキングを対応させ、
    近接ノードほど類似した susceptibility になるよう割り当てる。
    """
    g_und = graph.to_undirected()
    nodes = list(g_und.nodes())
    n = len(nodes)

    fiedler = nx.fiedler_vector(g_und)
    order = np.argsort(fiedler)
    q_sorted = np.linspace(q_min, q_max, n)
    q = {nodes[idx]: q_sorted[rank] for rank, idx in enumerate(order)}
    return q, nodes


def generate_q_obs(q, sigma_delta):
    """q_obs = clip(q + N(0, sqrt(sigma_delta)), 0, 1)"""
    q_obs = {}
    for node, q_val in q.items():
        q_observed = q_val + np.random.normal(loc=0, scale=np.sqrt(sigma_delta))
        q_obs[node] = np.clip(q_observed, 0, 1)
    return q_obs


def q_dict_to_array(q_dict, nodes):
    """固定ノード順でベクトル化"""
    return np.array([q_dict[v] for v in nodes], dtype=float)


def save_q_signals(cache_dir, graph_name, nodes, q_truth, q_obs_by_sigma):
    """q_truth と各 sigma_delta の q_obs を npy / npz で保存"""
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, 'nodes.npy'), np.array(nodes, dtype=object))
    np.save(os.path.join(cache_dir, 'q_truth.npy'), q_truth)

    q_obs_stack = np.stack(
        [q_obs_by_sigma[s] for s in SIGMA_DELTAS], axis=0
    )  # (len(SIGMA_DELTAS), |V|)
    np.savez(
        os.path.join(cache_dir, 'q_obs.npz'),
        sigma_deltas=np.array(SIGMA_DELTAS, dtype=float),
        q_obs=q_obs_stack,
        graph_name=graph_name,
        random_seed=RANDOM_SEED,
    )
    print(f'Saved: {cache_dir}/nodes.npy, q_truth.npy, q_obs.npz')


def load_q_signals(cache_dir):
    """保存済み信号を読み込む"""
    nodes = list(np.load(os.path.join(cache_dir, 'nodes.npy'), allow_pickle=True))
    q_truth = np.load(os.path.join(cache_dir, 'q_truth.npy'))
    data = np.load(os.path.join(cache_dir, 'q_obs.npz'))
    sigma_deltas = list(data['sigma_deltas'])
    q_obs_stack = data['q_obs']
    q_obs_by_sigma = {
        float(s): q_obs_stack[i] for i, s in enumerate(sigma_deltas)
    }
    graph_name = str(data['graph_name']) if 'graph_name' in data else ''
    return nodes, q_truth, q_obs_by_sigma, graph_name


def cache_exists(cache_dir):
    return all(
        os.path.isfile(os.path.join(cache_dir, f))
        for f in ('nodes.npy', 'q_truth.npy', 'q_obs.npz')
    )

# %% グラフ読み込み・q / q_obs 生成・保存

graph, graph_name = GRAPH_LOADER()
print(f'Graph: {graph_name}, |V|={len(graph.nodes())}, |E|={len(graph.edges())}')

if REGENERATE or not cache_exists(CACHE_DIR):
    print('Computing homophilic q (Fiedler vector) and q_obs ...')
    np.random.seed(RANDOM_SEED)
    q, nodes = generate_q_homophilic(graph)
    q_truth = q_dict_to_array(q, nodes)

    q_obs_by_sigma = {}
    print('q_obs summary:')
    for sigma_delta in SIGMA_DELTAS:
        q_obs = generate_q_obs(q, sigma_delta)
        q_obs_by_sigma[sigma_delta] = q_dict_to_array(q_obs, nodes)
        vals = q_obs_by_sigma[sigma_delta]
        print(
            f'  sigma_delta^2={sigma_delta}: '
            f'min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}'
        )

    save_q_signals(CACHE_DIR, graph_name, nodes, q_truth, q_obs_by_sigma)
else:
    print(f'Cache exists: {CACHE_DIR} (set REGENERATE=True to recompute)')
    nodes, q_truth, q_obs_by_sigma, graph_name = load_q_signals(CACHE_DIR)
    print(f'Loaded graph_name={graph_name}, |V|={len(nodes)}')

# %% プロット（描画パラメータは上の設定ブロックで調整）

graph, graph_name = GRAPH_LOADER()
nodes, q_truth, q_obs_by_sigma, _ = load_q_signals(CACHE_DIR)
g_und = graph.to_undirected()
pos = nx.spring_layout(g_und, k=0.01, seed=LAYOUT_SEED)

fig, axes = plt.subplots(1, 4, figsize=FIGSIZE, dpi=DPI)
nodes_collection = None

for ax, sigma_delta in zip(axes, SIGMA_DELTAS):
    node_colors = q_obs_by_sigma[sigma_delta]
    rho, _ = spearmanr(q_truth, node_colors)

    nx.draw_networkx_edges(
        g_und, pos, ax=ax, alpha=EDGE_ALPHA, width=0.3, arrows=False
    )
    nodes_collection = nx.draw_networkx_nodes(
        g_und,
        pos,
        nodelist=nodes,
        node_color=node_colors,
        cmap='RdYlBu_r',
        vmin=0,
        vmax=1,
        node_size=NODE_SIZE,
        ax=ax,
    )
    ax.set_title(rf'$\sigma_{{\delta}}^2 = {sigma_delta} ~~ (r = {rho:.3f})$', fontsize=22)
    ax.axis('off')

# サブプロット領域を先に確定してから、専用 axes にカラーバーを置く
plt.tight_layout(rect=[0, 0, SUBPLOT_RIGHT, 1])
cbar_ax = fig.add_axes([CBAR_LEFT, 0.12, CBAR_WIDTH, 0.76])
cbar = fig.colorbar(nodes_collection, cax=cbar_ax)
cbar.set_label(r'$q_{\mathrm{obs}}$', fontsize=CBAR_FONTSIZE)
cbar.ax.tick_params(labelsize=CBAR_FONTSIZE)

if OUTPUT_PATH:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches='tight', facecolor='white')
    print(f'Saved: {OUTPUT_PATH}')

plt.show()

# %%
