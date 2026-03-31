"""
MIA-NPP runtime analysis: measure execution time vs n and theta
on ER graphs with uniform edge probability p_e=0.5.
"""

import os
import pickle
import time
import random
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

import algorithm

# Default parameters (aligned with run_benchmark_experiment)
DEFAULT_N_LIST = [100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]
# DEFAULT_THETA_LIST = [0.1, 0.01, 0.001]
DEFAULT_THETA_LIST = [0.05, 0.01]
# DEFAULT_THETA_LIST = [0.1, 0.01]
K = 100
NUM_SEEDS = 5
AVG_DEGREE = 20
P_E = 0.2
RESULTS_CSV = "results/runtime_analysis_mia_npp.csv"
RESULTS_PNG = "runtime_analysis_mia_npp.png"
# グラフキャッシュ: 同じ n で一度構築したグラフを保存し、theta を変えて計測するときに再利用
GRAPH_CACHE_DIR = "results/runtime_graph_cache"


def get_truncated_normal(mu, sigma, lower=0, upper=1):
    """Sample from truncated normal distribution (same as main.py)."""
    a, b = (lower - mu) / sigma, (upper - mu) / sigma
    return truncnorm.rvs(a, b, loc=mu, scale=sigma)


def build_er_graph_for_runtime(
    n,
    avg_degree=AVG_DEGREE,
    p_e=P_E,
    mu_q=0.7,
    sigma_q=0.3,
    mu_eps=0.5,
    sigma_eps=0.1,
    random_seed=0,
):
    """
    Build ER graph for runtime analysis: LCC of G(n,p) with p = avg_degree/(n-1),
    directed, uniform p_e and truncated-normal q/epsilon.

    Returns
    -------
    G : nx.DiGraph
        Graph with p_e, -logp on edges and q, epsilon on nodes (seeds not yet set).
    lcc_nodes : list
        Node list of LCC (for seed selection).
    """
    np.random.seed(random_seed)
    random.seed(random_seed)
    p = avg_degree / max(n - 1, 1)
    G = nx.erdos_renyi_graph(n, p, seed=random_seed)
    lcc = max(nx.connected_components(G), key=len)
    G = G.subgraph(lcc).copy()
    G = G.to_directed()

    log_p = -np.log(p_e)
    for u, v in G.edges():
        G[u][v]["p_e"] = p_e
        G[u][v]["-logp"] = log_p

    # q and epsilon from truncated normal (seeds overwritten later)
    np.random.seed(42)
    random.seed(42)
    q = {node: get_truncated_normal(mu_q, sigma_q) for node in G.nodes()}
    epsilon = {node: get_truncated_normal(mu_eps, sigma_eps) for node in G.nodes()}
    for node in G.nodes():
        G.nodes[node]["q"] = np.clip(q[node], 0, 1)
        G.nodes[node]["epsilon"] = np.clip(epsilon[node], 0, 1)

    return G, list(lcc)


def _graph_cache_paths(cache_dir, n):
    """キャッシュファイルパス: er_n{n}.gpickle と er_n{n}_lcc.npy"""
    os.makedirs(cache_dir, exist_ok=True)
    gpath = os.path.join(cache_dir, f"er_n{n}.gpickle")
    lpath = os.path.join(cache_dir, f"er_n{n}_lcc.npy")
    return gpath, lpath


def save_graph_cache(G, lcc_nodes, cache_dir, n):
    """構築したグラフと LCC ノードリストをキャッシュに保存する。"""
    gpath, lpath = _graph_cache_paths(cache_dir, n)
    with open(gpath, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    np.save(lpath, np.array(lcc_nodes, dtype=np.int64))
    print(f"[runtime_analysis] Graph cache saved: n={n} -> {gpath}")


def load_graph_cache(cache_dir, n):
    """
    キャッシュからグラフと LCC ノードリストを読み込む。
    存在しなければ None を返す。
    """
    gpath, lpath = _graph_cache_paths(cache_dir, n)
    if not os.path.isfile(gpath) or not os.path.isfile(lpath):
        return None
    try:
        with open(gpath, "rb") as f:
            G = pickle.load(f)
        lcc_nodes = np.load(lpath).tolist()
        print(f"[runtime_analysis] Graph cache loaded: n={n}")
        return G, lcc_nodes
    except Exception as e:
        print(f"[runtime_analysis] Failed to load graph cache n={n}: {e}")
        return None


def get_or_build_graph(n, cache_dir=None, **build_kw):
    """
    キャッシュがあれば読み込み、なければ build_er_graph_for_runtime で構築して
    cache_dir が指定されていれば保存して返す。
    """
    if cache_dir is not None:
        loaded = load_graph_cache(cache_dir, n)
        if loaded is not None:
            return loaded
    G, lcc_nodes = build_er_graph_for_runtime(n, **build_kw)
    if cache_dir is not None:
        save_graph_cache(G, lcc_nodes, cache_dir, n)
    return G, lcc_nodes


def select_seeds(lcc_nodes, num_seeds=NUM_SEEDS, seed=42):
    """Select num_seeds nodes at random from LCC."""
    random.seed(seed)
    return list(random.sample(lcc_nodes, min(num_seeds, len(lcc_nodes))))


def run_one(n, theta, k=K, skip_large=False, graph_cache_dir=None):
    """
    Run MIA-NPP once for given n, theta. Returns (n, theta, time_sec) or (n, theta, None) on failure.
    グラフは graph_cache_dir が指定されていればキャッシュから読み込み／構築後に保存して再利用する。
    """
    if skip_large and n >= 500000:
        return (n, theta, None)
    try:
        G, lcc_nodes = get_or_build_graph(n, cache_dir=graph_cache_dir)
        S = select_seeds(lcc_nodes)
        for s in S:
            G.nodes[s]["q"] = 1.0
            G.nodes[s]["epsilon"] = 0.0

        t0 = time.perf_counter()
        algorithm.MIA_NPP(G, S, k=k, theta=theta, cache_dir=None)
        elapsed = time.perf_counter() - t0
        return (n, theta, elapsed)
    except Exception as e:
        print(f"[runtime_analysis] n={n}, theta={theta} failed: {e}")
        return (n, theta, None)


def load_results(csv_path):
    """Load existing (n, theta, time_sec) from CSV. Returns list of (n, theta, time_sec)."""
    if not os.path.exists(csv_path):
        return []
    data = []
    with open(csv_path, "r") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                n = int(parts[0])
                theta = float(parts[1])
                t = float(parts[2]) if parts[2].strip().lower() not in ("", "nan", "none") else None
                data.append((n, theta, t))
    return data


def save_results(data, csv_path):
    """Save list of (n, theta, time_sec) to CSV."""
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w") as f:
        f.write("n,theta,time_sec\n")
        for n, theta, t in data:
            f.write(f"{n},{theta},{t}\n")


def run_measurements(
    n_list=None,
    theta_list=None,
    csv_path=RESULTS_CSV,
    skip_large=False,
    n_max=None,
    use_existing=True,
    graph_cache_dir=GRAPH_CACHE_DIR,
):
    """
    Run MIA-NPP for each (n, theta). If use_existing and csv_path exists, only run missing (n, theta).
    グラフは graph_cache_dir にキャッシュし、同じ n で複数 theta を試すときは再利用する。
    """
    n_list = n_list or DEFAULT_N_LIST
    theta_list = theta_list or DEFAULT_THETA_LIST
    if n_max is not None:
        n_list = [n for n in n_list if n <= n_max]

    existing = {}
    if use_existing and os.path.exists(csv_path):
        for n, theta, t in load_results(csv_path):
            existing[(n, theta)] = t
        print(f"[runtime_analysis] Loaded {len(existing)} existing results from {csv_path}")

    results = []
    for n in n_list:
        for theta in theta_list:
            if (n, theta) in existing:
                results.append((n, theta, existing[(n, theta)]))
                continue
            print(f"[runtime_analysis] Running n={n}, theta={theta} ...")
            r = run_one(n, theta, skip_large=skip_large, graph_cache_dir=graph_cache_dir)
            results.append(r)
            if r[2] is not None:
                print(f"  -> {r[2]:.2f} sec")
            save_results(results, csv_path)
    return results


def plot_results(csv_path=RESULTS_CSV, out_path=RESULTS_PNG):
    """Plot n (log scale) vs time_sec, one curve per theta."""
    data = load_results(csv_path)
    if not data:
        print(f"[runtime_analysis] No data in {csv_path}, skipping plot.")
        return

    theta_list = sorted(set(t for _, t, _ in data))
    for theta in theta_list:
        points = [(n, t) for n, th, t in data if th == theta and t is not None]
        if not points:
            continue
        points.sort(key=lambda x: x[0])
        ns = [p[0] for p in points]
        ts = [p[1] for p in points]
        plt.plot(ns, ts, "o-", label=f"θ={theta}")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("n (number of nodes)")
    plt.ylabel("Time (sec)")
    plt.legend()
    plt.title("MIA-NPP runtime (ER, avg_degree=20, p_e=0.5, k=100)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[runtime_analysis] Plot saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="MIA-NPP runtime analysis")
    parser.add_argument("--n_max", type=int, default=None, help="Max n to run (e.g. 100000)")
    parser.add_argument("--skip_large", action="store_true", help="Skip n >= 500000")
    parser.add_argument("--no_existing", action="store_true", help="Ignore existing CSV and rerun all")
    parser.add_argument("--no_graph_cache", action="store_true", help="Do not cache/load graphs (rebuild every time)")
    parser.add_argument("--plot_only", action="store_true", help="Only load CSV and plot")
    args = parser.parse_args()

    if args.plot_only:
        plot_results()
        return

    graph_cache_dir = None if args.no_graph_cache else GRAPH_CACHE_DIR
    results = run_measurements(
        n_max=args.n_max,
        skip_large=args.skip_large,
        use_existing=not args.no_existing,
        graph_cache_dir=graph_cache_dir,
    )
    save_results(results, RESULTS_CSV)
    plot_results()


if __name__ == "__main__":
    main()
