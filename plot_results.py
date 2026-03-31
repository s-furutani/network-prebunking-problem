# %%

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import load_real_graph
import seaborn as sns
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

def plot_simulation_results(dataset_name='benchmark', edge_prob_model='WC'):
    if dataset_name == 'benchmark':
        graph_names = ['Reed98', 'LastFM', 'Deezer', 'Epinions', 'Twitter']
    elif dataset_name == 'fakenewsnet':
        graph_names = ['politifact', 'gossipcop']
    else:
        raise ValueError(f'Invalid dataset name: {dataset_name}')
    
    n = len(graph_names)
    num_row = 1
    num_col = n
    xaxis = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    theta_suffix = '_theta=1e-02'
    theta_suffix2 = '_theta=5e-02'
    rho_suffix = '_rho=1000'
    
    plt.rcParams["mathtext.fontset"] = 'cm'
    plt.figure(figsize=(4 * num_col, 4), dpi=300)
    alg_names = ['Random', 'Gullible', 'Degree', 'Distance', 'Adv.Greedy', 'CMIA-O', r'MIA-NPP ($\theta=0.01$)', r'MIA-NPP ($\theta=0.05$)', r'Greedy ($n_{\mathrm{MC}}=1000$)']#, r'Greedy ($n_{\mathrm{MC}}=2000$)']
    alg_path_names = {
        'Random': 'Random',
        'Gullible': 'Gullible',
        'Degree': 'Degree',
        'Distance': 'Distance',
        'Adv.Greedy': 'AdvancedGreedy'+rho_suffix,
        'CMIA-O': 'CMIA-O'+theta_suffix,
        r'MIA-NPP ($\theta=0.01$)': 'MIA-NPP'+theta_suffix,
        r'MIA-NPP ($\theta=0.05$)': 'MIA-NPP'+theta_suffix2,
        'Greedy': 'CELF'+rho_suffix,
    }
    cmap = plt.get_cmap('viridis_r')    
    color_list = [cmap(a/7) for a in range(7)]
    cmap_reds = plt.get_cmap('Reds_r')
    color_list_reds = [cmap_reds((a+1)/5) for a in range(4)]
    fmts = {'Greedy': ':', 'Greedy2000': ':', r'MIA-NPP ($\theta=0.01$)': 'o-', r'MIA-NPP ($\theta=0.05$)': 'o--', 'CMIA-O': '^-', 'Adv.Greedy': 'v-', 'Distance': 'd-', 'Degree': 'p-', 'Gullible': 'x-', 'Random': '*-'}
    colors = {'Random': color_list[0], 'Gullible': color_list[1], 'Degree': color_list[2], 'Distance': color_list[3], 'Adv.Greedy': color_list[4], 'CMIA-O': color_list[5], r'MIA-NPP ($\theta=0.01$)': color_list_reds[1], r'MIA-NPP ($\theta=0.05$)': color_list_reds[2], r'Greedy ($n_{\mathrm{MC}}=1000$)': 'black', r'Greedy ($n_{\mathrm{MC}}=2000$)': 'black'}
    linewidths = {'Greedy': 2, 'Greedy2000': 2, r'MIA-NPP ($\theta=0.01$)': 3, r'MIA-NPP ($\theta=0.05$)': 3, 'CMIA-O': 2, 'Adv.Greedy': 2, 'Distance': 2, 'Degree': 2, 'Gullible': 2, 'Random': 2}

    for i, graph_name in enumerate(graph_names):
        plt.subplot(num_row, num_col, i + 1)
        plt.title(graph_name)
        for j, alg_name in enumerate(alg_names):
            if dataset_name == 'benchmark':
                if alg_name == 'Greedy' or alg_name == r'MIA-NPP ($\theta=0.01$)':
                    path = f'results/{dataset_name}/{graph_name}_{edge_prob_model}/{alg_path_names[alg_name]}_sim_results_with_std.npz'
                else:
                    path = f'results/{dataset_name}/{graph_name}_{edge_prob_model}/{alg_path_names[alg_name]}_sim_results.npy'
            else:
                if alg_name == 'Greedy' or alg_name == r'MIA-NPP ($\theta=0.01$)':
                    path = f'results/{dataset_name}/{graph_name}/{alg_path_names[alg_name]}_sim_results_with_std.npz'
                else:
                    path = f'results/{dataset_name}/{graph_name}/{alg_path_names[alg_name]}_sim_results.npy'
            if not os.path.exists(path):
                continue
            if alg_name == 'Greedy' or alg_name == r'MIA-NPP ($\theta=0.01$)':
                results = np.load(path)
                mean = results['mean']
                std = results['std']
                ci = 1.96 * std / np.sqrt(2000)
                plt.plot(xaxis, mean/mean[0], fmts[alg_name], color=colors[alg_name], label=alg_name, markersize=8, linewidth=2, markerfacecolor='none')
                plt.fill_between(xaxis, mean/mean[0] - ci/mean[0], mean/mean[0] + ci/mean[0], alpha=0.2, color=colors[alg_name])
            else:
                results = np.load(path, allow_pickle=True)
                plt.plot(xaxis, results/results[0], fmts[alg_name], color=colors[alg_name], label=alg_name, markersize=8, linewidth=2, markerfacecolor='none')
            plt.xticks([0, 50, 100, 150, 200], [0, 50, 100, 150, 200], fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlim((-3, 203))
            # plt.ylim((ymin * 0.9, ymax * 1.1))
            plt.xlabel(r'# of prebunked nodes $k$', fontsize=15)
            if i == 0:
                plt.ylabel(r'Relative spread $y(k)/y(0)$', fontsize=15)
            plt.title(graph_name, fontsize=17)
        if i == 0:
            # Collect handles and labels from the current axes
            handles, labels = plt.gca().get_legend_handles_labels()
    # After all subplots, add a single legend above the figure
    if dataset_name == 'fakenewsnet':
        # 2行で表示
        plt.figlegend(
            handles, labels, loc='upper center', ncol=(len(handles) + 1) // 2,
            bbox_to_anchor=(0.5, 1.2), frameon=False, fontsize='x-large'
        )
    else:
        plt.figlegend(
            handles, labels, loc='upper center', ncol=len(handles),
            bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize='x-large'
        )

    plt.tight_layout()
    plt.show()

plot_simulation_results()
plot_simulation_results(edge_prob_model='TR')
plot_simulation_results(dataset_name='fakenewsnet')
# %%


def load_algorithm_timings_table(benchmark_dir='results/benchmark'):
    """
    results/benchmark/{graph_name}/algorithm_timings.csv を各グラフから読み込み、
    アルゴリズム x グラフ の実行時間を表形式で返す（表示用）。
    """
    if not os.path.isdir(benchmark_dir):
        return None
    graph_names = sorted([
        d for d in os.listdir(benchmark_dir)
        if os.path.isdir(os.path.join(benchmark_dir, d))
        and os.path.isfile(os.path.join(benchmark_dir, d, 'algorithm_timings.csv'))
    ])
    if not graph_names:
        return None
    # alg_name -> { graph_name -> time_sec }
    rows = {}
    for graph_name in graph_names:
        path = os.path.join(benchmark_dir, graph_name, 'algorithm_timings.csv')
        with open(path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                alg = row.get('alg_name', '').strip()
                if not alg:
                    continue
                try:
                    t = float(row.get('time_sec', 0) or 0)
                except (ValueError, TypeError):
                    t = float('nan')
                if alg not in rows:
                    rows[alg] = {}
                rows[alg][graph_name] = t
    # 行: アルゴリズム, 列: グラフ
    alg_order = sorted(rows.keys())
    return alg_order, graph_names, rows


def print_algorithm_timings_table(benchmark_dir='results/benchmark'):
    """各グラフの algorithm_timings.csv を読み、実行時間を表形式で一覧表示する。"""
    result = load_algorithm_timings_table(benchmark_dir)
    if result is None:
        print(f'No algorithm_timings.csv found under {benchmark_dir}')
        return
    # result unpack前に rows 未定義なので result から rows を取得する必要
    # result: (alg_order, graph_names, rows)
    _, _, rows = result

    # アルゴリズム順固定
    desired_alg_order = [
        'Random', 'Gullible', 'Degree', 'Distance',
        'AdvancedGreedy', 'CMIA-O', 'MIA-NPP', 'CELF'
    ]
    present_algs = set(rows.keys())
    alg_order = [a for a in desired_alg_order if a in present_algs]
    # 残りのアルゴリズムをアルファベット順で追加
    alg_order += sorted([a for a in present_algs if a not in alg_order])

    # グラフ(データセット)順固定 + サフィックス: _WCを先、_TRを後
    # 旧: Red98, 正: Reed98
    base_graph_order = ['Reed98', 'LastFM', 'Deezer', 'Epinions', 'Twitter']
    orig_graph_names = result[1]
    present_graphs = set(orig_graph_names)

    # With possible suffixes, flatten order: e.g., Reed98_WC, Reed98_TR, ...
    # 作成する: ["Reed98_WC", "Reed98_TR", ...] の順
    expanded_order = []
    for base in base_graph_order:
        for suffix in ['_WC', '_TR']:
            name = base + suffix
            if name in present_graphs:
                expanded_order.append(name)
    # 残りの(未登場)グラフ名（例: 新しい名前やsuffix）があればアルファベット昇順で追加
    remaining_graphs = sorted([g for g in present_graphs if g not in expanded_order])
    graph_names = expanded_order + remaining_graphs
    col_w = max(20, max(len(g) for g in graph_names) + 2)
    alg_w = max(25, max(len(a) for a in alg_order) + 2)
    header = 'Algorithm'.ljust(alg_w) + ''.join(g.ljust(col_w) for g in graph_names)
    print(header)
    print('-' * len(header))
    for alg in alg_order:
        cells = [rows[alg].get(g, float('nan')) for g in graph_names]
        str_cells = []
        for c in cells:
            if isinstance(c, float) and (c != c or c < 0):  # NaN or negative
                str_cells.append('—'.ljust(col_w))
            else:
                str_cells.append(f'{c:.2f}'.ljust(col_w))
        print(alg.ljust(alg_w) + ''.join(str_cells))
    print()


print_algorithm_timings_table()
print_algorithm_timings_table(benchmark_dir='results/fakenewsnet')
# %%


def plot_runtime_analysis_mia_npp(csv_path='results/runtime_analysis_mia_npp.csv'):
    """results/runtime_analysis_mia_npp.csv を読み、n vs T を両対数でプロット。γ=1.0, 1.5 の参照直線を表示。"""
    if not os.path.isfile(csv_path):
        print(f'File not found: {csv_path}')
        return
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n = int(row['n'])
                theta = float(row['theta'])
                t = row['time_sec'].strip().lower()
                time_sec = float(t) if t and t not in ('nan', '') else np.nan
            except (ValueError, KeyError):
                continue
            data.append((n, theta, time_sec))
    if not data:
        return
    # theta ごとに (n_list, time_list) を用意。凡例は 0.005, 0.01, 0.05 の順
    thetas_plot = (0.005, 0.01, 0.05)
    cmap_reds = plt.get_cmap('Reds_r')
    color_list_reds = [cmap_reds((a + 1) / 5) for a in range(4)]
    colors_theta = {0.005: color_list_reds[0], 0.01: color_list_reds[1], 0.05: color_list_reds[2]}
    labels_theta = {0.005: r'MIA-NPP ($\theta=0.005$)', 0.01: r'MIA-NPP ($\theta=0.01$)', 0.05: r'MIA-NPP ($\theta=0.05$)'}
    n_all = sorted(set(x[0] for x in data))
    n_min, n_max = min(n_all), max(n_all)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    for th in thetas_plot:
        pts = [(n, t) for n, theta, t in data if theta == th and not (isinstance(t, float) and t != t)]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        ns = np.array([p[0] for p in pts])
        ts = np.array([p[1] for p in pts])
        ax.loglog(ns, ts, 'o-', color=colors_theta[th], label=labels_theta[th], linewidth=2, markersize=8, markerfacecolor='none')
    # 参照直線: T = C * n^γ。最小 n のとある T を通るように C を設定
    n_ref = np.array([n_min, n_max], dtype=float)
    valid_t = [t for _, _, t in data if isinstance(t, (int, float)) and t == t and t > 0]
    T_anchor1 = 10 ** -1
    T_anchor15 = 10 ** 0.8
    C1 = T_anchor1 / (n_min ** 1.0)
    C15 = T_anchor15 / (n_min ** 1.5)
    ax.loglog(n_ref, C1 * (n_ref ** 1.0), 'k--', alpha=0.7, linewidth=1.5, label=r'$T \propto n^{1.0}$')
    ax.loglog(n_ref, C15 * (n_ref ** 1.5), 'k:', alpha=0.7, linewidth=1.5, label=r'$T \propto n^{1.5}$')
    ax.set_xlabel(r'# of nodes $n$', fontsize=13)
    ax.set_ylabel(r'Execution time $T$', fontsize=13)
    ax.tick_params(axis='both', labelsize=13)
    
    ax.legend(loc='upper left', fontsize=13, frameon=False)
    # ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.show()


plot_runtime_analysis_mia_npp()
# %%


def _sigdelta_fmt(s):
    """σ_δ をファイル名用の文字列に (main.py と同じ規則)。"""
    return '0' if s == 0 else f'{s:.0e}'


def plot_simulation_results_uncertain(graph_names, num_row, num_col, kmax, sig_eps_is_01):
    """plot_simulation_results と同じスタイルで uncertainty 実験の結果をプロット。"""
    n = len(graph_names)
    xaxis = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    plt.rcParams["mathtext.fontset"] = 'cm'
    # main.py の sigma_delta = [0, 0.1, 0.2, 0.5] に対応
    sigma_deltas = [0, 0.1, 0.2, 0.5]
    alg_labels = [r'$\sigma_{\delta}^2=0.0$', r'$\sigma_{\delta}^2=0.1$', r'$\sigma_{\delta}^2=0.2$', r'$\sigma_{\delta}^2=0.5$']
    num_alg = len(sigma_deltas)
    fmts = ['o-', '^-', 's-', 'v-']
    cmap = plt.get_cmap('Purples')
    colors = [cmap((a + 1) / (num_alg + 1)) for a in range(num_alg+1)][::-1]

    cmap_viridis = plt.get_cmap('viridis_r')
    color_list_viridis = [cmap_viridis(a/7) for a in range(7)]
    cmap_reds = plt.get_cmap('Reds_r')
    color_list_reds = [cmap_reds((a+1)/5) for a in range(4)]
    color_ref = {'Adv.Greedy': color_list_viridis[4], r'MIA-NPP ($\theta=0.01$)': color_list_reds[1]}

    plt.figure(figsize=(4 * num_col, 4), dpi=300)
    for i in range(n):
        plt.subplot(num_row, num_col, i + 1)
        ag_path = f'results/fakenewsnet/{graph_names[i]}/AdvancedGreedy_rho=1000_sim_results.npy'
        if os.path.exists(ag_path):
            y_AG = np.load(ag_path, allow_pickle=True)
            plt.plot(xaxis, y_AG / y_AG[0], ':', color=color_ref['Adv.Greedy'], markersize=10, linewidth=3, markerfacecolor='none')
        if sig_eps_is_01:
            mia_path = f'results/fakenewsnet/{graph_names[i]}/MIA-NPP_theta=1e-02_sim_results.npy'
            if os.path.exists(mia_path):
                y_MIA = np.load(mia_path, allow_pickle=True)
                plt.plot(xaxis, y_MIA / y_MIA[0], ':', color=color_ref[r'MIA-NPP ($\theta=0.01$)'], markersize=10, linewidth=3, markerfacecolor='none')
        # MIA-NPP with different σ_δ (results/uncertain/{graph_name}/MIA-NPP_theta=1e-02_sigdelta=*_sim_results.npy)
        for a in range(num_alg):
            file_base = f'MIA-NPP_theta=1e-02_sigdelta={_sigdelta_fmt(sigma_deltas[a])}'
            path = f'results/fakenewsnet/{graph_names[i]}/{file_base}_sim_results.npy'
            if not os.path.exists(path):
                continue
            y = np.load(path, allow_pickle=True)
            results0 = y[0]
            n_pts = min(len(xaxis), len(y))
            step = max(1, len(y) // n_pts) if len(y) > n_pts else 1
            y_plot = (y / results0)[::step][:n_pts]
            x_plot = xaxis[:n_pts]
            plt.plot(x_plot, y_plot, 'o-', color=colors[a], label=alg_labels[a], linewidth=2, markersize=10, markerfacecolor='none')
        # 比較用: AdvancedGreedy, MIA-NPP (perfect observation)
        plt.xticks([0, 50, 100, 150, 200], [0, 50, 100, 150, 200], fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlim((-3, 203))
        plt.xlabel(r'# of prebunked nodes $k$', fontsize=15)
        if i == 0:
            plt.ylabel(r'Relative spread $y(k)/y(0)$', fontsize=15)
        title = 'PolitiFact' if graph_names[i] == 'politifact' else 'GossipCop'
        plt.title(title, fontsize=17)
        if i == 0:
            handles, labels = plt.gca().get_legend_handles_labels()
    plt.figlegend(
        handles, labels, loc='upper center', ncol=len(handles),
        bbox_to_anchor=(0.5, 1.13), frameon=False, fontsize='x-large'
    )
    plt.tight_layout()
    plt.show()


plot_simulation_results_uncertain(graph_names=['politifact', 'gossipcop'], num_row=1, num_col=2, kmax=200, sig_eps_is_01=True)
# %%
