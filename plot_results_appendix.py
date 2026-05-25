# %%
"""
Appendix 用プロット: seed / parameter / asymmetric propagation sensitivity.

LastFM / Epinions × WC / TR の 1x4 レイアウトで表示する。
"""

import os

import matplotlib.pyplot as plt

from main import get_alpha_suffix, get_benchmark_directory, get_param_suffix
from plot_results import ALGS_WITH_CI, plot_sim_results_curve, resolve_sim_results_path

plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'

GRAPH_NAMES = ['LastFM', 'Epinions']
EDGE_PROB_MODELS = ['WC', 'TR']
PANELS = [(g, m) for g in GRAPH_NAMES for m in EDGE_PROB_MODELS]

XAXIS = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
THETA_SUFFIX = '_theta=1e-02'
THETA_SUFFIX2 = '_theta=5e-02'
RHO_SUFFIX = '_rho=1000'

STANDARD_ALG_NAMES = [
    'Random', 'Gullible', 'Degree', 'Distance', 'Adv.Greedy', 'CMIA-O',
    r'MIA-NPP ($\theta=0.01$)', r'MIA-NPP ($\theta=0.05$)', 'Greedy',
]
STANDARD_ALG_PATH_NAMES = {
    'Random': 'Random',
    'Gullible': 'Gullible',
    'Degree': 'Degree',
    'Distance': 'Distance',
    'Adv.Greedy': 'AdvancedGreedy' + RHO_SUFFIX,
    'CMIA-O': 'CMIA-O' + THETA_SUFFIX,
    r'MIA-NPP ($\theta=0.01$)': 'MIA-NPP' + THETA_SUFFIX,
    r'MIA-NPP ($\theta=0.05$)': 'MIA-NPP' + THETA_SUFFIX2,
    'Greedy': 'CELF' + RHO_SUFFIX,
}


def _get_standard_style():
    """plot_results.py の twitter_higgs と同じ色・線種。"""
    cmap = plt.get_cmap('viridis_r')
    color_list = [cmap(a / 7) for a in range(7)]
    cmap_reds = plt.get_cmap('Reds_r')
    color_list_reds = [cmap_reds((a + 1) / 5) for a in range(4)]
    fmts = {
        'Greedy': ':', r'MIA-NPP ($\theta=0.01$)': 'o-', r'MIA-NPP ($\theta=0.05$)': 'o--',
        'CMIA-O': '^-', 'Adv.Greedy': 'v-', 'Distance': 'd-', 'Degree': 'p-',
        'Gullible': 'x-', 'Random': '*-',
    }
    colors = {
        'Random': color_list[0], 'Gullible': color_list[1], 'Degree': color_list[2],
        'Distance': color_list[3], 'Adv.Greedy': color_list[4], 'CMIA-O': color_list[5],
        r'MIA-NPP ($\theta=0.01$)': color_list_reds[1],
        r'MIA-NPP ($\theta=0.05$)': color_list_reds[2], 'Greedy': 'black',
    }
    return fmts, colors


def _plot_algorithms(result_dir, file_suffix=''):
    """指定ディレクトリの sim 結果を現在の axes にプロット。"""
    fmts, colors = _get_standard_style()
    for alg_name in STANDARD_ALG_NAMES:
        alg_file_base = STANDARD_ALG_PATH_NAMES[alg_name] + file_suffix
        use_std = alg_name in ALGS_WITH_CI
        path, kind = resolve_sim_results_path(result_dir, alg_file_base, prefer_std=use_std)
        if path is None:
            continue
        plot_sim_results_curve(
            path, fmts[alg_name], kind, XAXIS, colors[alg_name], alg_name,
            plot_ci=use_std,
        )


def _format_subplot(i, graph_name, edge_prob_model):
    """1x4 サブプロットの共通軸設定。"""
    plt.xticks([0, 50, 100, 150, 200], [0, 50, 100, 150, 200], fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim((-3, 203))
    plt.xlabel(r'# of prebunked nodes $k$', fontsize=15)
    if i == 0:
        plt.ylabel(r'Relative spread $y(k)/y(0)$', fontsize=15)
    plt.title(f'{graph_name} ({edge_prob_model})', fontsize=17)


def _finalize_figure(handles, labels, show=True):
    if handles:
        plt.figlegend(
            handles, labels, loc='upper center', ncol=len(handles),
            bbox_to_anchor=(0.5, 1.15), frameon=False, fontsize='x-large',
        )
    plt.tight_layout()
    if show:
        plt.show()


def _plot_benchmark_panels(
    file_suffix='',
    seed_mode='default',
    result_subdir='',
    show=True,
):
    """LastFM / Epinions × WC / TR を 1x4 でプロット。"""
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.figure(figsize=(4 * len(PANELS), 4), dpi=300)
    handles, labels = [], []

    for i, (graph_name, edge_prob_model) in enumerate(PANELS):
        plt.subplot(1, len(PANELS), i + 1)
        result_dir = get_benchmark_directory(graph_name, edge_prob_model, seed_mode)
        if result_subdir:
            result_dir = os.path.join(result_dir, result_subdir)
        _plot_algorithms(result_dir, file_suffix=file_suffix)
        _format_subplot(i, graph_name, edge_prob_model)
        if i == 0:
            handles, labels = plt.gca().get_legend_handles_labels()

    _finalize_figure(handles, labels, show=show)


def plot_seed_sensitivity(seed_mode='default', show=True):
    """
    Seed sensitivity: LastFM / Epinions × WC / TR を 1x4 でプロット。

    例: plot_seed_sensitivity(seed_mode='nonzero')
        -> LastFM_WC_nonzero/, LastFM_TR_nonzero/, Epinions_WC_nonzero/, ...
    """
    _plot_benchmark_panels(seed_mode=seed_mode, show=show)


def plot_param_sensitivity(param_profile='default', seed_mode='default', show=True):
    """
    Parameter sensitivity: ファイル名 suffix 付き sim 結果を 1x4 でプロット。

    例: plot_param_sensitivity(param_profile='q0502_eps0301')
        -> {graph}_{WC|TR}/MIA-NPP_theta=1e-02_q0502_eps0301_sim_results.*
    """
    file_suffix = get_param_suffix(param_profile)
    _plot_benchmark_panels(
        file_suffix=file_suffix, seed_mode=seed_mode, show=show,
    )


def plot_asymmetric_propagation(alpha, seed_mode='default', show=True):
    """
    Asymmetric propagation: asymmetric/ 配下の alpha 付き sim 結果を 1x4 でプロット。

    例: plot_asymmetric_propagation(alpha=0.5)
        -> {graph}_{WC|TR}/asymmetric/MIA-NPP_theta=1e-02_alpha=5e-01_sim_results.*
    """
    file_suffix = get_alpha_suffix(alpha)
    _plot_benchmark_panels(
        file_suffix=file_suffix, seed_mode=seed_mode,
        result_subdir='asymmetric', show=show,
    )

# %%

# 使用例
# plot_seed_sensitivity(seed_mode='nonzero')
# plot_param_sensitivity(param_profile='q0502_eps0301')
# plot_asymmetric_propagation(alpha=0.5)

# %%
