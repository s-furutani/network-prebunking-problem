"""
非対称伝播設定 (p_e^- = alpha * p_e^+) での IC-N シミュレーション評価。

介入集合 X は対称設定で計算済みの .npy（suffix なし）を source ディレクトリから読み込み、
シミュレーション結果のみ source_directory/asymmetric/{alg}_alpha=..._sim_results.* に保存する。
"""

import argparse

import main


def main_cli():
    parser = argparse.ArgumentParser(description='Asymmetric propagation IC-N simulation evaluation')
    parser.add_argument('--graph', type=str, required=True,
                        help='Benchmark graph name (same as main.py --graph)')
    parser.add_argument('--edge_prob_model', choices=['WC', 'TR'], default='WC')
    parser.add_argument('--alpha', type=float, required=True,
                        help='Negative propagation factor: p_e_minus = alpha * p_e_plus')
    parser.add_argument('--seed_mode', choices=['default', 'nonzero', 'adversarial'], default='default',
                        help='Must match the source benchmark run that produced .npy files')
    parser.add_argument('--param_profile', choices=['default', 'q0502_eps0301', 'q_eps_uni'], default='default',
                        help='Must match the source benchmark run (for S, q, epsilon reconstruction)')
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--kmax', type=int, default=200)
    parser.add_argument('--with_std', action='store_true',
                        help='Save *_sim_results_with_std.npz with mean and std')
    args = parser.parse_args()

    source_directory = main.get_benchmark_directory(
        args.graph, args.edge_prob_model, args.seed_mode,
    )
    print(f'Source directory (intervention .npy): {source_directory}')
    print(f'Asymmetric propagation: alpha={args.alpha}')

    graph, graph_name, S = main.setup_benchmark_graph(
        args.graph,
        args.edge_prob_model,
        num_seeds=args.num_seeds,
        seed_mode=args.seed_mode,
        param_profile=args.param_profile,
        alpha=args.alpha,
    )
    print(f'graph: {graph_name}, |V|={len(graph.nodes())}, |E|={len(graph.edges())}, seed node: {S}')
    print(f'  {main.describe_param_profile(args.param_profile)}')

    main.run_asymmetric_simulation(
        graph, S, args.kmax, source_directory, args.alpha, with_std=args.with_std,
    )


if __name__ == '__main__':
    main_cli()
