import networkx as nx
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List
import os
try:
    from ..utils import *
except ImportError:
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

    from utils import *

def extract_path_features(path: List[str], graph: nx.Graph, node_attr='label', edge_attr='label') -> List[str]:
    features = []
    for i in range(len(path) - 1):
        features.append(str(graph.nodes[path[i]].get(node_attr, '')))
        edge_data = graph.get_edge_data(path[i], path[i+1])
        features.append(str(edge_data.get(edge_attr, '')) if edge_data else '')
    features.append(str(graph.nodes[path[-1]].get(node_attr, '')))
    return features

def longest_common_substring(seq1: List[str], seq2: List[str]) -> int:
    len1, len2 = len(seq1), len(seq2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    max_length = 0
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_length = max(max_length, dp[i][j])
    return max_length

def _find_max_lcs(args):
    target_seq, reference_sequences = args
    return max(longest_common_substring(target_seq, ref_seq) for ref_seq in reference_sequences)

def evaluate_lcs_similarity(G_ref: nx.Graph, G_pred_list: List[nx.Graph]) -> float:
    paths_ref = get_dfs_paths(G_ref)[:3000]
    seqs_ref = [extract_path_features(p, G_ref) for p in paths_ref]
    
    if not seqs_ref:
        return 0.0

    scores = []
    for G_pred in tqdm(G_pred_list, desc="Evaluating LCS", leave=False):
        paths_pred = get_dfs_paths(G_pred)[:1000]
        seqs_pred = [extract_path_features(p, G_pred) for p in paths_pred]
        
        if not seqs_pred:
            scores.append(0.0)
            continue

        task_args = [(s_pred, seqs_ref) for s_pred in seqs_pred]
        
        with Pool(cpu_count()) as pool:
            max_lcs_per_path = pool.map(_find_max_lcs, task_args)
        
        scores.append(sum(max_lcs_per_path) / len(max_lcs_per_path))
    
    return sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    datasets = ["cadets_e3", "theia_e3", 'clearscope_e5', 'theia_e5', 'optc_h501', 'optc_h201']
    llms = ['gpt-5.1', 'claude-sonnet-4.5', 'deepseek-v3.2', 'qwen3-max', 'ProvSyn']
    results = {}

    for dataset in datasets:
        G_real = load_graph(f"../data/realGraphs_test/{dataset}.graphml")
        
        for llm in llms:
            if llm == 'ProvSyn':
                G_syns_path = f"synGraphs_noname/{dataset}"
            else:
                G_syns_path = f"baselines/synGraphs/{dataset}/{llm}"
            G_syns = sample_graph(G_syns_path, 10)
            
            avg_score = evaluate_lcs_similarity(G_real, G_syns)
            print(f"Average LCS Score for {dataset} | {llm}: {avg_score:.2f}")
            results[f"{dataset}-{llm}"] = avg_score

    save_json(results, f"results/LCS_ProvSyn.json")