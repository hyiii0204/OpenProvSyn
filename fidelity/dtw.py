import networkx as nx
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from typing import List, Dict
from dtaidistance import dtw
try:
    from ..utils import *
except ImportError:
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

    from utils import *

def extract_path_features(path: List[str], graph: nx.Graph, n_map: Dict, e_map: Dict, node_attr='label', edge_attr='label') -> np.ndarray:
    features = []
    for node_id in path:
        val = str(graph.nodes[node_id].get(node_attr, ''))
        features.append(float(n_map.get(val, -1)))
    
    for i in range(len(path) - 1):
        edge_data = graph.get_edge_data(path[i], path[i+1])
        val = str(edge_data.get(edge_attr, '')) if edge_data else ''
        features.append(float(e_map.get(val, -1)))
        
    return np.array(features, dtype=np.double)

def get_mappings(graphs: List[nx.Graph], node_attr='label', edge_attr='label'):
    n_labels, e_labels = set(), set()
    for G in graphs:
        for _, d in G.nodes(data=True): 
            n_labels.add(str(d.get(node_attr, '')))
        for _, _, d in G.edges(data=True): 
            e_labels.add(str(d.get(edge_attr, '')))
    
    n_map = {l: i for i, l in enumerate(sorted(n_labels))}
    e_map = {l: i for i, l in enumerate(sorted(e_labels))}
    return n_map, e_map

def _find_min_dtw(args):
    target_seq, reference_sequences = args
    min_dist = float('inf')
    for ref_seq in reference_sequences:
        dist = dtw.distance(target_seq, ref_seq)
        if dist < min_dist:
            min_dist = dist
    return min_dist

def evaluate_dtw_similarity(G_ref: nx.Graph, G_pred_list: List[nx.Graph]) -> float:
    n_map, e_map = get_mappings([G_ref] + G_pred_list)
    
    paths_ref = get_dfs_paths(G_ref)[:2000]
    seqs_ref = [extract_path_features(p, G_ref, n_map, e_map) for p in paths_ref]
    
    if not seqs_ref:
        return 0.0

    graph_scores = []
    for G_pred in tqdm(G_pred_list, desc="Evaluating DTW"):
        paths_pred = get_dfs_paths(G_pred)[:500]
        seqs_pred = [extract_path_features(p, G_pred, n_map, e_map) for p in paths_pred]
        
        if not seqs_pred:
            graph_scores.append(0.0)
            continue

        task_args = [(s_p, seqs_ref) for s_p in seqs_pred]
        with Pool(cpu_count()) as pool:
            min_distances = pool.map(_find_min_dtw, task_args)
        
        graph_scores.append(np.mean(min_distances))
    
    return np.mean(graph_scores) if graph_scores else 0.0

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
            avg_dtw = evaluate_dtw_similarity(G_real, G_syns)
            
            print(f"Dataset: {dataset} | LLM: {llm} | Avg DTW: {avg_dtw:.4f}")
            results[f"{dataset}-{llm}"] = float(avg_dtw)

    save_json(results, f"results/DTW.json")