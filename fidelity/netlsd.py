import numpy as np
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
try:
    from ..utils import *
except ImportError:
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

    from utils import *


def netlsd_embedding(G, timescales=None):
    if G.is_directed(): 
        G = G.to_undirected()

    if timescales is None:
        timescales = np.logspace(-2, 2, 250)

    L = nx.normalized_laplacian_matrix(G).astype(float)
    eigvals = np.linalg.eigvalsh(L.A)

    heat_trace = np.array([
        np.sum(np.exp(-t * eigvals)) for t in timescales
    ])
    return heat_trace / heat_trace[0]


def evaluate_netlsd_similarity(G_ref, G_pred_list, ref_emb=None):
    if ref_emb is None:
        ref_emb = netlsd_embedding(G_ref)

    similarities = []
    for G_pred in tqdm(G_pred_list, desc="Evaluating NetLSD Similarity", leave=False):
        try:
            pred_emb = netlsd_embedding(G_pred)
            # similarities.append(cosine_similarity([ref_emb], [pred_emb])[0][0])
            dist = np.linalg.norm(ref_emb - pred_emb)
            similarities.append(np.exp(-dist))
        except:
            similarities.append(0.0)

    return float(np.mean(similarities)) if similarities else 0.0

if __name__ == "__main__":
    datasets = ["cadets_e3", "theia_e3", 'clearscope_e5', 'theia_e5', 'optc_h501', 'optc_h201']
    llms = ['gpt-5.1', 'claude-sonnet-4.5', 'deepseek-v3.2', 'qwen3-max', 'ProvSyn']
    results = {}

    for dataset in datasets:
        G_real = load_graph(f"../data/realGraphs_test/{dataset}.graphml")
        ref_emb = netlsd_embedding(G_real)

        for llm in llms:
            if llm == 'ProvSyn':
                G_syns_path = f"synGraphs_noname/{dataset}"
            else:
                G_syns_path = f"baselines/synGraphs/{dataset}/{llm}"

            G_syns = sample_graph(G_syns_path, 30)


            avg_sim = evaluate_netlsd_similarity(G_real, G_syns, ref_emb)

            print(f"Average NetLSD similarity for {llm} on {dataset}: {avg_sim:.4f}")
            results[f"{dataset}-{llm}"] = avg_sim

    save_json(results, "results/netlsd.json")
