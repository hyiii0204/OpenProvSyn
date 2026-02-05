import nltk
import os
import multiprocessing
from tqdm import tqdm
from nltk.translate.gleu_score import sentence_gleu
import networkx as nx
try:
    from ..utils import *
except ImportError:
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

    from utils import *

def get_node_names(G: nx.Graph, name_dict=None):
    if name_dict is None:
        name_dict = {}
    for node_id, data in G.nodes(data=True):
        node_type = data.get('label', 'unknown')
        node_name = data.get('name', node_id)
        if node_type not in name_dict:
            name_dict[node_type] = set()
        if str(node_name).strip() != "" and node_name not in name_dict[node_type]:
            name_dict[node_type].add(node_name)
    return name_dict

def calculate_distinct_1(texts):
    if not texts:
        return 0.0
    all_unigrams = []
    for text in texts:
        tokens = nltk.word_tokenize(str(text).lower())
        all_unigrams.extend(tokens)
    
    if not all_unigrams:
        return 0.0
    return len(set(all_unigrams)) / len(all_unigrams)

def _process_single_hypothesis(hypothesis, reference_list):
    max_gleu = 0.0
    hyp_tokens = nltk.word_tokenize(str(hypothesis).lower())
    
    for reference in reference_list:
        ref_tokens = nltk.word_tokenize(str(reference).lower())
        score = sentence_gleu([ref_tokens], hyp_tokens)
        if score > max_gleu:
            max_gleu = score
    return max_gleu

def calculate_max_gleu_parallel(reference_list, hypothesis_list):
    num_processes = os.cpu_count()
    tasks = [(hyp, list(reference_list)) for hyp in hypothesis_list]
    with multiprocessing.Pool(processes=num_processes) as pool:
        max_scores = list(tqdm(pool.starmap(_process_single_hypothesis, tasks), 
                               total=len(hypothesis_list), 
                               desc="  GLEU", 
                               leave=False))
    return max_scores

def evaluate_graph_text_similarity(G_ref, G_pred_list):
    ref_names_dict = get_node_names(G_ref)
    pred_names_dict = {}
    for G_pred in G_pred_list:
        pred_names_dict = get_node_names(G_pred, pred_names_dict)

    final_results = {}
    for node_type, pred_name_set in pred_names_dict.items():
        print(f"Evaluating Node Type: [{node_type}]")
        hyp_list = list(pred_name_set)
        
        # 1. Fidelity Metric: GLEU
        if node_type in ref_names_dict and ref_names_dict[node_type]:
            ref_list = list(ref_names_dict[node_type])
            max_gleu_scores = calculate_max_gleu_parallel(ref_list, hyp_list)
            avg_gleu = sum(max_gleu_scores) / len(max_gleu_scores) if max_gleu_scores else 0.0
        else:
            avg_gleu = 0.0

        # 2. Diversity Metric: Distinct-1
        dist1 = calculate_distinct_1(hyp_list)

        final_results[node_type] = {
            'GLEU': avg_gleu,
            'Distinct-1': dist1
        }
        print(f"  Results: {final_results[node_type]}")

    return final_results

if __name__ == "__main__":
    multiprocessing.freeze_support()

    datasets = ["cadets_e3", "theia_e3", 'clearscope_e5', 'theia_e5', 'optc_h501', 'optc_h201']
    llms = ['gpt-5.1', 'claude-sonnet-4.5', 'deepseek-v3.2', 'qwen3-max', 'ProvSyn']

    results = {}

    for dataset in datasets:
        G_real = load_graph(f"../data/realGraphs_test/{dataset}.graphml")
        
        for llm in llms:
            if llm == 'ProvSyn':
                G_syns_path = f"synGraphs_name/{dataset}"
            else:
                G_syns_path = f"baselines/synGraphs/{dataset}/{llm}"
            G_syns = sample_graph(G_syns_path, 30)
            
            similarity = evaluate_graph_text_similarity(G_real, G_syns)
            results[f"{dataset}-{llm}"] = similarity

    save_json(results, f"results/text_provSyn.json")