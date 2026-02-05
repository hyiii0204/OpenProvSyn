import os
import numpy as np
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
try:
    from ..utils import *
except ImportError:
    import os
    import sys

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    sys.path.insert(0, project_root)

    from utils import *


def wl_features(G, h=2, text_attr="name"):
    G = G.copy()

    for n, data in G.nodes(data=True):
        text = data.get(text_attr, "")
        G.nodes[n]["wl"] = f"{text}_{G.degree[n]}"

    features = []
    for _ in range(h):
        new_labels = {}
        for n in G.nodes:
            neigh = sorted(G.nodes[v]["wl"] for v in G.neighbors(n))
            label = G.nodes[n]["wl"] + "_" + "_".join(neigh)
            new_labels[n] = label
            features.append(label)
        for n in G.nodes:
            G.nodes[n]["wl"] = new_labels[n]

    return features

def train_graph2vec(graphs, dim=64):
    documents = []
    for i, G in enumerate(graphs):
        features = wl_features(G)
        documents.append(TaggedDocument(features, [f"g_{i}"]))

    model = Doc2Vec(
        vector_size=dim,
        window=0,
        min_count=1,
        dm=0,
        epochs=100,
        workers=4
    )
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=model.epochs)
    return model

def graph_embedding(model, idx):
    return model.dv[f"g_{idx}"]

def evaluate_graph2vec_similarity(G_ref, G_pred_list):
    graphs = [G_ref] + G_pred_list
    model = train_graph2vec(graphs)

    ref_vec = graph_embedding(model, 0)
    sims = []
    for i in range(1, len(graphs)):
        pred_vec = graph_embedding(model, i)
        sims.append(cosine_similarity([ref_vec], [pred_vec])[0][0])

    return float(np.mean(sims))

if __name__ == "__main__":
    datasets = ["cadets_e3", "theia_e3", 'clearscope_e5', 'theia_e5', 'optc_h501', 'optc_h201']
    llms = ['gpt-5.1', 'claude-sonnet-4.5', 'deepseek-v3.2', 'qwen3-max', 'ProvSyn']
    results = {}

    for dataset in datasets:
        G_real = load_graph(f"../data/realGraphs_test/{dataset}.graphml")

        for llm in llms:
            if llm == 'ProvSyn':
                path = f"synGraphs_name/{dataset}"
            else:
                path = f"baselines/synGraphs/{dataset}/{llm}"

            G_syns = sample_graph(path, 30)
            score = evaluate_graph2vec_similarity(G_real, G_syns)

            print(f"Average Similarity Score for {llm}: {score:.4f}")
            results[f"{dataset}-{llm}"] = score

    save_json(results, "results/graph2vec_provSyn.json")
