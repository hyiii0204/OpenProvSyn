import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from transformers import BertTokenizer, BertModel
import networkx as nx
import numpy as np
import os
from tqdm import tqdm

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('./Bert')
bert_model = BertModel.from_pretrained('./Bert', output_hidden_states=True).to(device)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.feature_mapping = torch.nn.Linear(in_channels, out_channels).to(device)
        self.conv1 = GATConv(out_channels, 8, heads=8, dropout=0.4).to(device)
        self.conv2 = GATConv(64, out_channels, heads=1, concat=False, dropout=0.4).to(device)

    def forward(self, data):
        x = F.elu(self.conv1(F.dropout(self.feature_mapping(data.x), p=0.4, training=self.training), data.edge_index))
        return self.conv2(F.dropout(x, p=0.4, training=self.training), data.edge_index)


def get_bert_embeddings(texts):
    embeddings = []
    bert_model.eval()
    for text in texts:
        if not text.strip():
            embeddings.append(np.zeros(768))
            continue
        inputs = {k: v.to(device) for k, v in tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).items()}
        with torch.no_grad():
            h = bert_model(**inputs).hidden_states
            embeddings.append(torch.stack(h[-4:]).mean(0).mean(1).cpu().numpy().flatten())
    return torch.tensor(np.array(embeddings), dtype=torch.float32).to(device)


def get_graph_triplet_vectors(G, gat_model):
    edge_labels = [G.edges[e].get('label', '') for e in G.edges()]
    if not edge_labels: 
        return torch.empty(0)

    n_bert = get_bert_embeddings([G.nodes[n].get('name', '') for n in G.nodes()])
    e_bert = get_bert_embeddings(edge_labels)
    n_idx = {node: i for i, node in enumerate(G.nodes())}
    e_idx = torch.tensor([(n_idx[u], n_idx[v]) for u, v in G.edges()], dtype=torch.long).t().to(device)

    gat_model.eval()
    with torch.no_grad():
        n_gat = gat_model(Data(x=n_bert, edge_index=e_idx))
        e_comp = gat_model.feature_mapping(e_bert)

    return torch.stack([torch.cat([n_gat[n_idx[u]], n_gat[n_idx[v]], e_comp[i]]) for i, (u, v) in enumerate(G.edges())])


def compute_bertscore_precision(cand_vecs, ref_vecs):
    if cand_vecs.size(0) == 0 or ref_vecs.size(0) == 0: 
        return 0.0
    
    sim = torch.matmul(F.normalize(cand_vecs, p=2, dim=-1), F.normalize(ref_vecs, p=2, dim=-1).t())
    max_sim = (torch.max(sim, dim=1)[0] - 0.85) / 0.15
    return torch.clamp(max_sim, min=0.0).mean().item()


def evaluate_graph_similarity(G_ref, G_pred_list, gat_model, ref_emb=None):
    r_emb = ref_emb if ref_emb is not None else get_graph_triplet_vectors(G_ref, gat_model)
    res = []

    for G_p in tqdm(G_pred_list, desc="Eval", leave=False):
        if len(G_p.edges()) == 0:
            res.append(0.0)
            continue
        c_vecs = get_graph_triplet_vectors(G_p, gat_model)
        res.append(compute_bertscore_precision(c_vecs, r_emb))

    return sum(res) / len(res) if res else 0.0


if __name__ == "__main__":
    from utils import load_graph, sample_graph, save_json

    datasets = ["cadets_e3", "theia_e3", 'clearscope_e5', 'theia_e5', 'optc_h501', 'optc_h201']
    llms = ['gpt-5.1', 'claude-sonnet-4.5', 'deepseek-v3.2', 'qwen3-max', 'ProvSyn']
    results = {}
    main_gat = GAT(768, 64).to(device)


    for ds in datasets:
        G_r = load_graph(f"../data/realGraphs_test/{ds}.graphml")
        r_vecs = get_graph_triplet_vectors(G_r, main_gat)

        for llm in llms:
            path = f"synGraphs_name/{ds}" if llm == "ProvSyn" else f"baselines/synGraphs/{ds}/{llm}"
            if not os.path.exists(path): 
                continue

            G_s = sample_graph(path, sample_size=10)
            score = evaluate_graph_similarity(G_r, G_s, main_gat, ref_emb=r_vecs)
            
            print(f"{ds}-{llm}: {score:.4f}")
            results[f"{ds}-{llm}"] = score

    save_json(results, "results/semantic_2.json")