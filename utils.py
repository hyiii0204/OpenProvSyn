import networkx as nx
import pickle
import os
import random
import json


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def load_graph(file_path):
    try:
        if file_path.endswith('.graphml'):
            return nx.read_graphml(file_path)
        elif file_path.endswith('.dat'):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError("Unsupported file format.")
    except Exception:
        print(f"Loading {file_path} fails.")
        return None

def save_graph(graph, file_path):
    if file_path.endswith('.graphml'):
        nx.write_graphml(graph, file_path)
    elif file_path.endswith('.dat'):
        with open(file_path, 'wb') as f:
            pickle.dump(graph, f)
    else:
        raise ValueError("Unsupported file format.")

def sample_graph(folder_path, sample_size):
    files = [f for f in os.listdir(folder_path) if f.endswith('.graphml') or f.endswith('.dat')]
    sampled_files = random.sample(files, min(sample_size, len(files)))
    
    graphs = []
    for file in sampled_files:
        file_path = os.path.join(folder_path, file)
        graph = load_graph(file_path)
        if graph:
            graphs.append(graph)
    
    return graphs
    

def dfs(graph, node, visited, path, paths):
    visited.add(node)
    path.append(node)
    
    if len(path) > 10 or not list(graph.neighbors(node)): 
        paths.append(list(path))
    else:
        has_unvisited = False
        for neighbor in graph.neighbors(node):
            if neighbor not in visited:
                has_unvisited = True
                dfs(graph, neighbor, visited, path, paths)
        if not has_unvisited:
            paths.append(list(path))
    path.pop()
    visited.remove(node)


def find_bidirectional_nodes(G):
    bidirectional = []
    processed = set()
    edges = set(G.edges())
    
    for u, v in edges:
        if (v, u) in edges and (u, v) not in processed:
            bidirectional.append(u)  
            processed.add((u, v))
            processed.add((v, u))  
    return bidirectional

def get_dfs_paths(G):
    start_nodes = [node for node in G.nodes() if G.in_degree(node) == 0]
    bidirectional_nodes = find_bidirectional_nodes(G)
    nodes = list(set(start_nodes + bidirectional_nodes))  
    # nodes = start_nodes

    all_paths = []
    for node in nodes:
        visited = set()
        path = []
        paths = []
        dfs(G, node, visited, path, paths)
        all_paths.extend(paths)
        if len(all_paths) > 10000:
            break
    return all_paths



def generate_random_walks(graph, num_walks=10, walk_length=80,  text_attribute=None):
    walks = []
    nodes = list(graph.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)
        for node in nodes:
            walk = [node]
            current_node = node
            for _ in range(walk_length-1):
                neighbors = list(graph.neighbors(current_node))
                if len(neighbors) > 0:
                    current_node = random.choice(neighbors)
                    walk.append(current_node)
                else:
                    break
            walks.append([str(x) for x in walk])

            if text_attribute:
                node_texts = []
                for n in walk:
                    if text_attribute in graph.nodes[n] and graph.nodes[n][text_attribute]:
                        node_texts.append(str(graph.nodes[n][text_attribute]))
                    else:
                        node_texts.append(str(n))
            
                walks.append(node_texts)
    return walks