import networkx as nx
import random
import json
from utils import *
from collections import Counter


def create_conversation(question, answer):
    return {
        "conversations": [
            {
                "from": "human",
                "value": f"Fill in the missing entity names in the JSON structure. This structure is related to activities in a Linux system and may involve various system entities, such as processes, files, or network-related components. Each entity should be named appropriately based on its type and its action on the next entity in the JSON. Please provide realistic and diverse names for these entities.\n{json.dumps(question)}"
            },
            {
                "from": "gpt",
                "value": json.dumps(answer)
            }
        ]
    }

def format_path_to_qa(G, path, mask_all=False):

    question = []
    answer = []
    nodes = []  

    type_map = {0: "process", 1: "file", 2: "network"}


    for i, node in enumerate(path):
        node_type = G.nodes[node].get("label", "")
        if type(node_type) != str:
            node_type = type_map[node_type]
        node_name = G.nodes[node].get("name", "")
        node_action = None

        if i + 1 < len(path):
            edge_data = G.get_edge_data(node, path[i + 1])
            node_action = edge_data.get("label", "") if edge_data else ""


        node_q = {"type": node_type, "name": node_name, "action": node_action}
        node_a = {"type": node_type, "name": node_name, "action": node_action}

        nodes.append((node_q, node_a)) 


    if mask_all:
        for node_q, node_a in nodes:
            node_q["name"] = "[FILL]"
            question.append(node_q)
            answer.append(node_a)
    else:
        num_nodes = len(nodes)
        if num_nodes > 1:  
            num_to_mask = random.randint(1, num_nodes - 1)
            indices_to_mask = random.sample(range(num_nodes), num_to_mask)

            for idx, (node_q, node_a) in enumerate(nodes):
                if idx in indices_to_mask:
                    node_q["name"] = "[FILL]"  
                question.append(node_q)
                answer.append(node_a)
        else:
            node_q, node_a = nodes[0]
            if random.random() > 0.5:
                node_q["name"] = "[FILL]"
            question.append(node_q)
            answer.append(node_a)

    return {"Q": question, "A": answer}

def create_conversations(G, paths):
    conversations = []  

    for path in paths:
        mask_all = True
        qa = format_path_to_qa(G, path, mask_all)
        conversations.append(create_conversation(qa['Q'], qa["A"]))
        mask_all = False
        qa = format_path_to_qa(G, path, mask_all)
        conversations.append(create_conversation(qa['Q'], qa["A"]))
    
    print(len(conversations))

    return conversations


def clean_path(G, path):
    while True:
        deleted = False
        for i, node in enumerate(path):
            if G.nodes[node].get("label", "") in ["<NA>", "None", 'NA']:
                path = path[i + 1:]  
                deleted = True
                break  
        if not deleted:
            break  
    return path  


if __name__ == "__main__":

    # datasets = ['nodlink', 'cadets', 'theia', 'trace']
    datasets = ['theia_e3']

    for dataset in datasets:

        file_path = f'../data/realGraphs/{dataset}.graphml' 

        G = load_graph(file_path)

        all_dfs_paths = get_dfs_paths(G)
        print(len(all_dfs_paths))
        cleaned_paths = []

        for path in all_dfs_paths:
            cleaned_paths.append(clean_path(G, path))

        output_file = f"lora_data/qa_{dataset}.json"

        # print(len(cleaned_paths))
        # filtered_paths = [p[:5] for p in cleaned_paths if len(p) >= 2]
        filtered_paths = [p for p in cleaned_paths if 2 <= len(p) <= 8]
        conversations = create_conversations(G, random.sample(filtered_paths, min(10000, len(filtered_paths))))

        save_json(conversations, output_file)


        path_lengths = [len(path) for path in filtered_paths]
        path_length_counts = Counter(path_lengths)

        print(f"Path length counts for dataset '{dataset}':")
        for length, count in sorted(path_length_counts.items()):
            print(f"Length {length}: {count} paths")

