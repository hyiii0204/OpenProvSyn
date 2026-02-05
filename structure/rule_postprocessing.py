import os
import pickle
import networkx as nx

def load_graph(file_path):
    if file_path.endswith('.graphml'):
        return nx.read_graphml(file_path)
    elif file_path.endswith('.dat'):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError("Unsupported file format.")

schema_e3 = {
    ("subject", "subject"): ["EVENT_READ", "EVENT_WRITE", "EVENT_OPEN", "EVENT_CONNECT", "EVENT_RECVFROM", "EVENT_SENDTO", "EVENT_CLONE", "EVENT_SENDMSG", "EVENT_RECVMSG"],
    ("subject", "file"): ["EVENT_WRITE", "EVENT_CONNECT", "EVENT_SENDMSG", "EVENT_SENDTO", "EVENT_CLONE"],
    ("subject", "netflow"): ["EVENT_WRITE", "EVENT_SENDTO", "EVENT_CONNECT", "EVENT_SENDMSG"],
    ("file", "subject"): ["EVENT_READ", "EVENT_OPEN", "EVENT_RECVFROM", "EVENT_EXECUTE", "EVENT_RECVMSG"],
    ("netflow", "subject"): ["EVENT_OPEN", "EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG"],
}

schema_optc = {
    ("subject", "subject"): ["CREATE", "OPEN", "TERMINATE"],
    ("netflow", "subject"): ["MESSAGE", "OPEN", "START"],
    ("subject", "file"): ["CREATE", "DELETE", "MODIFY", "RENAME", "WRITE"],
    ("subject", "netflow"): ["MESSAGE", "START"],
    ("file", "subject"): ["READ"]
}

schema_e5 = {
    ("subject", "subject"): ["EVENT_READ", "EVENT_WRITE", "EVENT_OPEN", "EVENT_CONNECT", "EVENT_RECVFROM", "EVENT_SENDTO", "EVENT_CLONE", "EVENT_SENDMSG", "EVENT_RECVMSG"],
    ("subject", "file"): ["EVENT_WRITE", "EVENT_CONNECT", "EVENT_SENDMSG", "EVENT_SENDTO", "EVENT_CLONE"],
    ("subject", "netflow"): ["EVENT_WRITE", "EVENT_SENDTO", "EVENT_CONNECT", "EVENT_SENDMSG"],
    ("file", "subject"): ["EVENT_READ", "EVENT_OPEN", "EVENT_RECVFROM", "EVENT_EXECUTE", "EVENT_RECVMSG"],
    ("netflow", "subject"): ["EVENT_OPEN", "EVENT_READ", "EVENT_RECVFROM", "EVENT_RECVMSG"],
}

def clean_attributes(G):
    for n, d in G.nodes(data=True):
        for k, v in d.items():
            if not isinstance(v, (str, int, float, bool)):
                d[k] = str(v)
    for u, v, d in G.edges(data=True):
        for k, v in d.items():
            if not isinstance(v, (str, int, float, bool)):
                d[k] = str(v)
    return G

def process_and_save_graphs(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        current_schema = None

        if any(key in file_path for key in ["optc_h201", "optc_h501"]):
            current_schema = schema_optc
        elif any(key in file_path for key in ["theia_e5", "clearscope_e5"]):
            current_schema = schema_e5
        elif any(key in file_path for key in ["theia_e3", "cadets_e3"]):
            current_schema = schema_e3
            
        if current_schema is None:
            continue

        undirected_G = load_graph(file_path)
            
        directed_G = undirected_G.to_directed()
        edges_to_remove = []
        for u, v, data in directed_G.edges(data=True):
            u_type = directed_G.nodes[u].get('label')
            v_type = directed_G.nodes[v].get('label')
            edge_label = data.get('label')
            
            allowed_labels = current_schema.get((u_type, v_type), [])
            if edge_label not in allowed_labels:
                edges_to_remove.append((u, v))
        
        directed_G.remove_edges_from(edges_to_remove)
        directed_G.remove_nodes_from(list(nx.isolates(directed_G)))
        
        output_filename = os.path.splitext(filename)[0] + ".graphml"
        output_path = os.path.join(output_dir, output_filename)
        final_G = clean_attributes(directed_G)
        
        try:
            nx.write_graphml(final_G, output_path)
            print(f"(Nodes: {final_G.number_of_nodes()}, Edges: {final_G.number_of_edges()})")
        except:
            pass

if __name__ == "__main__":
    input_folder = 'theia_e3'
    output_folder = 'directed_theia_e3'
    process_and_save_graphs(input_folder, output_folder)