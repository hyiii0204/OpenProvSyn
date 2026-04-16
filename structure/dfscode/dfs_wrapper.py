import os
import subprocess
import tempfile
import pickle
import networkx as nx


def get_min_dfscode(G, temp_path=tempfile.gettempdir()):
    input_fd, input_path = tempfile.mkstemp(dir=temp_path)

    with open(input_path, 'w') as f:
        vcount = len(G.nodes)
        f.write(str(vcount) + '\n')
        i = 0
        d = {}
        for x in G.nodes:
            d[x] = i
            i += 1
            f.write(str(G.nodes[x]['label']) + '\n')

        ecount = len(G.edges)
        f.write(str(ecount) + '\n')
        # For directed graphs: direction=1 means the original edge goes src→dest
        # For undirected graphs: direction=1 for all edges
        is_directed = G.is_directed()
        for (u, v) in G.edges:
            if is_directed:
                f.write(str(d[u]) + ' ' + str(d[v]) +
                        ' ' + str(G[u][v]['label']) + ' 1\n')
            else:
                f.write(str(d[u]) + ' ' + str(d[v]) +
                        ' ' + str(G[u][v]['label']) + ' 1\n')

    output_fd, output_path = tempfile.mkstemp(dir=temp_path)

    dfscode_bin_path = 'bin/dfscode'
    with open(input_path, 'r') as f:
        subprocess.call([dfscode_bin_path, output_path, '2'], stdin=f)

    with open(output_path, 'r') as dfsfile:
        dfs_sequence = []
        for row in dfsfile.readlines():
            splited_row = row.split()
            # 6-tuple: (timestamp1, timestamp2, label1, direction, edge_label, label2)
            splited_row = [splited_row[2 * i + 1] for i in range(6)]
            dfs_sequence.append(splited_row)

    os.close(input_fd)
    os.close(output_fd)

    try:
        os.remove(input_path)
        os.remove(output_path)
    except OSError:
        pass

    return dfs_sequence


def graph_from_dfscode(dfscode):
    """Reconstruct a graph from a direction-aware 6-tuple DFS code.

    Each dfscode entry is (i, j, l1, D_e, e, l2):
      - i, j: DFS timestamps of source and target
      - l1, l2: node labels
      - D_e: direction indicator (1 = edge goes i->j, 0 = edge goes j->i)
      - e: edge label

    Returns a DiGraph if any D_e=0 is present (directed), else Graph.
    """
    if not dfscode:
        return nx.Graph()

    has_reverse = any(int(code[3]) == 0 for code in dfscode)
    graph = nx.DiGraph() if has_reverse else nx.Graph()

    for dfscode_edge in dfscode:
        i, j, l1, d_e, e, l2 = dfscode_edge
        graph.add_node(int(i), label=l1)
        graph.add_node(int(j), label=l2)

        if int(d_e) == 1:
            graph.add_edge(int(i), int(j), label=e)
        else:
            graph.add_edge(int(j), int(i), label=e)

    return graph


if __name__ == '__main__':
    with open(os.path.expanduser('~/MTP/data/dataset/ENZYMES/graphs/graph180.dat'), 'rb') as f:
        G = pickle.load(f)

    dfs_code = get_min_dfscode(G)
    print(len(dfs_code), G.number_of_edges())
    for code in dfs_code:
        print(code)