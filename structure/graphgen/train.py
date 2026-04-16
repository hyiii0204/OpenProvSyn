import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.distributions import Categorical
import networkx as nx

from graphgen.model import create_model
from utils import load_model, get_model_attribute
from dfscode.dfs_wrapper import graph_from_dfscode


# ---------- Schema definitions for provenance graphs ----------

SCHEMA_E3 = {
    ("subject", "subject"): ["EVENT_READ", "EVENT_WRITE", "EVENT_OPEN", "EVENT_CONNECT",
                             "EVENT_RECVFROM", "EVENT_SENDTO", "EVENT_CLONE",
                             "EVENT_SENDMSG", "EVENT_RECVMSG"],
    ("subject", "file"): ["EVENT_WRITE", "EVENT_CONNECT", "EVENT_SENDMSG",
                          "EVENT_SENDTO", "EVENT_CLONE"],
    ("subject", "netflow"): ["EVENT_WRITE", "EVENT_SENDTO", "EVENT_CONNECT",
                             "EVENT_SENDMSG"],
    ("file", "subject"): ["EVENT_READ", "EVENT_OPEN", "EVENT_RECVFROM",
                          "EVENT_EXECUTE", "EVENT_RECVMSG"],
    ("netflow", "subject"): ["EVENT_OPEN", "EVENT_READ", "EVENT_RECVFROM",
                             "EVENT_RECVMSG"],
}

SCHEMA_E5 = {
    ("subject", "subject"): ["EVENT_READ", "EVENT_WRITE", "EVENT_OPEN", "EVENT_CONNECT",
                             "EVENT_RECVFROM", "EVENT_SENDTO", "EVENT_CLONE",
                             "EVENT_SENDMSG", "EVENT_RECVMSG"],
    ("subject", "file"): ["EVENT_WRITE", "EVENT_CONNECT", "EVENT_SENDMSG",
                          "EVENT_SENDTO", "EVENT_CLONE"],
    ("subject", "netflow"): ["EVENT_WRITE", "EVENT_SENDTO", "EVENT_CONNECT",
                             "EVENT_SENDMSG"],
    ("file", "subject"): ["EVENT_READ", "EVENT_OPEN", "EVENT_RECVFROM",
                          "EVENT_EXECUTE", "EVENT_RECVMSG"],
    ("netflow", "subject"): ["EVENT_OPEN", "EVENT_READ", "EVENT_RECVFROM",
                             "EVENT_RECVMSG"],
}

SCHEMA_OPTC = {
    ("subject", "subject"): ["CREATE", "OPEN", "TERMINATE"],
    ("netflow", "subject"): ["MESSAGE", "OPEN", "START"],
    ("subject", "file"): ["CREATE", "DELETE", "MODIFY", "RENAME", "WRITE"],
    ("subject", "netflow"): ["MESSAGE", "START"],
    ("file", "subject"): ["READ"],
}


def _get_schema(graph_type):
    """Select the correct schema based on dataset name."""
    if any(k in graph_type for k in ["e3", "cadets_e3", "theia_e3"]):
        return SCHEMA_E3
    elif any(k in graph_type for k in ["e5", "theia_e5", "clearscope_e5"]):
        return SCHEMA_E5
    elif any(k in graph_type for k in ["optc"]):
        return SCHEMA_OPTC
    else:
        return None


def _build_edge_mask(schema, node_forward, edge_forward, device):
    """Pre-compute a mask tensor of shape (num_node_labels, num_node_labels, num_edge_labels).
    mask[n1, n2, e] = 1 if edge label e is allowed between node labels n1 and n2.

    node_forward / edge_forward map string labels to integer indices
    (these are the *original* dicts without the +1 end-token).

    Returns None if schema is None (no constraint).
    """
    if schema is None:
        return None

    num_node_labels = len(node_forward)
    num_edge_labels = len(edge_forward)
    mask = torch.zeros(num_node_labels, num_node_labels, num_edge_labels, device=device)

    node_backward = {v: k for k, v in node_forward.items()}

    for n1_idx in range(num_node_labels):
        for n2_idx in range(num_node_labels):
            n1_label = node_backward[n1_idx]
            n2_label = node_backward[n2_idx]
            allowed = schema.get((n1_label, n2_label), [])
            for edge_label in allowed:
                if edge_label in edge_forward:
                    mask[n1_idx, n2_idx, edge_forward[edge_label]] = 1.0

    return mask


# ---------- Training ----------

def evaluate_loss(args, model, data, feature_map):
    x_len_unsorted = data['len'].to(args.device)
    x_len_max = max(x_len_unsorted)
    batch_size = x_len_unsorted.size(0)

    # sort input for packing variable length sequences
    x_len, sort_indices = torch.sort(x_len_unsorted, dim=0, descending=True)

    max_nodes = feature_map['max_nodes']
    len_node_vec = len(feature_map['node_forward']) + 1
    len_edge_vec = len(feature_map['edge_forward']) + 1
    len_direction_vec = len(feature_map['direction_forward']) + 1
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_direction_vec + len_edge_vec

    # Prepare targets with end_tokens already there
    t1 = torch.index_select(
        data['t1'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    t2 = torch.index_select(
        data['t2'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    v1 = torch.index_select(
        data['v1'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    de = torch.index_select(
        data['de'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    e = torch.index_select(
        data['e'][:, :x_len_max + 1].to(args.device), 0, sort_indices)
    v2 = torch.index_select(
        data['v2'][:, :x_len_max + 1].to(args.device), 0, sort_indices)

    x_t1, x_t2 = F.one_hot(t1, num_classes=max_nodes +
                           2)[:, :, :-1], F.one_hot(t2, num_classes=max_nodes + 2)[:, :, :-1]
    x_v1, x_v2 = F.one_hot(v1, num_classes=len_node_vec +
                           1)[:, :, :-1], F.one_hot(v2, num_classes=len_node_vec + 1)[:, :, :-1]
    x_de = F.one_hot(de, num_classes=len_direction_vec + 1)[:, :, :-1]
    x_e = F.one_hot(e, num_classes=len_edge_vec + 1)[:, :, :-1]

    # 6-tuple: (t1, t2, v1, D_e, e, v2)
    x_target = torch.cat((x_t1, x_t2, x_v1, x_de, x_e, x_v2), dim=2).float()

    # initialize dfs_code_rnn hidden according to batch size
    model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
        batch_size=batch_size)

    # Teacher forcing: Feed the target as the next input
    # Start token is all zeros
    dfscode_rnn_input = torch.cat(
        (torch.zeros(batch_size, 1, feature_len, device=args.device), x_target[:, :-1, :]), dim=1)

    # Forward propogation
    dfscode_rnn_output = model['dfs_code_rnn'](
        dfscode_rnn_input, input_len=x_len + 1)

    # Evaluating dfscode tuple: (t1, t2, v1, D_e, e, v2)
    timestamp1 = model['output_timestamp1'](dfscode_rnn_output)
    timestamp2 = model['output_timestamp2'](dfscode_rnn_output)
    vertex1 = model['output_vertex1'](dfscode_rnn_output)
    direction = model['output_direction'](dfscode_rnn_output)
    edge = model['output_edge'](dfscode_rnn_output)
    vertex2 = model['output_vertex2'](dfscode_rnn_output)

    if args.loss_type == 'BCE':
        x_pred = torch.cat(
            (timestamp1, timestamp2, vertex1, direction, edge, vertex2), dim=2)

        # Cleaning the padding i.e setting it to zero
        x_pred = pack_padded_sequence(x_pred, x_len + 1, batch_first=True)
        x_pred, _ = pad_packed_sequence(x_pred, batch_first=True)

        if args.weights:
            weight = torch.cat((feature_map['t1_weight'].to(args.device), feature_map['t2_weight'].to(args.device),
                                feature_map['v1_weight'].to(args.device), feature_map['de_weight'].to(args.device),
                                feature_map['e_weight'].to(args.device),
                                feature_map['v2_weight'].to(args.device)))

            weight = weight.expand(batch_size, x_len_max + 1, -1)
        else:
            weight = None

        loss_sum = F.binary_cross_entropy(
            x_pred, x_target, reduction='none', weight=weight)
        loss = torch.mean(
            torch.sum(loss_sum, dim=[1, 2]) / (x_len.float() + 1))

    elif args.loss_type == 'NLL':
        timestamp1 = timestamp1.transpose(dim0=1, dim1=2)
        timestamp2 = timestamp2.transpose(dim0=1, dim1=2)
        vertex1 = vertex1.transpose(dim0=1, dim1=2)
        direction = direction.transpose(dim0=1, dim1=2)
        edge = edge.transpose(dim0=1, dim1=2)
        vertex2 = vertex2.transpose(dim0=1, dim1=2)

        loss_t1 = F.nll_loss(
            timestamp1, t1, ignore_index=max_nodes + 1, weight=feature_map.get('t1_weight'))
        loss_t2 = F.nll_loss(
            timestamp2, t2, ignore_index=max_nodes + 1, weight=feature_map.get('t2_weight'))
        loss_v1 = F.nll_loss(vertex1, v1, ignore_index=len(
            feature_map['node_forward']) + 1, weight=feature_map.get('v1_weight'))
        loss_de = F.nll_loss(direction, de, ignore_index=len(
            feature_map['direction_forward']) + 1, weight=feature_map.get('de_weight'))
        loss_e = F.nll_loss(edge, e, ignore_index=len(
            feature_map['edge_forward']) + 1, weight=feature_map.get('e_weight'))
        loss_v2 = F.nll_loss(vertex2, v2, ignore_index=len(
            feature_map['node_forward']) + 1, weight=feature_map.get('v2_weight'))

        loss = loss_t1 + loss_t2 + loss_v1 + loss_de + loss_e + loss_v2

    return loss


# ---------- Generation with Schema-Constrained Decoding ----------

def predict_graphs(eval_args):
    train_args = eval_args.train_args
    feature_map = get_model_attribute(
        'feature_map', eval_args.model_path, eval_args.device)
    train_args.device = eval_args.device

    model = create_model(train_args, feature_map)
    load_model(eval_args.model_path, eval_args.device, model)

    for _, net in model.items():
        net.eval()

    max_nodes = feature_map['max_nodes']
    len_node_vec = len(feature_map['node_forward']) + 1
    len_edge_vec = len(feature_map['edge_forward']) + 1
    len_direction_vec = len(feature_map['direction_forward']) + 1
    # 6-tuple feature: t1, t2, v1, D_e, e, v2
    feature_len = 2 * (max_nodes + 1) + 2 * len_node_vec + len_direction_vec + len_edge_vec

    # Offset indices for one-hot positions in rnn_input
    offset_t1 = 0
    offset_t2 = max_nodes + 1
    offset_v1 = 2 * (max_nodes + 1)
    offset_de = 2 * (max_nodes + 1) + len_node_vec
    offset_e = 2 * (max_nodes + 1) + len_node_vec + len_direction_vec
    offset_v2 = 2 * (max_nodes + 1) + len_node_vec + len_direction_vec + len_edge_vec

    # Build schema edge mask (None if no schema applies)
    schema = _get_schema(train_args.graph_type)
    edge_mask = _build_edge_mask(
        schema, feature_map['node_forward'], feature_map['edge_forward'],
        eval_args.device)

    graphs = []

    for _ in range(eval_args.count // eval_args.batch_size):
        # initialize dfs_code_rnn hidden according to batch size
        model['dfs_code_rnn'].hidden = model['dfs_code_rnn'].init_hidden(
            batch_size=eval_args.batch_size)

        rnn_input = torch.zeros(
            (eval_args.batch_size, 1, feature_len), device=eval_args.device)
        # 6 columns: t1, t2, v1, D_e, e, v2
        pred = torch.zeros(
            (eval_args.batch_size, eval_args.max_num_edges, 6), device=eval_args.device)

        for i in range(eval_args.max_num_edges):
            rnn_output = model['dfs_code_rnn'](rnn_input)

            # Get raw outputs for all 6 fields
            timestamp1_logits = model['output_timestamp1'](
                rnn_output).reshape(eval_args.batch_size, -1)
            timestamp2_logits = model['output_timestamp2'](
                rnn_output).reshape(eval_args.batch_size, -1)
            vertex1_logits = model['output_vertex1'](
                rnn_output).reshape(eval_args.batch_size, -1)
            direction_logits = model['output_direction'](rnn_output).reshape(
                eval_args.batch_size, -1)
            edge_logits = model['output_edge'](rnn_output).reshape(
                eval_args.batch_size, -1)
            vertex2_logits = model['output_vertex2'](
                rnn_output).reshape(eval_args.batch_size, -1)

            # --- Sample t1, t2, v1, D_e, v2 first ---
            if train_args.loss_type == 'BCE':
                timestamp1 = Categorical(timestamp1_logits).sample()
                timestamp2 = Categorical(timestamp2_logits).sample()
                vertex1 = Categorical(vertex1_logits).sample()
                direction = Categorical(direction_logits).sample()
                vertex2 = Categorical(vertex2_logits).sample()
            else:  # NLL
                timestamp1 = Categorical(logits=timestamp1_logits).sample()
                timestamp2 = Categorical(logits=timestamp2_logits).sample()
                vertex1 = Categorical(logits=vertex1_logits).sample()
                direction = Categorical(logits=direction_logits).sample()
                vertex2 = Categorical(logits=vertex2_logits).sample()

            # --- Schema-constrained edge sampling ---
            if edge_mask is not None:
                # Get allowed edge indices for each (v1, v2) pair in the batch
                # vertex1, vertex2 are in [0, len_node_vec-1], where
                # indices < len(node_forward) are real labels,
                # index == len(node_forward) is end token.
                # For end-token v1/v2, allow all edges (will stop anyway).
                # Clamp indices to valid range for mask lookup.
                v1_for_mask = vertex1.clamp(max=len(feature_map['node_forward']) - 1)
                v2_for_mask = vertex2.clamp(max=len(feature_map['node_forward']) - 1)

                # batch_mask[b] = edge_mask[v1[b], v2[b]] shape: (num_edge_labels,)
                batch_mask = edge_mask[v1_for_mask, v2_for_mask]  # (B, num_edge_labels)

                # For samples where v1 or v2 is end-token, allow all edges
                is_end = (vertex1 >= len(feature_map['node_forward'])) | \
                         (vertex2 >= len(feature_map['node_forward']))
                batch_mask[is_end] = 1.0

                if train_args.loss_type == 'BCE':
                    # edge_logits are probabilities (Softmax output)
                    masked_probs = edge_logits * batch_mask
                    # Renormalize
                    masked_probs = masked_probs / (masked_probs.sum(dim=1, keepdim=True) + 1e-10)
                    edge = Categorical(masked_probs).sample()
                else:  # NLL
                    # edge_logits are log-probabilities; convert to logits for masking
                    masked_logits = edge_logits.clone()
                    mask_zero = (batch_mask == 0)
                    masked_logits[mask_zero] = -1e10
                    edge = Categorical(logits=masked_logits).sample()
            else:
                # No schema constraint
                if train_args.loss_type == 'BCE':
                    edge = Categorical(edge_logits).sample()
                else:
                    edge = Categorical(logits=edge_logits).sample()

            # --- Build next rnn_input ---
            rnn_input = torch.zeros(
                (eval_args.batch_size, 1, feature_len), device=eval_args.device)

            rnn_input[torch.arange(eval_args.batch_size), 0, offset_t1 + timestamp1] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0, offset_t2 + timestamp2] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0, offset_v1 + vertex1] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0, offset_de + direction] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0, offset_e + edge] = 1
            rnn_input[torch.arange(eval_args.batch_size), 0, offset_v2 + vertex2] = 1

            pred[:, i, 0] = timestamp1
            pred[:, i, 1] = timestamp2
            pred[:, i, 2] = vertex1
            pred[:, i, 3] = direction
            pred[:, i, 4] = edge
            pred[:, i, 5] = vertex2

        nb = feature_map['node_backward']
        eb = feature_map['edge_backward']
        for i in range(eval_args.batch_size):
            dfscode = []
            for j in range(eval_args.max_num_edges):
                # End token checks: any field at its end value stops generation
                if pred[i, j, 0] == max_nodes or pred[i, j, 1] == max_nodes \
                        or pred[i, j, 2] == len_node_vec - 1 \
                        or pred[i, j, 3] == len_direction_vec - 1 \
                        or pred[i, j, 4] == len_edge_vec - 1 \
                        or pred[i, j, 5] == len_node_vec - 1:
                    break

                # 6-tuple: (t_u, t_v, L_u, D_e, L_e, L_v)
                dfscode.append(
                    (int(pred[i, j, 0].data), int(pred[i, j, 1].data),
                     nb[int(pred[i, j, 2].data)],
                     int(pred[i, j, 3].data),   # D_e: 0 or 1
                     eb[int(pred[i, j, 4].data)],
                     nb[int(pred[i, j, 5].data)]))

            graph = graph_from_dfscode(dfscode)

            # Remove self loops
            graph.remove_edges_from(nx.selfloop_edges(graph))

            # Take maximum weakly connected component (works for both Graph and DiGraph)
            if len(graph.nodes()):
                if graph.is_directed():
                    max_comp = max(nx.weakly_connected_components(graph), key=len)
                else:
                    max_comp = max(nx.connected_components(graph), key=len)
                graph = graph.__class__(graph.subgraph(max_comp))

            graphs.append(graph)

    return graphs