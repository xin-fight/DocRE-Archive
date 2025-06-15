import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class GraphConvolutionLayer(nn.Module):
    """公式5，图卷积网络（GCN）"""
    def __init__(self, input_size, hidden_size, graph_drop):
        super(GraphConvolutionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)

        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

    def forward(self, input):
        # nodes_embed [bs, node, hidden]    node_adj [bs, node, node]
        nodes_embed, node_adj = input
        # [bs, node, hidden]
        h = torch.matmul(nodes_embed, self.W.unsqueeze(0))
        sum_nei = torch.zeros_like(h)
        # [bs, node, node] mul [bs, node, hidden] -> [bs, node, hidden]
        sum_nei += torch.matmul(node_adj, h)
        # [bs, node, 1]
        degs = torch.sum(node_adj, dim=-1).float().unsqueeze(dim=-1)
        norm = 1.0 / degs
        # [bs, node, hidden] * [bs, node, 1] -> [bs, node, hidden]
        dst = sum_nei * norm + self.bias
        out = self.drop(torch.relu(dst))
        return nodes_embed + out, node_adj


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key):
    N_bt, h, N_nodes, _ = query.shape
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
    return scores


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, edges, in_features: int, out_features: int, n_heads: int, dropout=0.0):
        super().__init__()

        self.h = n_heads
        self.d_k = out_features // n_heads
        self.edges = edges
        self.linear_layers = nn.ModuleList()
        for i in range(len(edges)):
            self.linear_layers.append(clones(nn.Linear(in_features, out_features), 2))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
        # h [bs, node, hidden]      adj_mat [bs, node, node]
        N_bt, N_nodes, _ = h.shape  # [bs, node, hidden]
        adj_mat = adj_mat.unsqueeze(1)  # [bs, 1, node, node]
        # bs, n_heads, N_nodes, N_nodes
        scores = torch.zeros(N_bt, self.h, N_nodes, N_nodes).to(h)

        """利用动态算法来对图进行剪枝和结构优化 —— 公式4"""
        # 四种类型的边
        for edge in range(len(self.edges)):
            # 公式4中计算Hv*WQ     Hv*WK     [bs, node, hidden] -> [bs, 2, node, hidden//2]
            q, k = [l(x).view(N_bt, -1, self.h, self.d_k).transpose(1, 2) for l, x in
                    zip(self.linear_layers[edge], (h, h))]
            # dj_mat != edge + 1这是再prepro.py中定义的边的记号：
            #   self-loop 1     mention-anaphor edges 2     co-reference edges 3     inter-entity edges 4
            scores += attention(q, k).masked_fill(adj_mat != edge + 1, 0)
        scores = scores.masked_fill(scores == 0, -1e9)
        scores = self.dropout(scores)
        attn = F.softmax(scores, dim=-1)
        # [bs, 2, node, node] -> [2, bs, node, node]
        return attn.transpose(0, 1)


class AttentionGCNLayer(nn.Module):
    def __init__(self, edges, input_size, nhead=2, graph_drop=0.0, iters=2, attn_drop=0.0):
        """
        edges ['self-loop', 'mention-anaphor', 'co-reference', 'inter-entity']
        """
        super(AttentionGCNLayer, self).__init__()
        self.nhead = nhead
        self.graph_attention = MultiHeadDotProductAttention(edges, input_size, input_size, self.nhead, attn_drop)

        """公式5，图卷积网络（GCN）"""
        self.gcn_layers = nn.Sequential(
            *[GraphConvolutionLayer(input_size, input_size, graph_drop) for _ in range(iters)])
        self.blocks = nn.ModuleList([self.gcn_layers for _ in range(self.nhead)])

        self.aggregate_W = nn.Linear(input_size * nhead, input_size)

    def forward(self, nodes_embed, node_adj):
        # [bs, node, hidden]   [bs, node, node]
        output = []
        """经过公式4之后得到 新的邻接矩阵"""
        # [2, bs, node, node]
        graph_attention = self.graph_attention(nodes_embed, node_adj)

        """公式5，图卷积网络（GCN）"""
        for cnt in range(0, self.nhead):
            hi, _ = self.blocks[cnt]((nodes_embed, graph_attention[cnt]))
            output.append(hi)

        # 将多头进行拼接
        output = torch.cat(output, dim=-1)
        return self.aggregate_W(output), graph_attention
