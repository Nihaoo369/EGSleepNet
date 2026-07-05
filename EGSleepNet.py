import math

import torch
from torch import nn
from torch.autograd import Variable
from args_WUU import Path, Config

import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
from net import GNNStack

def batch_to_single_adjacency(A):
    
    A_max, _ = A.max(dim=0)           
    A_thresh = A_max * (A_max > 0.05) 
    edge_index = (A_thresh > 0).nonzero(as_tuple=False).t()  # [2, num_edges]
    return A_thresh, edge_index

class CosineHypergraphLayer2(nn.Module):
    def __init__(self, in_dim, out_dim, num_hyperedges, top_k=5):
        super().__init__()
        self.num_hyperedges = num_hyperedges
        self.top_k = top_k

        self.hyperedge_embed = nn.Parameter(torch.randn(num_hyperedges, in_dim))  # [E, F]
        self.node_update = nn.Linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim)

        self.gate_proj = nn.Sequential(
            nn.Linear(in_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, N, _ = x.shape
        E = self.num_hyperedges

        x_norm = F.normalize(x, p=2, dim=2)
        h_norm = F.normalize(self.hyperedge_embed, p=2, dim=1)
        h_norm = h_norm.unsqueeze(0).expand(batch_size, -1, -1)

        sim = torch.bmm(x_norm, h_norm.transpose(1, 2))  # [B, N, E]
        gate = self.gate_proj(x)
        sim = sim * gate

        topk_val, topk_idx = torch.topk(sim, self.top_k, dim=2)
        H_sparse = torch.zeros_like(sim)
        H_sparse = H_sparse.view(batch_size * N, E)
        topk_idx_ = topk_idx.view(batch_size * N, self.top_k)
        topk_val_ = topk_val.view(batch_size * N, self.top_k)
        H_sparse.scatter_(1, topk_idx_, topk_val_)
        H_sparse = H_sparse.view(batch_size, N, E)

        edge_feats = torch.bmm(H_sparse.transpose(1, 2), x)
        edge_feats = edge_feats / (H_sparse.transpose(1, 2).sum(dim=2, keepdim=True) + 1e-6)

        node_feats = torch.bmm(H_sparse, edge_feats)
        node_feats = self.node_update(node_feats)

        x_down = self.node_update(x)
        out = self.norm(node_feats + x_down)

        return out, H_sparse

def hypergraph_to_adjacency(H):
    batch_size, N, E = H.shape
    H_T = H.transpose(1, 2)
    A = torch.bmm(H, H_T)  # [B, N, N]

    # 归一化 A = D^{-1/2} A D^{-1/2}
    D = A.sum(dim=2)  # [B, N]
    D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
    D_inv_sqrt = D_inv_sqrt.unsqueeze(2)  # [B, N, 1]
    A_norm = A * D_inv_sqrt * D_inv_sqrt.transpose(1, 2)  # [B, N, N]

    # 合并为全局邻接矩阵
    A_global, edge_index = batch_to_single_adjacency(A_norm)
    return A_global, edge_index

class EGSleepNet(nn.Module): 
    def __init__(self, config):
        super(EGSleepNet, self).__init__()

        self.position_single = PositionalEncoding(d_model=config.dim_model, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=config.dim_model, nhead=config.num_head, dim_feedforward=config.forward_hidden, dropout=config.dropout)
        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)
        self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder)

        self.drop = nn.Dropout(p=0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        self.position_multi = PositionalEncoding(d_model=config.dim_model * 3, dropout=0.1)
        encoder_layer_multi = nn.TransformerEncoderLayer(d_model=config.dim_model * 3, nhead=config.num_head,dim_feedforward=config.forward_hidden, dropout=config.dropout)
        self.transformer_encoder_multi = nn.TransformerEncoder(encoder_layer_multi, num_layers=config.num_encoder_multi)

        self.dct_layer = FcaBasicBlock(29, 29) 
        self.fc1 = nn.Sequential(
            nn.Linear(config.dim_model * 3, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(config.fc_hidden, config.num_classes)
        )

        self.hg1 = CosineHypergraphLayer2(128*3, 128*3, 30, 5)

        self.gat1 = GATConv(config.dim_model * 3, config.dim_model * 3 // config.gat_heads, heads=config.gat_heads, concat=True, dropout=config.dropout)
        self.gat2 = GATConv(config.dim_model * 3, config.dim_model * 3, heads=1, concat=False, dropout=config.dropout)
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128)
        )
        
    
    def forward(self, x):
        x1 = x[:, 0, :, :]
        x2 = x[:, 1, :, :]
        x3 = x[:, 2, :, :]
        x1 = self.position_single(x1)
        x2 = self.position_single(x2)
        x3 = self.position_single(x3)

        combined_x1 = torch.cat([x2, x3], dim=-1)
        combined_x1 = self.fusion_layer(combined_x1)
        x1 = x1 + combined_x1

        combined_x2 = torch.cat([x1, x3], dim=-1)
        combined_x2 = self.fusion_layer(combined_x2)
        x2 = x2 + combined_x2

        combined_x3 = torch.cat([x1, x2], dim=-1)
        combined_x3 = self.fusion_layer(combined_x3)
        x3 = x3 + combined_x3

        x1 = self.transformer_encoder_1(x1)     # (batch_size, 29, 128)
        x2 = self.transformer_encoder_2(x2)
        x3 = self.transformer_encoder_3(x3)


        x = torch.cat([x1, x2, x3], dim=2) # (batch_size, 29, 384)
        x = self.dct_layer(x)

        x = self.drop(x)
        x = self.layer_norm(x)
        
        mean_row = x.mean(dim=1, keepdim=True) 
        x = torch.cat([x, mean_row], dim=1)
        residual = x
        
        x, H1 = self.hg1(x)
        x = F.relu(x)
        residual_1 = x
        
        A_global, edge_index = hypergraph_to_adjacency(H1)

        edge_index = edge_index.to(x.device)
        data_list = []
        for i in range(x.shape[0]):
            node_feat = x[i]  
            data = Data(x=node_feat, edge_index=edge_index)
            data_list.append(data)
        batch_graph = Batch.from_data_list(data_list)
        out = F.elu(self.gat1(batch_graph.x, batch_graph.edge_index)) # 145, 128
        out = F.elu(self.gat2(out, batch_graph.edge_index))

        residual_2 = out
        out = residual.view(-1, 384) + residual_1.view(-1,384) + residual_2
        out = global_mean_pool(out, batch_graph.batch)
        ###
        # x = self.position_multi(x)
        # x = self.transformer_encoder_multi(x)
        # x = self.layer_norm(x + residual)       
        ###

        # x = x.view(x.size(0), -1)
        x = self.fc1(out)
        x = self.fc2(x)
        return x