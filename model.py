import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GraphConv


class FPNLayer(nn.Module):
    def __init__(self, num_src, num_dst, prior_mask, custom_thresholds=None):
        super().__init__()
        self.num_dst = num_dst
        self.register_buffer('prior_mask', prior_mask)
        self.exploration_rate = nn.Parameter(torch.tensor(0.0)) 
        self.D_matrix = None
        if custom_thresholds is not None:
            inv_thresholds = torch.logit(custom_thresholds, eps=1e-5)
            self.thresholds = nn.Parameter(inv_thresholds)
        else:
            self.thresholds = nn.Parameter(torch.rand(num_src, num_dst))
            nn.init.constant_(self.thresholds, 0.0)

    def forward(self, source_tokens, beta, gnn_sim_matrix=None):
        token_expanded = source_tokens.unsqueeze(1).expand(-1, self.num_dst)
        thresholds = torch.sigmoid(self.thresholds)
        gates = torch.sigmoid(beta * (token_expanded - thresholds))
        base_adj = gates * self.prior_mask
        self.D_matrix = (token_expanded - thresholds).detach().cpu()

        if gnn_sim_matrix is not None:
            alpha = torch.sigmoid(self.exploration_rate)
            explore_adj = gates * gnn_sim_matrix * (1 - self.prior_mask) * alpha 
            optimized_adj = base_adj + explore_adj
            flow_matrix = token_expanded * optimized_adj
            target_tokens, _ = torch.max(flow_matrix, dim=0) 
            return target_tokens, optimized_adj, explore_adj

        else:
            optimized_adj = base_adj
            flow_matrix = token_expanded * optimized_adj
            target_tokens, _ = torch.max(flow_matrix, dim=0)

            return target_tokens, optimized_adj


class FPN_GCN(nn.Module):
    def __init__(self, hidden_dim, in_m, in_g, in_d, 
                 N_M, N_G, N_D, m2g_mask, g2d_thresholds, g2d_mask):
        super().__init__()
        self.lin_m = nn.Linear(in_m, hidden_dim)
        self.lin_g = nn.Linear(in_g, hidden_dim)
        self.lin_d = nn.Linear(in_d, hidden_dim)
        self.logit_projector = nn.Linear(1, 1)
        nn.init.constant_(self.logit_projector.weight, 1.0)
        nn.init.constant_(self.logit_projector.bias, 0.0)
        self.hidden_dim = hidden_dim
        self.fpn_beta = 10.0
        self.N_g = N_G
        self.N_d = N_D
        
        self.gnn = HeteroConv({
            ('miRNA', 'targets', 'Target Gene'): GraphConv((-1, -1), hidden_dim),
            ('Target Gene', 'rev_targets', 'miRNA'): GraphConv((-1, -1), hidden_dim),
            ('Target Gene', 'associates', 'DOID'): GraphConv((-1, -1), hidden_dim),
            ('DOID', 'rev_associates', 'Target Gene'): GraphConv((-1, -1), hidden_dim),

        }, aggr='mean')
        
        self.bridge = nn.Linear(hidden_dim, 1)
        
        self.fpn_m2g = FPNLayer(N_M, N_G, m2g_mask)
        self.fpn_g2d = FPNLayer(N_G, N_D, g2d_mask, custom_thresholds=g2d_thresholds)
        

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            'miRNA': F.relu(self.lin_m(x_dict['miRNA'])),
            'Target Gene': F.relu(self.lin_g(x_dict['Target Gene'])),
            'DOID': F.relu(self.lin_d(x_dict['DOID']))
        }
        
        h_dict = self.gnn(h_dict, edge_index_dict)
        embed_m = h_dict['miRNA']
        embed_d = h_dict['DOID']
        embed_g = h_dict['Target Gene']

        mirna_tokens = torch.sigmoid(self.bridge(h_dict['miRNA'])).squeeze(-1)
        
        scale_g = math.sqrt(self.hidden_dim)
        sim_m2g = torch.sigmoid(torch.matmul(embed_m, embed_g.t()) / scale_g)
        sim_g2d = torch.sigmoid(torch.matmul(embed_g, embed_d.t()) / scale_g)

        gene_tokens, opt_adj_m2g, explore_adj_m2g = self.fpn_m2g(mirna_tokens, self.fpn_beta, gnn_sim_matrix=sim_m2g)
        _, opt_adj_g2d, explore_adj_g2d = self.fpn_g2d(gene_tokens, self.fpn_beta, gnn_sim_matrix=sim_g2d)

        
        main_pred_matrix = torch.matmul(opt_adj_m2g, opt_adj_g2d) / math.sqrt(self.N_g)
        path_scores = main_pred_matrix.unsqueeze(-1)
        logits = self.logit_projector(path_scores) # (Num_miRNA, Num_Disease, 1)
        final_pred_matrix = logits.squeeze(-1) # (Num_miRNA, Num_Disease)
        
        return final_pred_matrix, opt_adj_m2g, opt_adj_g2d, explore_adj_m2g, explore_adj_g2d
