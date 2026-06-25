import torch
import torch.nn as nn
from .layers import PositionEncoding, DataEmbedding, STAttentionBlock


class STGNet(nn.Module):
    def __init__(self,
                 origin_dim,
                 output_dim,
                 time_max,
                 data_dim,
                 heads_num,
                 hide_node_num,
                 T,
                 batch_len,
                 prediction_horizon,
                 exo_dim=0,
                 dropout=0.1,
                 g_flags=[0, 1, 1],
                 s_flags=[0, 0, 1]
                 ):
        super(STGNet, self).__init__()
        self.time_max = time_max
        self.T = T
        self.batch_len = batch_len
        self.prediction_horizon = prediction_horizon
        self.num_components = len(g_flags)
        self.time_embedding = PositionEncoding(time_max, T, 0)
        self.fusion_layer = nn.Linear(output_dim * 3, output_dim)
        self.exo_dim = exo_dim

        self.data_emb_bases = nn.ModuleList(
            [DataEmbedding(1, self.time_max, self.T, start=0) for _ in range(self.num_components)])

        if self.exo_dim > 0:
            self.exo_embedding = DataEmbedding(self.exo_dim, self.time_max, self.T, start=0)

        self.GModels = nn.ModuleList()
        self.CrossModels = nn.ModuleList()

        for i in range(self.num_components):
            self.GModels.append(
                STAttentionBlock(heads_num, data_dim, self.time_max, batch_len, g_flag=g_flags[i], s_flag=s_flags[i]))
            if self.exo_dim > 0:
                self.CrossModels.append(
                    STAttentionBlock(heads_num, data_dim, self.time_max, batch_len, g_flag=g_flags[i],
                                     s_flag=s_flags[i]))
            else:
                self.CrossModels.append(nn.Identity())

        self.linears = nn.ModuleList([nn.Linear(1, output_dim) for _ in range(self.num_components)])
        self.final_projections = nn.ModuleList(
            [nn.Linear(batch_len, self.prediction_horizon) for _ in range(self.num_components)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, endo_data, exo_data, start_batch, end_batch, epoch, step):
        return_graph = None
        batch_front = 1
        start_index = (start_batch + epoch * (batch_front - 1)) * step
        end_index_endo = start_index + self.batch_len
        time_emb_endo = self.time_embedding()[start_index:end_index_endo, :]

        if self.exo_dim > 0:
            exo_emb = self.exo_embedding.encoding(exo_data)
        else:
            exo_emb = None

        component_outputs = []
        for i in range(self.num_components):
            component_data = endo_data[:, :, i:i + 1]
            endo_emb = self.data_emb_bases[i].encoding(component_data)
            self_attn_out, graph = self.GModels[i](Q=endo_emb, K=endo_emb, V=endo_emb, time_embedding=time_emb_endo)
            hidden_endo = endo_emb + self_attn_out

            if self.exo_dim > 0:
                cross_attn_out, _ = self.CrossModels[i](Q=hidden_endo, K=exo_emb, V=exo_emb,
                                                        time_embedding=time_emb_endo)
                hidden_final = hidden_endo + cross_attn_out
            else:
                hidden_final = hidden_endo

            output_decoded = self.data_emb_bases[i].decoding(hidden_final)
            if i == 2:
                return_graph = graph

            out_linear = self.linears[i](output_decoded)
            out_trans = out_linear.transpose(1, 2)
            out_proj = self.dropout(self.final_projections[i](out_trans))
            out_final = out_proj.transpose(1, 2).contiguous()
            component_outputs.append(out_final)

        cat_output = torch.cat(component_outputs, dim=-1)
        total_output = self.fusion_layer(cat_output)

        if return_graph is None:
            return_graph = graph
        return total_output, return_graph