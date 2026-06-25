import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_ff, d_model):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionEncoding(nn.Module):
    def __init__(self, max_seq_length, T, start=0):
        super(PositionEncoding, self).__init__()
        self.max_seq_length = max_seq_length
        if not isinstance(T, torch.Tensor):
            self.T = torch.tensor(T)
        else:
            self.T = T
        self.T_max = torch.max(self.T).float()
        self.encoding_dim = self.T.shape[0] * 4
        self.pe = torch.zeros(max_seq_length, self.encoding_dim, requires_grad=False)
        self.position = torch.arange(start, max_seq_length, dtype=torch.float, requires_grad=False)
        varphi = 0.5
        for i in range(self.T.shape[0]):
            self.pe[:, 4 * i] = torch.cos(2 * math.pi * self.position / self.T[i] + varphi)
            self.pe[:, 1 + 4 * i] = torch.sin(2 * math.pi * self.position / self.T[i] + varphi)
            self.pe[:, 2 + 4 * i] = 2 * self.T[i] / self.max_seq_length * self.position + 1
            self.pe[:, 3 + 4 * i] = self.T[i] / self.max_seq_length * self.position ** 2 + 1

    def forward(self, x=None):
        return self.pe


class DataEmbedding(nn.Module):
    def __init__(self, x_dim, end, T, start=0):
        super(DataEmbedding, self).__init__()
        self.time_embedding = PositionEncoding(end, T, start)
        self.Wt = nn.Parameter(torch.rand(
            size=(x_dim, self.time_embedding Pe().shape[0] if hasattr(self, 'Pe') else self.time_embedding().shape[0])))
        embedding_dim = self.time_embedding().shape[1]
        self.decoding_layer = nn.Linear(embedding_dim, x_dim)

    def get_Wt(self):
        return torch.matmul(self.Wt, self.time_embedding())

    def encoding(self, x):
        W_t = self.get_Wt()
        return torch.matmul(x.float(), W_t)

    def decoding(self, y):
        W_t = self.get_Wt()
        return torch.matmul(y, torch.linalg.pinv(W_t))


class STAttentionBlock(nn.Module):
    def __init__(self, num_heads, d_model, time_max, batch_len, g_flag=1, s_flag=1):
        super(STAttentionBlock, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.seq_length = batch_len

        self.alpha = nn.Parameter(torch.ones(self.num_heads))
        self.beta_vector = nn.Parameter(torch.linspace(0.01, 1, self.num_heads))

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(self.d_k, self.d_k)
        self.W_c = nn.Linear(self.d_k, self.d_k)
        self.W_cor = nn.Linear(self.d_k, self.d_k)
        self.dropout = nn.Dropout(p=0.1)
        self.relu = nn.ReLU()

        self.time_data_dim = d_model
        self.tfc1 = nn.Linear(self.time_data_dim, 1)
        self.tfc2 = nn.Linear(self.seq_length, self.d_model)
        self.tfc3 = nn.Linear(self.time_data_dim, self.d_k)
        self.tfc4 = nn.Linear(self.seq_length, self.d_k)
        self.mu_origin_change = nn.Linear(self.seq_length, 1)
        self.sigma_origin_change = nn.Linear(self.seq_length, self.d_k)

        self.mixing_logits = nn.Parameter(torch.ones(2))
        self.graph_weight = nn.Parameter(torch.ones(1))
        self.g_flag = g_flag
        self.s_flag = s_flag

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, _ = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def split_heads_G(self, K):
        batch_size, seq_length, _ = K.size()
        K_heads = K.view(batch_size, seq_length, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        alpha = F.softmax(self.alpha, dim=0)
        graph_matrices, graph_enhanced_K = [], []

        for i in range(self.num_heads):
            K_head_i = K_heads[:, i, :, :]
            A = self.adjacent_matrix(K_head_i, beta=self.beta_vector[i])
            I = torch.eye(self.d_k, device=K.device).unsqueeze(0).expand(A.shape[0], -1, -1)
            A_hat = I + A
            D_hat = torch.stack([torch.sum(x, dim=1) * torch.eye(x.shape[0], device=x.device) for x in A_hat], dim=0)
            D_hat_05_inverse = torch.inverse(torch.sqrt(D_hat) + 1e-8)

            graph_matrix_i = self.W_g(alpha[i] * torch.matmul(torch.matmul(D_hat_05_inverse, A_hat), D_hat_05_inverse))
            graph_matrices.append(graph_matrix_i)
            graph_enhanced_K_i = torch.matmul(K_head_i, graph_matrix_i)
            graph_enhanced_K.append(graph_enhanced_K_i)

        graph_enhanced_K_stacked = torch.stack(graph_enhanced_K, dim=1)
        graph_matrices_stacked = torch.stack(graph_matrices, dim=1)
        return graph_enhanced_K_stacked, graph_matrices_stacked

    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.num_heads * self.d_k)

    def distance(self, K):
        distance_origin = torch.stack([torch.norm(x.t().unsqueeze(0) - x.t().unsqueeze(1), p=2, dim=2) for x in K],
                                      dim=0)
        distance_sort = torch.argsort(torch.argsort(distance_origin, dim=-1), dim=-1)
        return F.softmax(distance_sort.float(), dim=-1)

    def adjacent_matrix(self, K, beta):
        z = self.distance(K)
        A3 = torch.where(z > beta, torch.zeros_like(z), z)
        A4 = torch.where(z <= beta, torch.ones_like(z), A3)
        A0 = self.dropout(A4)
        A = torch.where(A0 + torch.eye(self.d_k, device=K.device) == 2, torch.eye(self.d_k, device=K.device), A0)
        return A

    def get_cor(self, K_heads):
        b, h, s, d_k = K_heads.shape
        K_heads_T = K_heads.permute(1, 0, 2, 3)
        cor_per_head = []
        for i in range(h):
            head_data = K_heads_T[i]
            noise = torch.randn_like(head_data) * 1e-4
            head_data_safe = head_data + noise
            raw_cor = torch.stack([torch.corrcoef(item.t()) for item in head_data_safe], dim=0)
            raw_cor = torch.nan_to_num(raw_cor, nan=0.0)
            cor_batch = torch.square(raw_cor) + 1e-6
            cor_per_head.append(cor_batch)
        WC_heads = torch.stack(cor_per_head, dim=1)
        return self.W_cor(WC_heads)

    def time_mlp(self, x):
        a = self.relu(self.tfc1(x).squeeze(dim=-1))
        mu = self.relu(self.tfc2(a))
        c = self.relu(self.tfc3(x))
        sigma = torch.exp(torch.tanh(self.relu(self.tfc4(c.transpose(-1, -2)))))
        return mu, (torch.add(sigma, sigma.transpose(-1, -2))) / 2

    def attn_sigma_mu(self, x, WG, WC, time_mu, time_sigma):
        mu_origin = self.mu_origin_change(x.transpose(-1, -2)).squeeze(dim=-1)
        time_mu_unsqueezed = time_mu.unsqueeze(1)
        time_mu_heads = self.split_heads(time_mu_unsqueezed)
        time_mu_heads_squeezed = time_mu_heads.squeeze(2)
        mu = torch.add(mu_origin, time_mu_heads_squeezed)

        sigma_agent = self.sigma_origin_change(x.transpose(-1, -2))
        sigma_origin = F.normalize(sigma_agent)
        agent1 = torch.matmul(WC, time_sigma.unsqueeze(1))
        L_raw = torch.exp(torch.tanh(torch.matmul(sigma_origin, torch.matmul(WG, agent1))))

        diag_elements = torch.diagonal(L_raw, dim1=-2, dim2=-1)
        L_diag = torch.diag_embed(F.softplus(diag_elements) + 1e-6)
        scale_tril = torch.tril(L_raw, diagonal=-1) + L_diag
        return mu, scale_tril

    def normal_sample(self, mu, scale_tril, normal):
        num_samples_for_stability = 1
        m = MultivariateNormal(loc=mu, scale_tril=scale_tril, validate_args=True)
        samples = m.rsample(sample_shape=torch.Size([num_samples_for_stability, self.seq_length]))
        samples_permuted = samples.permute(2, 3, 1, 0, 4)
        averaged_samples = torch.mean(samples_permuted, dim=3)
        return averaged_samples

    def forward(self, Q, K, V, time_embedding, mask=None):
        batch_size = Q.shape[0]
        time_embedding_batched = time_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        time_mlp_mu, time_mlp_sigma = self.time_mlp(time_embedding_batched)
        Q_h = self.split_heads(self.W_q(Q))
        K_h = self.split_heads(self.W_k(K))
        V_h = self.split_heads(self.W_v(V))
        K_g, WG_h = self.split_heads_G(K)

        if self.g_flag == 1:
            K_final = torch.add(K_h, K_g * self.graph_weight[0])
        else:
            K_final = K_h

        attn_output1 = self.scaled_dot_product_attention(Q_h, K_final, V_h, mask)

        if self.s_flag == 1:
            time_embedding_batched = time_embedding.unsqueeze(0).expand(batch_size, -1, -1)
            time_mlp_mu, time_mlp_sigma = self.time_mlp(time_embedding_batched)
            WC_h = self.get_cor(K_h)
            attn_mu, attn_cholesky_L = self.attn_sigma_mu(attn_output1, WG_h, WC_h, time_mlp_mu, time_mlp_sigma)
            attn_output2 = self.normal_sample(attn_mu, attn_cholesky_L, 0)
            attn_output = torch.add(attn_output1, self.mixing_logits[1] * attn_output2)
        else:
            attn_output = attn_output1

        output = self.W_o(self.combine_heads(attn_output))
        return output, WG_h.detach()