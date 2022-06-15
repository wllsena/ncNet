__author__ = "Yuyu Luo"
'''
Define the Coder of the model
'''

import torch
import torch.nn as nn

from model.SubLayers import (MultiHeadAttentionLayer,
                             PositionwiseFeedforwardLayer)


class Coder(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            hid_dim,  # == d_model
            n_layers,
            n_heads,
            pf_dim,
            dropout,
            device,
            TOK_TYPES,
            max_length=128):
        super().__init__()

        self.device = device

        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.tok_types_embedding = nn.Embedding(len(TOK_TYPES.vocab.itos), hid_dim)

        self.layers = nn.ModuleList(
            [CoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, src, src_mask, tok_types, batch_matrix):

        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)

        src = self.dropout((self.tok_embedding(src) * self.scale) +
                           self.tok_types_embedding(tok_types) + self.pos_embedding(pos))

        # src = [batch size, src len, hid dim]

        for layer in self.layers:
            src, attention = layer(src, src_mask, batch_matrix)

        output = self.fc_out(src)

        # src = [batch size, src len, hid dim]
        return output, attention


class CoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask, batch_matrix):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]

        # self attention
        _src, _attention = self.self_attention(src, src, src, src_mask, batch_matrix)

        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]

        # position-wise feedforward
        _src = self.positionwise_feedforward(src)

        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))

        # src = [batch size, src len, hid dim]
        # print('CoderLayer->forward:', src)
        return src, _attention
