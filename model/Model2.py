__author__ = "Yuyu Luo"

import numpy as np
import torch
import torch.nn as nn

from model.AttentionForcing import create_visibility_matrix

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GenSeq(nn.Module):
    def __init__(self, coder, src_pad_idx, device):
        super().__init__()

        self.coder = coder
        self.src_pad_idx = src_pad_idx
        self.device = device

    def make_visibility_matrix(self, src, SRC):
        '''
        building the visibility matrix here
        '''
        # src = [batch size, src len]
        batch_matrix = []
        for each_src in src:
            v_matrix = create_visibility_matrix(SRC, each_src)
            n_heads_matrix = [v_matrix] * 8  # TODO: 8 is the number of heads ...
            batch_matrix.append(np.array(n_heads_matrix))
        batch_matrix = np.array(batch_matrix)

        # batch_matrix = [batch size, n_heads, src_len, src_len]
        return torch.tensor(batch_matrix).to(device)

    def make_src_mask(self, src):
        # src = [batch size, src len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def forward(self, src, tok_types, SRC):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)

        batch_visibility_matrix = self.make_visibility_matrix(src, SRC)

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        output, attention = self.coder(src, src_mask, tok_types, batch_visibility_matrix)

        # enc_src = [batch size, src len, hid dim]

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention
