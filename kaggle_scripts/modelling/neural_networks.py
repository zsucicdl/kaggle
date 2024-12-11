import torch
from torch import nn


def seq_block(in_features, out_features, ks=3, drop_rate=0.2, dilation=1):
    padding = ((ks - 1) * dilation) // 2
    return nn.Sequential(
        nn.Conv1d(in_features, out_features, ks, padding=padding, dilation=dilation),
        nn.BatchNorm1d(out_features),
        nn.LeakyReLU(0.2),
        nn.Dropout(drop_rate)
    )


class SimpleConvNet(nn.Module):
    def __init__(self, in_c=2, out_c=2, hidden=128, emb_dim=8, ks=3, n_blks=3, dropout=0.2):
        super(SimpleConvNet, self).__init__()

        self.hr_emb = nn.Embedding(24, emb_dim)
        self.fc1_hr = nn.Linear(emb_dim, emb_dim)
        self.fc2_hr = nn.Linear(emb_dim, emb_dim)

        self.fc_in = nn.Linear(in_c + emb_dim, hidden)
        self.blks = nn.Sequential(
            *[seq_block(hidden, hidden, ks, drop_rate=dropout, dilation=2 ** i) for i in range(n_blks)]
        )
        self.fc_out = nn.Linear(hidden, out_c)

        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x, h):
        e = self.hr_emb(h)
        e = self.fc1_hr(e)
        e = self.lrelu(e)
        e = self.fc2_hr(e)

        x = torch.cat([x, e.squeeze(2)], dim=-1)
        x = self.fc_in(x)

        x = x.permute(0, 2, 1)

        for b in self.blks:
            x = b(x)

        x = x.permute(0, 2, 1)

        x = self.fc_out(x)
        return x
