"""Beta-VAE-B encoder/decoder (Burgess et al., arXiv:1804.03599)."""

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


class BetaVAE_B(nn.Module):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, input_dim=40, intermediate_dim=40 * 32):
        super(BetaVAE_B, self).__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Linear(intermediate_dim, intermediate_dim//2),          # B,  32, 16, 16
            nn.ReLU(True),
            # nn.Linear(intermediate_dim//2, intermediate_dim//4),          # B,  32,  8,  8
            # nn.ReLU(True),
            # nn.Linear(intermediate_dim//4, intermediate_dim//8),          # B,  32,  4,  4
            # nn.ReLU(True),
            nn.Linear(intermediate_dim//2, 256),              # B, 256
            nn.ReLU(True),
            # nn.Linear(256, 256),                 # B, 256
            # nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B,  32, 32, 32
            nn.ReLU(True),
            # nn.Linear(256, 256),  # B,  32, 16, 16
            # nn.ReLU(True),
            nn.Linear(256, intermediate_dim // 2),  # B,  32,  8,  8
            nn.ReLU(True),
            # nn.Linear(intermediate_dim // 8, intermediate_dim // 4),  # B,  32,  4,  4
            # nn.ReLU(True),
            # nn.Linear(intermediate_dim // 4, intermediate_dim // 2),  # B, 256
            # nn.ReLU(True),
            nn.Linear(intermediate_dim // 2, intermediate_dim),  # B, 256
            nn.ReLU(True),
            nn.Linear(intermediate_dim, input_dim),  # B, z_dim*2
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass