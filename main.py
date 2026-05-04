import warnings

warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
import torch.nn.functional as F
from load_adult import AdultDataSet
from torch.utils.data import DataLoader
from b_vae import BetaVAE_B
import numpy as np
from torch.optim.lr_scheduler import ExponentialLR

device = torch.device(0 if torch.cuda.is_available() else "cpu")
epochs = 1000

z_dim = 20
input_dim = 14
intermediate_dim = 14 * 32
beta = 10
C_max = 1

batch_size = 128
lr = 1e-3

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


if __name__ == "__main__":
    net = BetaVAE_B(z_dim, input_dim, intermediate_dim)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, 0.99)
    dataset_train = AdultDataSet("adult.data")
    dataset_test = AdultDataSet("adult.test")

    train_data_loader = DataLoader(dataset_train, num_workers=0, batch_size=batch_size)
    test_data_loader = DataLoader(dataset_test, num_workers=0, batch_size=batch_size)
    C_max = torch.FloatTensor([C_max]).to(device)
    global_iter = 0
    max_iter = epochs * len(train_data_loader)
    C_stop_iter = max_iter // 15
    epoch = 0
    out = False
    while not out:
        net.train()
        train_loss = []
        print(epoch)
        for x, y in train_data_loader:
            global_iter += 1
            x = x.to(torch.float32)
            x = x.to(device)
            x_recon, mu, logvar = net(x)
            recon_loss = reconstruction_loss(x, x_recon, 'gaussian')
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            C = torch.clamp(C_max / C_stop_iter * global_iter, 0, C_max.data[0])
            beta_vae_loss = recon_loss + beta * (total_kld - C).abs()
            train_loss.append(beta_vae_loss.item())
            optimizer.zero_grad()
            beta_vae_loss.backward()
            optimizer.step()
        print(np.array(train_loss).mean())
        scheduler.step()
        net.eval()
        eval_loss_recon = []
        eval_loss_kld = []
        for x, y in test_data_loader:
            x = x.to(torch.float32)
            x = x.to(device)
            x_recon, mu, logvar = net(x)
            recon_loss = reconstruction_loss(x, x_recon, 'gaussian')
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            beta_vae_loss = recon_loss + beta * (total_kld - C).abs()
            eval_loss_recon.append(recon_loss.item())
            eval_loss_kld.append((total_kld - C).abs().item())

        print(np.array(eval_loss_recon).mean())
        print(np.array(eval_loss_kld).mean())
        print("------------------------")
        epoch += 1
        if global_iter >= max_iter:
            out = True
            break

