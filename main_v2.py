import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

try:
    from lazypredict.Supervised import LazyClassifier
except ImportError:
    LazyClassifier = None

from b_vae import BetaVAE_B, reparametrize
from load_adult import AdultDataSet

if torch.cuda.is_available():
    device = torch.device(0)
elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

epochs = 100
encoding_mode = "onehot"  # reserved; see AdultDataSet
z_dim = 10
intermediate_dim = 256
beta = 10
C_max = 5
batch_size = 512
gamma = 100
alpha = 1
lr = 1e-3
seed = 64
np.random.seed(seed)
torch.manual_seed(seed)


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


def rbfsigma(data):
    if len(data.shape) == 1:
        data = data.unsqueeze(1)

    data_norm = torch.sum(data**2, dim=1).reshape(-1, 1)
    dist_matrix = data_norm + data_norm.t() - 2.0 * torch.mm(data, data.t())
    dist_values = dist_matrix.triu(diagonal=1).flatten()
    sigma = torch.sqrt(0.5 * torch.median(dist_values[dist_values > 0]))
    return sigma


def GaussianKernel(x, s, sigma):
    if len(x.shape) == 1:
        x = x.unsqueeze(1)
    if len(s.shape) == 1:
        s = s.unsqueeze(1)

    n_x = x.shape[0]
    n_s = s.shape[0]

    x_norm = torch.pow(torch.norm(x, dim=1).reshape([1, n_x]), 2)
    s_norm = torch.pow(torch.norm(s, dim=1).reshape([1, n_s]), 2)

    ones_x = torch.ones([1, n_x], dtype=torch.float32, device=x.device)
    ones_s = torch.ones([1, n_s], dtype=torch.float32, device=x.device)

    kernel = torch.exp(
        (-torch.mm(torch.t(x_norm), ones_s) - torch.mm(torch.t(ones_x), s_norm) + 2 * torch.mm(x, torch.t(s)))
        / (2 * sigma**2)
    )

    return kernel


def new_hsic_loss(zz, ss, n):
    zz = zz.to(dtype=torch.float64)
    ss = ss.to(dtype=torch.float64)
    sigma_z = rbfsigma(zz)
    sigma_s = rbfsigma(ss)

    H = torch.eye(n, device=ss.device, dtype=torch.float64) - torch.ones(n, n, device=ss.device, dtype=torch.float64) / n

    K_s = GaussianKernel(ss, ss, sigma_s)
    K_sm = torch.mm(H, torch.mm(K_s, H))
    K_z = GaussianKernel(zz, zz, sigma_z)
    K_zm = torch.mm(H, torch.mm(K_z, H))

    denom = torch.sqrt(torch.trace(torch.mm(K_sm, K_sm)) * torch.trace(torch.mm(K_zm, K_zm)))
    hsic = (torch.trace(torch.mm(K_zm, K_sm)) / denom).to(torch.float32)
    return hsic


def subset_to_arrays(subset):
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=0)
    xb, yb, sb = next(iter(loader))
    return xb.numpy(), yb.numpy(), sb.numpy()


def main():
    dataset = AdultDataSet("adult", encoding_mode, with_sensitive_attribute=True)
    dataset_train, dataset_test = torch.utils.data.random_split(dataset, [0.8, 0.2])
    train_data_loader = DataLoader(dataset_train, num_workers=0, batch_size=batch_size)
    test_data_loader = DataLoader(dataset_test, num_workers=0, batch_size=batch_size)
    input_dim = dataset[0][0].shape[0]

    net = BetaVAE_B(z_dim, input_dim, intermediate_dim)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, 0.99)
    c_max_tensor = torch.FloatTensor([C_max]).to(device)
    global_iter = 0
    max_iter = epochs * len(train_data_loader)
    c_stop_iter = max_iter // 5

    train_loss_to_plot = []
    train_loss_recon_to_plot = []
    train_loss_kld_to_plot = []
    train_loss_hsic_to_plot = []
    eval_loss_to_plot = []
    best_recon_loss = float("inf")

    for epoch in range(epochs):
        net.train()
        train_loss = []
        train_loss_recon = []
        train_loss_kld = []
        train_loss_hsic = []

        print(epoch)
        for x, y, s in train_data_loader:
            global_iter += 1
            x = x.to(torch.float32).to(device)
            s = s.to(torch.float32).to(device)
            x_recon, mu, logvar = net(x)
            recon_loss = F.binary_cross_entropy_with_logits(
                x_recon.to(torch.float32), y.unsqueeze(1).to(torch.float32).to(device)
            )
            total_kld, _, _ = kl_divergence(mu, logvar)
            c = torch.clamp(c_max_tensor / c_stop_iter * global_iter, 0, c_max_tensor.data[0])
            z = reparametrize(mu, logvar)
            hsic_loss = new_hsic_loss(z, s, s.shape[0])
            beta_vae_loss = alpha * recon_loss + beta * (total_kld - c).abs() + gamma * hsic_loss

            train_loss_recon.append(recon_loss.item())
            train_loss_kld.append((total_kld - c).abs().item())
            train_loss.append(beta_vae_loss.item())
            train_loss_hsic.append(hsic_loss.item())

            optimizer.zero_grad()
            beta_vae_loss.backward()
            optimizer.step()

        print(np.array(train_loss).mean())
        print(np.array(train_loss_recon).mean())
        print(np.array(train_loss_kld).mean())
        print(np.array(train_loss_hsic).mean())

        train_loss_to_plot.append(np.array(train_loss).mean())
        train_loss_recon_to_plot.append(np.array(train_loss_recon).mean())
        train_loss_kld_to_plot.append(np.array(train_loss_kld).mean())
        train_loss_hsic_to_plot.append(np.array(train_loss_hsic).mean())

        scheduler.step()
        net.eval()
        eval_loss_recon = []
        eval_loss_kld = []
        eval_loss_hsic = []
        eval_loss = []

        print("------")
        with torch.no_grad():
            for x, y, s in test_data_loader:
                x = x.to(torch.float32).to(device)
                s = s.to(torch.float32).to(device)
                x_recon, mu, logvar = net(x)
                recon_loss = F.binary_cross_entropy_with_logits(
                    x_recon.to(torch.float32), y.unsqueeze(1).to(torch.float32).to(device)
                )
                total_kld, _, _ = kl_divergence(mu, logvar)
                c = torch.clamp(c_max_tensor / c_stop_iter * global_iter, 0, c_max_tensor.data[0])
                z = reparametrize(mu, logvar)
                hsic_loss = new_hsic_loss(z, s, s.shape[0])
                beta_vae_loss = alpha * recon_loss + beta * (total_kld - c).abs() + gamma * hsic_loss

                eval_loss_hsic.append(hsic_loss.item())
                eval_loss.append(beta_vae_loss.item())
                eval_loss_recon.append(recon_loss.item())
                eval_loss_kld.append((total_kld - c).abs().item())

        mean_eval_recon = float(np.array(eval_loss_recon).mean())
        print(np.array(eval_loss).mean())
        print(mean_eval_recon)
        print(np.array(eval_loss_kld).mean())
        print(np.array(eval_loss_hsic).mean())

        eval_loss_to_plot.append(np.array(eval_loss).mean())

        print("------------------------")
        if mean_eval_recon <= best_recon_loss:
            best_recon_loss = mean_eval_recon
            torch.save(net.state_dict(), "best_model.pth")

        if global_iter >= max_iter:
            break

    plt.figure(figsize=(12, 6))
    plt.plot(train_loss_to_plot, label="Total train loss")
    plt.plot(train_loss_kld_to_plot, label="KL divergence")
    plt.plot(train_loss_hsic_to_plot, label="HSIC loss")
    plt.plot(train_loss_recon_to_plot, label="Reconstruction loss")
    plt.legend()
    plt.savefig("training_losses.png", dpi=150)
    plt.close()

    net = BetaVAE_B(z_dim, input_dim, intermediate_dim)
    net.to(device)
    net.load_state_dict(torch.load("best_model.pth", map_location=device))
    X_train, y_train, _s_train = subset_to_arrays(dataset_train)
    X_test, y_test, s_test = subset_to_arrays(dataset_test)

    with torch.no_grad():
        net.eval()
        X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
        distributions = net._encode(X_train_t)
        mu = distributions[:, :z_dim]
        logvar = distributions[:, z_dim:]
        X_train_encoded = reparametrize(mu, logvar)

        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        distributions = net._encode(X_test_t)
        mu = distributions[:, :z_dim]
        logvar = distributions[:, z_dim:]
        X_test_encoded = reparametrize(mu, logvar)

    X_test_encoded = X_test_encoded.cpu().numpy()
    X_train_encoded = X_train_encoded.cpu().numpy()

    if LazyClassifier is not None:
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, _predictions = clf.fit(X_train_encoded, X_test_encoded, y_train, y_test)
        print(models)

    lgbm = LGBMClassifier(objective="binary", random_state=5)
    lgbm.fit(X_train_encoded, y_train)
    y_pred = lgbm.predict(X_test_encoded)

    print(accuracy_score(y_test, y_pred))
    print(f1_score(y_test, y_pred))

    df = pd.DataFrame({"sex": s_test, "pred": y_pred})
    female = df[(df["sex"] == 1) & (df["pred"] == 1)].shape[0] / df[df["sex"] == 1].shape[0]
    male = df[(df["sex"] == 0) & (df["pred"] == 1)].shape[0] / df[df["sex"] == 0].shape[0]
    dp = male - female
    print(f"female:{female}")
    print(f"male:{male}")
    print(f"DP:{dp}")


if __name__ == "__main__":
    main()
