import numpy as np
import torch;
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import matplotlib.pyplot as plt;

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


class RandomColorMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.mnist = datasets.MNIST(root, download=download, transform=transform)

    def __getitem__(self, index):
        img, label = self.mnist[index]
        color = torch.randint(0, 256, (3,)).to(torch.uint8)
        img = img.repeat(3, 1, 1)

        img[0, :, :][img[0, :, :] > 0] = color[0] / 255
        img[1, :, :][img[1, :, :] > 0] = color[1] / 255
        img[2, :, :][img[2, :, :] > 0] = color[2] / 255

        return img, label

    def __len__(self):
        return len(self.mnist)


transform = transforms.Compose([transforms.ToTensor()])
mnist_data = RandomColorMNIST(root='path/to/data', download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=128, shuffle=False)


def sample_gumbel(shape, eps=1e-20):
    # Sample from Gumbel(0, 1)
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-20):
    dims = len(logits.size())
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    y = logits + gumbel_noise
    return F.softmax(y / tau, dim=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    bs, N, K = logits.size()
    y_soft = gumbel_softmax_sample(logits.view(bs * N, K), tau=tau, eps=eps)
    if hard:
        k = torch.argmax(y_soft, dim=-1)
        y_hard = F.one_hot(k, num_classes=K)

        # 1. makes the output value exactly one-hot
        # 2.makes the gradient equal to y_soft gradient
        y = y_hard - y_soft.detach() + y_soft
    else:
        y = y_soft

    return y.reshape(bs, N * K)


class ContinousEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(ContinousEncoder, self).__init__()
        self.linear1 = nn.Linear(2352, 512)
        self.to_mean_logvar = nn.Linear(512, 2 * latent_dims)

    def reparametrization_trick(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu, log_var = torch.split(self.to_mean_logvar(x), 2, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reparametrization_trick(mu, log_var)


class ContinousDecoder(nn.Module):
    def __init__(self, latent_dims):
        super(ContinousDecoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 2352)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 3, 28, 28))


class DiscreteEncoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim):
        super(DiscreteEncoder, self).__init__()
        self.fc1 = nn.Linear(2352, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, latent_dim * categorical_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 2352)
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.relu(self.fc3(h2))


class DiscreteDecoder(nn.Module):
    def __init__(self, latent_dim, categorical_dim):
        super(DiscreteDecoder, self).__init__()
        self.fc4 = nn.Linear(latent_dim * categorical_dim, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 2352)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.N = latent_dim
        self.K = categorical_dim

    def forward(self, x, temp, hard):
        q_y = x.view(x.size(0), self.N, self.K)
        z = gumbel_softmax(q_y, temp, hard)
        h4 = self.relu(self.fc4(z))
        h5 = self.relu(self.fc5(h4))
        return self.sigmoid(self.fc6(h5)), F.softmax(q_y, dim=-1).reshape(x.size(0) * self.N, self.K)


class CombinedAutoencoder(nn.Module):
    def __init__(self, latent_dims, N, K):
        super().__init__()
        self.contencoder = ContinousEncoder(latent_dims)
        self.contdecoder = ContinousDecoder(latent_dims)
        self.discencoder = DiscreteEncoder(N, K)
        self.discdecoder = DiscreteDecoder(N, K)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(latent_dims, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 2352)
        self.N = N
        self.K = K
        self.cont_dim = latent_dims
        self.to_mean_logvar = nn.Linear(62, 2 * latent_dims)
        self.kl = 0

    def reparametrization_trick(self, mu, log_var):
        # Using reparameterization trick to sample from a gaussian
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var / 2) * eps

    def encoder(self, x):
        z1 = self.contencoder(x)
        z2 = self.discencoder(x)
        z = torch.cat((z1,z2), dim=1)
        z = self.relu(torch.flatten(z, start_dim=1))
        mu, log_var = torch.split(self.to_mean_logvar(z), 2, dim=-1)
        self.kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return self.reparametrization_trick(mu, log_var)

    def decoder(self, z):
        h1 = self.relu(self.fc1(z))
        h2 = self.relu(self.fc2(h1))
        output = self.sigmoid(self.fc3(h2))
        return output.reshape((-1, 3, 28, 28))

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output.reshape((-1, 3, 28, 28))


z_dim = 2
N = 3
K = 20  # one-of-K vector

temp = 1.0
hard = False
temp_min = 0.5
ANNEAL_RATE = 0.00003

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CombinedAutoencoder(z_dim, N, K).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
model.load_state_dict(torch.load("hw2_318736501_both_model_q1.pkl"))

def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
plot_latent(model, data_loader)

def plot_reconstructed(autoencoder, r0=(-2, 2), r1=(-2, 2), n=12):
    w = 28
    img = []
    for i, z2 in enumerate(np.linspace(r1[1], r1[0], n)):
        for j, z1 in enumerate(np.linspace(*r0, n)):
            z = torch.Tensor([[z1, z2]]).to(device)
            x_hat = autoencoder.decoder(z)
            img.append(x_hat)

    img = torch.cat(img)
    img = torchvision.utils.make_grid(img, nrow=12).permute(1, 2, 0).detach().numpy()
    plt.imshow(img, extent=[*r0, *r1])


plot_reconstructed(model)


