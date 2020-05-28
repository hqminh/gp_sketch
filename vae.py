from utility import *


class InferenceNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, norm=False):
        super(InferenceNet, self).__init__()
        self.device = get_cuda_device()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc21 = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.fc22 = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.bn = nn.BatchNorm1d(hidden_dim).to(self.device)
        self.norm = norm

    def forward(self, x, n_sample=100):
        x = self.fc1(x)
        x_reshape = x.view(-1, self.fc1.out_features)
        if (x_reshape.shape[0] > 1) and (self.norm is True):
            shape = x.shape
            x = x.view(-1, self.fc1.out_features)
            x = F.relu(self.bn(x)).view(shape)
        else:
            x = F.relu(x)
        mu, log_var = self.fc21(x), self.fc22(x)
        std = torch.exp(0.5 * log_var)
        if n_sample is None:
            return mu, log_var
        eps = torch.empty(n_sample, mu.shape[0], self.output_dim).normal_(0, 1).to(self.device)
        return mu + eps * std, mu, log_var


class MixGauss(nn.Module):
    def __init__(self, input_dim, output_dim, k=10):
        super(MixGauss, self).__init__()
        self.device = get_cuda_device()
        self.k = k
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.p = [InferenceNet(input_dim, output_dim) for i in range(k)]
        self.l = nn.Parameter((1.0 / k) * torch.ones(1, k), requires_grad=True)

    def cat_sampling(self, n_sample):
        # u shape nxk
        u = -torch.log(-torch.log(torch.FloatTensor(n_sample, self.k).uniform_(0, 1).to(self.device)))
        # pr shape 1xk
        pr = torch.log(self.l / (1.0 - self.l))
        return torch.argmax(u + pr, dim=1)

    def forward(self, x=None, n_sample=None):
        if x is None:
            x = torch.ones(1, self.input_dim)

        if n_sample is not None:
            cs = self.cat_sampling(n_sample)
            ns = [0] * self.k
            for i in range(n_sample):
                ns[cs[i]] += 1
            s = [self.p[cs[i]](x, n_sample=ns[i])[0] for i in range(self.k)]
            return torch.cat(s, dim=0)

        else:


class VAE(nn.Module):
    def __init__(self, xdim, zdim, k=10):
        super(VAE, self).__init__()
        self.xdim = xdim
        self.zdim = zdim
        self.k = k
        self.qzx = MixGauss(xdim, zdim, k)
        self.pz = MixGauss(xdim, zdim, k)
        self.pxz = InferenceNet(zdim, xdim)

    def forward(self):
        ret
