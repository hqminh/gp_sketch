from utility import *
from kernel import *


class Sampler(nn.Module):
    def __init__(self, noise, n_dim):
        super(Sampler, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise = ts(noise).to(self.device)
        self.n_dim = n_dim
        self.cov = CovFunction(n_dim).to(self.device)

    def recursive_sampling(self, X, tolerance):
        m = X.shape[0]
        print(m)
        if m <= 192 * np.log(1.0 / tolerance):
            return torch.eye(m)

        selected = sample_rows(0.5 * np.ones(m), m)
        Xs = X[selected, :].view(-1, self.n_dim)
        Ss = torch.eye(m)[:, selected].view(m, -1).to(self.device)
        St = self.recursive_sampling(Xs, tolerance / 3.0)
        print(St.shape[1])
        Sh = torch.mm(Ss, St)
        selected = torch.argmax(Sh, dim=1)
        Xh = X[selected, :].view(-1, self.n_dim)
        Khh = self.cov(Xh)
        phi = torch.inverse(Khh + torch.exp(2.0 * self.noise) * torch.eye(selected.shape[0]))
        l = np.zeros(m)
        for i in range(m):
            Kii = self.cov(X[i, :].view(1, -1))[0, 0]
            Kih = self.cov(X[i, :].view(1, -1), Xh)
            l[i] = Kii - torch.mm(Kih, torch.mm(phi, Kih.t()))[0, 0]
        scale = 16.0 * np.log(np.sum(l) / tolerance)
        p = np.minimum(np.ones(m), l * scale)
        selected = sample_rows(p, m)
        S = (1.0 / torch.sqrt(torch.tensor(p[selected]).view(1, -1))) * torch.eye(m)[:, selected].view(m, -1)
        return S.float()


class SGP(nn.Module):
    def __init__(self, data):
        super(SGP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.n_data = data['X'].shape[0]
        self.n_dim = data['X'].shape[1]
        self.cov = CovFunction(self.n_dim).to(self.device)
        self.mean = MeanFunction().to(self.device)
        self.lik = LikFunction().to(self.device)
        self.sampler = Sampler(self.lik.noise, self.n_dim).to(self.device)

    def NLL(self):
        Kss, Kxs = self.nystrom()
        y = self.data['Y'] - self.mean(self.data['X'])
        Kxx = torch.mm(Kxs, torch.mm(Kss, Kxs.t()))
        Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(self.n_data))
        nll = -0.5 * torch.logdet(Q_inv) + 0.5 * torch.mm(y.t(), torch.mm(Q_inv, y))[0, 0]
        return nll

    def nystrom(self, x=None):
        S = self.sampler.recursive_sampling(self.data['X'], tolerance=0.1)
        selected = torch.argmax(S, dim=0)
        Xs = self.data['X'][selected, :]
        Kss = self.cov(Xs)
        Kxs = self.cov(self.data['X'], Xs)
        if x is None:
            return Kss, Kxs

        Kts = self.cov(x, Xs)
        return Kts, Kss, Kxs

    def forward(self, x):
        y = self.data['Y'] - self.mean(self.data['X'])
        Kts, Kss, Kxs = self.nystrom(x)
        Ksx = Kxs.t()
        t1 = torch.inverse(Kss + torch.exp(-2.0 * self.lik.noise) * torch.mm(Ksx, Kxs))
        y_pred = self.mean(x) + torch.exp(-2.0 * self.lik.noise) * torch.mm(Kts, torch.mm(t1, torch.mm(Ksx, y)))
        return y_pred


class GP(nn.Module):
    def __init__(self, data):
        super(GP, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data = data
        self.n_data = data['X'].shape[0]
        self.n_dim = data['X'].shape[1]
        self.cov = CovFunction(self.n_dim).to(self.device)
        self.mean = MeanFunction().to(self.device)
        self.lik = LikFunction().to(self.device)
        self.sampler = Sampler(self.lik.noise, self.n_dim).to(self.device)

    def NLL(self):
        y = self.data['Y'] - self.mean(self.data['X'])
        Kxx = self.cov(self.data['X'])
        Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(self.n_data))
        nll = -0.5 * torch.logdet(Q_inv) + 0.5 * torch.mm(y.t(), torch.mm(Q_inv, y))[0, 0]
        return nll

    def forward(self, x):
        y = self.data['Y'] - self.mean(self.data['X'])
        ktx = self.cov(x, self.data['X'])
        ktt = self.cov(x)
        Kxx = self.cov(self.data['X'])
        Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(self.n_data))
        ktx_Q_inv = torch.mm(ktx, Q_inv)
        y_pred = self.mean(x) + torch.mm(ktx_Q_inv, y)
        y_var = ktt - torch.mm(ktx_Q_inv, ktx.t())

        return y_pred, y_var

