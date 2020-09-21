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
    def __init__(self, data, cov_func='exponential', ls=None):
        super(GP, self).__init__()
        self.device = get_cuda_device()
        self.data = data
        self.n_data = data['X'].shape[0]
        self.n_dim = data['X'].shape[1]
        if cov_func == 'exponential':
            self.cov = CovFunction(self.n_dim, ls=ls).to(self.device)
        elif 'spectral' in cov_func:
            self.n_eps = float(cov_func.split('_')[1])
            self.cov = SpectralCov(self.n_dim, n_eps=self.n_eps, ls=ls).to(self.device)
        elif 'chi' in cov_func:
            self.n_eps = float(cov_func.split('_')[1])
            self.cov = ChiSpectralCov(self.n_dim, n_eps=self.n_eps, ls=ls).to(self.device)
        self.mean = MeanFunction().to(self.device)
        self.lik = LikFunction().to(self.device)

    def NLL_old(self):
        y = self.data['Y'].float() - self.mean(self.data['X']).float()
        Kxx = self.cov(self.data['X'])
        Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(self.n_data).to(self.device)).float()
        nll = -0.5 * torch.logdet(Q_inv) + 0.5 * torch.mm(y.t(), torch.mm(Q_inv, y))[0, 0]
        return nll

    def NLL(self):
        y = self.data['Y'].float() - self.mean(self.data['X']).float()
        Kxx = self.cov(self.data['X'])
        torch.diagonal(Kxx).fill_(torch.exp(2.0 * self.cov.sn) + torch.exp(2.0 * self.lik.noise))
        L = torch.cholesky(Kxx, upper=False)
        Linv = torch.mm(torch.inverse(L), y)
        nll = 0.5 * torch.sum(torch.log(L.diag())) + 0.5 * torch.mm(Linv.t(), Linv)
        return nll

    def NLL_batch(self, X, Y):
        y = Y.float() - self.mean(X).float()
        Kxx = self.cov(X)
        torch.diagonal(Kxx).fill_(torch.exp(2.0 * self.cov.sn) + torch.exp(2.0 * self.lik.noise))
        L = torch.cholesky(Kxx, upper=False)
        Linv = torch.mm(torch.inverse(L), y)
        nll = 0.5 * torch.sum(torch.log(L.diag())) + 0.5 * torch.mm(Linv.t(), Linv)
        return nll

    def forward(self, x, batch_train=None, batch_label=None, grad=False):
        if not grad:
            torch.no_grad()
        if batch_label is None:
            y = self.data['Y'].float() - self.mean(self.data['X']).float()
            ktx = self.cov(x, self.data['X']).float()
            Kxx = self.cov(self.data['X'])
        else:
            y = batch_label.float() - self.mean(batch_train).float()
            ktx = self.cov(x, batch_train).float()
            Kxx = self.cov(batch_train)

        ktt = self.cov(x)
        Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(self.n_data).to(self.device)).float()
        ktx_Q_inv = torch.mm(ktx, Q_inv)
        y_pred = self.mean(x) + torch.mm(ktx_Q_inv, y)
        y_var = ktt - torch.mm(ktx_Q_inv, ktx.t())
        return y_pred, y_var


class GPCluster(nn.Module):
    def __init__(self, data, cov_func='exponential', ls=None):
        super(GPCluster, self).__init__()
        self.device = get_cuda_device()
        self.data = data
        self.n_dim = self.data[0]['X'].shape[1]
        if cov_func == 'exponential':
            self.cov = CovFunction(self.n_dim, ls=ls).to(self.device)
        elif 'spectral' in cov_func:
            n_eps = float(cov_func.split('_')[1])
            self.cov = SpectralCov(self.n_dim, n_eps=n_eps, ls=ls).to(self.device)
        elif 'chi' in cov_func:
            n_eps = float(cov_func.split('_')[1])
            self.cov = ChiSpectralCov(self.n_dim, n_eps=n_eps, ls=ls).to(self.device)
        self.mean = MeanFunction().to(self.device)
        self.lik = LikFunction().to(self.device)
        self.y_mean, self.n_data = self.get_mean()

    def get_mean(self):
        y_mean = 0.0
        n_data = []
        for i in range(self.data['k']):
            n_data.append(self.data[i]['Y'].shape[0])
            y_mean += torch.sum(self.data[i]['Y'])
        return y_mean / sum(n_data), n_data

    def NLL(self):
        for i in range(self.data['k']):
            y = self.data[i]['Y'].float() - self.mean(self.data[i]['X']).float()
            Kxx = self.cov(self.data[i]['X'])
            Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(self.n_data[i]).to(self.device)).float()
            nll = -0.5 * torch.logdet(Q_inv) + 0.5 * torch.mm(y.t(), torch.mm(Q_inv, y))[0, 0]
        return nll

    def NLL_batch(self, X, Y):
        y = Y.float() - self.mean(X).float()
        Kxx = self.cov(X)
        torch.diagonal(Kxx).fill_(torch.exp(2.0 * self.cov.sn) + torch.exp(2.0 * self.lik.noise))
        L = torch.cholesky(Kxx, upper=False)
        Linv = torch.mm(torch.inverse(L), y)
        nll = 0.5 * torch.sum(torch.log(L.diag())) + 0.5 * torch.mm(Linv.t(), Linv)
        return nll

    def forward(self, x, grad=False):
        if not grad:
            torch.no_grad()

        ktt = self.cov(x)
        y_pred = torch.zeros(x.shape[0], 1).to(self.device)
        y_var = torch.zeros(x.shape[0], x.shape[0]).to(self.device)
        for i in range(self.data['k']):
            y = self.data[i]['Y'].float() - self.mean(self.data[i]['X'])
            ktx = self.cov(x, self.data[i]['X']).float()
            Kxx = self.cov(self.data[i]['X'])
            Q_inv = torch.inverse(Kxx + torch.exp(2.0 * self.lik.noise) * torch.eye(self.n_data[i]).to(self.device)).float()
            ktx_Q_inv = torch.mm(ktx, Q_inv)
            y_pred += self.mean(x) + torch.mm(ktx_Q_inv, y)
            y_var += ktt - torch.mm(ktx_Q_inv, ktx.t())

        return y_pred, y_var
