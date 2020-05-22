from utility import *

class CovFunction(nn.Module):
    def __init__(self, n_dim, c=0.0, ls=None):
        super(CovFunction, self).__init__()
        self.n_dim = n_dim
        if ls is None:
            self.weights = nn.Parameter(torch.tensor(np.random.randn(self.n_dim, 1)), requires_grad=True)
        else:
            self.weights = nn.Parameter(ls, requires_grad=True)
        self.sn = nn.Parameter(torch.tensor(c), requires_grad=False)
        self.device = get_cuda_device()

    def forward(self, U, V = None):  # U is of size _ by n_dim, V is of size _ by n_dim
        if V is None:
            V = U
        assert (len(U.size()) == 2) and (len(V.size()) == 2), "Input matrices must be 2D"
        assert U.size(1) == V.size(1), "Input matrices must agree on the second dimension"
        scales = torch.exp(-1.0 * self.weights).float().view(1, -1) / np.sqrt(self.n_dim)
        a = torch.sum((U * scales) ** 2, 1).reshape(-1, 1)
        b = torch.sum((V * scales) ** 2, 1) - 2 * torch.mm((U * scales), (V * scales).t())
        res = torch.exp(2.0 * self.sn) * torch.exp(-0.5 * (a.float() + b.float()))
        return res

    def forward_shit(self, U, V = None):  # U is of size _ by n_dim, V is of size _ by n_dim
        if V is None:
            V = U

        assert (len(U.size()) == 2) and (len(V.size()) == 2), "Input matrices must be 2D"
        assert U.size(1) == V.size(1), "Input matrices must agree on the second dimension"
        diff = V[None, :, None, :] - U[:, None, :, None]  # m by n by d by d
        scales = torch.exp(-1.0 * self.weights).float() * torch.eye(self.n_dim).float().to(self.device) # diagonal by by d -- containing length-scale on diag
        cov = (diff.float() * scales[None, None, :, :]) ** 2  # n by m by d by d @ 1 by 1 by d by d
        cov = torch.exp(-0.5 * torch.sum(cov, dim=[2, 3])) # n by m by d by d ==> n by m after sum
        res = torch.exp(2.0 * self.sn) * cov
        return res

class ChiSpectralCov(CovFunction):
    def __init__(self, n_dim, eps=None, n_eps=0, c=0.0, ls=None):
        super(ChiSpectralCov, self).__init__(n_dim, c, ls)
        if n_eps == 0:
            self.n_eps = eps.shape[1]
            self.eps = eps
        else:
            self.n_eps = n_eps
            self.log_n_eps = int(n_dim * np.log(n_eps) / 0.1)
            self.sign = (2.0 * torch.randint(0, 2, (n_dim, self.log_n_eps)) - 1.0).to(self.device)
            self.eps = self.sign * ts(np.random.chisquare(self.n_eps, (n_dim, self.log_n_eps))).float().to(self.device)

    def phi(self, X):
        diag = torch.diag(torch.exp(0.5 * self.weights.view(-1)))
        phi = torch.mm(X, diag)
        phi = torch.mm(phi.float(), self.eps)
        cphi = torch.cos(phi)
        sphi = torch.sin(phi)
        res = torch.cat([cphi, sphi], dim=1)
        return res

    def forward(self, U, V=None):
        pu = self.phi(U)
        if V is None:
            return (1.0 / self.log_n_eps) * torch.exp(2.0 * self.sn) * torch.mm(pu, pu.t())
        else:
            pv = self.phi(V)
            return (1.0 / self.log_n_eps) * torch.exp(2.0 * self.sn) * torch.mm(pu, pv.t())


class SpectralCov(CovFunction):
    def __init__(self, n_dim, eps=None, n_eps=0, c=0.0, ls=None):
        super(SpectralCov, self).__init__(n_dim, c, ls)
        if n_eps == 0:
            self.n_eps = eps.shape[1]
            self.eps = eps
        else:
            self.n_eps = n_eps
            self.eps = torch.rand((n_dim, int(n_eps))).to(self.device)

    def phi(self, X):
        diag = torch.exp(0.5 * self.weights.view(1, -1))
        phi = X * diag.float()
        phi = torch.mm(phi.float(), self.eps)
        res = torch.cat([torch.cos(phi), torch.sin(phi)], dim=1)
        return res

    def forward(self, U, V=None):
        pu = self.phi(U)
        if V is None:
            return (1.0 / self.n_eps) * torch.exp(2.0 * self.sn) * torch.mm(pu, pu.t())
        else:
            pv = self.phi(V)
            return (1.0 / self.n_eps) * torch.exp(2.0 * self.sn) * torch.mm(pu, pv.t())


class MeanFunction(nn.Module):  # simply a constant mean function
    def __init__(self, c=0.0, opt=True):
        super(MeanFunction, self).__init__()
        self.mean = nn.Parameter(torch.tensor(c), requires_grad=opt)

    def forward(self, U):  # simply the constant mean function; input form: (_, x_dim)
        assert len(U.size()) == 2, "Input matrix must be 2D"
        n = U.size(0)
        return torch.ones(n, 1).to(torch.device('cuda')) * self.mean


class LikFunction(nn.Module):  # return log likelihood of a Gaussian N(z | x, noise^2 * I)
    def __init__(self, c=0.0, opt=True):
        super(LikFunction, self).__init__()
        self.noise = nn.Parameter(torch.tensor(c), requires_grad=opt)

    def forward(self, o, x):  # both are n_sample, n_dim -- assert that they have the same dim
        assert (len(o.size()) == 2) and (len(o.size()) == 2), \
            "Input matrices must be 2D"
        assert (o.size(0) == x.size(0)) and (o.size(1) == x.size(1)), \
            "Input matrices are supposed to be of the same size"
        diff = o - x
        n, d = o.size(0), o.size(1)
        output = -0.5 * n * d * torch.log(torch.tensor(2 * np.pi)) - 0.5 * n * self.noise \
                 -0.5 * torch.exp(-2.0 * self.noise) * torch.trace(torch.mm(diff.t(), diff))
        return output