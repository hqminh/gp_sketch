from utility import *

import warnings
warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)


class GaussianNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=10, norm=True, bias=None):
        super(GaussianNet, self).__init__()
        self.device = get_cuda_device()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.fc21 = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.fc22 = nn.Linear(hidden_dim, output_dim).to(self.device)
        self.bn = nn.BatchNorm1d(hidden_dim).to(self.device)
        self.norm = norm
        if bias is None:
            self.bias = torch.zeros(1, output_dim).to(self.device)
        else:
            self.bias = bias

    def forward(self, x):
        x = self.fc1(x)
        x_reshape = x.view(-1, self.fc1.out_features)
        if (x_reshape.shape[0] > 1) and (self.norm is True):
            shape = x.shape
            x = x.view(-1, self.fc1.out_features)
            x = F.relu(self.bn(x)).view(shape)
        else:
            x = F.relu(x)
        mu, log_var = self.fc21(x), self.fc22(x)  # mu: n_data by output_dim; log_var: n_data by output_dim
        return (mu + self.bias), log_var

    def sample(self, x, n_sample=100):
        x = self.fc1(x)
        x_reshape = x.view(-1, self.fc1.out_features)
        if (x_reshape.shape[0] > 1) and (self.norm is True):
            shape = x.shape
            x = x.view(-1, self.fc1.out_features)
            x = F.relu(self.bn(x)).view(shape)
        else:
            x = F.relu(x)
        mu, log_var = self.fc21(x), self.fc22(x)  # mu: n_data by output_dim; log_var: n_data by output_dim
        std = torch.exp(0.5 * log_var)
        eps = torch.empty(n_sample, mu.shape[0], self.output_dim).normal_(0, 1).to(self.device)
        return (mu + self.bias) + eps * std


class MixtureGaussianNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_component, prior_bias=False):
        super(MixtureGaussianNet, self).__init__()
        self.device = get_cuda_device()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_component = n_component
        self.components = []
        for i in range(self.n_component):
            bias = None
            if prior_bias:
                bias = (i - n_component / 2.0) * torch.ones(1, output_dim).to(self.device)
            self.components.append(GaussianNet(input_dim, output_dim, bias=bias))
        self.weights = nn.Linear(input_dim, n_component).to(self.device)
        #self.weights = nn.Parameter(torch.ones(1, self.n_component) * np.log((1.0 / self.n_component)), requires_grad=True).to(self.device)


    def forward(self, Z, X=None):  # return log p(z|x) for each pair (z, x) or log p(z) for each z if X is None
        if X is None:
            X = torch.zeros((Z.shape[0], self.input_dim)).to(self.device)
        res = torch.zeros((Z.shape[0], self.n_component)).to(self.device)
        for i in range(self.n_component):
            mean, logvar = self.components[i](X)  # mean, logvar: Z.shape[0] by Z.shape[1]
            res[:, i] -= 0.5 * torch.sum(((mean - Z) ** 2) * torch.exp(-2.0 * logvar), dim=1)
            res[:, i] -= 0.5 * torch.sum(logvar, dim=1)

        w = F.softmax(F.relu(self.weights(torch.ones(1, self.input_dim).to(self.device))))
        #res += self.weights - torch.logsumexp(self.weights, dim=1)
        res += torch.log(w)
        return torch.logsumexp(res, dim=1, keepdim=True)  # output has shape Z.shape[0] by 1

    def sample(self, X=None, n_sample=10):  # return n_sample from p(.|x) or p(.) if X is None
        if X is None:
            X = torch.zeros((1, self.input_dim)).to(self.device)
        U = torch.empty(n_sample, X.shape[0], self.n_component).uniform_(0, 1).to(self.device)
        #w = torch.exp(self.weights - torch.logsumexp(self.weights, dim=1))
        w = F.softmax(F.relu(self.weights(torch.ones(1, self.input_dim).to(self.device))))
        chosen = torch.argmax(torch.log(w[None, :, :] / (1 - w[None, :, :]))
                              - torch.log(-torch.log(U)), dim=2)
        res = torch.zeros(n_sample, X.shape[0], self.output_dim).to(self.device)
        for r in range(n_sample):
            for c in range(X.shape[0]):
                res[r, c, :] = self.components[chosen[r, c]].sample(X[c, :].reshape(1, self.input_dim),
                                                                    n_sample=1).reshape(-1)
        return res


class MixtureVAE(nn.Module):
    def __init__(self, dataset, input_dim, output_dim, n_component, n_sample=100):
        super(MixtureVAE, self).__init__()
        self.device = get_cuda_device()
        self.dataset = dataset
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_component = n_component
        self.n_sample = n_sample
        self.qz_x = MixtureGaussianNet(self.input_dim, self.output_dim, self.n_component, prior_bias=False)
        self.px_z = GaussianNet(self.output_dim, self.input_dim, bias=None)
        self.pz = MixtureGaussianNet(self.input_dim, self.output_dim, self.n_component, prior_bias=True)
        self.model = nn.ModuleList([self.qz_x, self.px_z, self.pz, self.qz_x.weights, self.pz.weights]
                                   + self.qz_x.components + self.pz.components)

    def ELBO(self, X):
        Z = self.qz_x.sample(X, n_sample=self.n_sample)  # n_sample by X.shape[0] by output_dim
        elbo = ts(0.0).to(self.device)
        for i in range(self.n_sample):
            log_qzx = torch.sum(self.qz_x(Z[i], X)) / X.shape[0]
            log_pz = torch.sum(self.pz(Z[i]))
            mu_pxz, var_pxz = self.px_z(Z[i])
            log_pxz = -0.5 * (torch.sum((X - mu_pxz) ** 2 * torch.exp(-2.0 * var_pxz) + var_pxz)) / X.shape[0]
            elbo += log_pz + log_pxz - log_qzx

        zero = torch.zeros((1, self.input_dim)).to(self.device)
        c = torch.cat([component(zero)[0] for component in self.qz_x.components], dim=0) ** 2
        lv = torch.zeros(len(self.qz_x.components)).to(self.device)
        for i in range(lv.shape[0]):
            lv[i] = torch.sum(self.qz_x.components[i](zero)[1])

        w = F.softmax(F.relu(self.qz_x.weights(torch.ones(1, self.qz_x.input_dim).to(self.device))))

        a = torch.sum(c ** 2, 1).reshape(-1, 1)
        b = torch.sum(c ** 2, 1) - 2 * torch.mm(c, c.t())
        dist = a.float() + b.float()
        torch.diagonal(dist).fill_(10000000000)
        return elbo / self.n_sample, torch.min(dist), torch.max(lv), w

    def dELBO(self, X, alpha=8.0, beta=1.2):
        Z = self.qz_x.sample(X, n_sample=self.n_sample)  # n_sample by X.shape[0] by output_dim
        elbo = ts(0.0).to(self.device)
        for i in range(self.n_sample):
            qzx = self.qz_x(Z[i], X)
            log_qzx = torch.sum(qzx) / X.shape[0]
            log_qz = torch.logsumexp(qzx, dim=0)[0] - np.log(X.shape[0])
            log_pz = torch.sum(self.pz(Z[i]))
            mu_pxz, var_pxz = self.px_z(Z[i])
            log_pxz = (-0.5 * (torch.sum((X - mu_pxz) ** 2 * torch.exp(-2.0 * var_pxz) + var_pxz))) / X.shape[0]
            #print(log_qzx.shape, log_qz.shape, log_pxz.shape, log_pz.shape)
            elbo += log_pxz - beta * (log_qzx - log_pz) - alpha * (log_qz - log_pz)

        return elbo / self.n_sample

    def forward(self, data, encode=True):
        if encode is True:
            res = self.qz_x.sample(data)
            return torch.mean(res, dim=0)  # data.shape[0] by output_dim
        else:
            res, _ = self.px_z(data)  # data.shape[0] by input_dim
            return res

    def train_vae(self, n_iter=100, l1=1.0, l2=1.0, l3=1.0):
        optimizer = opt.Adam(self.model.parameters())
        for i in range(n_iter):
            ave_elbo = 0.0
            for (X, _) in self.dataset:
                self.model.train()
                optimizer.zero_grad()
                elbo, min_dist, max_rad, balance = self.ELBO(X.to(self.device))
                loss = -elbo - l1 * (1.0 - i / n_iter) * min_dist + l2 * max_rad + l3 * torch.sum(balance ** 2)
                print(i, min_dist.item(), max_rad.item())
                print(balance)
                ave_elbo += elbo
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            print(ave_elbo.item())

    def train_dvae(self, n_iter=100):
        optimizer = opt.Adam(self.model.parameters())
        for i in range(n_iter):
            ave_loss = 0.0
            for (X, _) in self.dataset:
                self.model.train()
                optimizer.zero_grad()
                delbo = self.dELBO(X.to(self.device))
                loss = -delbo
                ave_loss += loss / X.shape[0]
                print(loss.item())
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            print('Ave Loss=', ave_loss)


if __name__ == '__main__':
    set_seed(1001)
    train, n_train = abalone_data(is_train=True)
    test, n_test = abalone_data(is_train=False)
    dataset = TensorDataset(train['X'], train['Y'])
    data = DataLoader(dataset, batch_size=100, shuffle=True)
    input_dim = train['X'].shape[1]
    output_dim = 2
    n_component = 10
    vae = MixtureVAE(data, input_dim, output_dim, n_component, n_sample=10)
    vae.train_vae(n_iter=100)



