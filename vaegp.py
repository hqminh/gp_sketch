from utility import *
from gp import *
from mvae import *
from experiments import *


class VAEGP(nn.Module):
    def __init__(self, train, vae_cluster=8, embed_dim=2, gp_method='vaegp_32'):
        super(VAEGP, self).__init__()
        self.device = get_cuda_device()
        self.dataset = TensorDataset(train['X'], train['Y'])
        self.data = DataLoader(self.dataset, batch_size=200, shuffle=True)
        self.n_eps = int(gp_method.split('_')[1])
        self.vae = MixtureVAE(self.data, train['X'].shape[1], embed_dim, vae_cluster, n_sample=self.n_eps)
        self.original = train
        self.train = {'X': self.vae(train['X']), 'Y': train['Y']}
        self.gp = Experiment.create_gp_object(self.train, gp_method)
        self.model = nn.ModuleList([self.vae.qz_x, self.vae.px_z, self.vae.qz_x.weights, self.gp.cov, self.gp.mean]
                                       + self.vae.qz_x.components)
        self.gp_model = nn.ModuleList([self.gp.cov, self.gp.mean])
        self.history = []

    def train_gp(self, seed=0, burn_iter=100, n_iter=300, lmbda=1.0, pred_interval=5, test=None, verbose=True):
        set_seed(seed)
        print('SEED=', seed)
        optimizer = opt.Adam(self.model.parameters())
        for i in range(burn_iter):
            batch_nll = 0.0
            batch_elbo = 0.0
            batch_loss = 0.0
            batch_validation = 0.0
            for (X, Y) in self.data:
                X.to(self.device)
                Y.to(self.device)
                Z = self.vae(X)
                self.gp.data['X'] = Z
                self.gp.data['Y'] = Y
                self.gp.n_data = Z.shape[0]
                self.gp.n_dim = Z.shape[1]
                self.model.train()
                optimizer.zero_grad()
                delbo = self.vae.dsELBO(X, alpha=1.0, beta=1.0, gamma=1.0, verbose=False)
                nll = self.gp.NLL_batch(Z, Y)
                validation = rmse(self.gp(Z)[0], Y)
                loss = - delbo + nll + lmbda * validation
                batch_validation += validation * X.shape[0]
                batch_nll += nll * X.shape[0]
                batch_elbo += delbo * X.shape[0]
                batch_loss += loss * X.shape[0]
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            if i % pred_interval == 0:
                record = {'nll': batch_nll.item() / self.train['X'].shape[0],
                          'elbo': batch_elbo.item() / self.train['X'].shape[0],
                          'loss': batch_loss.item() / self.train['X'].shape[0],
                          'val': batch_validation.item() / self.train['X'].shape[0],
                          'iter': i}
                if test is not None:
                    self.gp.data['X'] = self.vae(self.original['X'])
                    self.gp.data['Y'] = self.original['Y']
                    self.gp.n_data = self.gp.data['X'].shape[0]
                    self.gp.n_dim = self.gp.data['X'].shape[1]
                    err = rmse(self.gp(self.vae(test['X']))[0], test['Y'])
                    record['rmse'] = err.item()

                if verbose:
                    print(record)

                self.history.append(record)

        for i in range(n_iter):
            batch_nll = 0.0
            for (X, Y) in self.data:
                X.to(self.device)
                Y.to(self.device)
                Z = self.vae(X)
                self.gp.data['X'] = Z
                self.gp.data['Y'] = Y
                self.gp.n_data = Z.shape[0]
                self.gp.n_dim = Z.shape[1]
                self.gp_model.train()
                optimizer.zero_grad()
                nll = self.gp.NLL_batch(Z, Y)
                batch_nll += nll * X.shape[0]
                nll.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            if i % pred_interval == 0:
                record = {'nll': batch_nll.item() / self.train['X'].shape[0],
                          'iter': i}

                if test is not None:
                    self.gp.data['X'] = self.vae(self.original['X'])
                    self.gp.data['Y'] = self.original['Y']
                    self.gp.n_data = self.gp.data['X'].shape[0]
                    self.gp.n_dim = self.gp.data['X'].shape[1]
                    err = rmse(self.gp(self.vae(test['X']))[0], test['Y'])
                    record['rmse'] = err.item()

                if verbose:
                    print(record)

                self.history.append(record)

        return self.history

    def forward(self, X):
        return self.gp(X)


class GP_wrapper(nn.Module):
    def __init__(self, train, gp_method='ssgp_32'):
        super(GP_wrapper, self).__init__()
        self.device = get_cuda_device()
        self.train = train
        self.gp_method = gp_method
        self.gp = Experiment.create_gp_object(self.train, self.gp_method)
        self.dataset = TensorDataset(self.train['X'], self.train['Y'])
        self.data = DataLoader(self.dataset, batch_size=200, shuffle=True)
        self.model = nn.ModuleList([self.gp.cov, self.gp.mean])
        self.history = []

    def train_gp(self, seed=0, n_iter=300, pred_interval=5, test=None, verbose=True):
        set_seed(seed)
        print('SEED=', seed)
        optimizer = opt.Adam(self.model.parameters())
        for i in range(n_iter):
            batch_nll = 0.0
            for (X, Y) in self.data:
                self.model.train()
                optimizer.zero_grad()
                nll = self.gp.NLL_batch(X, Y)
                batch_nll += nll * X.shape[0]
                nll.backward()
                optimizer.step()
                torch.cuda.empty_cache()
            if i % pred_interval == 0:
                record = {'nll': batch_nll.item() / self.train['X'].shape[0],
                          'iter': i}

                if test is not None:
                    err = rmse(self.gp(test['X'])[0], test['Y'])
                    record['rmse'] = err.item()

                if verbose:
                    print(record)

                self.history.append(record)
        return self.history


def deploy(prefix, method, dataset, plot=True):
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    train, test = Experiment.load_data(dataset)
    seed = [1001, 1002, 1003, 1004, 1005]
    res = dict()
    for s in seed:
        if 'vaegp' in method:
            vaegp = VAEGP(train, gp_method=method)
            try:
                res[s] = vaegp.train_gp(seed=s, n_iter=300, lmbda=1000.0, pred_interval=10, test=test, verbose=True)
                torch.save(vaegp, prefix + str(s) + '.pth')
            except RuntimeError:
                res[s] = vaegp.history
                torch.save(vaegp, prefix + str(s) + '.pth')
        else:
            gp = GP_wrapper(train, gp_method=method)
            try:
                res[s] = gp.train_gp(seed=s, n_iter=300, pred_interval=10, test=test, verbose=True)
                torch.save(gp, prefix + str(s) + '_' + method + '.pth')
            except RuntimeError:
                res[s] = gp.history
                torch.save(gp, prefix + str(s) + '_' + method + '.pth')

    if plot:
        mean_rmse = np.zeros(len(res[seed[0]]))
        var_rmse = np.zeros(len(res[seed[0]]))
        for s in seed:
            mean_rmse += 1.0 / len(seed) * np.array([h['rmse'] for h in res[s]])
        for s in seed:
            var_rmse += 1.0 / len(seed) * ((np.array([h['rmse'] for h in res[s]]) - mean_rmse) ** 2)

        iter = np.array([h['iter'] for h in res[seed[0]]])

        plt.figure()
        plt.errorbar(iter, mean_rmse, var_rmse ** 0.5, linestyle='--',
                     marker='^', linewidth=2, markersize=12, label='')
        plt.legend(loc="upper right")
        plt.xlabel("No. of Training Iterations")
        plt.ylabel("RMSE")
        plt.savefig(prefix + 'result.png')
        plt.clf()


if __name__ == '__main__':
    #deploy(prefix='3JunExp1', method='ssgp_32', dataset='abalone', plot=True)
    torch.cuda.set_device(int(sys.argv[4]))
    deploy(sys.argv[1], sys.argv[2], sys.argv[3], plot=False)

