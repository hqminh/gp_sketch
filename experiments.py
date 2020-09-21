from utility import *
from mvae import *
from gp import *


class Experiment:
    def __init__(self, dataset='abalone', method='full', embedding=True, vae_model=None):
        self.dataset = dataset
        self.method = method
        self.embedding = embedding
        self.vae = vae_model

    @staticmethod
    def train_gp(gp_obj, test, n_iter=500, record_interval=10):
        idx = []
        nll = []
        error = []
        model = nn.ModuleList([gp_obj.cov, gp_obj.mean])
        optimizer = opt.Adam(model.parameters())
        for i in range(n_iter + 1):
            model.train()
            optimizer.zero_grad()
            loss = gp_obj.NLL()

            if i < n_iter + 1:
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

            if record_interval == -1:
                continue
            elif (i + 1) % record_interval == 0:
                y_pred, y_var = gp_obj(test['X'])
                error.append(rmse(y_pred, test['Y']))
                nll.append(loss.item())
                idx.append(i + 1)
                print('Training Iteration', i + 1, 'rmse:', error[-1], 'nll:', nll[-1])

        if record_interval == -1:
            y_pred, y_var = gp_obj(test['X'])
            return rmse(y_pred, test['Y'])
        if record_interval == -2:
            return gp_obj.cov(gp_obj.data['X']), gp_obj.cov.weights
        else:
            return error, nll, idx

    @staticmethod
    def load_data(dataset):
        if dataset == 'abalone':
            train, n_train = abalone_data(is_train=True)
            test, n_test = abalone_data(is_train=False)
        elif dataset == 'diabetes':
            train, n_train = diabetes_data(is_train=True)
            test, n_test = diabetes_data(is_train=False)
        elif dataset == 'gas':
            full_train, full_test = gas_sensor_data(is_preload=True)
            p1 = torch.randperm(full_test['X'].size(0))
            idx_test = p1[:2000]
            test = {'X': full_test['X'][idx_test],
                    'Y': full_test['Y'][idx_test]
                    }
            p2 = torch.randperm(full_train['X'].size(0))
            idx_train = p2[:20000]
            train = {'X': full_train['X'][idx_train],
                     'Y': full_train['Y'][idx_train]
                     }
        else:
            raise NotImplementedError
        return train, test

    @staticmethod
    def cluster_data(data, k=10):
        X = dt(data['X'])
        kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
        cluster = {'k': k,
                   'label': kmeans.labels_,
                   'centroids': kmeans.cluster_centers_,
                   }
        for i in range(k):
            cluster[i] = {'idx': []}

        for i in range(X.shape[0]):
            cid = cluster['label'][i]
            cluster[cid]['idx'].append(i)

        for i in range(k):
            cluster[i]['X'] = data['X'][cluster[i]['idx']]
            cluster[i]['Y'] = data['Y'][cluster[i]['idx']]

        return cluster

    @staticmethod
    def create_gp_object(train, method):
        if 'ssgpc' in method:
            n_eps = method.split('_')[1]
            cluster = Experiment.cluster_data(train)
            gpc = GPCluster(cluster, 'spectral_' + n_eps)
            return gpc
        elif ('ssgp' in method) or ('vaegp' in method):
            n_eps = method.split('_')[1]
            gpc = GP(train, 'spectral_' + n_eps)
            return gpc
        elif 'full' in method:
            gpf = GP(train, 'exponential')
            return gpf

    def deploy(self, seed=1001, savefile=None):
        set_seed(seed)
        train, test = self.load_data(self.dataset)

        if self.embedding:
            if self.vae is None:
                self.vae = MixtureVAE(train, train['X'].shape[1], 2, 10, n_sample=10)
                n_iter = 20
                epoch_iter = 5
                for i in range(n_iter):
                    self.vae.train_dsvae(n_iter=epoch_iter, alpha=1.0, beta=1.0, gamma=1.0, verbose=True)

            train['X'] = ts(dt(self.vae(train['X']))).to(get_cuda_device())
            test['X'] = ts(dt(self.vae(test['X']))).to(get_cuda_device())

        gp_obj = Experiment.create_gp_object(train, self.method)
        result = dict()
        result['err'], result['nll'], result['idx'] = Experiment.train_gp(
            gp_obj,
            test=test,
            n_iter=500,
            record_interval=25
        )

        if savefile is not None:
            pickle.dump(result, open(savefile, 'wb'))

        return result