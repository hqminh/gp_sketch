from utility import *
from mvae import *
from gp import *


class Experiment():
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
                ypred, yvar = gp_obj(test['X'])
                error.append(rmse(ypred, test['Y']))
                nll.append(loss.item())
                idx.append(i + 1)
                print('Training Iteration', i + 1, 'rmse:', error[-1], 'nll:', nll[-1])

        if record_interval == -1:
            ypred, yvar = gp_obj(test['X'])
            return rmse(ypred, test['Y'])
        if record_interval == -2:
            return gp_obj.cov(gp_obj.data['X']), gp_obj.cov.weights
        else:
            return error, nll, idx

    def load_data(self):
        train = None
        test = None
        if self.dataset == 'abalone':
            train, n_train = abalone_data(is_train=True)
            test, n_test = abalone_data(is_train=False)
        elif self.dataset == 'diabetes':
            train, n_train = diabetes_data(is_train=True)
            test, n_test = diabetes_data(is_train=False)

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
        if 'ssgp' in method:
            n_eps = method.split('_')[1]
            cluster = Experiment.cluster_data(train)
            gpc = GPCluster(cluster, 'spectral' + n_eps)
            return gpc

        elif 'full' in method:
            gpf = GP(train, 'exponential')
            return gpf

    def deploy(self, seed=1001, savefile=None):
        set_seed(seed)
        train, test = self.load_data()
        if self.embedding:
            if self.vae is None:
                pass
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