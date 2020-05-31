from gp import *
from kernel import *
from utility import *
from sklearn.cluster import KMeans
from mvae import *


def train_gp(gp_obj, test, n_iter=500, record_interval=10):
    idx = []
    nll = []
    error = []
    model = nn.ModuleList([gp_obj.cov, gp_obj.mean])
    optimizer = opt.Adam(model.parameters())
    #optimizer = opt.SGD(model.parameters(), lr=0.01, momentum=0.9)
    for i in range(n_iter + 1):
        model.train()
        optimizer.zero_grad()
        loss = gp_obj.NLL()
        if loss < 0:
            print('Converged')
            break
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


def data_cluster(data, k=10):
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


def plot_result():
    res_gpc = pickle.load(open('./gpc.pkl', 'rb'))
    res_gpf = pickle.load(open('./gpf.pkl', 'rb'))
    res_gps = pickle.load(open('./gps.pkl', 'rb'))
    plt.figure()
    plt.plot(res_gpc['idx'], res_gpc['err'])
    plt.plot(res_gps['idx'], res_gps['err'])
    plt.plot(res_gpf['idx'], res_gpf['err'])
    plt.savefig('gpc_vs_gpf_vs_gps_rmse.png')


def run_gpc(seed, data, test, n_eps=1500):
    set_seed(seed)
    cluster = data_cluster(data)
    gpc = GPCluster(cluster, 'spectral_'+str(n_eps))
    res_gpc = dict()
    res_gpc['err'], res_gpc['nll'], res_gpc['idx'] = train_gp(gpc, test=test, n_iter=800, record_interval=25)
    pickle.dump(res_gpc, open('./gpc_' + str(n_eps) + '.pkl', 'wb'))


def run_gps(seed, data, test, n_eps=1500):
    set_seed(seed)
    gps = GP(data, 'spectral_'+str(n_eps))
    res_gps = dict()
    res_gps['err'], res_gps['nll'], res_gps['idx'] = train_gp(gps, test=test, n_iter=800, record_interval=25)
    pickle.dump(res_gps, open('./gps_' + str(n_eps) + '.pkl', 'wb'))


def run_gpf(seed, data, test):
    set_seed(seed)
    gpf = GP(data, 'exponential')
    res_gpf = dict()
    res_gpf['err'], res_gpf['nll'], res_gpf['idx'] = train_gp(gpf, test=test, n_iter=800, record_interval=25)
    pickle.dump(res_gpf, open('./gpf.pkl','wb'))


def analyze_cluster(seed):
    set_seed(seed)
    data, _ = abalone_data(is_train=True)
    test, _ = abalone_data(is_train=False)
    cluster = data_cluster(data)
    distance = torch.zeros(cluster['k'], cluster['k']).to(get_cuda_device())
    radii = torch.zeros(cluster['k']).to(get_cuda_device())
    for i in range(cluster['k']):
        for j in range(cluster[i]['X'].shape[0]):
            ci = cluster['centroids'][i]
            xj = cluster[i]['X'][j]
            rij = torch.sqrt(torch.sum(torch.pow(ci - xj, 2.0)))
            radii[i] = torch.max(radii[i], rij)


def load_data(seed, prefix='./', encode=True, l1=0.0, l2=0.0, l3=0.0):
    f = open(prefix + 'exp_desc.txt', 'w')
    f.write(str(l1) + ' ' + str(l2) + ' ' + str(l3))
    f.close()
    train, n_train = abalone_data(is_train=True)
    test, n_test = abalone_data(is_train=False)
    set_seed(seed)
    if encode:
        dataset = TensorDataset(train['X'], train['Y'])
        data = DataLoader(dataset, batch_size=100, shuffle=True)
        input_dim = train['X'].shape[1]
        output_dim = 2
        n_component = 10
        vae = MixtureVAE(data, input_dim, output_dim, n_component, n_sample=10)
        n_iter = 20
        epoch_iter = 5
        for i in range(n_iter):
            vae.train_vae(n_iter=epoch_iter, l1=l1, l2=l2, l3=l3)
            torch.save(vae, prefix + 'encoder_' + str(i * epoch_iter) + '.pth')
            z = dt(vae(train['X'], encode=True))
            xc, yc = z[:, 0], z[:, 1]
            plt.figure()
            plt.scatter(xc, yc)
            plt.savefig(prefix + 'embed_scatter_' + str(i * epoch_iter) + '.png')
        train['X'] = vae(train['X'], encode=True)
        test['X'] = vae(test['X'], encode=True)

    return train, test


def load_data_2(seed, prefix='./', name='abalone', method='dvae', alpha=8.0, beta=1.2, gamma=1.0, encode=True, odim=2, k=10, verbose=True):
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    if name == 'abalone':
        train, n_train = abalone_data(is_train=True)
        test, n_test = abalone_data(is_train=False)
    elif name == 'diabetes':
        train, n_train = diabetes_data(is_train=True)
        test, n_test = diabetes_data(is_train=False)
    set_seed(seed)
    if encode:
        f = open(prefix + 'exp_result.txt', 'w')
        dataset = TensorDataset(train['X'], train['Y'])
        data = DataLoader(dataset, batch_size=100, shuffle=True)
        input_dim = train['X'].shape[1]
        output_dim = odim
        n_component = k
        vae = MixtureVAE(data, input_dim, output_dim, n_component, n_sample=10)
        n_iter = 20
        epoch_iter = 5
        for i in range(n_iter):
            f.write('iter=' + str(i) + '\n')
            if method == 'dvae':
                vae.train_dvae(n_iter=epoch_iter, alpha=alpha, beta=beta, verbose=verbose)
            elif method == 'dsvae':
                vae.train_dsvae(n_iter=epoch_iter, alpha=alpha, beta=beta, gamma=gamma, verbose=verbose)
            elif method == 'dvvae':
                vae.train_dvvae(n_iter=epoch_iter, alpha=alpha, beta=beta, gamma=gamma, verbose=verbose)
            torch.save(vae, prefix + 'encoder_' + str(i * epoch_iter) + '.pth')
            z = dt(vae(train['X'], encode=True))
            if odim == 2:
                xc, yc = z[:, 0], z[:, 1]
                plt.figure()
                plt.scatter(xc, yc)
                plt.savefig(prefix + 'embed_scatter_' + str(i * epoch_iter) + '.png')
            f.write('Cluster membership\n')
            print('Cluster membership\n')
            kmeans = KMeans(n_clusters=10, random_state=0).fit(z)
            cluster = [[] for _ in range(10)]

            for j in range(z.shape[0]):
                cid = kmeans.labels_[j]
                cluster[cid].append(j)
            f.write(str([len(cluster[j]) for j in range(10)]) + '\n')
            f.write('Min cluster distance\n')
            print(str([len(cluster[j]) for j in range(10)]) + '\n')
            print('Min cluster distance\n')
            min_dist = None
            for u in range(10):
                for v in range(u + 1, 10):
                    duv = np.linalg.norm(kmeans.cluster_centers_[u] - kmeans.cluster_centers_[v])
                    if min_dist is None:
                        min_dist = duv
                    else:
                        min_dist = min(min_dist, duv)
            f.write(str(min_dist) + '\n')
            f.write('Max cluster radius\n')
            print(str(min_dist) + '\n')
            print('Max cluster radius\n')
            max_rad = None
            for j in range(z.shape[0]):
                cj = kmeans.labels_[j]
                rj = np.linalg.norm(z[j] - kmeans.cluster_centers_[cj])
                if max_rad is None:
                    max_rad = rj
                else:
                    max_rad = max(max_rad, rj)
            f.write(str(max_rad) + '\n')
            print(str(max_rad) + '\n')
            f.flush()
        f.close()

        train['X'] = vae(train['X'], encode=True)
        test['X'] = vae(test['X'], encode=True)

    return train, test


if __name__ == '__main__':
    #train, test = load_data_2(1001, './29MayExp1/', name='abalone', method='dvae', alpha=100.0, beta=1.0, gamma=10.0)
    #train, test = load_data_2(1001, './29MayExp2/', name='abalone', method='dvae', alpha=1.0, beta=1.0, gamma=1.0, k=2)
    #train, test = load_data_2(1001, './29MayExp3/', name='abalone', method='dvae', alpha=1.0, beta=1.0, gamma=1.0, k=4)
    #train, test = load_data_2(1001, './29MayExp4/', name='abalone', method='dvae', alpha=1.0, beta=1.0, gamma=1.0, k=6)
    #train, test = load_data_2(1001, './29MayExp5/', name='abalone', method='dvae', alpha=1.0, beta=1.0, gamma=1.0, k=8)
    #train, test = load_data_2(1001, './29MayExp6/', name='abalone', method='dvae', alpha=1.0, beta=1.0, gamma=1.0, k=6)
    #train, test = load_data_2(1001, './29MayExp7/', name='abalone', method='dvae', alpha=1.0, beta=1.0, gamma=1.0, k=8)
    train, test = load_data_2(1001, './29MayExp8/', name='abalone', method='dsvae', alpha=5.0, beta=1.0, gamma=5.0, k=8)

