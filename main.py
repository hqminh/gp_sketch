from gp import *
from kernel import *
from utility import *
from sklearn.cluster import KMeans
import torch
import gc

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_seed = seed
    np_seed = seed
    np.random.seed(np_seed)
    torch.manual_seed(torch_seed)


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

        if record_interval == -1:
            continue
        elif (i + 1) % record_interval == 0:
            with torch.no_grad():
                ypred, yvar = gp_obj(test['X'])
                error.append(rmse(ypred, test['Y']))
                nll.append(loss.item())
                idx.append(i + 1)
                print('Training Iteration', i + 1, 'rmse:', error[-1], 'nll:', nll[-1])

    if record_interval == -1:
        ypred, yvar = gp_obj(test['X'])
        torch.cuda.empty_cache()
        gc.collect()
        return rmse(ypred, test['Y'])
    if record_interval == -2:
        torch.cuda.empty_cache()
        gc.collect()
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


def run_gpc(seed, n_eps=1500):
    set_seed(seed)
    data, _ = abalone_data(is_train=True)
    test, _ = abalone_data(is_train=False)
    cluster = data_cluster(data)
    gpc = GPCluster(cluster, 'spectral_'+str(n_eps))

    res_gpc = dict()
    res_gpc['err'], res_gpc['nll'], res_gpc['idx'] = train_gp(gpc, test=test, n_iter=800, record_interval=25)
    pickle.dump(res_gpc, open('./gpc_' + str(n_eps) + '.pkl', 'wb'))


def run_gps(seed, n_eps=1500):
    set_seed(seed)
    data, _ = abalone_data(is_train=True)
    test, _ = abalone_data(is_train=False)
    gps = GP(data, 'spectral_'+str(n_eps))
    res_gps = dict()
    res_gps['err'], res_gps['nll'], res_gps['idx'] = train_gp(gps, test=test, n_iter=800, record_interval=25)
    pickle.dump(res_gps, open('./gps_' + str(n_eps) + '.pkl', 'wb'))


def run_gpf(seed):
    set_seed(seed)
    data, _ = abalone_data(is_train=True)
    test, _ = abalone_data(is_train=False)
    gpf = GP(data, 'exponential')
    res_gpf = dict()
    res_gpf['err'], res_gpf['nll'], res_gpf['idx'] = train_gp(gpf, test=test, n_iter=800, record_interval=25)
    pickle.dump(res_gpf, open('./gpf.pkl','wb'))


def analyze_cluster(seed):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data, _ = abalone_data(is_train=True)
    test, _ = abalone_data(is_train=False)

    cluster = data_cluster(data)
    distance = torch.zeros(cluster['k'], cluster['k']).to(device)
    radii = torch.zeros(cluster['k']).to(device)

    for i in range(cluster['k']):
        for j in range(cluster[i]['X'].shape[0]):
            ci = cluster['centroids'][i]
            ci = torch.Tensor(ci).to(device)
            xj = cluster[i]['X'][j].to(device)
            rij = torch.sqrt(torch.sum(torch.pow(ci - xj, 2.0)))
            radii[i] = torch.max(radii[i], rij)


if __name__ == '__main__':
    # seed = 1001, 2002, 3003, 4004, 5005
    run_gpc(1001, 2500)
    run_gps(1001, 500)
    run_gpf(1001)
    plot_result()
    analyze_cluster(1001)


