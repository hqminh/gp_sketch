from gp import *
from kernel import *
from utility import *


def synthetic():
    set_seed()
    device = get_cuda_device()
    data, test = generate_data(1000, 100, 5)
    full_ls = torch.tensor(np.random.randn(5, 1))
    spec_ls = copy.deepcopy(full_ls)
    gp_full = GP(data, ls=full_ls).to(device)
    gp_spec = GP(data, cov_func='spectral_30', ls=spec_ls).to(device)
    print('Full GP Training')
    full_rmse, full_loss, idx = train_gp(gp_full, test, n_iter=500, record_interval=10)

    print('Spectral GP Training')
    spec_rmse, spec_loss, _ = train_gp(gp_spec, test, n_iter=500, record_interval=10)

    plt.figure()
    plt.plot(idx, spec_loss, 'b+')
    plt.plot(idx, full_loss, 'r-')
    plt.savefig('./synthetic_loss.png')
    plt.figure()
    plt.plot(idx, spec_rmse, 'b+')
    plt.plot(idx, full_rmse, 'r-')
    plt.savefig('./synthetic.png')
    plt.figure()


def synthetic_2():
    set_seed()

    device = get_cuda_device()
    data, test = generate_data(1000, 100, 5)
    ls = torch.tensor(np.zeros((5, 1)))
    full_ls = copy.deepcopy(ls)
    gp_full = GP(data, ls=full_ls).to(device)
    n_eps = [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    spec_rmse = []
    full_rmse = []

    print('Full GP Training')
    frmse = train_gp(gp_full, test, n_iter=200, record_interval=-1)

    for ne in n_eps:
        print('Spectral GP Training', ne)
        spec_ls = copy.deepcopy(ls)
        cov_func = 'spectral_' + str(ne)
        gp_spec = GP(data, cov_func=cov_func, ls=spec_ls).to(device)
        spec_rmse.append(train_gp(gp_spec, test, n_iter=200, record_interval=-1))
        full_rmse.append(frmse)
        del gp_spec

    plt.figure()
    plt.plot(n_eps, spec_rmse, 'b+')
    plt.plot(n_eps, full_rmse, 'r-')
    plt.savefig('./synthetic_rmse_sample.png')


def abalone():
    set_seed()
    device = get_cuda_device()
    data, n_data = abalone_data(is_train=True)
    data['X'] = data['X'][:1000]
    data['Y'] = data['Y'][:1000]
    print(data['X'])
    test, _ = abalone_data(is_train=False)
    test['X'] = test['X'][:100]
    test['Y'] = test['Y'][:100]
    full_ls = torch.tensor(0.1 * np.ones((data['X'].shape[1], 1)))
    spec_ls = copy.deepcopy(full_ls)
    gp_full = GP(data, ls=full_ls).to(device)
    gp_spec = GP(data, cov_func='spectral_30', ls=spec_ls).to(device)
    gp_chi = GP(data, cov_func='chi_30', ls=spec_ls).to(device)
    print('Full GP Training')
    full_rmse, full_loss, idx = train_gp(gp_full, test, n_iter=500, record_interval=10)

    print('Spectral GP Training')
    spec_rmse, spec_loss, _ = train_gp(gp_spec, test, n_iter=500, record_interval=10)

    print('ChiSpectral GP Training')
    spec_rmse, spec_loss, _ = train_gp(gp_chi, test, n_iter=500, record_interval=10)

    plt.figure()
    plt.plot(idx, spec_loss, 'b+')
    plt.plot(idx, full_loss, 'r-')
    plt.savefig('./abalone_loss.png')
    plt.figure()
    plt.plot(idx, spec_rmse, '-b')
    plt.plot(idx, full_rmse, '-r')
    plt.savefig('./abalone_rmse.png')
    plt.figure()


def abalone_2():
    set_seed()
    device = get_cuda_device()
    data, n_data = abalone_data(is_train=True)
    data['X'] = data['X'][:500]
    data['Y'] = data['Y'][:500]
    test, _ = abalone_data(is_train=False)
    test['X'] = test['X'][:50]
    test['Y'] = test['Y'][:50]
    ls = torch.tensor(np.zeros((data['X'].shape[1], 1)))
    full_ls = copy.deepcopy(ls)
    gp_full = GP(data, ls=full_ls).to(device)
    n_eps = [1000, 2500, 5000, 10000]
    chi_rmse = []
    spec_rmse = []
    full_rmse = []

    print('Full GP Training')
    frmse = train_gp(gp_full, test, n_iter=200, record_interval=-1)

    for ne in n_eps:
        print('Sparse GP Training', ne)
        spec_ls = copy.deepcopy(ls)
        cov_func = 'spectral_' + str(ne)
        chi_cov_func = 'chi_' + str(ne)
        gp_spec = GP(data, cov_func=cov_func, ls=spec_ls).to(device)
        gp_chi = GP(data, cov_func=chi_cov_func, ls=spec_ls).to(device)
        spec_rmse.append(train_gp(gp_spec, test, n_iter=200, record_interval=-1))
        chi_rmse.append(train_gp(gp_chi, test, n_iter=200, record_interval=-1))
        full_rmse.append(frmse)
        del gp_spec

    plt.figure()
    plt.plot(n_eps, spec_rmse, '-b')
    plt.plot(n_eps, chi_rmse, '-c')
    plt.plot(n_eps, full_rmse, '-r')
    plt.savefig('./abalone_rmse_sample.png')


def inspect_condition():
    set_seed()
    device = get_cuda_device()
    data, n_data = abalone_data(is_train=True)
    data['X'] = data['X'][:500]
    data['Y'] = data['Y'][:500]
    test, _ = abalone_data(is_train=False)
    X = dt(data['X'])
    kmeans = cluster.KMeans(n_clusters=5, random_state=0).fit(X)
    label = kmeans.labels_
    center = kmeans.cluster_centers_
    count = [0] * 5
    mdist = [0.0] * 5
    for i in range(len(label)):
        count[label[i]] += 1
        mdist[label[i]] = np.max(np.linalg.norm(mdist[label[i]], X[i] - center[label[i]]))
    print(count)
    print(mdist)
    exit()


    test['X'] = test['X'][:50]
    test['Y'] = test['Y'][:50]
    ls = torch.tensor(np.zeros((data['X'].shape[1], 1)))
    full_ls = copy.deepcopy(ls)
    gp_full = GP(data, ls=full_ls).to(device)
    kxx, trained_ls = train_gp(gp_full, test, n_iter=200, record_interval=-2)
    kxx = dt(kxx)
    kxx = kxx / np.max(kxx)
    spec_ker = SpectralCov(n_dim=data['X'].shape[1], n_eps=20000, ls=trained_ls)
    chi_ker = ChiSpectralCov(n_dim=data['X'].shape[1], n_eps=20000, ls=trained_ls)
    skxx = dt(spec_ker(data['X']))
    sscale = np.max(skxx)
    skxx = skxx / sscale
    ckxx = dt(chi_ker(data['X']))
    cscale = np.max(ckxx)
    ckxx = ckxx / cscale
    plt.figure()
    plt.imshow(kxx, cmap='hot', interpolation='nearest')
    plt.savefig('kernel.png')
    plt.figure()
    plt.imshow(skxx, cmap='hot', interpolation='nearest')
    plt.savefig('skernel.png')
    plt.figure()
    plt.imshow(ckxx, cmap='hot', interpolation='nearest')
    plt.savefig('ckernel.png')
    print(sscale, cscale)

def old_config():
    # train, test = load_data(1001, './24MayExp1/', encode=True, l1=1.0, l2=10.0, l3=10.0)
    # train, test = load_data(1001, './24MayExp2/', encode=True, l1=10.0, l2=10.0, l3=10.0)
    # train, test = load_data(1001, './24MayExp3/', encode=True, l1=100.0, l2=10.0, l3=10.0)
    # train, test = load_data(1001, './24MayExp4/', encode=True, l1=1000.0, l2=10.0, l3=10.0)
    # train, test = load_data(1001, './27MayExp1/', encode=True, l1=5000.0, l2=10.0, l3=10.0)
    # train, test = load_data(1001, './27MayExp2/', encode=True, l1=10000.0, l2=10.0, l3=100.0)
    # train, test = load_data(1001, './27MayExp3/', encode=True, l1=5000.0, l2=50.0, l3=10.0)
    # train, test = load_data(1001, './27MayExp4/', encode=True, l1=10000.0, l2=50.0, l3=100.0)
    # train, test = load_data_2(1001, './27MayExp5/', encode=True, odim=2)
    # train, test = load_data_2(1001, './27MayExp6/', encode=True, odim=3)
    # train, test = load_data_2(1001, './27MayExp7/', encode=True, odim=4)
    # train, test = load_data_2(1001, './27MayExp8/', encode=True, odim=5)
    # train, test = load_data_2(1001, './28MayExp1/', encode=True, odim=2)
    # train, test = load_data_2(1001, './28MayExp2/', encode=True, odim=3)
    # train, test = load_data_2(1001, './28MayExp3/', encode=True, odim=4)
    # train, test = load_data_2(1001, './28MayExp4/', encode=True, odim=5)
    # train, test = load_data_2(1001, './28MayExp5/', name='abalone', method='dsvae', alpha=5.0, beta=1.0, gamma=5.0)
    # train, test = load_data_2(1001, './28MayExp6/', name='abalone', method='dsvae', alpha=16.0, beta=2.4, gamma=5.0)
    # train, test = load_data_2(1001, './28MayExp7/', name='abalone', method='dsvae', alpha=32.0, beta=5.0, gamma=5.0)
    # train, test = load_data_2(1001, './28MayExp8/', name='abalone', method='dsvae', alpha=100.0, beta=1.0, gamma=5.0)
    # train, test = load_data_2(1001, './28MayExp9/', name='abalone', method='dsvae', alpha=8.0, beta=1.2, gamma=5.0)
    # train, test = load_data_2(1001, './28MayExp10/', name='abalone', method='dsvae', alpha=3.0, beta=0.5, gamma=5.0)
    # train, test = load_data_2(1001, './29MayExp1/', name='abalone', method='dvvae', alpha=100.0, beta=1.0, gamma=1.0)
    # train, test = load_data_2(1001, './29MayExp2/', name='abalone', method='dvvae', alpha=100.0, beta=1.0, gamma=2.0)
    # train, test = load_data_2(1001, './29MayExp3/', name='abalone', method='dvvae', alpha=100.0, beta=1.0, gamma=4.0)
    # train, test = load_data_2(1001, './29MayExp4/', name='abalone', method='dvvae', alpha=100.0, beta=1.0, gamma=8.0)
    # train, test = load_data_2(1001, './29MayExp5/', name='abalone', method='dvvae', alpha=100.0, beta=1.0, gamma=16.0)
    # train, test = load_data_2(1001, './29MayExp6/', name='abalone', method='dvvae', alpha=100.0, beta=1.0, gamma=32.0)
    train, test = load_data_2(1001, './29MayExp7/', name='abalone', method='dvae', alpha=100.0, beta=1.0, gamma=10.0)
    # train, test = load_data_2(1001, './29MayExp8/', name='abalone', method='dsvae', alpha=300.0, beta=50.0, gamma=1000.0)
    # run_gpc(1001, train, test, 2500)
    # run_gpc(1001, train, test, 2500)
    # run_gps(1001, train, test, 500)
    # run_gpf(1001, train, test)
    # plot_result()
    # analyze_cluster(1001)
    '''
    device = get_cuda_device()
    data, _ = abalone_data(is_train=True)
    recon_loss = []
    for i in range(2):
        vae = torch.load('./24MayExp' + str(i + 3) +'/encoder_65.pth')
        Z = vae(data['X'], encode=True)
        X_recon = vae(Z, encode=False)
        recon_loss.append(torch.mean((X_recon - data['X']) ** 2) ** 0.5)
        kmeans = KMeans(n_clusters=10, random_state=0).fit(dt(Z))
        cluster = [[] for j in range(10)]
        for j in range(Z.shape[0]):
            cid = kmeans.labels_[j]
            cluster[cid].append(j)
        print([len(cluster[j]) for j in range(10)])
    print(recon_loss)
    '''

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
