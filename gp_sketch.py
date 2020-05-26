from utility import *
from kernel import *
from gp import *

import gc
import torch

if __name__ == '__main__':
    np_seed = 1000
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_seed = 123456
    torch.manual_seed(torch_seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_data = 2000
    n_test = 100
    n_dim = 5
    noise = 0.1
    n_iter = 100

    # synthetic
    # data, test = generate_data(n_data, n_test, n_dim)

    # abalone data
    data, n_data = abalone_data(is_train=True)
    test, n_test = abalone_data(is_train=False)
    n_dim = 8

    gp = GP(data)
    model = nn.ModuleList([gp.cov, gp.mean])

    model = model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)

    optimizer = opt.Adam(model.parameters())
    for i in range(n_iter + 1):
        model.train()
        optimizer.zero_grad()
        loss = gp.NLL()
        print('Training Iteration ' + str(i) + ' : ', loss.item())
        if i < n_iter + 1:
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            gc.collect()
    print('Done')

    sgp = SGP(data)
    sgp.cov.weights = gp.cov.weights
    sgp.cov.sn = gp.cov.sn
    sgp.mean.mean = gp.mean.mean

    sgp_pred = sgp(test['X'])
    gp_pred = gp(test['X'])[0]
    print(rmse(sgp_pred, test['Y']), rmse(gp_pred, test['Y']))
