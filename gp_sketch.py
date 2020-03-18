import matplotlib.pyplot as plt
import numpy as np
import torch

from gp import GP
from gp import SGP
from pylab import rcParams
from utility import abalone_data
from utility import generate_data
from utility import generate_synthetic
from utility import get_min_eigen_val
from utility import rmse

from torch import nn
from torch import optim as opt


if __name__ == '__main__':
    np_seed = 1000
    np.random.seed(np_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch_seed = 123456
    torch.manual_seed(torch_seed)


    # --- CONSTANSTS ----
    n_data = 2000
    n_test = 100
    n_dim = 5
    n_iter = 50
    tolerance = 0.1

    # --- DATA ----
    # noise = 0.1
    # data, test = generate_data(n_data, n_test, n_dim)

    # synthetic data
    # data = generate_synthetic(n_data, n_dim)
    # test = generate_synthetic(n_test, n_dim)

    # abalone data
    data, n_data = abalone_data(is_train=True)
    test, n_test = abalone_data(is_train=False)
    n_dim = 8

    # --- TRAINING ----
    gp = GP(data)
    model = nn.ModuleList([gp.cov, gp.mean])
    optimizer = opt.Adam(model.parameters())
    for i in range(n_iter + 1):
        model.train()
        optimizer.zero_grad()
        loss = gp.NLL()
        print('Training Iteration ' + str(i) + ' : ', loss.item())
        if i < n_iter + 1:
            loss.backward()
            optimizer.step()
    print('Done')

    # --- PREDICTION ----
    sgp = SGP(data)
    sgp.cov.weights = gp.cov.weights
    sgp.cov.sn = gp.cov.sn
    sgp.mean.mean = gp.mean.mean

    sgp_pred = sgp(test['X'], tolerance)
    gp_pred = gp(test['X'])[0]

    print(rmse(sgp_pred, test['Y']), rmse(gp_pred, test['Y']))
    print("Diff between GP and SGP: {}".format(rmse(sgp_pred, gp_pred)))

    min_eig_value = get_min_eigen_val(gp)
    print("min_eig_value={}, threshhold={} "
          .format(min_eig_value,
                  tolerance / (1. - tolerance) / min_eig_value)
          )

    # --- PLOT ----
    rcParams['figure.figsize'] = 16, 9
    solutions_diff = torch.abs(sgp_pred - gp_pred).data.numpy()
    plt.close()
    plt.plot(range(n_test), solutions_diff,
             '-o',
             color='b', lw=1.5,
             label='|SGP-GP|')

    plt.ylabel('diffirence bw sketched GP and GP', fontsize=24)
    plt.xlabel('test points', fontsize=24)
    plt.xticks(fontsize=18)
    # plt.locator_params(nbins=11, axis='x')
    plt.yticks(fontsize=18)
    # plt.locator_params(nbins=11, axis='y')
    plt.legend(fontsize=17.5,
               # loc=(1.04, 0)
               )
    plt.grid()
    plt.tight_layout()
    # plt.title('GP vs SGP', fontsize=30)
    plt.savefig('sgp_diff_gp.pdf')
    plt.show()
