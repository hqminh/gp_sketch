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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- CONSTANSTS ----
    n_data = 2000
    n_test = 100
    n_dim = 200
    n_iter = 200
    # tolerance = 0.1
    tolerances = np.linspace(1e-3, 0.5, num=50)
    lr = 0.2

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
    min_eigens = []
    gp = GP(data).to(device)
    model = nn.ModuleList([gp.cov, gp.mean]).to(device)
    optimizer = opt.Adam(model.parameters(), lr=lr)


    diffs = []
    for tolerance in tolerances:
        diffs_tolerance = []
        for i in range(n_iter + 1):
            model.train()
            optimizer.zero_grad()
            loss = gp.NLL()
            print('Training Iteration ' + str(i) + ' : ', loss.item())
            if i < n_iter + 1:
                loss.backward()
                optimizer.step()
            min_eig_value = get_min_eigen_val(gp).data.numpy()
            print(min_eig_value)
            min_eigens.append(np.abs(min_eig_value))


            if (i + 1) % 10 == 0:
                # --- PREDICTION ----
                sgp = SGP(data)
                sgp.cov.weights = gp.cov.weights
                sgp.cov.sn = gp.cov.sn
                sgp.mean.mean = gp.mean.mean

                sgp_pred = sgp(test['X'], tolerance)
                gp_pred = gp(test['X'])[0]

                print(rmse(sgp_pred, test['Y']), rmse(gp_pred, test['Y']))
                diff = rmse(sgp_pred, gp_pred).data.numpy()[0][0]

                diffs_tolerance.append(diff)

                min_eig_value = get_min_eigen_val(gp)

                print("min_eig_value={}, threshhold={} "
                      .format(min_eig_value,
                              tolerance / (1. - tolerance) / min_eig_value)
                      )

                print("Tolerance: {}\titer {}: mineig={}, Diff GP-SGP: {}".
                      format(tolerance, i + 1, min_eig_value, diff))



                # # --- PLOT ----
                # rcParams['figure.figsize'] = 16, 9
                # solutions_diff = torch.abs(sgp_pred - gp_pred).data.numpy()
                # plt.close()
                # plt.plot(range(n_test), solutions_diff,
                #          '-o',
                #          color='b', lw=1.5,
                #          label='|SGP-GP|')
                #
                # plt.ylabel('diffirence bw sketched GP and GP', fontsize=24)
                # plt.xlabel('test points', fontsize=24)
                # plt.xticks(fontsize=18)
                # # plt.locator_params(nbins=11, axis='x')
                # plt.yticks(fontsize=18)
                # # plt.locator_params(nbins=11, axis='y')
                # plt.legend(fontsize=17.5,
                #            # loc=(1.04, 0)
                #            )
                # plt.grid()
                # plt.tight_layout()
                # # plt.title('GP vs SGP', fontsize=30)
                # plt.savefig('sgp_diff_gp_iter{}.pdf'.format(i+1))
                #
                # # plt.show()
                #
                # # eigen
                # plt.close()
                # plt.plot(range(i + 1), min_eigens, '-d', color='r', lw=1.5,
                #          # label='min_eigen_vals'
                #          )
                # plt.ylabel('minimum eigen values', fontsize=24)
                # plt.xlabel('iters', fontsize=24)
                # plt.xticks(fontsize=18)
                # # plt.locator_params(nbins=11, axis='x')
                # plt.yticks(fontsize=18)
                # # plt.locator_params(nbins=11, axis='y')
                # plt.legend(fontsize=17.5,
                #            # loc=(1.04, 0)
                #            )
                # plt.grid()
                # plt.tight_layout()
                # # plt.title('GP vs SGP', fontsize=30)
                # plt.savefig('min_eigen_iter{}.pdf'.format(i+1))
                # # plt.show()
                #
                # plt.close()
                # plt.plot(range(i + 1), np.log(np.array(min_eigens)), '-x',
                #          color='g', lw=1.5,
                #          # label='min_eigen_vals'
                #          )
                # plt.ylabel('minimum eigen values', fontsize=24)
                # plt.xlabel('iters', fontsize=24)
                # plt.xticks(fontsize=18)
                # # plt.locator_params(nbins=11, axis='x')
                # plt.yticks(fontsize=18)
                # # plt.locator_params(nbins=11, axis='y')
                # # plt.legend(fontsize=17.5,
                # # loc=(1.04, 0)
                # # )
                # plt.grid()
                # plt.tight_layout()
                # # plt.title('GP vs SGP', fontsize=30)
                # plt.savefig('log_min_eigen_iter{}.pdf'.format(i+1))
                # # plt.show()

        diffs.append(diffs_tolerance)
        np.save('diffs', np.array(diffs))

    print('Done')



