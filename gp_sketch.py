from utility import *
from kernel import *
from gp import *

if __name__ == '__main__':
    n_data = 2000
    n_test = 100
    n_dim = 5
    noise = 0.1
    n_iter = 10
    data, test = generate_data(n_data, n_test, n_dim)

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

    sgp = SGP(data)
    sgp.cov.weights = gp.cov.weights
    sgp.cov.sn = gp.cov.sn
    sgp.mean.mean = gp.mean.mean

    sgp_pred = sgp(test['X'])
    gp_pred = gp(test['X'])
    print(rmse(sgp_pred, test['Y']), rmse(gp_pred, test['Y']))
