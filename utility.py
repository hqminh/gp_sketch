import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np


def generate_data(n_data, n_test, n_dim):
    data = dict()
    test = dict()
    A = ts(np.random.random((n_dim, 1)))
    data['X'] = ts(np.random.random((n_data, n_dim)))
    data['Y'] = torch.mm(data['X'], A)
    test['X'] = ts(np.random.random((n_test, n_dim)))
    test['Y'] = torch.mm(test['X'], A)

    '''
        selected = sample_rows(np.ones(n_data) / np.sqrt(n_data), n_data)
        print(selected)
        for id in range(n_data):
            if id in selected:
                data['X'][id, :] = 1.0 * data['X'][id, :]
            else:
                data['X'][id, :] = 0.0 * data['X'][id, :]
        '''

    return data, test


def sample_rows(prob, n_rows):
    selected = []
    for i in range(n_rows):
        p = np.random.random()
        if p < prob[i]:
            selected.append(i)

    return selected


def rmse(y1, y2):
    diff = y1 - y2
    return torch.sqrt(torch.dot(diff, diff) / diff.shape[0])


def ts(X):
    return torch.tensor(X)


def dt(X):
    return X.detach().numpy()


def get_cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device