import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
import copy
from matplotlib import pyplot as plt
from sklearn import cluster
import pickle

def generate_data(n_data, n_test, n_dim):
    device = get_cuda_device()
    data = dict()
    test = dict()
    A = ts(np.random.random((n_dim, 1))).to(device)
    data['X'] = ts(np.random.random((n_data, n_dim))).to(device)
    data['Y'] = torch.mm(data['X'], A)
    test['X'] = ts(np.random.random((n_test, n_dim))).to(device)
    test['Y'] = torch.mm(test['X'], A)

    return data, test


def abalone_data(is_train=True):
    tail = 'train' if is_train else 'test'
    datapath = './data/abalone/abalone.{}'.format(tail)
    device = get_cuda_device()
    X, y = [], []
    with open(datapath) as f:
        for line in f:
            line = line.strip().split(',')

            # convert a line to numbers
            if line[0] == 'M': line[0] = 1
            elif line[0] == 'F': line[0] = 2
            else: line[0] = 3

            data = [float(x) for x in line[:-1]]
            target = float(line[-1])
            target = target - 1 if target != 29 else (target - 2) # index form 0

            X.append(data)
            y.append(target)

    X = np.array(X)
    y = np.array(y)
    y = y.reshape(len(y), 1)

    x_tensor = torch.from_numpy(X).to(device)
    y_tensor = torch.from_numpy(y).to(device)
    xmean = torch.mean(x_tensor, dim=0)
    xstd = torch.std(x_tensor, dim=0)
    print(x_tensor.shape, xmean.shape, xstd.shape)
    x_tensor = (x_tensor - xmean) / (xstd)
    #y_tensor = y_tensor - torch.mean(y_tensor)
    return {'X': x_tensor.float(), 'Y': y_tensor.float()}, len(X)


def sample_rows(prob, n_rows):
    selected = []
    for i in range(n_rows):
        p = np.random.random()
        if p < prob[i]:
            selected.append(i)

    return selected


def rmse(y1, y2):
    diff = y1.float() - y2.float()
    return torch.sqrt(torch.dot(diff.view(-1), diff.view(-1)) / diff.shape[0])


def ts(X):
    return torch.tensor(X)


def dt(X):
    return X.detach().cpu().numpy()


def get_cuda_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # set device