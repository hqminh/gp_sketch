import numpy as np
import torch
from scipy.stats import ortho_group


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


def generate_synthetic(len_data, n_features):
    """Synthetic data for Linear Regression problem"""

    rank = 1.

    # Synthetic hyper-params
    Lm = (2 ** rank + 1) ** 2
    d = n_features

    # create a diagnomal matrix Dm
    row_Dm = np.random.uniform(0.5 * np.sqrt(Lm), np.sqrt(Lm), size=d)
    index = np.random.randint(0, d)
    row_Dm[index] = np.sqrt(Lm)
    Dm = np.diag(row_Dm)

    # orthornomal row U and V
    U = ortho_group.rvs(dim=d)  # d x d
    V = generate_orthonomal(d, len_data)

    # data
    Xm = U.dot(Dm).dot(V)
    x = Xm.T  # n x d

    # labels
    w_star = np.ones(d)
    # noise = np.random.normal(0., 1., len_data)
    y = x.dot(w_star)
    y = y.reshape(len(y), 1)

    x_tensor = torch.from_numpy(x)
    y_tensor = torch.from_numpy(y)

    return {'X': x_tensor, 'Y': y_tensor}


def abalone_data(is_train=True):
    tail = 'train' if is_train else 'test'
    datapath = './data/abalone/abalone.{}'.format(tail)


    X, y = [], []
    # dataset = []
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

    x_tensor = torch.from_numpy(X)
    y_tensor = torch.from_numpy(y)

    return {'X': x_tensor, 'Y': y_tensor}, len(X)


def generate_orthonomal(d, n):
    """Generate d x n matrix that has d rows orthonormal to each other"""
    A = np.zeros((d, n))
    for i in range(d):
        A[i] = np.random.random(n)
        for j in range(i):
            A[i] -= A[i].dot(A[j]) * A[j]
        A[i] /= np.linalg.norm(A[i])

    return A


def sample_rows(prob, n_rows):
    selected = []
    for i in range(n_rows):
        p = np.random.random()
        if p < prob[i]:
            selected.append(i)

    return selected


def rmse(y1, y2):
    diff = y1 - y2
    return torch.sqrt(torch.mm(diff.t(), diff) / diff.shape[0])


def ts(X):
    return torch.tensor(X)


def dt(X):
    return X.detach().numpy()


def get_cuda_device():
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")  # set device


def get_min_eigen_val(gp):
    """gp object has data inside: gp.data and covariance kerneled matrix: 
        gp.cov 
    """
    values, vectors = torch.eig(gp.cov(gp.data['X']), eigenvectors=True)
    return torch.min(values)


