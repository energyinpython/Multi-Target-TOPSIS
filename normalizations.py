import numpy as np

# targeted linear normalization
def targeted_linear_normalization(X, TX, types):
    CX = np.vstack((X, TX))
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]

    nmatrix = np.zeros(np.shape(X))
    nmatrix[:, ind_profit] = X[:, ind_profit] / (np.amax(CX[:, ind_profit], axis = 0))
    nmatrix[:, ind_cost] = np.amin(CX[:, ind_cost], axis = 0) / X[:, ind_cost]
    return nmatrix


# targeted minmax normalization
def targeted_minmax_normalization(X, TX, criteria_type):
    CX = np.vstack((X, TX))
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(criteria_type == 1)[0]
    ind_cost = np.where(criteria_type == -1)[0]

    x_norm[:, ind_profit] = (X[:, ind_profit] - np.amin(CX[:, ind_profit], axis = 0)
                             ) / (np.amax(CX[:, ind_profit], axis = 0) - np.amin(CX[:, ind_profit], axis = 0))

    x_norm[:, ind_cost] = (np.amax(CX[:, ind_cost], axis = 0) - X[:, ind_cost]
                           ) / (np.amax(CX[:, ind_cost], axis = 0) - np.amin(CX[:, ind_cost], axis = 0))

    return x_norm

# multimoora normalization
def moora_normalization(X, types):
    x_norm = X / ((np.sum(X ** 2, axis = 0))**(0.5))
    return x_norm


# linear normalization
def linear_normalization(X, types):
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]
    x_norm = np.zeros(np.shape(X))

    x_norm[:, ind_profit] = X[:, ind_profit] / (np.amax(X[:, ind_profit], axis = 0))
    x_norm[:, ind_cost] = np.amin(X[:, ind_cost], axis = 0) / X[:, ind_cost]
    return x_norm


# min-max normalization
def minmax_normalization(X, types):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]

    x_norm[:, ind_profit] = (X[:, ind_profit] - np.amin(X[:, ind_profit], axis = 0)
                             ) / (np.amax(X[:, ind_profit], axis = 0) - np.amin(X[:, ind_profit], axis = 0))

    x_norm[:, ind_cost] = (np.amax(X[:, ind_cost], axis = 0) - X[:, ind_cost]
                           ) / (np.amax(X[:, ind_cost], axis = 0) - np.amin(X[:, ind_cost], axis = 0))

    return x_norm


# max normalization
def max_normalization(X, types):
    maximes = np.amax(X, axis=0)
    ind = np.where(types == -1)[0]
    X = X/maximes
    X[:,ind] = 1-X[:,ind]
    return X


# sum normalization
def sum_normalization(X, types):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]

    x_norm[:, ind_profit] = X[:, ind_profit] / np.sum(X[:, ind_profit], axis = 0)

    x_norm[:, ind_cost] = (1 / X[:, ind_cost]) / np.sum((1 / X[:, ind_cost]), axis = 0)

    return x_norm


# vector normalization
def vector_normalization(X, types):
    x_norm = np.zeros((X.shape[0], X.shape[1]))
    ind_profit = np.where(types == 1)[0]
    ind_cost = np.where(types == -1)[0]

    x_norm[:, ind_profit] = X[:, ind_profit] / (np.sum(X[:, ind_profit] ** 2, axis = 0))**(0.5)

    x_norm[:, ind_cost] = 1 - (X[:, ind_cost] / (np.sum(X[:, ind_cost] ** 2, axis = 0))**(0.5))

    return x_norm

