import numpy as np

def get_matrix_A_linear_regression(x, y):
    # ==================================================
    # complete the code
    # ==================================================
    n = len(x)
    ones = np.ones(n)
    A = np.column_stack((ones, x, y))
    return A


def compute_prediction(A, theta):
    # ==================================================
    # complete the code
    # ==================================================
    f = np.dot(A, theta)
    return f


def compute_residual(A, z, theta):
    # ==================================================
    # complete the code
    # ==================================================
    f = compute_prediction(A, theta)
    res = f - z
    return res


def compute_gradient(A, z, theta):
    # ==================================================
    # complete the code
    # ==================================================
    n = len(z)
    grad = (1 / n) * np.dot(A.T, compute_residual(A, z, theta))
    return grad
    
    
def compute_loss(A, z, theta):
    # ==================================================
    # complete the code
    # ==================================================
    n = len(z)
    loss = (1 / (2 * n)) * np.sum(compute_residual(A, z, theta) ** 2)
    return loss