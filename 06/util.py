import numpy as np

def get_matrix_A_regression_polynomial(x, p):
    # ==================================================
    # complete the code
    # ==================================================
    n = len(x)
    A = np.zeros((n, p + 1))

    for i in range(p + 1):
        A[:, i] = x ** i
    return A


def compute_regression_polynomial(A, y, alpha):
    # ==================================================
    # complete the code
    # ==================================================
        
    n, m = A.shape
    theta = np.linalg.inv(A.T @ A + alpha * np.identity(m)) @ A.T @ y
    f = A @ theta
    
    return f, theta


def compute_loss_regression_polynomial(A, y, theta, alpha):
    # ==================================================
    # complete the code
    # ==================================================    
    
    n = len(y)
    error = y - A @ theta
    loss_data = np.linalg.norm(error) ** 2 / (2 * n)
    loss_reg = alpha * np.linalg.norm(theta) ** 2 / 2
    loss = loss_data + loss_reg
    return loss


