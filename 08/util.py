import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

    
def compute_gradient_descent_step(x, y, label, theta, lr):
    # ==================================================
    # complete the code
    # ==================================================
    ones = np.array([1 for i in range(len(x))])
    p = np.transpose(np.concatenate((ones, x.flatten(), y.flatten())).reshape(3, len(x)))
    
    linear_regression_output = np.matmul(p, theta)

    h = 1/(1+np.exp(-(linear_regression_output)))
    loss = (1/len(h)) * np.sum((-label*np.log(h) - (1-label)*np.log(1-h)))

    grad = (1/len(p)) * np.matmul(np.transpose(p), (1/(1+np.exp(-(linear_regression_output)))- label))

    theta = theta - lr*grad

    return theta, loss 


def plot_data_label(x, y, label):
    # ==================================================
    # complete the code
    # ==================================================
    x_0 = x[label == 0]
    x_1 = x[label == 1]
    y_0 = y[label == 0]
    y_1 = y[label == 1]

    plt.figure(figsize=(6,6))

    plt.scatter(x_0,y_0, marker='.', c='b', label="label 0", s=12)
    plt.scatter(x_1,y_1, marker='.', c='r', label="label 1", s=12)

    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.show()


def plot_classifier(x, y, label, theta):
    # ==================================================
    # complete the code
    # ==================================================    
    plt.figure(figsize=(6,6))

    x_0 = x[label == 0]
    x_1 = x[label == 1]
    y_0 = y[label == 0]
    y_1 = y[label == 1]

    x_val = np.arange(min(x)-5, max(x)+5, 0.01)
    y_val = np.arange(min(y)-5, max(y)+5, 0.01)
    x, y = np.meshgrid(x_val, y_val)
    z = theta[0] + theta[1] * x + theta[2] * y

    lev = np.linspace(-25, 25, 150)
    plt.contourf(x, y, z, cmap='RdBu_r', levels=lev)
    plt.colorbar().set_ticks(np.arange(-25, 25, 5))

    plt.scatter(x_0,y_0, marker='.', c='b', label="label 0", s=12)
    plt.scatter(x_1,y_1, marker='.', c='r', label="label 1", s=12)
    
    f = (-theta[0]-theta[1]*x_val)/theta[2]
    plt.plot(x_val, f, color = 'k')
    
    plt.axis('equal')
    plt.legend(fontsize=10)
    plt.tight_layout()

    plt.show()