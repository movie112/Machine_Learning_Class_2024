import numpy as np

def convert_color_to_gray(image_color):
    # ==================================================
    # complete the code
    # ==================================================
    red = image_color[:, :, 0]
    green = image_color[:, :, 1]
    blue = image_color[:, :, 2]

    image_gray = (0.3*red + 0.5*green + 0.2*blue) / 3 
    return image_gray


def normalize_01(data_input):
    # ==================================================
    # complete the code
    # ==================================================
    data_output = (data_input - np.min(data_input)) / (np.max(data_input) - np.min(data_input))
    return data_output


def compute_derivative_first_row_forward(I):
    # ==================================================
    # complete the code
    # ==================================================
    # neumann boundary condition
    I_pad = np.pad(I, ((0, 1), (0, 0)), 'edge') 
    I_der = np.zeros(I.shape)
    for i in range(I.shape[0]):
        I_der[i, :] = (I_pad[i+1, :] - I_pad[i, :])
    return I_der


def compute_derivative_first_col_forward(I):
    # ==================================================
    # complete the code
    # ==================================================
    I_pad = np.pad(I, ((0, 0), (0, 1)), 'edge') 
    I_der = np.zeros(I.shape)
    for i in range(I.shape[1]):
        I_der[:, i] = (I_pad[:, i+1] - I_pad[:, i])
    return I_der


def compute_derivative_first_row_central(I):
    # ==================================================
    # complete the code
    # ==================================================
    I_pad = np.pad(I, ((0, 1), (0, 0)), 'edge')
    I_der = np.zeros(I.shape)
    for i in range(I.shape[0]):
        I_der[i, :] = (I_pad[i+1, :] - I_pad[i-1, :]) / 2
    return I_der


def compute_derivative_first_col_central(I):
    # ==================================================
    # complete the code
    # ==================================================
    I_pad = np.pad(I, ((0, 0), (0, 1)), 'edge')
    I_der = np.zeros(I.shape)
    for i in range(I.shape[1]):
        I_der[:, i] = (I_pad[:, i+1] - I_pad[:, i-1]) / 2
    return I_der


def compute_derivative_second_row(I):
    # ==================================================
    # complete the code
    # ==================================================
    I_pad = np.pad(I, ((1, 1), (0, 0)), 'edge')
    I_der = np.zeros(I.shape)
    for i in range(I.shape[0]):
        I_der[i, :] = (I_pad[i+1, :] - 2*I_pad[i, :] + I_pad[i-1, :])
    return I_der


def compute_derivative_second_col(I):
    # ==================================================
    # complete the code
    # ==================================================
    I_pad = np.pad(I, ((0, 0), (1, 1)), 'edge')
    I_der = np.zeros(I.shape)
    for i in range(I.shape[1]):
        I_der[:, i] = (I_pad[:, i+1] - 2*I_pad[:, i] + I_pad[:, i-1])
    return I_der


def compute_norm_gradient_central(I):
    # ==================================================
    # complete the code
    # ==================================================
    I_row   = compute_derivative_second_row(I)
    I_col   = compute_derivative_second_col(I)
    I_norm  = np.sqrt(I_row**2 + I_col**2)
    return I_norm


def compute_laplacian(I):
    # ==================================================
    # complete the code
    # ==================================================
    I_pad       = np.pad(I, (1, 1), 'symmetric')
    I_der1      = I_pad[2:, 1:-1]
    I_der2      = I_pad[:-2, 1:-1]
    I_der3      = I_pad[1:-1, 2:]
    I_der4      = I_pad[1:-1, :-2]
    I_laplace   = I_der1 + I_der2 + I_der3 + I_der4 - 4.0 * I
    return I_laplace
