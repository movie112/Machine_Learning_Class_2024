import numpy as np

def get_fun1(x):
    # ==================================================
    # complete the code
    # ==================================================
    y = np.cos(x)
    return y 
    

def get_derivative_fun1_first(x):
    # ==================================================
    # complete the code
    # ==================================================
    y =  -np.sin(x)
    return y    
    
    
def get_derivative_fun1_second(x):
    # ==================================================
    # complete the code
    # ==================================================
    y = -np.cos(x)
    return y    
    
    
def get_taylor_approximate_fun1_first(x, z):
    # ==================================================
    # complete the code
    # ==================================================
    y = get_fun1(x) + get_derivative_fun1_first(x) * (x-z)
    return y    

    
def get_taylor_approximate_fun1_second(x, z):
    # ==================================================
    # complete the code
    # ==================================================
    y = get_fun1(x) + get_derivative_fun1_first(x) * (x-z) + 0.5 * get_derivative_fun1_second(x) * (x-z)**2
    return y        


def get_fun2(x):
    # ==================================================
    # complete the code
    # ==================================================
    y = np.exp(x)
    return y 
    

def get_derivative_fun2_first(x):
    # ==================================================
    # complete the code
    # ==================================================
    y = np.exp(x)
    return y    
    
    
def get_derivative_fun2_second(x):
    # ==================================================
    # complete the code
    # ==================================================
    y = np.exp(x)
    return y    
    
    
def get_taylor_approximate_fun2_first(x, z):
    # ==================================================
    # complete the code
    # ==================================================
    y = np.exp(z) + np.exp(z) * (x-z)
    return y    

    
def get_taylor_approximate_fun2_second(x, z):
    # ==================================================
    # complete the code
    # ==================================================
    y = np.exp(z) + np.exp(z) * (x-z) + 0.5 * np.exp(z) * (x-z)**2
    return y        