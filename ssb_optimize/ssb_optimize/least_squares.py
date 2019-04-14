from optimizer import nelder_mead
import numpy as np

def test_func(x, a, b, c):
    return a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1]

def least_squares_objective_function(theta, func, x, fx):
    """
    The least squares objective function is the heart of fitting functions.
    
    In all cases, x is a list of lists or a list of tuples that represent vector inputs
    and fx is the scalar output corresponding to the vector input.

    Long term goal --> Minimize func(x, *theta, *args, **kwargs) subject to bounds 
    and contraints where theta is a list of scalar arguements to be adjusted and 
    *args/*kwargs are other positional/keyword arguements to be passed to func.
        lsof(x, fx, theta, func, args=(), kwargs={}, bounds=None, constraints=None)

    Short term goal --> Minimize func(x, *theta)

    interface well with nelder_mead(x0,func,args=(),kwargs={}, method=None, bounds=None, constraints=(), tol=None, options=None)

    Parameters
    ----------
    x : list of lists or list of tuples, required
    fx : list of scalars, required
    theta : list, required    
    func : callable, ``func(x, *theta, *args, **kwargs)``
    args : tuple, optional
    kwargs : dictionary, optional
    bounds : list of bounds tuples, **special format**, optional
    constraints : list of constraint dictionaries, **special format**, optional 
    
    Returns
    ----------
    objective_function_value : scalar
    """

    evaluated_func = np.apply_along_axis(func, 1, x_array, *theta)
    
    return np.sum((fx-evaluated_func)**2) 
    
x = [[-2.0, -2.0],
    [-1.0, -2.0],
    [0.0, -2.0],
    [1.0, -2.0],
    [2.0, -2.0],
    [-2.0, -2.0],
    [-1.0, -1.0],
    [0.0, -1.0],
    [1.0, -1.0],
    [2.0, -1.0],
    [-2.0, 0.0],
    [-1.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [2.0, 0.0],
    [-2.0, 1.0],
    [-1.0, 1.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 1.0],
    [-2.0, 2.0],
    [-1.0, 2.0],
    [0.0, 2.0],
    [1.0, 2.0],
    [2.0, 2.0]]

fx_exact = [0.16, 0.34, 1.04, 2.26, 4, 0.16, 0.04, 0.26, 1, 2.26, 1.04, 0.26, 0, 0.26, 1.04, 2.26, 1, 0.26, 0.04, 0.34, 4, 2.26, 1.04, 0.34, 0.16]
fx_noise = [0.14, 0.35, 0.94, 1.98, 3.34, 0.14, 0.05, 0.2, 1.02, 2.37, 0.97, 0.29, 0, 0.2, 0.97, 2.59, 0.98, 0.3, 0.03, 0.36, 4.75, 2.56, 1.17, 0.29, 0.2]

x_array = np.array(x)
fx_exact_array = np.array(fx_exact)
fx_noise_array = np.array(fx_noise)
theta = [0.0,0.0,0.0]

print(nelder_mead(theta,least_squares_objective_function, args = (test_func, x_array, fx_noise_array)))
