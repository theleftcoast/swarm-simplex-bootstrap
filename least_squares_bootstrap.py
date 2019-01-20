from optimizer import nelder_mead
import numpy as np
from multiprocessing import Pool

NUM_CPUS = 4

def test_func(x, a, b, c):
    return a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1]


def cuda_lsqs(theta, func, x, fx):
    """The power of NVIDIA compels you! Run faster"""
    # TODO: use the GPU to do all the function evaluation?

def vectorized_lsq(theta, func, x, fx):
    """
    Let's try this with some vectorizing, shall we?
    
    This is *not* parallelized, but it gets rid of a lot of the iterator
    protocol overhead, which is cool (I guess?)
    """
    # Assuming the function is, in fact, vectorizable
    vec_func = np.vectorize(func)
    evaluated_func = vec_func(x_array, *theta)
    return np.sum((fx-evaluated_func)**2) 

    
def pooled_lsq(theta, func, x, fx):
    """
    Farm out the "model function" evaluation to a multiprocessing pool.
    """
    # Assuming it's not:
    with Pool(processes=NUM_CPUS) as pool:
        # function closure to pre-fill kwargs, since pool.map() only acceps 
        # a single positional argument
        _func = lambda x: func(x, *theta)  
        evaluated_func = pool.map(_func, x_array)
    return np.sum((fx-evaluated_func)**2) 



def bootstrap_least_squares_objective_function(theta, func, x, fx, bootstrap_index, bootstrap_count):
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
    x_sample = x[bootstrap_index]
    fx_sample = fx[bootstrap_index]
    evaluated_func = parallelize(func, x_sample, *theta)
    return np.sum(((fx_sample - evaluated_func)**2)*bootstrap_count)


def parallelize(func, x, *args, **kwargs):
    """
    Run the objective function
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    return np.apply_along_axis(func, 1, x, *args, **kwargs)


def boostrap_resample(x, x_size):
    """
    returns index, count arrays for a random sample of the input data

    The goal is to not evaluate the same function with the same paramters multiple times,

    Runs an object function on a randomly re-sampled vector. based on the
    initial input
    
    re-sample an initial vector into something bigger. Something more. 
    Something we can use over and over again to check cross-correlation between
    our regressed parameters based on their covariance
    
    Such correlation. Much meta regression. Wow.
    """
    x_array = np.array(x)
    x_size = np.size(x_array[:, 0])
    indexes = np.arange(x_size)
    samples = np.random.choice(indexes, size=x_size, replace=True)
    bootstrap_index, bootstrap_count = np.unique(samples, return_counts=True)
    return bootstrap_index, bootstrap_count


    
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

x_size = np.size(x_array[:,0])
indexes = np.arange(x_size)
samples = np.random.choice(indexes, size=x_size, replace=True)
bootstrap_index, bootstrap_count = np.unique(samples, return_counts=True)
print(samples)
print(bootstrap_index)
print(bootstrap_count)

print(nelder_mead(theta,bootstrap_least_squares_objective_function, args = (test_func, x_array, fx_exact_array, bootstrap_index, bootstrap_count)))