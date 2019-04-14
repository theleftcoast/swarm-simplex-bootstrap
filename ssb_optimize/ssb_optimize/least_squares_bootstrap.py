#import optimizer as opt
import numpy as np
import multiprocessing
import pandas as pd
import matplotlib.pyplot as plt

NUM_CPUS = multiprocessing.cpu_count()
NUM_PROCESSES = 1 if (NUM_CPUS - 1) <= 1 else NUM_CPUS

def bootstrap_sample(array_size, sample_size):
    """
    Returns an array of integers which is a uniform random sample consisting of [sample_size] elements taken from an
    array of indeces where n = [array_size].  This array of multipliers can be used to generate bootstrap confidence
    intervals.
    """
    indeces = np.arange(array_size)
    samples = np.random.choice(indeces, size=sample_size, replace=True)
    bootstrap_result = np.zeros(array_size,dtype=np.int32)
    bootstrap_indeces, bootstrap_count = np.unique(samples, return_counts=True)
    bootstrap_result[bootstrap_indeces] = bootstrap_count
    return bootstrap_result

def function_wrapper(argument):
    '''
    Takes a single argument, unpacks it to args and kwargs components, and passes them to func. This gets around the
    fact that mp.Pool.map() and mp.Pool.starmap() only take one iterable argument.   This doesn't allow us to pass
    multiple args and kwargs which is a problem.  Build a single argument from all input args and kwargs and then call
    func_wrapper in the Pool method.

    arguments = [(args, kwargs) for j in jobs_with_different_args_and_kwargs]

    This wrapper supports the least_squares_objective_function and will return one evaluated element of the objective
    function.  It also checks if either the weight or bootstrapping multipliers is zero before evaluating the func to
    improve efficiency.

    objective_function_element_i = (b_i*w_i)*(func(theta, xi, *args_i, **kwargs_i) - fx)**2
    '''
    func, theta, x, fx, w, b, args, kwargs = argument
    return (w*b)*(func(x, *theta, *args, **kwargs)-fx)**2.0 if w > 0.0 and b > 0 else 0.0

def least_squares_objective_function(theta, func, x, fx, w = None, b = None, args = None, kwargs = None, multiprocess = False):
    """
    The least squares objective function is the core of parameter fitting.  This implementation facilitates weights as
    well as bootstrapping.
    
    In all cases, x is a list of lists or a list of tuples that represent vector inputs to func and fx is the
    corresponding real scalar output.  The difference between fx and func(theta, xi, *args_i, **kwargs_s) is used to
    measure the goodness of fit as follows.

    objective_function = sum_over_i{(b_i*w_i)*(func(theta, xi, *args_i, **kwargs_i) - fx_i)**2}

    In this implementation, the objective function is broken down into pieces to improve the code structure as well
    as to facilitate parallel processing.  The following structure allows us to evaluate each result_i in an
    embarrassingly parallel fashion if the objection function is very expensive to evaluate.

    objective_function = sum_over_i{result_i}
    result_i = (b_i*w_i)*(func(theta, xi, *args_i, **kwargs_i) - fx)**2

    The user can toggle between 'multiprocess = False' and 'multiprocess = True' to test and see if there is a
    performance improvement. The Python multiprocessing library can result in slower performance because of the system
    overhead required.

    Parameters
    ----------
    theta : list, required
    x : list of lists or list of tuples, required
    fx : list of scalars, required
    w : list of scalars, optional
    b : list of integers, optional

    func : callable, ``func(x, *theta, *args, **kwargs)``
    args : tuple or list of tuples, optional
    kwargs : dictionary or list of dictionaries, optional
    bounds : list of bounds tuples, **special format**, optional
    constraints : list of constraint dictionaries, **special format**, optional 
    
    Returns
    ----------
    objective_function_value : scalar
    """
    x_array = np.array(x)
    fx_array = np.array(fx)

    if w is None:
        w_array = np.ones(len(x))
    else:
        w_array = np.array(w)

    if b is None:
        b_array = np.ones(len(x))
    else:
        b_array = np.array(b)

    func_list = [func]*len(x)

    theta_list = [theta]*len(x)

    if args is None:
        args_list = [()]*len(x)
    elif isinstance(args, (list, tuple)):
        args_list = args
    else:
        raise TypeError("args must be a list or tuple of length len(x) containing args lists or args tuples")

    if kwargs is None:
        kwargs_list = [{}]*len(x)
    elif isinstance(kwargs, (list, tuple)):
        kwargs_list = kwargs
    else:
        raise TypeError("kwargs must be a list or tuple of length len(x) containing kwargs dictionaries")

    arguments = list(zip(func_list, theta_list, x_array, fx_array, w_array, b_array, args_list, kwargs_list))

    if multiprocess:
        with multiprocessing.Pool(NUM_PROCESSES) as p:
            results = p.map(function_wrapper, arguments)
    else:
        results = np.apply_along_axis(function_wrapper, 1, arguments)

    return np.sum(results)

# def quadratic(x, a, b, c, d, e, f):
#     return a*x[0]**2 + b*x[1]**2 + c*x[0] + d*x[1] + e*x[0]*x[1] + f
#
# x = [[-2.0, -2.0],
#     [-1.0, -2.0],
#     [0.0, -2.0],
#     [1.0, -2.0],
#     [2.0, -2.0],
#     [-2.0, -2.0],
#     [-1.0, -1.0],
#     [0.0, -1.0],
#     [1.0, -1.0],
#     [2.0, -1.0],
#     [-2.0, 0.0],
#     [-1.0, 0.0],
#     [0.0, 0.0],
#     [1.0, 0.0],
#     [2.0, 0.0],
#     [-2.0, 1.0],
#     [-1.0, 1.0],
#     [0.0, 1.0],
#     [1.0, 1.0],
#     [2.0, 1.0],
#     [-2.0, 2.0],
#     [-1.0, 2.0],
#     [0.0, 2.0],
#     [1.0, 2.0],
#     [2.0, 2.0]]
#
# fx_exact = [0.16, 0.34, 1.04, 2.26, 4, 0.16, 0.04, 0.26, 1, 2.26, 1.04, 0.26, 0, 0.26, 1.04, 2.26, 1, 0.26, 0.04, 0.34, 4.3, 2.26, 1.04, 0.34, 0.16]
# fx_noise = [0.14, 0.35, 0.94, 1.98, 3.34, 0.14, 0.05, 0.2, 1.02, 2.37, 0.97, 0.29, 0, 0.2, 0.97, 2.59, 0.98, 0.3, 0.03, 0.36, 4.75, 2.56, 1.17, 0.29, 0.2]
# theta = [0.261, 0.261, 0.0, 0.0,-0.481, 0]
#
# weight = None # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# bootstrap = None # bootstrap_sample(len(x),len(x))
#
# print(least_squares_objective_function(theta, quadratic, x, fx_exact, w = weight, b = bootstrap, args = None, kwargs = None, multiprocess = False))
# print(opt.nelder_mead(theta, least_squares_objective_function, args=(quadratic, x, fx_exact, weight, bootstrap)))
#
# result = []
# for i in range(100):
#     bootstrap = bootstrap_sample(len(x), len(x))
#     result.append(opt.nelder_mead(theta, least_squares_objective_function, args=(quadratic, x, fx_exact, weight, bootstrap)))
#
# df = pd.DataFrame(result, columns = ['a', 'b', 'c', 'd', 'e', 'f'])
# print(df.describe())
# pd.plotting.scatter_matrix(df)
# plt.show()