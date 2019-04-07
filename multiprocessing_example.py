import numpy as np
import functools
import multiprocessing
import time

def example_func(x, a, b, c):
    return a*x[0]**2 + b*x[1]**2 + c*x[0]*x[1]

def function_wrapper(argument):
    ''' Takes a single argument, unpacks it to args and kwargs components, and passes them to func. This gets around the
    fact that mp.Pool.map() and mp.Pool.starmap() only take one iterable argument.   This doesn't allow us to pass
    multiple args and kwargs which is a problem.  Build a single argument from all input args and kwargs and then call
    func_wrapper in the Pool method.

    arguments = [(args, kwargs) for j in jobs_with_different_args_and_kwargs]

    '''
    func, x, theta, args, kwargs = argument
    return func(x, *theta, *args, **kwargs)

x = [[1.0,2.0],[1.1,2.2],[2.0,3.0],[2.2,3.3]]
theta = [(3.3,),(4.4,),(5.5,),(6.6,)]
args = [(1.1,),(2.2,),(3.3,),(4.4,)]
kwargs = [{"c":7.6},{"c":3.4},{"c":1.2},{"c":8.7}]
function = [example_func]*len(x)

arguments = list(zip(function,x,theta,args,kwargs))

if __name__ == '__main__':
    cpu_count = multiprocessing.cpu_count()
    processes = 1 if (cpu_count - 1) <= 1 else cpu_count
    results = None

    t0_pool = time.time()

    with multiprocessing.Pool(processes) as p:
        results = p.map(function_wrapper, arguments)

    t1_pool = time.time()

    print(t1_pool-t0_pool)
    print(results)

def apply_approach():
    t0_apply = time.time()
    results = np.apply_along_axis(function_wrapper, 1, arguments)
    t1_apply = time.time()

    print(t1_apply-t0_apply)
    print(results)

apply_approach()

'''
def parallelize(func, x, *args, **kwargs):
    """
    Run the objective function
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    return np.apply_along_axis(func, 1, x, *args, **kwargs)

def vectorized_lsq(theta, func, x, fx):
    """
    Let's try this with some vectorizing, shall we?

    This is *not* parallelized, but it gets rid of a lot of the iterator
    protocol overhead, which is cool (I guess?)
    """
    # Assuming the function is, in fact, vectorizable
    vec_func = np.vectorize(func)
    evaluated_func = vec_func(x_array, *theta)
    return np.sum((fx - evaluated_func) ** 2)

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
    return np.sum((fx - evaluated_func) ** 2) '''