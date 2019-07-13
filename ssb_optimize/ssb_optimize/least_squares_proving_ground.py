import objective_functions as obj
import optimizer as opt
import numpy as np
import itertools


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

# fx was generated by passing x to the 'sectioned quadratic' function and rounding the results.
fx_full = [0.16, 0.34, 1.04, 2.26, 4, 0.16, 0.04, 0.26, 1, 2.26, 1.04, 0.26, 0, 0.26, 1.04, 2.26, 1, 0.26, 0.04,
     0.34, 4.3, 2.26, 1.04, 0.34, 0.16]
fx_a = [3.22, 1.61, 1.07, 1.61, 3.22, 3.22, 0.81, 0.27, 0.81, 2.42, 2.15, 0.54, 0.0, 0.54, 2.15, 2.42, 0.81, 0.27, 0.81,
        2.42, 3.22, 1.61, 1.07, 1.61, 3.22]
fx_b = [0.0012, -0.011, -0.021, -0.036, -0.048, 0.0011, 0.0006, -0.012, -0.024, -0.037, 0.025, 0.012, 0.0, -0.012,
        -0.025, 0.037, 0.024, 0.012, -0.0006, -0.013, 0.041, 0.036, 0.024, 0.011, -0.0012]
fx_c = [-1.99, -1.0, -0.022, 0.96, 1.95, -1.98, -0.51, -0.022, 0.47, 0.96, -0.022, -0.022, -0.022, -0.022, -0.022,
        0.96, 0.47, -0.022, -0.51, -1.0, 1.95, 0.96, -0.022, -1.0, -1.98]
fx = list(itertools.chain(fx_full, fx_a, fx_b, fx_c))

func = obj.sectioned_quadratic
theta = [0.261, 0.261, 0.0, 0.0, -0.481, 0.0]
theta_min = [0.26886495, 0.26838753, -0.01241092, 0.01181188, -0.49181186, -0.02209403]

w_full = [1.0] * len(x)
w_a = [0.05] * len(x)
w_b = [0.1] * len(x)
w_c = [0.2] * len(x)
w = list(itertools.chain(w_full, w_a, w_b, w_c))

option_full = [{'section':'full'}] * len(x)
option_a = [{'section':'a'}] * len(x)
option_b = [{'section':'b'}] * len(x)
option_c = [{'section':'c'}] * len(x)
option = tuple(itertools.chain(option_full, option_a, option_b, option_c))

x_repeat = list(itertools.chain(x, x, x, x))

print(w)
print(fx)
print(option)
print(x_repeat)

b = None # opt._bootstrap_sample(len(x), len(x))

precision = 3

minimum = opt.nelder_mead(theta, opt.least_squares_objective_function,
                          args=(func, x_repeat, fx),
                          kwargs={'w':w,
                                  'b':b,
                                  'args':None,
                                  'kwargs':option,
                                  'bounds':None,
                                  'constraints':None,
                                  'multiprocess':False},
                          max_iter=1000)
print(minimum)

bootstrap_min = opt.least_squares_bootstrap(theta, func, x_repeat, fx,
                                        weight=w, args=None, kwargs=option,
                                        bounds=None, constraints=None,
                                        multiprocess=False,
                                        samples=100, max_iter=1000)
print(bootstrap_min)

