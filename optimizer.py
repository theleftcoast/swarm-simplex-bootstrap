import numpy as np
import itertools
import functools
import multiprocessing
import numbers
import types

DEBUG = False  # Controls debugging print statements

def bounds_check(n, bounds=None):
    """
    Check bounds list passed to optimizer for internal consistency.
    """
    if bounds is None:
        bounds = [(None, None)] * n
    if len(bounds) != n:
        raise ValueError('length of bounds list must equal dimension of x passed to func')

    lower = np.zeros(n)
    upper = np.zeros(n)
    for i, bound in enumerate(bounds):
        if bound is None:
            lower[i] = -np.inf
            upper[i] = np.inf
        elif not isinstance(bound, tuple) or len(bound) != 2:
            raise TypeError('bounds[%d] must be a (min, max) tuple' % i)
        else:
            l, u = bound
            if l is None:
                lower[i] = -np.inf
            elif isinstance(l, numbers.Number):
                lower[i] = l
            else:
                raise TypeError('bounds[%d] min value must be a number' % i)
            if u is None:
                upper[i] = np.inf
            elif isinstance(u, numbers.Number):
                upper[i] = u
            else:
                raise TypeError('bounds[%d] max value must be a number' % i)
    bounds = np.array(list(zip(lower, upper)))

    for i, bound in enumerate(bounds):
        l, u = bound
        if u - l <= 0:
            raise ValueError('bounds[%d] min must be less than max' % i)
    return bounds

def constraints_check(constraints=None):
    """
    Check constraints dictionary passed to optimizer for internal consistency.
    """
    if constraints is not None:
        if not isinstance(constraints, list):
            raise TypeError('constraints must be a list of dictionaries')
        for i, const in enumerate(constraints):
            if not isinstance(const, dict):
                raise TypeError('constraints[%d] is not a dictionary' % i)
            if 'type' not in const.keys():
                raise TypeError('constraints[%d] dictionary does not have required "type" key' % i)
            if const['type'] not in ['>0', '>=0', '<0', '<=0']:
                raise ValueError('constraints[%d]["type"] must be >0, >=0, <0, or <=0' % i)
            if 'func' not in const.keys():
                raise TypeError('constraints[%d] dictionary does not have required "func" key' % i)
            if not callable(const['func']):
                raise TypeError('constraints[%d]["func"] must be callable' % i)
            if 'args' not in const.keys():
                const['args'] = ()
            if not isinstance(const['args'], tuple):
                raise TypeError('constraints[%d]["args"] must be a tuple' % i)
            if 'kwargs' not in const.keys():
                const['kwargs'] = {}
            if not isinstance(const['kwargs'], dict):
                raise TypeError('constraints[%d]["kwargs"] must be a dictionary' % i)
    return constraints

def penalized_func(x, func, args=(), kwargs={}, bounds=None, constraints=None):
    penalty_value = penalty(x, bounds, constraints)
    return func(x, *args, **kwargs) + penalty_value

def penalty(x, bounds=None, constraints=None):
    penalty_value = 0.0

    if bounds is not None:
        for i, bound in enumerate(bounds):
            if bound[0] > x[i] or bound[1] < x[i]:
                penalty_value = np.inf
                break

    if constraints is not None:
        for i, const in enumerate(constraints):
            const_value = const['func'](x, *const['args'], **const['kwargs'])
            if const['type'] == '>0' and const_value <= 0.0:
                penalty_value = np.inf
                break
            if const['type'] == '>=0' and const_value < 0.0:
                penalty_value = np.inf
                break
            if const['type'] == '<0' and const_value >= 0.0:
                penalty_value = np.inf
                break
            if const['type'] == '<=0' and const_value > 0.0:
                penalty_value = np.inf
                break

    return penalty_value

def feasible_points_grid(bounds, constraints=None, grid_size=7, inf_repl=10.0**4):
    """
    Explore the space defined by bounds and constraints using grid generated points.
    """
    if not isinstance(grid_size, int):
        raise TypeError('grid_size must be an integer')
    if not isinstance(inf_repl, numbers.Number):
        raise TypeError('inf_repl must be a number')
    # Transform infinite bound values (-inf and inf) into finite bound values
    lower = np.zeros(len(bounds))
    upper = np.zeros(len(bounds))
    for i, bound in enumerate(bounds):
        l, u = bound
        if np.isinf(l):
            lower[i] = -inf_repl
        else:
            lower[i] = l
        if np.isinf(u):
            upper[i] = inf_repl
        else:
            upper[i] = u
    finite_bounds = np.array(list(zip(lower, upper)))

    # Create a list of arrays that contain evenly spaced numbers spanning each bounded dimension
    bound_arrays = [np.linspace(*bound, grid_size) for bound in finite_bounds]

    # Create list of points from the cartesian product of all bound arrays covering entire bounded space
    list_of_points = np.array(list(itertools.product(*bound_arrays)))

    # Evaluate the penalty fuction at each point in the list of points
    penalty_function_values = np.apply_along_axis(penalty, axis=1, arr=list_of_points, bounds=bounds,
                                                  constraints=constraints)

    # Create summary statistic and list of points where all constraints are satisfied
    number_of_points = len(list_of_points)
    count_penalty_function_violated = np.count_nonzero(penalty_function_values)
    fraction_violated = count_penalty_function_violated / number_of_points
    indices_violated = np.nonzero(penalty_function_values)
    feasible_points = np.delete(list_of_points, indices_violated, axis=0)

    return fraction_violated, np.array(feasible_points)

def feasible_points_random(bounds, constraints=None, point_count=45, max_iter=None, inf_repl=10.0 ** 4):
    """
    Explore the space defined by bounds and constraints using randomly generated points.
    Potential issue: Might be able to remove infinite value check.
    """
    if not isinstance(point_count, int):
        raise TypeError('point_count must be an integer')
    if max_iter is None:
        max_iter = point_count * 20
    if not isinstance(max_iter, int):
        raise TypeError('max_iter must be an integer')
    if not isinstance(inf_repl, numbers.Number):
        raise TypeError('inf_repl must be a number')

    # Transform infinite bound values (-inf and inf) into finite bound values
    lower = np.zeros(len(bounds))
    upper = np.zeros(len(bounds))
    for i, bound in enumerate(bounds):
        l, u = bound
        if np.isinf(l):
            lower[i] = -inf_repl
        else:
            lower[i] = l
        if np.isinf(u):
            upper[i] = inf_repl
        else:
            upper[i] = u
    finite_bounds = np.array(list(zip(lower, upper)))

    # Create a list of arrays that contain evenly spaced numbers spanning each bounded dimension
    feasible_points = []
    infeasible_points = []

    while len(feasible_points) < point_count and len(feasible_points) + len(infeasible_points) < max_iter:
        random_vector = [np.random.uniform(bound[0], bound[1]) for bound in finite_bounds]
        penalty_function_value = penalty(random_vector, bounds=bounds, constraints=constraints)
        if penalty_function_value == 0.0:
            feasible_points.append(random_vector)
        else:
            infeasible_points.append(random_vector)

    number_of_points = len(feasible_points) + len(infeasible_points)
    fraction_violated = len(infeasible_points) / number_of_points

    return fraction_violated, np.array(feasible_points)

def best_point(points, func, args=(), kwargs={}):
    """
    Return the point corresponding to the lowest evaluated value of func.
    """
    # Evaluate function for every feasible vector in point
    cpu_count = multiprocessing.cpu_count()
    processes = max([1, cpu_count - 1])
    mp_pool = multiprocessing.Pool(processes)
    func_values = np.array(mp_pool.map(functools.partial(func, *args, **kwargs), points))
    ordered = np.argsort(func_values)

    return points[ordered[0], :]

def create_simplex(initial,size):
    """ Create initial simplex. """
    n = len(initial)
    p = (size/(n*2**0.5))*(n-1+(n+1)**0.5)
    q = (size/(n*2**0.5))*(-1+(n+1)**0.5)
    identity = np.identity(n)
    result = np.zeros((n,n))
    for i in range(n):
        result[i] += initial + p*identity[i]
        for j in range(n):
            if j == i:
                continue
            result[i] += q*identity[j]             
    return np.row_stack((initial,result))

def nelder_mead(x0,func,args=(),kwargs={},bounds=None,constraints=None,small_tol=10.0**-14,flat_tol=10.0**-14,max_iter=10000):
    """ 
    Nelder-Mead simplex minimization algorithm.  Implementation details can be found in "Implementing 
    the Nelder-Mead simplex algorithm with adaptive parameters" by Gao and Han

    TODO: Incorporate bounds and constraints into basic Nelder-Mead algorithm.
    TODO: Improve handling of shrink step function evaluation (parallelize??)
    """

    # Validate bounds list and constraints dictionary
    n = len(x0)
    b = bounds_check(n, bounds)
    c = constraints_check(constraints)

    # Initialize simplex
    simplex = create_simplex(x0,0.01)
    f_simplex = np.apply_along_axis(func, 1, simplex, *args, **kwargs)
    ordered = np.argsort(f_simplex)

    # Calculate adaptive parameters improve convergence for higher dimensional problems
    n = len(simplex[0])
    r = 1.0 # reflection (standard method = 1)
    e = 1.0+2.0/n # expansion (standard method = 2)
    c = 0.75-1.0/(2.0*n) # contraction (standard method = 1/2)
    s = 1.0-1.0/n # shrink (standard method = 1/2)

    # Termination criteria (used to set these here, moved them up to the function definition)
    # small_tol = 10**-14
    # flat_tol = 10**-14
    # max_iter = 10000

    # Initialize termination variables
    small = 1.0
    flat = 1.0
    counter = 0
    
    # Initialize algorithm performance variables
    reflection_count = 0
    expansion_count = 0    
    outside_contraction_count = 0
    inside_contraction_count = 0
    shrink_count = 0

    while small > small_tol and flat > flat_tol and counter < max_iter:
        # Worst, second worst, and best simplex points.
        lowest = simplex[ordered[0],:]
        second_highest = simplex[ordered[-2],:]
        highest = simplex[ordered[-1],:]
        centroid = simplex[ordered[0:-1],:].mean(axis = 0)
        # Objective function evaluated at each simplex point from above.
        f_highest = f_simplex[ordered[-1]]
        f_second_highest = f_simplex[ordered[-2]]
        f_lowest = f_simplex[ordered[0]]    

        # Evaluate reflection.
        reflection = centroid + r*(centroid-highest)
        f_reflection = func(reflection, *args, **kwargs)

        if f_reflection < f_lowest:
            # Evaluate expansion.
            expansion = centroid + e * (reflection - centroid)
            f_expansion = func(expansion, *args, **kwargs)
            if f_expansion < f_reflection:
                # Replace highest by expansion
                simplex[ordered[-1],:] = expansion
                f_simplex[ordered[-1]] = f_expansion
                expansion_count += 1
            else:
                # Replace highest by reflection
                simplex[ordered[-1],:] = reflection
                f_simplex[ordered[-1]] = f_reflection
                reflection_count += 1
        elif f_reflection < f_second_highest:
            # Replace highest by reflection
            simplex[ordered[-1],:] = reflection
            f_simplex[ordered[-1]] = f_reflection
            reflection_count += 1
        elif f_reflection < f_highest:
            # Evaluate outside contraction.
            outside_contraction = centroid + c * (reflection - centroid)
            f_outside_contraction = func(outside_contraction, *args, **kwargs)
            if f_outside_contraction < f_reflection:
                # Replace highest by reflection
                simplex[ordered[-1],:] = outside_contraction
                f_simplex[ordered[-1]] = f_outside_contraction
                outside_contraction_count += 1
            else:
                # Replace all but best by shrink
                for i in ordered:
                    simplex[i, :] = lowest + s * (simplex[i, :] - lowest)
                f_simplex = np.apply_along_axis(func, 1, simplex, *args, **kwargs)
                shrink_count += 1
        else:
            # Evaluate inside contraction.
            inside_contraction = centroid - c * (reflection - centroid)
            f_inside_contraction = func(inside_contraction, *args, **kwargs)
            if f_inside_contraction < f_highest:
                # Replace highest by contraction
                simplex[ordered[-1],:] = inside_contraction
                f_simplex[ordered[-1]] = f_inside_contraction
                inside_contraction_count += 1
            else:
                # Replace all but best by shrink
                for i in ordered:
                    simplex[i,:] = lowest + s*(simplex[i,:]-lowest)
                f_simplex = np.apply_along_axis(func, 1, simplex, *args, **kwargs)
                shrink_count += 1

        ordered = np.argsort(f_simplex)
        flat = np.absolute(f_simplex[ordered[-1]]-f_simplex[ordered[0]])
        small = np.linalg.norm(simplex[ordered[-1]]-simplex[ordered[0]])
        counter = counter + 1
    if DEBUG:
        print(reflection_count)
        print(expansion_count)
        print(outside_contraction_count)
        print(inside_contraction_count)
        print(shrink_count)
    return simplex[ordered[0]]