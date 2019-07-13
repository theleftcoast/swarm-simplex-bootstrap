import numpy as np
import itertools
import numbers
import multiprocessing


def bounds_check(n, bounds=None):
    """Check bounds list of size 'n' for consistency and return the list with basic problems corrected.

    A valid bounds list is a list of bound tuples. --> [(bound_tuple), ... ,(bound_tuple)]

    A bound tuple specifies the minimum and maximum values allowed for each of the 'n' dimensions of the problem
    space. If there are no boundaries for a particular dimension of the problem space (i.e. infinity or -infinity),
    then specify 'None' for that element of the bound tuple.

    bound_tuple --> (min, max), (None, max), (min, None), (None, None)

    Args:
        n (int): Size of bounds list.
        bounds (list): List of bounds tuples.

    Returns:
        list: Validated list of bounds tuples.

    Raises:
        TypeError: n must be an integer
        ValueError: length of bounds list must equal dimension n passed to bounds_check
        TypeError: bound must be a (min, max) tuple
        TypeError: bounds min value must be a number
        TypeError: bounds max value must be a number
        ValueError: bounds min must be less than max
    """
    # Check validity of 'n'.
    if not isinstance(n, int):
        raise TypeError('n must be an integer')
    # Check validity of the bounds list.
    if bounds is None:
        bounds = [(None, None)]*n
    if len(bounds) != n:
        raise ValueError('length of bounds list must equal dimension n passed to bounds_check')
    lower = np.zeros(n)
    upper = np.zeros(n)
    for i, bound in enumerate(bounds):
        if bound is None:
            lower[i] = -np.inf
            upper[i] = np.inf
        elif not isinstance(bound, tuple) or len(bound) != 2:
            raise TypeError('bounds[%d] must be a (min, max) tuple' % i)
        else:
            lb, ub = bound
            if lb is None:
                lower[i] = -np.inf
            elif isinstance(lb, numbers.Number):
                lower[i] = lb
            else:
                raise TypeError('bounds[%d] min value must be a number' % i)
            if ub is None:
                upper[i] = np.inf
            elif isinstance(ub, numbers.Number):
                upper[i] = ub
            else:
                raise TypeError('bounds[%d] max value must be a number' % i)
    bounds = np.array(list(zip(lower, upper)))
    for i, bound in enumerate(bounds):
        lb, ub = bound
        if ub - lb <= 0:
            raise ValueError('bounds[%d] min must be less than max' % i)
    return bounds


def constraints_check(constraints=None):
    """Check constraints list for consistency and return the list with basic problems corrected.

    A valid constraints list is a list of constraint dictionaries. --> [{const_dict}, ... ,{const_dict}]

    A constraint dictionary specifies individual constraint function, arguments (args and kwargs), and a constraint
    type.  The value corresponding the 'func' key is a callable function which defines the constraint.  Traditional
    functions ('def func(x): ...') and lambda functions ('func = lambda x: ...') both work.  The function signature
    must be of the form 'func(x, *args, **kwargs)' where 'x' is a scalar or a list of scalars representing a vector.
    The return value from 'func' must be a scalar.  Values corresponding to the 'args' and 'kwargs' keys are any
    additional arguments to be passed to 'func' (both are optional).  Positional arguments (args) are specified as
    a tuple and keyword arguments (kwargs) are specified as a dictionary. The value corresponding to the 'type' key
    specifies the inequality that the scalar returned from 'func' must obey for the constraint to be satisfied.
    Values for the inequality specification may be '>0', '>=0', '<0', '<=0'.

    const_dict --> {'type': ineq_spec_string, 'func': callable_func, 'args': (args_tuple), 'kwargs': {kwargs_dict}}

    Args:
        constraints (list): List of constraint dictionaries.

    Returns:
        list: Validated list of constraint dictionaries.

    Raises:
        TypeError: constraints must be a list of dictionaries
        TypeError: constraint is not a dictionary
        TypeError: constraint dictionary does not have required "type" key
        ValueError: constraint["type"] must be >0, >=0, <0, or <=0
        TypeError: constraint dictionary does not have required "func" key
        ValueError: constraint["func"] must be callable
        TypeError: constraint["args"] must be a tuple
        TypeError: constraint["kwargs"] must be a dictionary
    """
    if constraints is not None:
        if not isinstance(constraints, list):
            raise TypeError('constraints must be a list of dictionaries')
        for i, const in enumerate(constraints):
            if not isinstance(const, dict):
                raise TypeError('constraint[%d] is not a dictionary' % i)
            if 'type' not in const.keys():
                raise TypeError('constraint[%d] dictionary does not have required "type" key' % i)
            if const['type'] not in ['>0', '>=0', '<0', '<=0']:
                raise ValueError('constraint[%d]["type"] must be >0, >=0, <0, or <=0' % i)
            if 'func' not in const.keys():
                raise TypeError('constraint[%d] dictionary does not have required "func" key' % i)
            if not callable(const['func']):
                raise TypeError('constraint[%d]["func"] must be callable' % i)
            if 'args' not in const.keys():
                const['args'] = ()
            if not isinstance(const['args'], tuple):
                raise TypeError('constraint[%d]["args"] must be a tuple' % i)
            if 'kwargs' not in const.keys():
                const['kwargs'] = {}
            if not isinstance(const['kwargs'], dict):
                raise TypeError('constraint[%d]["kwargs"] must be a dictionary' % i)
    return constraints


def _penalty(x, bounds=None, constraints=None):
    """Return a penalty value (which is np.inf) if the input vector x violates either the bounds or constraints."""
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


def _penalized_func(x, func, args=None, kwargs=None, bounds=None, constraints=None):
    """Evaluate function and add a penalty (which is np.inf) if bounds or constraints are violated."""
    args = args or ()
    kwargs = kwargs or {}
    penalty_value = _penalty(x, bounds, constraints)
    return func(x, *args, **kwargs) + penalty_value


def _feasible_points_grid(bounds, constraints=None, grid_size=7, inf_repl=10.0 ** 4):
    """Generate list of feasible points starting from a grid which satisfy the bounds and constraints."""
    # Check validity of inputs.
    if not isinstance(grid_size, int):
        raise TypeError('grid_size must be an integer')
    if not isinstance(inf_repl, numbers.Number):
        raise TypeError('inf_repl must be a number')
    # Replace infinite bound values (-np.inf and np.inf) with finite value specified in inf_repl.
    lower = np.zeros(len(bounds))
    upper = np.zeros(len(bounds))
    for i, bound in enumerate(bounds):
        lb, ub = bound
        if np.isinf(lb):
            lower[i] = -inf_repl
        else:
            lower[i] = lb
        if np.isinf(ub):
            upper[i] = inf_repl
        else:
            upper[i] = ub
    finite_bounds = np.array(list(zip(lower, upper)))
    # Create a list of arrays containing evenly spaced numbers spanning each bounded dimension.
    bound_arrays = [np.linspace(*bound, grid_size) for bound in finite_bounds]
    # Create list of points from the cartesian product of all bound arrays covering entire bounded space.
    list_of_points = np.array(list(itertools.product(*bound_arrays)))
    # Evaluate penalty function at each point in list_of_points.
    penalty_function_values = np.apply_along_axis(_penalty, axis=1, arr=list_of_points, bounds=bounds,
                                                  constraints=constraints)
    # Create summary statistics and list of points where all constraints are satisfied.
    number_of_points = len(list_of_points)
    count_penalty_function_violated = np.count_nonzero(penalty_function_values)
    fraction_violated = count_penalty_function_violated/number_of_points
    indices_violated = np.nonzero(penalty_function_values)
    feasible_points = np.delete(list_of_points, indices_violated, axis=0)
    return fraction_violated, np.array(feasible_points)


def feasible_points_random(bounds, constraints=None, point_count=50, max_iter=None, inf_repl=10.0**4):
    """Generate an array of feasible points using random sampling which satisfy the bounds and constraints.

    The variables point_count, max_iter, and inf_repl specify how the point generation procedure functions.  The
    procedure randomly generates a point, evaluates the point against both the bounds and constraints, and stores that
    point if it satisfies both the bounds and constraints.  The loop continues until the point_count is satisfied or
    until max_iter has been reached.  If the loop terminates due to reaching the max_iter criteria, that is an
    indication that the feasible space (the region where both bounds and constraints are both satisfied) is a small
    region inside of the bounded space.  To overcome this problem, you can increase max_iter from it's default value
    (which is point_count*100) to a much larger value.  If any of the bound tuples contains +/- np.inf, this means that
    there is no boundary for that particular dimension of the problem space.  The procedure replaces 'None' with the
    value specified in inf_repl to create a bounded region for the point generation procedure to work inside.

    Args:
        bounds (list): Validated list of bound tuples (i.e. bounds returned by bounds_check).
        constraints (list): Validated list of constraint dictionaries (i.e. constraints returned by constraints_check).
        point_count(int): Number of random points to be generated.
        max_iter (int): Termination criteria based on maximum number of iterations.
        inf_repl (scalar): Value used to replace +/- np.inf in any of the bound tuples specified in bounds.

    Returns:
        np.array: Array of points that satisfy the bounds and constraints.

    Raises:
        RuntimeWarning: len(feasible_points) is less than the specified point_count
    """
    # Validate input variables
    if not isinstance(point_count, int):
        raise TypeError('point_count must be an integer')
    if max_iter is None:
        max_iter = point_count*1000
    if not isinstance(max_iter, int):
        raise TypeError('max_iter must be an integer')
    if not isinstance(inf_repl, numbers.Number):
        raise TypeError('inf_repl must be a number')
    # Replace infinite bound values (-np.inf and np.inf) with finite value specified in inf_repl.
    lower = np.zeros(len(bounds))
    upper = np.zeros(len(bounds))
    for i, bound in enumerate(bounds):
        lb, ub = bound
        if np.isinf(lb):
            lower[i] = -inf_repl
        else:
            lower[i] = lb
        if np.isinf(ub):
            upper[i] = inf_repl
        else:
            upper[i] = ub
    finite_bounds = np.array(list(zip(lower, upper)))
    # Create lists to hold the feasible and infeasible points that are generated.
    feasible_points = []
    infeasible_points = []
    # Generate random values within the limits defined by bounds and evaluate if each point satisfies constraints.
    while len(feasible_points) < point_count and len(feasible_points) + len(infeasible_points) < max_iter:
        random_vector = [np.random.uniform(bound[0], bound[1]) for bound in finite_bounds]
        penalty_function_value = _penalty(random_vector, bounds=bounds, constraints=constraints)
        if penalty_function_value == 0.0:
            feasible_points.append(random_vector)
        else:
            infeasible_points.append(random_vector)
    # Check that the loop did not exit due to the max_iter criteria.
    if len(feasible_points) < point_count:
        raise RuntimeWarning('len(feasible_points) is less than the specified point_count')
    # Create summary statistics.
    number_of_points = len(feasible_points) + len(infeasible_points)
    fraction_violated = len(infeasible_points)/number_of_points
    return fraction_violated, np.array(feasible_points)


def best_point(points, func, args=None, kwargs=None):
    """Return the point corresponding to the lowest evaluated value of func.

    Args:
        points (np.array): Array of feasible points (i.e. np.array returned by feasible_points_random).
        func (callable): Scalar function to be minimized.  Signature must be func(x, *args, **kwargs) --> scalar.
        args (tuple, optional): Additional positional arguments required by func (if any).
        kwargs (dict, optional): Additional keyword arguments required by func (if any).

    Returns:
        np.array: Point corresponding to the lowest evaluated value of func.

    """
    args = args or ()
    kwargs = kwargs or {}
    func_values = np.apply_along_axis(func, 1, points, *args, **kwargs)
    ordered = np.argsort(func_values)
    return points[ordered[0], :]


def _create_simplex(initial, size):
    """Create initial simplex for the nelder_mead function."""
    n = len(initial)
    p = (size/(n*2**0.5))*(n - 1 + (n + 1)**0.5)
    q = (size/(n*2**0.5))*(-1 + (n + 1)**0.5)
    identity = np.identity(n)
    result = np.zeros((n, n))
    for i in range(n):
        result[i] += initial + p*identity[i]
        for j in range(n):
            if j == i:
                continue
            result[i] += q*identity[j]             
    return np.row_stack((initial, result))


def _infinity_check(x):
    """ Check for +/- np.inf elements. """
    element_check = np.isinf(x)
    return np.any(element_check)


def nelder_mead(x0, func, args=None, kwargs=None, bounds=None, constraints=None, small_tol=10.0**-15,
                flat_tol=10.0**-13, max_iter=10000, max_bisect_iter=100, initial_size=0.01):
    """Minimize a scalar function using the Nelder-Mead simplex algorithm.

    Implementation details can be found in...

    Gao, F., & Han, L. (2010). Implementing the Nelder-Mead simplex algorithm with adaptive parameters. Computational
        Optimization and Applications, 51(1), 259–277. https://doi.org/10.1007/s10589-010-9329-3

    Args:
        x0 (list): Vector representing the initial starting point for optimization algorithm.
        func (callable): Scalar function to be minimized.  Signature must be func(x, *args, **kwargs) --> scalar.
        args (tuple, optional): Additional positional arguments required by func (if any).
        kwargs (dict, optional): Additional keyword arguments required by func (if any).
        bounds (list): Validated list of bound tuples (i.e. bounds returned by bounds_check).
        constraints (list): Validated list of constraint dictionaries (i.e. constraints returned by constraints_check).
        small_tol (scalar, optional): Termination criteria based on distance between best and worst point in simplex.
        flat_tol (scalar, optional): Termination criteria based on distance between best and worst point in simplex.
        max_iter (int, optional): Termination criteria based on maximum number of algorithm iterations.
        max_bisect_iter (int, optional): Termination criteria for bisection loop in initial simplex generation.
        initial_size (scalar, optional): Initial simplex size.

    Returns:
        (np.array): Vector representing the local minimum of func.

    Raises:
        TypeError: x0 must be a list of scalars or a numpy array
        TypeError: func must be callable
        TypeError: max_iter must be an integer
        TypeError: max_bisect_iter must be an integer
        TypeError: small_tol must be a number
        TypeError: flat_tol must be a number
        TypeError: initial_size must be a number
        TypeError: args must be a tuple
        TypeError: kwargs must be a dictionary
        ValueError: x0 must be inside the problem space defined by the bounds and constraints
        ValueError: x0 is too close to the edge of the problem space defined by the bounds and constraints
    """
    # Validate bounds list and constraints dictionary are formatted correctly
    n = len(x0)
    bound = bounds_check(n, bounds)
    const = constraints_check(constraints)
    # Validate input variable types
    args = args or ()
    kwargs = kwargs or {}
    if not isinstance(x0, (list, np.ndarray)):
        raise TypeError('x0 must be a list of scalars or a numpy array')
    if not callable(func):
        raise TypeError('func must be callable')
    if not isinstance(max_iter, int):
        raise TypeError('max_iter must be an integer')
    if not isinstance(max_bisect_iter, int):
        raise TypeError('max_bisect_iter must be an integer')
    if not isinstance(small_tol, numbers.Number):
        raise TypeError('small_tol must be a number')
    if not isinstance(flat_tol, numbers.Number):
        raise TypeError('flat_tol must be a number')
    if not isinstance(initial_size, numbers.Number):
        raise TypeError('initial_size must be a number')
    if not isinstance(args, tuple):
        raise TypeError('args must be a tuple')
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs must be a dict')
    # Validate the initial point is in the problem space defined by bounds and constraints.
    if _infinity_check(_penalty(x0, bound, const)):
        raise ValueError('x0 must be inside the problem space defined by the bounds and constraints')
    # Initialize simplex.
    simplex = _create_simplex(x0, initial_size)
    f_simplex = np.apply_along_axis(_penalized_func, 1, simplex, func, args=args, kwargs=kwargs, bounds=bound,
                                    constraints=const)
    # Check that the objective function evaluates to a finite value for all points in the simplex. If the objective
    # function evaluates to +/- np.inf, then this signals a bound or constraint violation.  One reason this can
    # happen is that the simplex is too big to fit inside the bounded and constrained problem space.  A good check
    # to see if bounds and constraints that define the problem space are reasonably 'well conditioned' is to run
    # feasible_points_random and check that fraction_violated is in the range 0.05-1.00 (or so).  If fraction_
    # violated is <0.05, then we know that the bounded and constrained problem space (the feasible problem space) is
    # a tiny fraction of the bounded space which means we should re-evaluate the problem space passed to nedler_mead.
    # If fraction_violated is in the range 0.05-1.00, then we bisect the simplex size, generate a new simplex, and
    # re-evaluate the objective function at all points. If the bisection routine doesn't yield a valid simplex (where
    # the objective function is finite for all points) after 100 iterations, then the initial point could be very
    # close to a bound or constraint. If this is the case, then restart the nelder_mead algorithm with an new x0 value
    # that is further away from the bounds and/or constraints that define the problem space.
    counter = 0
    while _infinity_check(f_simplex) is True and counter <= max_bisect_iter:
        initial_size = initial_size/2.0
        simplex = _create_simplex(x0, initial_size)
        f_simplex = np.apply_along_axis(_penalized_func, 1, simplex, func, args=args, kwargs=kwargs, bounds=bound,
                                        constraints=const)
        counter = counter + 1
    if counter >= max_bisect_iter:
        raise ValueError('x0 is too close to the edge of the problem space defined by the bounds and constraints')
    ordered = np.argsort(f_simplex)
    # Calculate adaptive parameters improve convergence for higher dimensional problems
    n = len(simplex[0])
    r = 1.0  # reflection (standard method = 1)
    e = 1.0 + 2.0/n  # expansion (standard method = 2)
    c = 0.75 - 1.0/(2.0*n)  # contraction (standard method = 1/2)
    s = 1.0 - 1.0/n  # shrink (standard method = 1/2)
    # Initialize termination variables
    small = 1.0
    flat = 1.0
    counter = 0
    # Initialize performance tracking variables
    reflection_count = 0
    expansion_count = 0    
    outside_contraction_count = 0
    inside_contraction_count = 0
    shrink_count = 0
    # Begin Nelder-Mead iterations.
    while small > small_tol and flat > flat_tol and counter < max_iter:
        # Worst, second worst, and best simplex points.
        lowest = simplex[ordered[0], :]
        # second_highest = simplex[ordered[-2], :]
        highest = simplex[ordered[-1], :]
        centroid = simplex[ordered[0:-1], :].mean(axis=0)
        # Objective function evaluated at each simplex point from above.
        f_highest = f_simplex[ordered[-1]]
        f_second_highest = f_simplex[ordered[-2]]
        f_lowest = f_simplex[ordered[0]]
        # Evaluate reflection.
        reflection = centroid + r*(centroid - highest)
        f_reflection = _penalized_func(reflection, func, args=args, kwargs=kwargs, bounds=bound, constraints=const)
        # Compare reflection, expansion, contraction, and shrink operations.
        if f_reflection < f_lowest:
            # Evaluate expansion.
            expansion = centroid + e*(reflection - centroid)
            f_expansion = _penalized_func(expansion, func, args=args, kwargs=kwargs, bounds=bound, constraints=const)
            if f_expansion < f_reflection:
                # Replace highest by expansion
                simplex[ordered[-1], :] = expansion
                f_simplex[ordered[-1]] = f_expansion
                expansion_count += 1
            else:
                # Replace highest by reflection
                simplex[ordered[-1], :] = reflection
                f_simplex[ordered[-1]] = f_reflection
                reflection_count += 1
        elif f_reflection < f_second_highest:
            # Replace highest by reflection
            simplex[ordered[-1], :] = reflection
            f_simplex[ordered[-1]] = f_reflection
            reflection_count += 1
        elif f_reflection < f_highest:
            # Evaluate outside contraction.
            outside_contraction = centroid + c*(reflection - centroid)
            f_outside_contraction = _penalized_func(outside_contraction, func, args=args, kwargs=kwargs, bounds=bound,
                                                    constraints=const)
            if f_outside_contraction < f_reflection:
                # Replace highest by reflection
                simplex[ordered[-1], :] = outside_contraction
                f_simplex[ordered[-1]] = f_outside_contraction
                outside_contraction_count += 1
            else:
                # Replace all but best by shrink
                for i in ordered:
                    simplex[i, :] = lowest + s*(simplex[i, :] - lowest)
                f_simplex = np.apply_along_axis(_penalized_func, 1, simplex, func, args=args, kwargs=kwargs,
                                                bounds=bound, constraints=const)
                shrink_count += 1
        else:
            # Evaluate inside contraction.
            inside_contraction = centroid - c*(reflection - centroid)
            f_inside_contraction = _penalized_func(inside_contraction, func, args=args, kwargs=kwargs, bounds=bound,
                                                   constraints=const)
            if f_inside_contraction < f_highest:
                # Replace highest by contraction
                simplex[ordered[-1], :] = inside_contraction
                f_simplex[ordered[-1]] = f_inside_contraction
                inside_contraction_count += 1
            else:
                # Replace all but best by shrink
                for i in ordered:
                    simplex[i, :] = lowest + s*(simplex[i, :] - lowest)
                f_simplex = np.apply_along_axis(_penalized_func, 1, simplex, func, args=args, kwargs=kwargs,
                                                bounds=bound, constraints=const)
                shrink_count += 1
        # Evaluate termination criteria.
        ordered = np.argsort(f_simplex)
        if _infinity_check(f_simplex) is False:
            flat = np.absolute(f_simplex[ordered[-1]] - f_simplex[ordered[0]])
            small = np.linalg.norm(simplex[ordered[-1]] - simplex[ordered[0]])
        counter = counter + 1
    return simplex[ordered[0]]


def particle_swarm(func, args=None, kwargs=None, bounds=None, constraints=None, small_tol=10.0**-11, flat_tol=10.0**-9,
                   max_iter=2000, neighborhood_size=5, swarm_size=50):
    """Minimize a scalar function using the Particle Swarm algorithm.

    Local-best algorithm with ring social network structure.  Implementation details can be found in...

    Particle Swarm Optimization. (n.d.). In Computational Intelligence (pp. 289–358). John Wiley & Sons, Ltd.
        https://doi.org/10.1002/9780470512517.ch16

    Args:
        func (callable): Scalar function to be minimized. Signature must be func(x, *args, **kwargs) --> scalar.
        args (tuple, optional): Additional positional arguments required by func (if any).
        kwargs (dict, optional): Additional keyword arguments required by func (if any).
        bounds (list): Validated list of bound tuples (i.e. bounds returned by bounds_check).
        constraints (list): Validated list of constraint dictionaries (i.e. constraints returned by constraints_check).
        small_tol (scalar, optional): Termination criteria based on distance between best and worst point in simplex.
        flat_tol (scalar, optional): Termination criteria based on distance between best and worst point in simplex.
        max_iter (int, optional): Termination criteria based on maximum number of algorithm iterations.
        neighborhood_size (int, optional): Neighborhood size for local-best algorithm.
        swarm_size (int, optional): Number of particles in the swarm.

    Returns:
        np.array: Vector representing the local minimum of func.

    Raises:
        TypeError: func must be callable
        TypeError: max_iter must be an integer
        TypeError: neighborhood_size must be an integer
        TypeError: swarm_size must be an integer
        TypeError: small_tol must be a number
        TypeError: flat_tol must be a number
        TypeError: args must be a tuple
        TypeError: kwargs must be a dictionary
        ValueError: neighborhood_size must be less than or equal to swarm_size

    TODO: return nelder_mead_initial_size (need to look at how this impacts existing use cases and test cases).
    """
    # Validate bounds list and constraints dictionary are formatted correctly
    n = len(bounds)
    bound = bounds_check(n, bounds)
    const = constraints_check(constraints)
    # Validate input variable types
    args = args or ()
    kwargs = kwargs or {}
    if not callable(func):
        raise TypeError('func must be callable')
    if not isinstance(max_iter, int):
        raise TypeError('max_iter must be an integer')
    if not isinstance(neighborhood_size, int):
        raise TypeError('neighborhood_size must be an integer')
    if not isinstance(swarm_size, int):
        raise TypeError('swarm_size must be an integer')
    if not isinstance(small_tol, numbers.Number):
        raise TypeError('small_tol must be a number')
    if not isinstance(flat_tol, numbers.Number):
        raise TypeError('flat_tol must be a number')
    if not isinstance(args, tuple):
        raise TypeError('args must be a tuple')
    if not isinstance(kwargs, dict):
        raise TypeError('kwargs must be a dict')
    # Validate neighborhood size relative to swarm size
    if neighborhood_size > swarm_size:
        raise ValueError("neighborhood_size must be less than or equal to swarm_size")
    # Initialize swarm with random points that satisfy conditions laid out in bounds and constraints.
    fraction_violated, feasible_points = feasible_points_random(bounds=bound, constraints=const,
                                                                point_count=swarm_size)
    # Initialize particle swarm algorithm constants.
    swarm_size = len(feasible_points)
    dimension = len(feasible_points[0])
    cognitive_parameter = 1.49
    social_parameter = 1.49
    velocity_weight = 0.73
    # Initialize swarm position and velocity.
    current_position = feasible_points.copy()
    current_velocity = np.zeros(shape=(swarm_size, dimension))
    # Initialize personal best and neighborhood best variables.
    personal_best_position = np.zeros(shape=(swarm_size, dimension))
    personal_best_value = np.full(shape=swarm_size, fill_value=np.inf)
    neighborhood_best_position = np.zeros(shape=(swarm_size, dimension))
    neighborhood_best_value = np.full(shape=swarm_size, fill_value=np.inf)
    # The distance between the best two swarm points is a great estimate for initial_size in the Nelder-Mead algorithm.
    nelder_mead_initial_size = 0.0
    # Begin particle swarm iterations.
    counter = 0
    while counter < max_iter:
        # Calculate function values at current swarm position.
        current_combined_value = np.apply_along_axis(_penalized_func, 1, current_position, func,
                                                     args=args, kwargs=kwargs, bounds=bound, constraints=const)
        # Evaluate termination variables.
        ordered = np.argsort(current_combined_value)
        flat = np.absolute(current_combined_value[ordered[0]] - current_combined_value[ordered[1]])
        small = np.linalg.norm(current_position[ordered[0]] - current_position[ordered[1]])
        nelder_mead_initial_size = small
        # Evaluate if termination criteria are met and break while loop if so.
        if small < small_tol:
            break
        elif flat < flat_tol:
            break
        # Update personal best values and positions.
        personal_best_value_update = np.less(current_combined_value, personal_best_value)
        personal_best_position_update = np.tile(personal_best_value_update, (dimension, 1)).transpose()
        personal_best_value = np.where(personal_best_value_update, current_combined_value, personal_best_value)
        personal_best_position = np.where(personal_best_position_update, current_position, personal_best_position)
        # Update neighborhood best values and positions.
        for i in np.arange(swarm_size):
            positions = np.take(current_position, range(i, i + neighborhood_size), axis=0, mode='wrap')
            values = np.take(current_combined_value, range(i, i + neighborhood_size), axis=0, mode='wrap')
            ordered = np.argsort(values)
            neighborhood_best_position[i] = positions[ordered[0]]
            neighborhood_best_value[i] = values[ordered[0]]
        # Generate random numbers for use in evaluating velocity components.
        cognitive_random = np.random.uniform(size=(swarm_size, dimension))
        social_random = np.random.uniform(size=(swarm_size, dimension))
        # Evaluate velocity components.
        cognitive_component = cognitive_parameter*cognitive_random*(personal_best_position - current_position)
        social_component = social_parameter*social_random*(neighborhood_best_position - current_position)
        momentum_component = velocity_weight*current_velocity
        # Evaluate velocity and use velocity to calculate new position.
        new_velocity = momentum_component + social_component + cognitive_component
        new_position = current_position + new_velocity
        # Update current position and iteration variable
        current_position = new_position.copy()
        current_velocity = new_velocity.copy()
        counter = counter + 1
    final_ordered = np.argsort(personal_best_value)
    return personal_best_position[final_ordered[0]]


NUM_CPUS = multiprocessing.cpu_count()
NUM_PROCESSES = 1 if (NUM_CPUS - 1) <= 1 else NUM_CPUS


def _bootstrap_sample(array_size, sample_size):
    """
    Returns an array of integers which is a uniform random sample consisting of [sample_size] elements taken from an
    array of indices where n = [array_size].  This array of multipliers can be used to generate bootstrap confidence
    intervals.
    """
    indices = np.arange(array_size)
    samples = np.random.choice(indices, size=sample_size, replace=True)
    bootstrap_result = np.zeros(array_size, dtype=np.int32)
    bootstrap_indices, bootstrap_count = np.unique(samples, return_counts=True)
    bootstrap_result[bootstrap_indices] = bootstrap_count
    return bootstrap_result


def _least_squares_function_wrapper(argument):
    """Takes single argument, unpacks it to args and kwargs components, and passes them to func.

    This gets around the fact that mp.Pool.map() and mp.Pool.starmap() only take one iterable argument. Unfortunately,
    this doesn't allow us to pass multiple args and kwargs which is a problem.  Build a single argument from all input
    args and kwargs and then call func_wrapper.

    arguments = [(args, kwargs) for j in jobs_with_different_args_and_kwargs]

    This wrapper supports the least_squares_objective_function and will return one evaluated element of the objective
    function.  It also checks if either the weight or bootstrapping multipliers is zero before evaluating the func to
    improve efficiency.

    objective_function_element_i = (b_i*w_i)*(func(theta, xi, *args_i, **kwargs_i) - fx)**2
    """
    func, theta, x, fx, w, b, args, kwargs = argument
    # New call signature --> penalized_func(x, func, args=(), kwargs={}, bounds=None, constraints=None)
    # Old call signature --> func(x, *theta, *args, **kwargs)
    # Disagreement between structures requires us to combine theta and args into one tuple to pass.
    # arguments = np.array(list(zip(func_list, theta_list, x_array, fx_array, w_array, b_array, args_list, kwargs_list,
    #                               bound_list, const_list)))
    # Numpy converts all objects of type list to np.ndarray.  To make other parts of this program behave correctly,
    # we have to convert all these objects back to list.
    # if isinstance(theta, (np.ndarray, np.generic)):
    #     theta = theta.tolist()
    # if isinstance(x, (np.ndarray, np.generic)):
    #     x = x.tolist()
    # if isinstance(bound, (np.ndarray, np.generic)):
    #     bound = [tuple(i) for i in bound.tolist()]
    return (w * b) * (func(x, *theta, *args, **kwargs) - fx) ** 2.0 if w > 0.0 and b > 0 else 0.0


def least_squares_objective_function(theta, func, x, fx, w=None, b=None, args=None, kwargs=None, bounds=None,
                                     constraints=None):
    """Returns the scalar result of evaluation of the least squares objective function.

    least_squares_objective_function = sum_over_i{(b_i*w_i)*(func(theta, xi, *args_i, **kwargs_i) - fx_i)**2}

    The least squares objective function is the core of parameter fitting.  This implementation facilitates weights
    as well as bootstrapping. The difference between fx and func(theta, xi, *args_i, **kwargs_i) is a measure  of the
    goodness of fit of func to the data.

    In this implementation, the objective function is broken down into pieces to improve code structure as well
    as to facilitate parallel processing.  The function structure allows us to evaluate each result_i in an
    embarrassingly parallel fashion if the objection function is very expensive to evaluate.

    least_squares_objective_function = sum_over_i{result_i}
    result_i = (b_i*w_i)*(func(theta, xi, *args_i, **kwargs_i) - fx)**2

    The impact of bounds and constraints is added after the least_squares_objective_function has been evaluated.

    The user can toggle between 'multiprocess = False' and 'multiprocess = True' to test and see if there is a
    performance improvement. The Python multiprocessing library often results in slower performance because of the
    system overhead required to manage multiple processes.

    Args:
        theta (list): Vector representing the starting point for optimization algorithm.
        x (list): List of tuples or list of lists representing input values of a data set.
        fx (list): List of scalars representing output values of a data set.
        w (list, optional): List of scalars representing the weight (often '1.0/std_error_i').
        b (list, optional): List of integers representing the bootstrap multiplier.
        func (callable): Scalar function to be minimized. Signature must be func(x, *theta *args, **kwargs) --> scalar.
        args (tuple, optional): Additional positional arguments required by func.
        kwargs (dict, optional): Additional keyword arguments required by func.
        bounds (list): Validated list of bound tuples (i.e. bounds returned by bounds_check).
        constraints (list): Validated list of constraint dictionaries (i.e. constraints returned by constraints_check).

    Returns:
        scalar: objective_function_value

    Raises:
        TypeError: args must be a list or tuple of length len(x) containing args lists or args tuples
        TypeError: kwargs must be a list or tuple of length len(x) containing kwargs dictionaries
    """
    x_array = np.array(x)
    if len(x) == len(fx):
        fx_array = np.array(fx)
    else:
        raise TypeError("fx must be a list of length len(x)")
    if w is None:
        w_array = np.ones(len(x))
    elif len(x) == len(w):
        w_array = np.array(w)
    else:
        raise TypeError("w must be a list of length len(x)")
    if b is None:
        b_array = np.ones(len(x))
    elif len(x) == len(b):
        b_array = np.array(b)
    else:
        raise TypeError("b must be a list of length len(x)")
    func_list = [func] * len(x)
    theta_list = [theta] * len(x)
    if args is None:
        args_list = [()] * len(x)
    elif isinstance(args, (list, tuple)) and len(x) == len(args):
        args_list = args
    else:
        raise TypeError("args must be a list or tuple of length len(x) containing args lists or args tuples")
    if kwargs is None:
        kwargs_list = [{}] * len(x)
    elif isinstance(kwargs, (list, tuple)) and len(x) == len(kwargs):
        kwargs_list = kwargs
    else:
        raise TypeError("kwargs must be a list or tuple of length len(x) containing kwargs dictionaries")
    n = len(theta)
    bound = bounds_check(n, bounds)
    const = constraints_check(constraints)
    arguments = np.array(list(zip(func_list, theta_list, x_array, fx_array, w_array, b_array, args_list, kwargs_list)))
    # multiprocess option makes more sense for bootstrapping rather than least squares objective function.  Simplify
    # my life and remove this at the moment.  It is a headache to have a chain of functions where both parent and child
    # functions have the ability to spawn processes.  Pain >> Gain
    # if multiprocess:
    #     with multiprocessing.Pool(NUM_PROCESSES) as p:
    #         results = p.map(_function_wrapper, arguments)
    # else:
    results = np.apply_along_axis(_least_squares_function_wrapper, 1, arguments)
    return np.sum(results) + _penalty(theta, bound, const)


def _least_squares_bootstrap_function_wrapper(argument):
    """Takes single argument, unpacks it to args and kwargs components, and passes them to func.

    This gets around the fact that mp.Pool.map() and mp.Pool.starmap() only take one iterable argument. Unfortunately,
    this doesn't allow us to pass multiple args and kwargs which is a problem.  Build a single argument from all input
    args and kwargs and then call func_wrapper.

    arguments = [(args, kwargs) for j in jobs_with_different_args_and_kwargs]

    This wrapper supports bootstrap iterations for repreat optimization of the least_squares_objective_function.
    """
    func, theta, x, fx, w, b, args, kwargs, bound, const, small, flat, max, max_bisect, initial_size = argument
    result = nelder_mead(theta, least_squares_objective_function,
                         args=(func, x, fx),
                         kwargs={'w': w,
                                 'b': b,
                                 'args': args,
                                 'kwargs': kwargs,
                                 'bounds': bound,
                                 'constraints': const},
                         small_tol=small,
                         flat_tol=flat,
                         max_iter=max,
                         max_bisect_iter=max_bisect,
                         initial_size=initial_size)
    return result


def least_squares_bootstrap(theta, func, x, fx, weight=None, args=None, kwargs=None, bounds=None, constraints=None,
                            multiprocess=False, samples=500, small_tol=10.0**-15, flat_tol=10.0**-13, max_iter=10000,
                            max_bisect_iter=100, initial_size=0.01):
    """Returns list of tuples containing the results (thetas) of repeated least squares fitting of func to x and fx.

    Repeated evaluation of the 'least_squares_objective_function' where the 'bootstrap' multiplier is used to drive
    random sampling from 'x', 'fx', and 'weight' with replacement.  THe bootstrap multiplier ('bi') is implemented in
    a separate function and returns a list of scalar multipliers that allow random sampling with replacement. This
    resulting list of tuples can be used to estimate the distribution of each element of theta.

    least_squares_objective_function = sum_over_i{(b_i*w_i)*(func(theta, xi, *args_i, **kwargs_i) - fx_i)**2}

    Args:
        theta (list): Vector representing the initial starting point for optimization algorithm.
        func (callable): Scalar function to be minimized, ``func(x, *theta, *args, **kwargs)``.
        x (list): List of tuples or list of lists representing input values of a data set for func.
        fx (list): List of scalars representing output values of a data set for func.
        weight (list, optional): List of scalars representing the weight.
        args (tuple, optional): Additional positional arguments required by func (if any).
        kwargs (dict, optional): Additional keyword arguments required by func.
        bounds (list): Validated list of bound tuples (i.e. bounds returned by bounds_check).
        constraints (list): Validated list of constraint dictionaries (i.e. constraints returned by constraints_check).
        samples (int, optional): Specification for the number of bootstrap samples to be run.
        multiprocess (bool, optional): Boolean indicator that enables parallel processing.
        small_tol (scalar, optional): Nelder-Mead algorithm termination criteria.
        flat_tol (scalar, optional): Nelder-Mead algorithm termination criteria.
        max_iter (int, optional): Nelder-Mead algorithm termination criteria.
        max_bisect_iter (int, optional): Termination criteria for simplex generation loop in Nelder-Mead algorithm.
        initial_size (scalar, optional): Initial simplex size in Nelder-Mead algorithm.

    Returns:
        (list of tuples): Vector containing the results of repeated least squares fitting of func to x and fx.
    """
    n = len(x)
    b_list = [_bootstrap_sample(n, n) for _ in range(samples)]
    x_list = [x] * samples
    if len(x) == len(fx):
        fx_list = [fx] * samples
    else:
        raise TypeError("fx must be a list of length len(x)")
    if weight is None:
        w_list = [np.ones(len(x))] * samples
    elif len(x) == len(weight):
        w_list = [weight] * samples
    else:
        raise TypeError("weight must be a list of length len(x)")
    func_list = [func] * samples
    theta_list = [theta] * samples
    if args is None:
        args_list = [[()] * len(x)] * samples
    elif isinstance(args, (list, tuple)) and len(x) == len(args):
        args_list = [args] * samples
    else:
        raise TypeError("args must be a list or tuple of length len(x) containing args lists or args tuples")
    if kwargs is None:
        kwargs_list = [[{}] * len(x)] * samples
    elif isinstance(kwargs, (list, tuple)) and len(x) == len(kwargs):
        kwargs_list = [kwargs] * samples
    else:
        raise TypeError("kwargs must be a list or tuple of length len(x) containing kwargs dictionaries")
    bound = bounds_check(len(theta), bounds)
    const = constraints_check(constraints)
    bound_list = [bound] * samples
    const_list = [const] * samples
    small_tol_list = [small_tol] * samples
    flat_tol_list = [flat_tol] * samples
    max_iter_list = [max_iter] * samples
    max_bisect_iter_list = [max_bisect_iter] * samples
    initial_size_list = [initial_size] * samples
    arguments = np.array(list(zip(func_list, theta_list,
                                  x_list, fx_list, w_list, b_list,
                                  args_list, kwargs_list, bound_list, const_list,
                                  small_tol_list, flat_tol_list,
                                  max_iter_list, max_bisect_iter_list, initial_size_list)))
    if multiprocess:
        with multiprocessing.Pool(NUM_PROCESSES) as p:
            result = p.map(_least_squares_bootstrap_function_wrapper, arguments)
    else:
        result = np.apply_along_axis(_least_squares_bootstrap_function_wrapper, 1, arguments)
    return result
