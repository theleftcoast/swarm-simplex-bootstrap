import optimizer as opt
import numpy as np

# ------------------------------
# TEST FUNCTIONS FOR MINIMIZATION
# ------------------------------
def booth(args):
    """
    Global minimum: f(1.0,3.0) = 0.0
    Search domain: -10.0 <= x, y <= 10.0
    """
    x = args[0]
    y = args[1]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2


def rosenbrock(args):
    """
    Global minimum: f(1,...,1) = 0.0
    Search domain: -inf <= xi <= inf, 1 <= i <= n
    """
    rosen = 0
    for i in range(len(args ) -1):
        rosen += 10.0 *((args[i]**2 ) -args[ i +1] )** 2 +( 1 -args[i] )**2
    return rosen

def ackley(args):
    """
    Global minimum: f(0,0) = 0.0
    Search domain: -5.0 <= x, y <= 5.0
    """
    firstSum = 0.0
    secondSum = 0.0
    for c in args:
        firstSum += c ** 2.0
        secondSum += np.cos(2.0 * np.pi * c)
    n = float(len(args))
    return -20.0 * np.exp(-0.2 * np.sqrt(firstSum / n)) - np.exp(secondSum / n) + 20 + np.e

# ------------------------------
# NELDER-MEAD MINIMIZATION
# ------------------------------
# The nelder_mead algorithm is a very general direct search minimization method. The algorithm makes few assumptions
# about the function to be minimized (such as continuity or differentiability) so it is applicable to minimizing a
# wide range of functions.  The main weakness of the nelder_mead algorithm for complex problems is that it can
# prematurely converge to a local minimum in search of a global minium.

# Unconstrained Minimization with Nelder-Mead Simplex, Booth function example (successful convergence)
initial_pt = [2.0, 2.0]
func = booth
minimum = opt.nelder_mead(initial_pt, func)
print(minimum)

# Unconstrained Minimization with Nelder-Mead Simplex, Rosenbrock function example (successful convergence)
initial_pt = [2.0, 2.0]
func = rosenbrock
minimum = opt.nelder_mead(initial_pt, func)
print(minimum)

# Unconstrained Minimization with Nelder-Mead Simplex, Ackley function example (successful convergence)
initial_pt = [0.1, 0.1]
func = ackley
minimum = opt.nelder_mead(initial_pt, func)
print(minimum)

# ------------------------------
# BOUNDS
# ------------------------------
# A bound specifies the minimum and maximum values allowed for each of the 'n' dimensions of the problem space. These
# bounds define valid regions of the problem space where a minimization algorithm can search. If there are not any
# bounds for a particular dimension of the problem space (i.e. infinity or -infinity), then specify 'None' for that
# element of the bound.
#
# A valid bounds list is a list of bound tuples or bound lists.
#
#                  --> [(bound_tuple), ... ,(bound_tuple)]
#                  --> [[bound_list], ... ,[bound_list]]
#
# Bounds tuples and lists are defined using the following syntax.
#
#                  --> (min, max), (None, max), (min, None), (None, None)
#                  --> [min, max], [None, max], [min, None], [None, None]
#
# The 'bounds_check' function checks bounds lists for consistency and returns the list with basic problems corrected.
#
# Bounds specification is optional for the nelder_mead algorithm. However, bounds specification is very important for
# the particle_swarm algorithm because bounds are used to generate the initial particle swarm. BOUNDS SPECIFICATION IS
# REQUIRED when calling the particle_swarm algorithm.

infinite_bounds = [(None, None), (None, None)]
finite_bounds = [(-5.0, 5.0), (-5.0, 5.0)]

inf_bounds_size = len(infinite_bounds)
fin_bounds_size = len(finite_bounds)

inf_bounds_checked = opt.bounds_check(inf_bounds_size, infinite_bounds)
fin_bounds_checked = opt.bounds_check(fin_bounds_size, finite_bounds)

print(inf_bounds_checked)
print(fin_bounds_checked)

# ------------------------------
# BOUNDED NELDER-MEAD MINIMIZATION
# ------------------------------

# Bounded Minimization with Nelder-Mead Simplex, Booth function example (successful convergence)
initial_pt = [2.0, 2.0]
func = booth
minimum = opt.nelder_mead(initial_pt, func, bounds=finite_bounds)
print(minimum)

# Bounded Minimization with Nelder-Mead Simplex, Rosenbrock function example (successful convergence)
initial_pt = [2.0, 2.0]
func = rosenbrock
minimum = opt.nelder_mead(initial_pt, func, bounds=finite_bounds)
print(minimum)

# Bounded Minimization with Nelder-Mead Simplex, Ackley function example (convergence failure) --> This starting point
# is far from the global minimum for the Ackley function.  This causes the  Nelder-Mead algorithm to become trapped in
# a local minimum.
initial_pt = [2.0, 2.0]
func = ackley
minimum = opt.nelder_mead(initial_pt, func, bounds=finite_bounds)
print(minimum)

# Bounded Minimization with Nelder-Mead Simplex, Ackley function example (successful convergence) --> This starting
# point is close to the global minimum for the Ackley function.  There are no local minima between this starting point
# and the global minimum. The nelder_mead algorithm quickly converges to the global minimum.
initial_pt = [0.1, 0.1]
func = ackley
minimum = opt.nelder_mead(initial_pt, func, bounds=finite_bounds)
print(minimum)

# ------------------------------
# PARTICLE SWARM MINIMIZATION
# ------------------------------
# The particle_swarm algorithm is an evolutionary minimization method. The algorithm makes few assumptions about the
# function to be minimized (such as continuity or differentiability) so it is applicable to minimizing a wide range of
# functions. The main strength of the particle_swarm algorithm is that it effectively identifies the global minimum in
# problem spaces that contain many additional local minima. Though the algorithm can identify the global minimum for
# MOST problems problems, there is no guarantee that it will identify the global minimum for EVERY problem. The main
# weakness of the algorithm is that it is not efficient at converging to a tight estimate of the global minimum after
# the neighborhood of the global minimum is identified.
#
# Bounds specification is required for this implementation fo the particle_swarm algorithm.  This is because bounds are
# used to generate the initial particle swarm.

# Bounded Minimization with Particle Swarm, Booth function example (successful convergence)
func = booth
minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds)
print(minimum)

# Bounded Minimization with Particle Swarm, Rosenbrock function example (successful convergence)
func = rosenbrock
minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds)
print(minimum)

# Bounded Minimization with Particle Swarm, Ackley function example (successful convergence)
func = ackley
minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds)
print(minimum)

# ------------------------------
# PARTICLE SWARM INITIAL MINIMIZATION WITH NELDER-MEAD REFINEMENT
# ------------------------------
# The particle_swarm and nelder_mead algorithms can be used together to efficiently minimize complex objective
# functions. The particle_swarm algorithm  used first to find a good estimate for the neighborhood of the global
# minimum (loose convergence critera are used). The particle_swarm algorithm also generates an estimate for the initial
# size of the simplex in the nelder_mead algorithm. The initial estimate of the global minimum and simplex size are
# then passed to the nelder_mead algorithm.  The nelder_mead algorithm can converge to a tight estimate of the global
# minimum. Though this procedure can identify the global minimum for MOST problems problems, there is no guarantee that
# it will identify the global minimum for EVERY problem.

# Bounded Minimization with Combined Procedure, Booth function example (successful convergence)
func = booth
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, initial_size=nelder_mead_initial_size)
print(nm_minimum)

# Bounded Minimization with Combined Procedure, Rosenbrock function example (successful convergence)
func = rosenbrock
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, initial_size=nelder_mead_initial_size)
print(nm_minimum)

# Bounded Minimization with Combined Procedure, Ackley function example (successful convergence)
func = ackley
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, initial_size=nelder_mead_initial_size)
print(nm_minimum)

# ------------------------------
# CONSTRAINTS
# ------------------------------
# Constraints are functional requirements that define valid regions of the problem space that a minimization algorithm
# can search. A complete constraint specification must include a constraint function, any optional arguments (args and
# kwargs), and constraint type (inequality type). More than one constraint can be specified to define a problem space.
#
# A valid constraints list is a list of constraint dictionaries.
#
#                  --> [{const_dict}, ... ,{const_dict}]
#                  --> {'type': ineq_spec_string, 'func': callable_func, 'args': (args_tuple), 'kwargs': {kwargs_dict}}
#
# The 'constraints_check' function checks constraints lists for consistency and returns the list with basic problems
# corrected.


def func_a(x):
    """Cubic curve"""
    return (x[0] - 1.0) ** 3 - x[1] + 1.0


def func_b(x):
    """Line"""
    return x[0] + x[1] - 2.0


def func_c(x):
    """Circle"""
    return x[0]**2 + x[1]**2 - 2.0


const_a = [{'type': '<=0', 'func': func_a},
           {'type': '<=0', 'func': func_b}]

const_b = [{'type': '<=0', 'func': func_c}]

const_a_checked = opt.constraints_check(const_a)
const_b_checked = opt.constraints_check(const_b)

print(const_a_checked)
print(const_b_checked)

# ------------------------------
# BOUNDED AND CONSTRAINED MINIMIZATION
# ------------------------------
# It is straight forward to minimize a function after bounds and constraints have been specified. Either the
# nelder_mead or particle_swarm algorithms can be used with bounds and constraints. However, the combined procedure
# (particle_swarm with nelder_mead refinement) is recommended.

# Bounded and Constrained Minimization with Combined Procedure, Booth function example (successful convergence) --> The
# minimum of the booth function subject to these bounds and constraints is no longer the global minimum of the
# unconstrained booth function.  The bounded and constrained minimum lies on the edge of the constrained problem space.
func = booth
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_a)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_a,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)

# Bounded and Constrained Minimization with Combined Procedure, Booth function example (successful convergence) --> The
# minimum of the booth function subject to these bounds and constraints is no longer the global minimum of the
# unconstrained booth function.  The bounded and constrained minimum lies on the edge of the constrained problem space.
func = booth
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_b)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_b,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)

# Bounded and Constrained Minimization with Combined Procedure, Rosenbrock function example (successful convergence)
# --> The minimum of the rosenbrock function subject to these bounds and constraints is still the global minimum
# of the unconstrained rosenbrock function. The global minimum lies on the edge of the constrained problem space.
func = rosenbrock
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_a)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_a,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)

# Bounded and Constrained Minimization with Combined Procedure, Rosenbrock function example (successful convergence)
# --> The minimum of the rosenbrock function subject to these bounds and constraints is still the global minimum
# of the unconstrained rosenbrock function. The global minimum lies on the edge of the constrained problem space.
func = rosenbrock
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_b)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_b,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)

# Bounded and Constrained Minimization with Combined Procedure, Ackley function example (successful convergence)
func = ackley
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_a)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_a,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)

# ------------------------------
# MODEL PARAMETERIZATION
# ------------------------------


# Function to be fit.
def quadratic(x,a,b,c,d,e,f):
    """General quadratic function"""
    return a*x[0]**2 + b*x[1]**2 + c*x[0] + d*x[1] + e*x[0]*x[1] + f

# x vectors for the data set.
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

# fx values for the data set. --> These values were generated by passing the x vector to the 'quadratic' function and
# rounding the results. The coefficients that define the quadratic function were [0.26, 0.26, 0.0, 0.0, -0.48, 0.0].
# Rounding the exact fx output introduces small deviations so the resulting data set serves as a good 'quasi-
# experimental' test set (where x represents input conditions and fx represents measured responses which include noise).
fx = [0.16, 0.34, 1.04, 2.26, 4.0, 0.16, 0.04, 0.26, 1.0, 2.26, 1.04, 0.26, 0.0, 0.26, 1.04, 2.26, 1.0, 0.26, 0.04,
      0.34, 4.3, 2.26, 1.04, 0.34, 0.16]

func = quadratic
theta_initial_guess = [0.26, 0.26, 0.0, 0.0, -0.48, 0.0]
theta_exact = [0.268864, 0.268387, -0.012410, 0.011811, -0.491811, -0.022094]
theta = opt.nelder_mead(theta_initial_guess, opt.least_squares_objective_function, args=(func, x, fx))
print(theta)

# Weights can be added which allows additional influence to be attached to certain data points. Even weights for each
# term will yield the same result as unweighted least squares.
even_weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, 0.5]
theta = opt.nelder_mead(theta_initial_guess, opt.least_squares_objective_function, args=(func, x, fx, even_weight))
print(theta)

# Uneven weights will emphasize certain terms which will impact the parameterization.  Uneven weights often arise when
# fitting data where experimental uncertainty can be different for each measurement.
uneven_weight = [0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4,
                 0.5, 0.3, 0.4, 0.5, 0.3]
theta = opt.nelder_mead(theta_initial_guess, opt.least_squares_objective_function, args=(func, x, fx, uneven_weight))
print(theta)

# ------------------------------
# BOOTSTRAPPING
# ------------------------------
# The least_squares_bootstrap function drives repeated evaluation of the least_squares_objective_function where the
# input parameters to each evaluation are sampled from 'x', 'fx', and 'weight' with replacement.  The bootstrapping
# technique uses the results (i.e. fitted model parameters) from each repeat evaluation to derive summary statistics
# which describe the overall result set (i.e. fitted model parameters with their uncertainties).
bootstrap_set = opt.least_squares_bootstrap(theta_initial_guess, func, x, fx,
                                            weight=uneven_weight,
                                            bounds=None, constraints=None,
                                            multiprocess=False,
                                            samples=100, max_iter=1000)
print(bootstrap_set)