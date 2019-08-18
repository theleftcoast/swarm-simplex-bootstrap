import optimizer as opt
import numpy as np

# ------------------------------
# TEST FUNCTIONS FOR MINIMIZATION
# ------------------------------


def booth(args):
    """ Booth function

    Global minimum: f(1.0,3.0) = 0.0
    Search domain: -10.0 <= x, y <= 10.0
    """
    return (args[0] + 2*args[1] - 7)**2 + (2*args[0] + args[1] - 5)**2


def rosenbrock(args):
    """Rosenbrock function

    Global minimum: f(1,...,1) = 0.0
    Search domain: -inf <= xi <= inf, 1 <= i <= n
    """
    rosen = 0
    for i in range(len(args) - 1):
        rosen += 10.0*((args[i]**2) - args[i + 1])**2 + (1 - args[i])**2
    return rosen


def ackley(args):
    """Ackley function

    Global minimum: f(0,0) = 0.0
    Search domain: -5.0 <= x, y <= 5.0
    """
    first_sum = 0.0
    second_sum = 0.0
    for c in args:
        first_sum += c ** 2.0
        second_sum += np.cos(2.0 * np.pi * c)
    n = float(len(args))
    return -20.0*np.exp(-0.2*np.sqrt(first_sum/n)) - np.exp(second_sum/n) + 20.0 + np.e


# ------------------------------
# NELDER-MEAD MINIMIZATION
# ------------------------------
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
# Bounded Minimization with Nelder-Mead Simplex, Ackley function example (convergence failure)
initial_pt = [2.0, 2.0]
func = ackley
minimum = opt.nelder_mead(initial_pt, func, bounds=finite_bounds)
print(minimum)
# Bounded Minimization with Nelder-Mead Simplex, Ackley function example (successful convergence)
initial_pt = [0.1, 0.1]
func = ackley
minimum = opt.nelder_mead(initial_pt, func, bounds=finite_bounds)
print(minimum)

# ------------------------------
# PARTICLE SWARM MINIMIZATION
# ------------------------------
# Bounded Minimization with Particle Swarm, Booth function example (successful convergence)
func = booth
minimum, _ = opt.particle_swarm(func, bounds=finite_bounds)
print(minimum)
# Bounded Minimization with Particle Swarm, Rosenbrock function example (successful convergence)
func = rosenbrock
minimum, _ = opt.particle_swarm(func, bounds=finite_bounds)
print(minimum)
# Bounded Minimization with Particle Swarm, Ackley function example (successful convergence)
func = ackley
minimum, _ = opt.particle_swarm(func, bounds=finite_bounds)
print(minimum)

# ------------------------------
# PARTICLE SWARM INITIAL MINIMIZATION WITH NELDER-MEAD REFINEMENT
# ------------------------------
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


def cubic(args):
    """Cubic curve"""
    return (args[0] - 1.0) ** 3 - args[1] + 1.0


def line(args):
    """Line"""
    return args[0] + args[1] - 2.0


def circle(args):
    """Circle"""
    return args[0]**2 + args[1]**2 - 2.0


const_a = [{'type': '<=0', 'func': cubic},
           {'type': '<=0', 'func': line}]
const_b = [{'type': '<=0', 'func': circle}]
const_a_checked = opt.constraints_check(const_a)
const_b_checked = opt.constraints_check(const_b)
print(const_a_checked)
print(const_b_checked)

# ------------------------------
# BOUNDED AND CONSTRAINED MINIMIZATION
# ------------------------------
# Bounded and Constrained Minimization with Combined Procedure, Booth function example (successful convergence)
func = booth
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_a)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_a,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)
# Bounded and Constrained Minimization with Combined Procedure, Booth function example (successful convergence)
func = booth
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_b)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_b,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)
# Bounded and Constrained Minimization with Combined Procedure, Rosenbrock function example (successful convergence)
func = rosenbrock
ps_minimum, nelder_mead_initial_size = opt.particle_swarm(func, bounds=finite_bounds, constraints=const_a)
nm_minimum = opt.nelder_mead(ps_minimum, func, bounds=finite_bounds, constraints=const_a,
                             initial_size=nelder_mead_initial_size)
print(nm_minimum)
# Bounded and Constrained Minimization with Combined Procedure, Rosenbrock function example (successful convergence)
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
# MODEL Regression
# ------------------------------


# Function to be fit.
def quadratic(args, a, b, c, d, e, f):
    """General quadratic function"""
    return a*args[0]**2 + b*args[1]**2 + c*args[0] + d*args[1] + e*args[0]*args[1] + f


# 'x' vector.
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
# 'fx' values
fx = [0.16, 0.34, 1.04, 2.26, 4.0, 0.16, 0.04, 0.26, 1.0, 2.26, 1.04, 0.26, 0.0, 0.26, 1.04, 2.26, 1.0, 0.26, 0.04,
      0.34, 4.3, 2.26, 1.04, 0.34, 0.16]
# Regression procedure
func = quadratic
theta_initial_guess = [0.26, 0.26, 0.0, 0.0, -0.48, 0.0]
theta_exact = [0.268864, 0.268387, -0.012410, 0.011811, -0.491811, -0.022094]
theta = opt.nelder_mead(theta_initial_guess, opt.least_squares_objective_function, args=(func, x, fx))
print(theta)
# Even weights
even_weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
               0.5, 0.5, 0.5, 0.5]
theta = opt.nelder_mead(theta_initial_guess, opt.least_squares_objective_function, args=(func, x, fx, even_weight))
print(theta)
# Uneven weights
uneven_weight = [0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.3, 0.4,
                 0.5, 0.3, 0.4, 0.5, 0.3]
theta = opt.nelder_mead(theta_initial_guess, opt.least_squares_objective_function, args=(func, x, fx, uneven_weight))
print(theta)

# ------------------------------
# BOOTSTRAPPING
# ------------------------------
bootstrap_set = opt.least_squares_bootstrap(theta_initial_guess, func, x, fx,
                                            weight=uneven_weight,
                                            bounds=None, constraints=None,
                                            multiprocess=False,
                                            samples=25, max_iter=1000)
print(bootstrap_set)
