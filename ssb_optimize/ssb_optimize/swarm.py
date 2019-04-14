import numpy as np
import functools
import multiprocessing
from optimizer import bounds_check, constraints_check, penalty, feasible_points_random

DEBUG = True  # Controls debugging print statements

def particle_swarm(func,args=(),kwargs={}, bounds=None, constraints=None):

    # Initialize parallel processing variables
    cpu_count = multiprocessing.cpu_count()
    processes = 1 if (cpu_count - 1) <= 1 else cpu_count - 1
    mp_pool = multiprocessing.Pool(processes)

    # Initialize swarm with random points that satisfy conditions laid out in bounds and constraints.
    fraction_violated, feasible_points = feasible_points_random(bounds=bounds, constraints=constraints)

    # Initialize particle swarm algorithm constants.
    swarm_size = len(feasible_points)
    dimension = len(feasible_points[0])
    cognitive_parameter = 1.49
    social_parameter = 1.49
    velocity_weight = 0.73
    neighborhood_size = 5

    # Termination criteria
    small_tol = 10**-7
    flat_tol = 10**-7
    max_iter = 2000

    # Initialize termination tracking and iteration tracking variables
    small = 1.0
    flat = 1.0
    counter = 0

    if DEBUG:
        print("Swarm Size: {}".format(swarm_size))
        print("Problem Dimension: {}".format(dimension))

    # Initialize swarm position and velocity.
    current_position = feasible_points.copy()
    current_velocity = np.zeros(shape=(swarm_size,dimension))

    # Create velocity clamping vector to prevent swarm explosion.
    velocity_limit = np.zeros(shape=swarm_size)

    if DEBUG:
        print("Initial Position:")
        print(current_position)
        print("Initial Velocity:")
        print(current_velocity)

    # Initialize function evaluation variables for swarm.
    # current_func_value = np.zeros(shape=swarm_size)
    # current_penalty_value = np.zeros(shape=swarm_size)
    # current_combined_value = np.zeros(shape=swarm_size)

    # Initialize personal best and neighborhood best variables.
    personal_best_position = np.zeros(shape=(swarm_size,dimension))
    personal_best_value = np.full(shape=swarm_size,fill_value=np.inf)
    neighborhood_best_position = np.zeros(shape=(swarm_size,dimension))
    neighborhood_best_value = np.full(shape=swarm_size,fill_value=np.inf)

    while counter < max_iter:

        # Calculate function values at current swarm position.
        current_func_value = np.array(mp_pool.map(functools.partial(func, *args, **kwargs), current_position))
        current_penalty_value = np.array(mp_pool.map(functools.partial(penalty, bounds=b, constraints=c), current_position))
        current_combined_value = current_func_value + current_penalty_value

        # Evaluate termination variables.
        ordered = np.argsort(current_combined_value)
        flat = np.absolute(current_combined_value[ordered[0]]-current_combined_value[ordered[1]])
        small = np.linalg.norm(current_position[ordered[0]]-current_position[ordered[1]])

        # Evaluate if termination criteria are met and break while loop if so.
        if small < small_tol:
            print("Break due to small_tol criteria.")
            break

        if flat < flat_tol:
            print("Break due to flat_tol criteria.")
            break

        if DEBUG:
            print("-----------------------------")
            print("Counter: {}".format(counter))
            print("Current Position:")
            print(current_position)
            print("Current Combined Value:")
            print(current_combined_value)
            print("Current Minimum Value: {}".format(current_combined_value[ordered[0]]))

        # Update personal best values and positions.
        personal_best_value_update = np.less(current_combined_value,personal_best_value)
        personal_best_position_update = np.tile(personal_best_value_update,(dimension,1)).transpose()
        personal_best_value = np.where(personal_best_value_update,current_combined_value,personal_best_value)
        personal_best_position = np.where(personal_best_position_update,current_position,personal_best_position)

        # Update neighborhood best values and positions.
        for i in np.arange(swarm_size):
            positions = np.take(current_position,range(i,i+neighborhood_size),axis=0, mode='wrap')
            values = np.take(current_combined_value,range(i,i+neighborhood_size),axis=0, mode='wrap')
            ordered = np.argsort(values)
            neighborhood_best_position[i] = positions[ordered[0]]
            neighborhood_best_value[i] = values[ordered[0]]

        if DEBUG:
            print("Personal Best Value:")
            print(personal_best_value)
            print("Personal Best Position:")
            print(personal_best_position)
            print("Neighborhood Best Value:")
            print(neighborhood_best_value)
            print("Neighborhood Best Position:")
            print(neighborhood_best_position)

        # Generate random numbers used in calculating velocity components.
        cognitive_random = np.random.uniform(size=(swarm_size,dimension))
        social_random = np.random.uniform(size=(swarm_size,dimension))

        # Calculate velocity components.
        cognitive_component = cognitive_parameter*cognitive_random*(personal_best_position-current_position)
        social_component = social_parameter*social_random*(neighborhood_best_position-current_position)
        momentum_component = velocity_weight*current_velocity

        # Calculate velocity and use velocity to calculate new position.  Need to implement velocity clamping.
        new_velocity = momentum_component + social_component + cognitive_component
        new_velocity_clamped = new_velocity
        new_position = current_position + new_velocity_clamped

        if DEBUG:
            print("Velocity:")
            print(new_velocity)

        # Update current position and iteration variable
        current_position = new_position.copy()
        current_velocity = new_velocity.copy()
        counter = counter + 1

    final_ordered = np.argsort(personal_best_value)

    return personal_best_position[final_ordered[0]]


def rosen(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

def booth(x):
    """
    Name: Booth
    Global minimum: f(1.0,3.0) = 0.0
    Search domain: -10.0 <= x, y <= 10.0
    """
    return (x[0] + 2.0 * x[1] - 7.0) ** 2 + (2.0 * x[0] + x[1] - 5.0) ** 2

def rastrigin(args):
    """
    Name: Rastrigin
    Global minimum: f(0,...,0) = 0.0
    Search domain: -5.12 <= xi <= 5.12, 1 <= i <= n
    """
    a = 10.0
    rast = len(args ) *a
    for i in args:
        rast += i** 2 - a * np.cos(2 * np.pi * i ** 2)
    return rast

def const1(x):
    return (x[0] - 1.0) ** 3 - x[1] + 1.0

def const2(x):
    return x[0] + x[1] - 2.0

def const3(x):
    return x[0] ** 2 + x[1] ** 2 - 2.0

x = [0, 0]
bounds = [(-15, 15), (-5, 25)]

constraints = [{'type': '<=0', 'func': const1},
               {'type': '<=0', 'func': const2},
               {'type': '<=0', 'func': const3}]

b = bounds_check(x, bounds=bounds)
c = constraints_check(constraints=[])

func = rastrigin
args = ()
kwargs = {}

print(particle_swarm(func,bounds=b,constraints=c))