import optimizer as opt
import objective_functions as of
import constraint_functions as cf

def constraint_a(x):
    return (x[0] - 1.0) ** 3 - x[1] + 1.0

def constraint_b(x):
    return x[0] + x[1] - 2.0

def constraint_c(x):
    return x[0] ** 2 + x[1] ** 2 - 2.0

x = [0, 0]
dim = len(x)
bounds = [(-15, 15), (-5, 25)]

constraints = [{'type': '<=0', 'func': constraint_a},
               {'type': '<=0', 'func': constraint_b},
               {'type': '<=0', 'func': constraint_c}]

bound = opt.bounds_check(dim, bounds=bounds)
const = opt.constraints_check(constraints=constraints)

print(bound)
print(const)

func = of.sphere
fraction_violated, feasbile_points = opt.feasible_points_random(bounds=bound, constraints=const)
x_0 = opt.best_point(feasbile_points, func, args=(), kwargs={})

print(x_0)

x_ps = opt.particle_swarm(func,args=(),kwargs={}, bounds=bounds, constraints=constraints)
x_nm = opt.nelder_mead(x_ps,func,args=(),kwargs={}, bounds=bounds, constraints=constraints)

print(x_ps)
print(x_nm)



x = [1.5, 1.5]
dim = len(x)

bounds = [(-1.0, 3.0), (-1.0, 3.0)]
constraints = [{'type': '<=0', 'func': cf.circle}]
               # {'type': '<=0', 'func': cf.line}]

bound = opt.bounds_check(dim, bounds=bounds)
const = opt.constraints_check(constraints=constraints)

print(bound)
print(const)

func = of.sphere
fraction_violated, feasbile_points = opt.feasible_points_random(bounds=bound, constraints=const)
x_0 = opt.best_point(feasbile_points, func, args=(), kwargs={})

print(x_0)

x_ps = opt.particle_swarm(func,args=(),kwargs={}, bounds=bounds, constraints=constraints)
x_nm = opt.nelder_mead(x_ps,func,args=(),kwargs={}, bounds=bounds, constraints=constraints)

print(x_ps)
print(x_nm)

x = [1.5, 1.5]
dim = len(x)

bounds = [(-1.0, 3.0), (-1.0, 3.0)]
constraints = [{'type': '<=0', 'func': cf.circle}]

bound = opt.bounds_check(dim, bounds=bounds)
const = opt.constraints_check(constraints=constraints)

print(bound)
print(const)

func = of.rosenbrock
fraction_violated, feasbile_points = opt.feasible_points_random(bounds=bound, constraints=const)
x_0 = opt.best_point(feasbile_points, func, args=(), kwargs={})

print(x_0)

x_ps = opt.particle_swarm(func,args=(),kwargs={}, bounds=bounds, constraints=constraints)
x_nm = opt.nelder_mead(x_ps,func,args=(),kwargs={}, bounds=bounds, constraints=constraints)

print(x_ps)
print(x_nm)

x = [1.5, 1.5]
dim = len(x)

bounds = [(-1.0, 3.0), (-1.0, 3.0)]
constraints = [{'type': '<=0', 'func': cf.circle}]

bound = opt.bounds_check(dim, bounds=bounds)
const = opt.constraints_check(constraints=constraints)

print(bound)
print(const)

func = of.beale
fraction_violated, feasbile_points = opt.feasible_points_random(bounds=bound, constraints=const)
x_0 = opt.best_point(feasbile_points, func, args=(), kwargs={})

print(x_0)

x_ps = opt.particle_swarm(func,args=(),kwargs={}, bounds=bounds, constraints=constraints)
x_nm = opt.nelder_mead(x_ps,func,args=(),kwargs={}, bounds=bounds, constraints=constraints)

print(x_ps)
print(x_nm)