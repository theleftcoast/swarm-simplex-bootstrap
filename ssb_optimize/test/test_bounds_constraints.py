# from ssb_optimize import objective_functions as obj, constraint_functions as const
from ssb_optimize.optimizer import nelder_mead, constraints_check, bounds_check
from unittest import TestCase, main
import numpy as np

# Initial vector and length for testing bounds_check and constraints_check
x0 = [1.2, 2.3]
n = 2

# Define bounds lists for testing bounds_check
bounds_well_defined = [(-1.5, 1.5), (-1.5, 2.5)]
bounds_none_defined = [None, (-1.5, None)]
bounds_partially_defined = [(-1.5, 1.5), (-1.5,)]
bounds_wrong_order = [(1.5, -1.5), (-1.5, 2.5)]
bounds_wrong_length = [(1.5, -1.5), (-1.5, 2.5), (4.3, 3.6)]
bounds_wrong_type = [(1.5, -1.5), ('a', 2.5)]


# Define simple functions for building constraints dictionaries to test testing constraints_check
def rosen(x):
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def const1(x):
    return (x[0] - 1) ** 3 - x[1] + 1


def const2(x):
    return x[0] + x[1] - 2


def const3(x):
    return x[0] ** 2 + x[1] ** 2 - 2


def lemniscate_bernoulli(x, a=1.0):
    return (x[0] ** 2 + x[1] ** 2) ** 2 - 2 * (x[0] ** 2 - x[1] ** 2) * a ** 2


def deltoid(x, a=1.0):
    return (x[0]**2+x[1]**2)**2+18*(x[0]**2+x[1]**2)*a**2-27*a**4-8*a*(x[0]**3-3*x[0]*x[1]**2)


def sphere(x, a=1.0):
    return a*np.sum(x**2)


def line(x, a=1.0, b=0.0):
    return a*x[0]-x[1]+b


# Define constraints dictionaries for testing constraints_check
consts_lambda = [{'type': '<=0', 'func': lambda x: (x[0] - 1) ** 3 - x[1] + 1},
                 {'type': '<=0', 'func': lambda x: x[0] + x[1] - 2},
                 {'type': '<=0', 'func': lambda x: x[0] ** 2 + x[1] ** 2 - 2}]

consts_function = [{'type': '<=0', 'func': const1},
                   {'type': '<=0', 'func': const2},
                   {'type': '<=0', 'func': const3}]

consts_some_kwargs = [{'type': '<0', 'func': sphere, 'kwargs': {'a': 1.0}},
                      {'type': '<=0', 'func': line, 'kwargs': {'a': 1.0, 'b': 0.0}},
                      {'type': '>0', 'func': deltoid},
                      {'type': '>=0', 'func': lemniscate_bernoulli}]

consts_missing_reqtypekey = [{'type': '<0', 'func': sphere, 'kwargs': {'a': 1.0}},
                             {'type': '<=0', 'func': line, 'kwargs': {'a': 1.0, 'b': 0.0}},
                             {'type': '>0', 'func': deltoid},
                             {'typ': '>=0', 'func': lemniscate_bernoulli}]

consts_missing_reqfunckey = [{'type': '<0', 'func': sphere, 'kwargs': {'a': 1.0}},
                             {'type': '<=0', 'f': line, 'kwargs': {'a': 1.0, 'b': 0.0}},
                             {'type': '>0', 'func': deltoid},
                             {'type': '>=0', 'func': lemniscate_bernoulli}]

consts_not_dictionary = [{'type': '<0', 'func': sphere, 'kwargs': {'a': 1.0}},
                         [],
                         {'type': '>0', 'func': deltoid},
                         {'type': '>=0', 'func': lemniscate_bernoulli}]

consts_wrong_type = [{'type': '<0', 'func': sphere, 'kwargs': {'a': 1.0}},
                     {'type': '<=0', 'func': line, 'kwargs': {'a': 1.0, 'b': 0.0}},
                     {'type': '=0', 'func': deltoid},
                     {'type': '>=0', 'func': lemniscate_bernoulli}]

consts_not_callable = [{'type': '<0', 'func': sphere, 'kwargs': {'a': 1.0}},
                       {'type': '<=0', 'func': line, 'kwargs': {'a': 1.0, 'b': 0.0}},
                       {'type': '>0', 'func': x0},
                       {'type': '>=0', 'func': lemniscate_bernoulli}]

consts_args_not_tuple = [{'type': '<0', 'func': sphere, 'args': {'a': 1.0}},
                         {'type': '<=0', 'func': line, 'kwargs': {'a': 1.0, 'b': 0.0}},
                         {'type': '>0', 'func': deltoid},
                         {'type': '>=0', 'func': lemniscate_bernoulli}]

consts_kwargs_not_dictionary = [{'type': '<0', 'func': sphere, 'kwargs': []},
                                {'type': '<=0', 'func': line, 'kwargs': {'a': 1.0, 'b': 0.0}},
                                {'type': '>0', 'func': deltoid},
                                {'type': '>=0', 'func': lemniscate_bernoulli}]


class TestBoundsConstraints(TestCase):

    def test_bounds_partially_defined(self):
        self.assertRaises(TypeError, bounds_check, n, bounds_partially_defined)

    def test_bounds_wrong_order(self):
        self.assertRaises(ValueError, bounds_check, n, bounds_wrong_order)

    def test_bounds_wrong_length(self):
        self.assertRaises(ValueError, bounds_check, n, bounds_wrong_length)

    def test_bounds_wrong_type(self):
        self.assertRaises(TypeError, bounds_check, n, bounds_wrong_type)

    def test_consts_missing_reqtypekey(self):
        self.assertRaises(TypeError, constraints_check, consts_missing_reqtypekey)

    def test_consts_missing_reqfunckey(self):
        self.assertRaises(TypeError, constraints_check, consts_missing_reqfunckey)

    def test_consts_not_dictionary(self):
        self.assertRaises(TypeError, constraints_check, consts_not_dictionary)

    def test_consts_wrong_type(self):
        self.assertRaises(ValueError, constraints_check, consts_wrong_type)

    def test_consts_not_callable(self):
        self.assertRaises(TypeError, constraints_check, consts_not_callable)

    def test_consts_args_not_tuple(self):
        self.assertRaises(TypeError, constraints_check, consts_args_not_tuple)

    def test_consts_kwargs_not_dictionary(self):
        self.assertRaises(TypeError, constraints_check, consts_kwargs_not_dictionary)


if __name__ == "__main__":
    main()
