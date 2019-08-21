from test import objective_functions as obj
from ssb_optimize.optimizer import nelder_mead
from unittest import TestCase, main


class ObjectiveFunctionTestCase(TestCase):
    func = obj.sphere
    initial_pt = [3.2, 2.1, 1.0]
    global_minimum = [0, 0, 0]
    precision = 4

    def __init__(self, *args, **kwargs):
        super(ObjectiveFunctionTestCase, self).__init__(*args, **kwargs)
        # Weird python thing -- if you define a function as a class attribute, calling instance.func(*args, **kwargs)
        # will force the call to actually be something like func(instance, *args, **kwargs); this is why there's
        # a `self` parameter as the first thing to normal methods in a class definition
        #
        # here, we actively DO NOT want that; the objective functions take a vector as their first argument, so we use
        # a cute little hack:
        # functions defined as *instance attributes* do NOT get the bound method treatment; they won't have their
        # first argument forced to be self.
        #
        # consequently, we grab the function off of the class, and "demote" it to be an instance attribute instead,
        # doding the self-currying thing
        #
        # Deep Python Magic:
        # https://docs.python.org/3.3/howto/descriptor.html#functions-and-methods
        self.func = self.__class__.func

    def assertArrayAlmostEqual(self, first, second, *args, **kwargs):
        """
        the vanilla AssertAlmostEqual method doesn't like numpy arrays; it fails on the `if first == second` call, since
        it doesn't know how to interpret an array of True/False

        Dodge this problem by just comparing each element of the array
        """
        self.assertEqual(len(first), len(second), msg="Arrays of different sizes cannot possibly be equal")
        for f, s in zip(first, second):
            self.assertAlmostEqual(f, s, *args, **kwargs)

    def test_nelder_mead(self):
        """
        Run the nelder mead solver on the objective function, seeded with the initial point;
        assert it gets at least kind of close to the correct value
        """
        minimum = nelder_mead(self.initial_pt, self.func)
        self.assertArrayAlmostEqual(minimum, self.global_minimum, places=self.precision)


class SphereTest(ObjectiveFunctionTestCase):
    func = obj.sphere
    initial_pt = [3.2, 2.1, 1.0, -1.0, -2.0]
    global_minimum = [0.0, 0.0, 0.0, 0.0, 0.0]


class RosenbrockTest(ObjectiveFunctionTestCase):
    func = obj.rosenbrock
    initial_pt = [2.2, 1.1, 0.5, -0.5, -1.0]
    global_minimum = [1.0, 1.0, 1.0, 1.0, 1.0]


class RastriginTest(ObjectiveFunctionTestCase):
    func = obj.rastrigin
    initial_pt = [0.05, 0.04, 0.06, -0.03, -0.02]
    global_minimum = [0.0, 0.0, 0.0, 0.0, 0.0]


class AckleyTest(ObjectiveFunctionTestCase):
    func = obj.ackley
    initial_pt = [0.3, 0.1]
    global_minimum = [0.0, 0.0]


class BealeTest(ObjectiveFunctionTestCase):
    func = obj.beale
    initial_pt = [3.5, 0.25]
    global_minimum = [3.0, 0.5]


class GoldsteinPriceTest(ObjectiveFunctionTestCase):
    func = obj.goldstein_price
    initial_pt = [1.0, -1.0]
    global_minimum = [0.0, -1.0]


class BoothTest(ObjectiveFunctionTestCase):
    func = obj.booth
    initial_pt = [1.0, -1.0]
    global_minimum = [1.0, 3.0]


class MatyasTest(ObjectiveFunctionTestCase):
    func = obj.matyas
    initial_pt = [1.0, -1.0]
    global_minimum = [0.0, 0.0]


class HimmelblauATest(ObjectiveFunctionTestCase):
    func = obj.himmelblau
    initial_pt = [3.1, 1.9]
    global_minimum = [3.0, 2.0]


class HimmelblauBTest(ObjectiveFunctionTestCase):
    func = obj.himmelblau
    initial_pt = [-2.7, 3.0]
    global_minimum = [-2.805118, 3.131312]


class HimmelblauCTest(ObjectiveFunctionTestCase):
    func = obj.himmelblau
    initial_pt = [-3.6, -3.4]
    global_minimum = [-3.779310, -3.283186]


class HimmelblauDTest(ObjectiveFunctionTestCase):
    func = obj.himmelblau
    initial_pt = [3.4, -1.6]
    global_minimum = [3.584428, -1.848126]


class ThreeHumpCamelTest(ObjectiveFunctionTestCase):
    func = obj.three_hump_camel
    initial_pt = [0.1, -0.1]
    global_minimum = [0.0, 0.0]


class McCormickTest(ObjectiveFunctionTestCase):
    func = obj.mccormick
    initial_pt = [1.0, -1.0]
    global_minimum = [-0.54719, -1.54719]


if __name__ == "__main__":
    main()
