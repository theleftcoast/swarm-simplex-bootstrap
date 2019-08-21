from test import objective_functions as obj
from ssb_optimize.optimizer import particle_swarm
from unittest import TestCase, main


class ObjectiveFunctionTestCase(TestCase):
    func = obj.sphere
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    global_minimum = [0, 0, 0]
    precision = 3

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

    def test_particle_swarm(self):
        """
        Run the particle solver on the objective function, seeded with the initial point;
        assert it gets at least kind of close to the correct value
        """
        minimum, nelder_mead_initial = particle_swarm(self.func, bounds=self.bounds)
        self.assertArrayAlmostEqual(minimum, self.global_minimum, places=self.precision)


class SphereTest(ObjectiveFunctionTestCase):
    func = obj.sphere
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    global_minimum = [0.0, 0.0, 0.0]


class RosenbrockTest(ObjectiveFunctionTestCase):
    func = obj.rosenbrock
    bounds = [(-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)]
    global_minimum = [1.0, 1.0, 1.0]


class RastriginTest(ObjectiveFunctionTestCase):
    func = obj.rastrigin
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]
    global_minimum = [0.0, 0.0, 0.0]


class AckleyTest(ObjectiveFunctionTestCase):
    func = obj.ackley
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    global_minimum = [0.0, 0.0]


class BealeTest(ObjectiveFunctionTestCase):
    func = obj.beale
    bounds = [(-4.0, 4.0), (-4.0, 4.0)]
    global_minimum = [3.0, 0.5]


class GoldsteinPriceTest(ObjectiveFunctionTestCase):
    func = obj.goldstein_price
    bounds = [(-2.0, 2.0), (-2.0, 2.0)]
    global_minimum = [0.0, -1.0]


class BoothTest(ObjectiveFunctionTestCase):
    func = obj.booth
    bounds = [(-4.0, 4.0), (-4.0, 4.0)]
    global_minimum = [1.0, 3.0]


class MatyasTest(ObjectiveFunctionTestCase):
    func = obj.matyas
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    global_minimum = [0.0, 0.0]


class ThreeHumpCamelTest(ObjectiveFunctionTestCase):
    func = obj.three_hump_camel
    bounds = [(-1.0, 1.0), (-1.0, 1.0)]
    global_minimum = [0.0, 0.0]


class McCormickTest(ObjectiveFunctionTestCase):
    func = obj.mccormick
    bounds = [(-1.5, 4.0), (-3.0, 4.0)]
    global_minimum = [-0.54719, -1.54719]


if __name__ == "__main__":
    main()
