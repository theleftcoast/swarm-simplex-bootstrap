import objective_functions as obj
from optimizer import nelder_mead
from unittest import TestCase, main

class ObjectiveFunctionTestCase(TestCase):
    func = obj.sphere
    bounds = [(-1.5, 1.5), (-0.5, 2.5)]
    constraints = [{'type': '<=0', 'func': lambda x: (x[0] - 1.0) ** 3 - x[1] + 1.0},
                     {'type': '<=0', 'func': lambda x: x[0] + x[1] - 2.0}]
    initial_pt = [1.2, -0.1]
    global_minimum = [0.0, 0.0]
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
        # doing the self-currying thing
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
        minimum = nelder_mead(self.initial_pt, self.func, bounds=self.bounds, constraints=self.constraints)
        print(minimum)
        self.assertArrayAlmostEqual(minimum, self.global_minimum, places=self.precision)

    def test_particle_swarm(self):
        """TBD"""
        pass

class SphereTest(ObjectiveFunctionTestCase):
    func = obj.sphere
    bounds = [(-1.5, 1.5), (-0.5, 2.5)]
    constraints = [{'type': '<=0', 'func': lambda x: x[0]**2 + x[1]**2}]
    initial_pt = [1.2, -0.1]
    global_minimum = [0, 0]

'''class RosenbrockTest(ObjectiveFunctionTestCase):
    func = obj.rosenbrock
    initial_pt = [2.2, 1.1, 0.5, -0.5, -1.0]
    global_minimum = [1.0, 1.0, 1.0, 1.0, 1.0]'''