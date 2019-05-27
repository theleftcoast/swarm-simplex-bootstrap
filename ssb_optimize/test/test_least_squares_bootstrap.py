from ssb_optimize import objective_functions as obj
from ssb_optimize import optimizer as opt
from unittest import TestCase, main

class LSBTestCase(TestCase):
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
    # fx was generated by passing x to the 'quadratic' function' and rounding the results.
    fx = [0.16, 0.34, 1.04, 2.26, 4, 0.16, 0.04, 0.26, 1, 2.26, 1.04, 0.26, 0, 0.26, 1.04, 2.26, 1, 0.26, 0.04,
                0.34, 4.3, 2.26, 1.04, 0.34, 0.16]

    func = obj.quadratic
    theta = [0.261, 0.261, 0.0, 0.0, -0.481, 0.0]
    weight = None # [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
             # 0.5, 0.5, 0.5, 0.5]
    bootstrap = None # lsb.bootstrap_sample(len(x), len(x))

    precision = 3
    theta_min = [0.26886495, 0.26838753, -0.01241092, 0.01181188, -0.49181186, -0.02209403]

    def __init__(self, *args, **kwargs):
        super(LSBTestCase, self).__init__(*args, **kwargs)
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
        The vanilla AssertAlmostEqual method doesn't like numpy arrays; it fails on the `if first == second` call, since
        it doesn't know how to interpret an array of True/False

        Dodge this problem by just comparing each element of the array
        """
        self.assertEqual(len(first), len(second), msg="Arrays of different sizes cannot possibly be equal")
        for f, s in zip(first, second):
            self.assertAlmostEqual(f, s, *args, **kwargs)

    def test_nelder_mead(self):
        """
        """
        minimum = opt.nelder_mead(self.theta, opt.least_squares_objective_function,
                              args=(obj.quadratic, self.x, self.fx, self.weight, self.bootstrap))
        self.assertArrayAlmostEqual(minimum, self.theta_min, places=self.precision)

# class WeightsTest():
#     func = obj.sphere
#     bounds = [(-1.5, 1.5), (-0.5, 2.5)]
#     constraints = [{'type': '<=0', 'func': const.f3}]
#     initial_pt_valid = [-0.5, 1.0]
#     initial_pt_invalid = [-500.0, 1000.0]
#     global_minimum = [0.0, 0.0]

if __name__ == "__main__":
    main()
