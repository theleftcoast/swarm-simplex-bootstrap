from test import objective_functions as obj
from ssb_optimize import optimizer as opt
from unittest import TestCase, main


class LeastSquaresTestCase(TestCase):
    # x was generated as a grid across the domain [(-2.0, 2.0), (-2.0, 2.0)]
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
    # fx was generated by passing x to the 'quadratic' function and rounding the results. The coefficients (thetas) that
    # define the quadratic function behavior are [0.26886495, 0.26838753, -0.01241092, 0.01181188, -0.49181186,
    # -0.02209403]. Rounding the exact fx output introduces small deviations.  These deviations create a data set that
    # serves as a good 'quasi-experimental' test set (where x represents input conditions and fx represents measured
    # responses with some noise).
    fx = [0.16, 0.34, 1.04, 2.26, 4.0, 0.16, 0.04, 0.26, 1.0, 2.26, 1.04, 0.26, 0.0, 0.26, 1.04, 2.26, 1.0, 0.26, 0.04,
          0.34, 4.3, 2.26, 1.04, 0.34, 0.16]
    func = obj.quadratic
    theta_min = [0.26886495, 0.26838753, -0.01241092, 0.01181188, -0.49181186, -0.02209403]
    theta = [0.261, 0.261, 0.0, 0.0, -0.481, 0.0]
    weight = None
    bootstrap = None
    precision = 3

    def __init__(self, *args, **kwargs):
        super(LeastSquaresTestCase, self).__init__(*args, **kwargs)
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
                                  args=(self.func, self.x, self.fx, self.weight, self.bootstrap))
        self.assertArrayAlmostEqual(minimum, self.theta_min, places=self.precision)


class WeightedTest(LeastSquaresTestCase):
    weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
              0.5, 0.5, 0.5, 0.5]
    bootstrap = None
    theta_min = [0.26886494, 0.26838754, -0.01241092, 0.01181187, -0.49181187, -0.02209404]


class BootstrappedTest(LeastSquaresTestCase):
    weight = None
    bootstrap = None
    theta_min = [0.26886494, 0.26838754, -0.01241092, 0.01181187, -0.49181187, -0.02209404]


class WeightedBootstrappedTest(LeastSquaresTestCase):
    weight = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
              0.5, 0.5, 0.5, 0.5]
    bootstrap = [1, 2, 1, 0, 1, 0, 3, 0, 0, 1, 1, 2, 1, 1, 2, 1, 0, 1, 0, 3, 0, 0, 1, 1, 2]
    theta_min = [0.26, 0.26, 0.0, 0.0, -0.48, 0.0]


if __name__ == "__main__":
    main()