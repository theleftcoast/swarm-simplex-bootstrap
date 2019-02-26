"""
A support suite of constraint functions that we can use to test whether an optimization algorithm is actually working.

These are relatively simple; they're good for constraining typical optimization routines.

Source: https://en.wikipedia.org/wiki/Test_functions_for_optimization --> "Test functions for constrained optimization."
"""
import numpy as np


def f1(x):
    """
    Name: First constraint for the Wikipedia Rosenbrock function.
    """
    return (x[0] - 1.0) ** 3 - x[1] + 1.0

def f2(x):
    """
    Name: Second constraint for the Wikipedia Rosenbrock function.
    """
    return x[0] + x[1] - 2.0

def f3(x):
    """
    Name: Third constraint for the Wikipedia Rosenbrock function.
    """
    return x[0]**2 + x[1]**2 - 2.0



