"""
A standard test suite of objective functions that we can use to test whether an optimization algorithm is actually working.

These have odd topologies and a well-known global minimum; they're good for trapping bad optimization routines.
"""
import numpy as np


def quadratic(x,a,b,c,d,e,f):
    """
    Name: Quadratic
    """
    return a*x[0]**2 + b*x[1]**2 + c*x[0] + d*x[1] + e*x[0]*x[1] + f


def sectioned_quadratic(x, a, b, c, d, e, f, section=None):
    """
    Name: Sectioned Quadratic
    """
    value = 0.0
    if section == 'full':
        value = a*x[0]**2 + b*x[1]**2 + c*x[0] + d*x[1] + e*x[0]*x[1] + f
    elif section == 'a':
        value = a*x[0]**2 + b*x[1]**2
    elif section == 'b':
        value = c*x[0] + d*x[1]
    elif section == 'c':
        value = e * x[0]*x[1] + f
    else:
        value = a*x[0]**2 + b*x[1]**2 + c*x[0] + d*x[1] + e*x[0]*x[1] + f
    return value


def sphere(args):
    """
    Name: Sphere    
    Global minimum: f(0,...,0) = 0.0
    Search domain: -inf <= xi <= inf, 1 <= i <= n
    """
    return np.sum(args**2)


def rosenbrock(args):
    """
    Name: Rosenbrock    
    Global minimum: f(1,...,1) = 0.0
    Search domain: -inf <= xi <= inf, 1 <= i <= n
    """
    rosen = 0
    for i in range(len(args ) -1):
        rosen += 10.0 *((args[i]**2 ) -args[ i +1] )** 2 +( 1 -args[i] )**2
    return rosen


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


def ackley(args):
    """
    Name: Ackley    
    Global minimum: f(0,0) = 0.0
    Search domain: -5.0 <= x, y <= 5.0
    """
    firstSum = 0.0
    secondSum = 0.0
    for c in args:
        firstSum += c ** 2.0
        secondSum += np.cos(2.0 * np.pi * c)
    n = float(len(args))
    return -20.0 * np.exp(-0.2 * np.sqrt(firstSum / n)) - np.exp(secondSum / n) + 20 + np.e


def beale(args):
    """
    Name: Beale    
    Global minimum: f(3.0,0.5) = 0.0
    Search domain: -4.5 <= x, y <= 4.5
    """
    x = args[0]
    y = args[1]
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3) ** 2


def goldstein_price(args):
    """
    Name: Goldstein-Price    
    Global minimum: f(0.0,-1.0) = 3.0
    Search domain: -2.0 <= x, y <= 2.0
    """
    x = args[0]
    y = args[1]
    l = 1 + (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2) * (x + y + 1) ** 2
    m = 30 + (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2) * (2 * x - 3 * y) ** 2
    return l * m


def booth(args):
    """
    Name: Booth    
    Global minimum: f(1.0,3.0) = 0.0
    Search domain: -10.0 <= x, y <= 10.0
    """
    x = args[0]
    y = args[1]
    return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

def matyas(args):
    """
    Name: Matyas    
    Global minimum: f(0.0,0.0) = 0.0
    Search domain: -10.0 <= x, y <= 10.0
    """
    x = args[0]
    y = args[1]
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y


def himmelblau(args):
    """
    Name: Himmelblau    
    Local minimum: f(3.0,2.0) = 0.0
    Local minimum: f(-2.805118,3.131312) = 0.0
    Local minimum: f(-3.779310,-3.283186) = 0.0
    Local minimum: f(3.584428,-1.848126) = 0.0
    Search domain: -5.0 <= x, y <= 5.0
    """
    x = args[0]
    y = args[1]
    return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7.0) ** 2


def three_hump_camel(args):
    """
    Name: Three hump camel    
    Global minimum: f(0.0,0.0) = 0.0
    Search domain: -5.0 <= x, y <= 5.0
    """
    x = args[0]
    y = args[1]
    return 2.0 * x ** 2 - 1.05 * x ** 4 + (x / 6.0) ** 6 + x * y + y ** 2


def mccormick(args):
    """
    Name: McCormic    
    Global minimum: f(-0.54719,-1.54719) = -1.9133
    Search domain: -1.5 <= x <= 4.0, -3.0 <= y <= 4.0
    """
    x = args[0]
    y = args[1]
    return np.sin(x + y) + (x - y) ** 2 - 1.5 * x + 2.5 * y + 1
