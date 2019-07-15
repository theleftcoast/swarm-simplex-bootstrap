# Swarm-Simplex-Bootstrap
Swarm-Simplex-Bootstrap is a python implementation of the Particle Swarm and Nelder-Mead Simplex minimization algorithms. Both  algorithms make few assumptions about the function to be minimized (such as continuity or differentiability) so they are applicable to minimizing a wide range of functions. Bounds, linear constraints, and nonlinear constraints are supported.

The emphasis of this library is paramterizing models using experimental data where the model parameters can be subject to bounds and constraints. Model parameterization is carried out by minimizing the Least Squares objective function. Model parameter uncertainty is estimated by Bootstrapping.

### Table of Contents
Installation
Testing
Project Structure, Versioning, and Documentation

### Installation
ssb_optimize can be easily installed as a python package using pip. Package dependencies include numpy, itertools, numbers, and multiprocessing.
```console
python -m pip install ssb_optimize
```
Initial development was done in Python 3.6 so I suspect the package will work with any Python 3 installation.  That being said, it hasn't been tested with any other versions of Python (if somebody would like to help with this, please let me know). 


### Testing
To run unit tests, open a shell environment and run the following command in the top-level 'ssb_optimize' directory.
```console
python -m unittest discover -v
```
### Project Structure, Versioning, and Documentation
Project structure follows [python-guide.org recommendations](https://docs.python-guide.org/writing/structure/).  

Docstring format follows [Google style recoomendations](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).  

Versioning for publishing to PyPI follows the "major.minor.patch" format based on [https://semver.org/ recommendations](https://semver.org/).
+ major version - when you make incompatible API changes,
+ minor version - when you add functionality in a backwards-compatible manner, and
+ patch version - when you make backwards-compatible bug fixes.

[Markdown cheat sheet](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet#code)

### General Information
Start off by importing the 'optimizer' module.   
```python
import optimizer as opt
```
The functions contained in the 'optimizer' module are summarized below.
+ **bounds_check**: Check bounds list of size 'n' for consistency and return the list with basic problems corrected.
+ **constraints_check**: Check constraints list for consistency and return the list with basic problems corrected.
+ **feasible_points_random**: Generate an array of points using random sampling which satisfy the bounds and constraints.
+ **best_point**: Return the point corresponding to the lowest evaluated value of 'func'.
+ **nelder_mead**: Minimize a scalar function using the Nelder-Mead simplex algorithm.
+ **particle_swarm**: Minimize a scalar function using the Particle Swarm algorithm.
+ **least_squares_objective_function**: Returns the scalar result of evaluation of the least squares objective function.
+ **least_squares_bootstrap**: Returns list of tuples containing the results (thetas) of repeated least squares fitting of func to x and fx.

The docstrings for these functions contain additional information about how each function works, arguments, return values, and error handling.
```python
print(function.__doc__)
help(function)
```
### Test Functions for Minimization
Test functions in this tutorial all come from the [Test functions for optimization](https://en.wikipedia.org/wiki/Test_functions_for_optimization) Wikipedia page with additional information available from the [BenchmarkFncs Toolbox](http://benchmarkfcns.xyz/).  These test functions are used to evaluate performance of minimization algorithms in situations relevant to real world applications.
##### Booth Function
The Booth function is continuous and convex with one global minimum. The Booth function has a 'smooth' approach to the global minimum.  This mimics the approach to a minimum for many function surfaces when very near (arbitrarily close to) a minimum.

The global minimum is:

![Booth Function Minimum](https://wikimedia.org/api/rest_v1/media/math/render/svg/08846444f0b442990de2fde559aeb9336564d9eb "Booth Function Minimum")

![Booth Function](https://upload.wikimedia.org/wikipedia/commons/thumb/6/6e/Booth%27s_function.pdf/page1-320px-Booth%27s_function.pdf.jpg "Booth Function")

##### Rosenbrock Function
The Rosenbrock function is continuous and convex with one globl minimum. The global minimum is inside a long, narrow, parabolic shaped flat valley. To find the valley is trivial but converging to the global minimum is difficult.

The global minimum is: 

![Rosenbrock Global Optimum](https://wikimedia.org/api/rest_v1/media/math/render/svg/a70149aee356dd9ee6f42993df754bd28e6a9560 "Rosenbrock Global Optimum")

![Rosenbrock Function](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/Rosenbrock%27s_function_in_3D.pdf/page1-320px-Rosenbrock%27s_function_in_3D.pdf.jpg "Rosenbrock Function")

##### Ackley Function
The Ackley function is continuous and non-convex having many local minima but only one one global minimum. Many minimizing algorithms will become trapped in one of the many local minimum during a search for the global minimum. 
The global minimum is:

![Ackley Global Optimum](https://wikimedia.org/api/rest_v1/media/math/render/svg/73eda9d9d0146876d534fe2d9d228990825ec759 "Ackley Global Optimum")

![Ackley Function](https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Ackley%27s_function.pdf/page1-320px-Ackley%27s_function.pdf.jpg "Ackley Function")


### Unconstrained Minimization

### Bounds and Constraints


### Constrained Minimization


### Model Parameterization


### Bootstrapping
