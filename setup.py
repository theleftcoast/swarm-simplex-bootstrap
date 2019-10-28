import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ssb_optimize",
    version="0.1.0",
    author="John Towne",
    author_email="towne.john@gmail.com",
    description="Particle Swarm and Nelder-Mead Simplex optimization algorithms with Bootstrap confidence intervals.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/theleftcoast/swarm-simplex-bootstrap",
    packages=setuptools.find_packages(),
    install_requires=['numpy', ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)