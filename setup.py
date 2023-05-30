from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, find_packages

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension("freegroup.derivatives",
        ["src/freegroup/derivatives/main.cpp"],
        define_macros = [('VERSION_INFO', __version__)],
        ),
]

setup(
    name="freegroup",
    version=__version__,
    author="F. Pavutnitskiy, D. Pasechnyuk, K. Brilliantov, G. Magai",
    description="Module for operating with free group and calculating homotopy groups of spheres.",
    packages = find_packages(
        where='src',
    ),
    package_dir={"":"src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires = [
    	numpy,
    	iteration_utilities,
    	parsec,
    ]
)
