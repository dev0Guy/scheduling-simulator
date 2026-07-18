from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

numpy_include = np.get_include()

extensions = [
    Extension(
        "scheduling_simulator.core.job",
        ["src/scheduling_simulator/core/job.pyx"],
        include_dirs=[numpy_include],
        define_macros=[("CYTHON_TRACE", "1")]
    ),
    Extension(
        "scheduling_simulator.core.machine",
        ["src/scheduling_simulator/core/machine.pyx"],
        include_dirs=[numpy_include],
        define_macros=[("CYTHON_TRACE", "1")]
    ),
    Extension(
        "scheduling_simulator.core.cluster",
        ["src/scheduling_simulator/core/cluster.pyx"],
        include_dirs=[numpy_include],
        define_macros=[("CYTHON_TRACE", "1")]
    ),
]

setup(
    name="scheduling_simulator",
    package_dir={"": "src"},
    packages=find_packages("src"),
    ext_modules=cythonize(
        extensions,
        include_path=[
            "src",
            "src/scheduling_simulator",
            "src/scheduling_simulator/core",
        ],
        compiler_directives={
            "language_level": "3",
            "boundscheck": True,
            "wraparound": True,
            "linetrace": True,
            "binding": True,
        },
    ),
)
