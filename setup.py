from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(name="core.job", sources=["src/core/job.pyx"]),
    Extension(name="core.machine", sources=["src/core/machine.pyx"]),
    Extension(name="core.cluster", sources=["src/core/cluster.pyx"], include_dirs=[np.get_include()]),
]

setup(
    package_dir={"": "src"},
    packages=["core"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": True,
            "wraparound": True,
        },
        annotate=True,
    ),
)
