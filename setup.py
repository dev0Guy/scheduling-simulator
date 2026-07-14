from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(name="core.action", sources=["src/core/action.pyx"]),
    Extension(name="core.observation", sources=["src/core/observation.pyx"]),
    Extension(name="core.job", sources=["src/core/job.pyx"]),
]

setup(
    package_dir={"": "src"},
    packages=["core"],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
        annotate=True,
    ),
)
