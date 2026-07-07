from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "delta_ego_core",
        ["customLib/deltaEGO_v2/binding_src/deltaEGO.cpp"],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-O3', '-march=native', '-std=c++17'],
    ),
]

setup(
    name="delta_ego_core",
    version="0.1.0",
    ext_modules=ext_modules,
    zip_safe=False,
    python_requires=">=3.13",
)
