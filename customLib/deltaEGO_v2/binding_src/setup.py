import os
import sys
from setuptools import setup, Extension
import pybind11

# 운영체제에 따른 컴파일러 최적화 옵션
if sys.platform == "win32":
    extra_compile_args = ['/O2', '/arch:AVX512'] # Windows (MSVC)
else:
    extra_compile_args = ['-O3', '-march=native', '-fPIC'] # Linux/WSL (GCC/Clang)

ext_modules = [
    Extension(
        "delta_ego_core",                     # 생성될 파이썬 모듈 이름
        sources=["deltaEGO.cpp"],       # 방금 작성한 C++ 파일 이름
        include_dirs=[
            pybind11.get_include(),
            # json.hpp가 다른 폴더에 있다면 여기에 경로 추가 (예: './include')
        ],
        libraries=["yaml-cpp"],          
        language='c++',
        extra_compile_args=extra_compile_args,
    ),
]

setup(
    name="delta_ego_core",
    version="2.0",
    author="deltaAnima",
    description="High-performance Emotion Engine optimized for 9950X",
    ext_modules=ext_modules,
)
