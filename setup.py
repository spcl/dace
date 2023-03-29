# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import glob
import os
import shutil
import subprocess

from setuptools import find_packages, setup

# Find runtime and external library files by obtaining the module path and
# trimming the absolute path of the resulting files.
dace_path = os.path.dirname(os.path.abspath(__file__)) + '/dace/'
runtime_files = [f[len(dace_path):] for f in glob.glob(dace_path + 'runtime/include/**/*', recursive=True)]
library_files = [f[len(dace_path):] for f in glob.glob(dace_path + 'libraries/**/include/**/*', recursive=True)]
cmake_files = [f[len(dace_path):] for f in glob.glob(dace_path + 'codegen/**/*.cmake', recursive=True)]
viewer_files = [
    f[len(dace_path):] for f in (glob.glob(dace_path + 'viewer/webclient/dist/*.js', recursive=True) +
                                 glob.glob(dace_path + 'viewer/webclient/external_libs/**/*', recursive=True) +
                                 glob.glob(dace_path + 'viewer/webclient/*.css', recursive=True) +
                                 glob.glob(dace_path + 'viewer/webclient/*.html', recursive=True) +
                                 glob.glob(dace_path + 'viewer/templates/**/*', recursive=True) +
                                 glob.glob(dace_path + 'viewer/**/LICENSE', recursive=True))
]
cub_files = [f[len(dace_path):] for f in glob.glob(dace_path + 'external/cub/cub/**/*', recursive=True)
             ] + [dace_path + 'external/cub/LICENSE.TXT']
hlslib_files = [f[len(dace_path):] for f in glob.glob(dace_path + 'external/hlslib/cmake/**/*', recursive=True)] + [
    f[len(dace_path):] for f in glob.glob(dace_path + 'external/hlslib/include/**/*', recursive=True)
] + [dace_path + 'external/hlslib/LICENSE.md']
rtllib_files = [f[len(dace_path):] for f in glob.glob(dace_path + 'external/rtllib/cmake/**/*', recursive=True)] + [
    f[len(dace_path):] for f in glob.glob(dace_path + 'external/rtllib/templates/**/*', recursive=True)
]

# See if CMake is available and if not, install as a dependency
cmake_requires = ['scikit-build', 'cmake']
try:
    cmake_path = shutil.which('cmake')
    if cmake_path:
        # CMake is available, check version
        output = subprocess.check_output([cmake_path, '--version']).decode('utf-8')
        cmake_version = tuple(int(t) for t in output.splitlines()[0].split(' ')[-1].split('.'))
        # If version meets minimum requirements, CMake is not necessary
        if cmake_version >= (3, 15):
            cmake_requires = []
except (subprocess.CalledProcessError, OSError, IndexError, ValueError):
    # Any failure in getting the CMake version counts as "not found"
    pass

with open("README.md", "r") as fp:
    long_description = fp.read()

with open(os.path.join(dace_path, "version.py"), "r") as fp:
    version = fp.read().strip().split(' ')[-1][1:-1]

setup(name='dace',
      version=version,
      url='https://github.com/spcl/dace',
      author='SPCL @ ETH Zurich',
      author_email='talbn@inf.ethz.ch',
      description='Data-Centric Parallel Programming Framework',
      long_description=long_description,
      long_description_content_type='text/markdown',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: BSD License",
          "Operating System :: OS Independent",
      ],
      python_requires='>=3.6, <3.12',
      packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      package_data={
          '': [
              '*.yml', 'codegen/CMakeLists.txt', 'codegen/tools/*.cpp', 'external/moodycamel/*.h',
              'external/moodycamel/LICENSE.md', 'codegen/Xilinx_HLS.tcl.in'
          ] + runtime_files + cub_files + viewer_files + hlslib_files + library_files + rtllib_files + cmake_files
      },
      include_package_data=True,
      install_requires=[
         'numpy', 'networkx >= 2.5', 'astunparse', 'sympy<=1.9', 'pyyaml', 'ply', 'websockets', 'requests', 'flask',
          'fparser', 'aenum >= 3.1', 'dataclasses; python_version < "3.7"', 'dill',
          'pyreadline;platform_system=="Windows"', 'typing-compat; python_version < "3.8"'
      ] + cmake_requires,
      extras_require={
          'testing': ['coverage', 'pytest-cov', 'scipy', 'absl-py', 'opt_einsum', 'pymlir', 'click'],
          'docs': ['jinja2<3.1.0', 'sphinx-autodoc-typehints', 'sphinx-rtd-theme>=0.5.1']
      },
      entry_points={
          'console_scripts': [
              'dacelab = dace.cli.dacelab:main',
              'sdfv = dace.cli.sdfv:main',
              'sdfgcc = dace.cli.sdfgcc:main',
              'fcfd = dace.cli.fcdc:main',
              'daceprof = dace.cli.daceprof:main',
          ],
      })
