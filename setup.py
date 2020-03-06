from setuptools import setup, find_packages
import glob
import os

# Find runtime and external library files by obtaining the module path and
# trimming the absolute path of the resulting files.
dace_path = os.path.dirname(os.path.abspath(__file__)) + '/dace/'
diode_path = os.path.dirname(os.path.abspath(__file__)) + '/diode/'
runtime_files = [
    f[len(dace_path):]
    for f in glob.glob(dace_path + 'runtime/include/**/*', recursive=True)
]
diode_files = [
    f[len(diode_path):]
    for f in (glob.glob(diode_path + 'webclient/**/*', recursive=True) +
              glob.glob(diode_path + '**/LICENSE', recursive=True))
]
cub_files = [
    f[len(dace_path):]
    for f in glob.glob(dace_path + 'external/cub/cub/**/*', recursive=True)
] + [dace_path + 'external/cub/LICENSE.TXT']
hlslib_files = [
    f[len(dace_path):]
    for f in glob.glob(dace_path + 'external/hlslib/cmake/**/*',
                       recursive=True)
] + [
    f[len(dace_path):]
    for f in glob.glob(dace_path + 'external/hlslib/include/**/*',
                       recursive=True)
] + [dace_path + 'external/hlslib/LICENSE.md']

with open("README.md", "r") as fp:
    long_description = fp.read()

setup(name='dace',
      version='0.9.5',
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
      python_requires='>=3.6',
      packages=find_packages(
          exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
      package_data={
          '': [
              '*.yml', 'codegen/CMakeLists.txt', 'codegen/tools/*.cpp',
              'external/moodycamel/*.h', 'external/moodycamel/LICENSE.md',
              'codegen/Xilinx_HLS.tcl.in'
          ] + runtime_files + cub_files + diode_files + hlslib_files
      },
      include_package_data=True,
      install_requires=[
          'numpy', 'networkx >= 2.2', 'astunparse', 'sympy', 'scipy', 'pyyaml',
          'absl-py', 'ply', 'websockets', 'graphviz', 'requests', 'flask',
          'scikit-build', 'cmake', 'aenum'
      ],
      tests_require=['coverage'],
      scripts=['scripts/diode', 'scripts/dacelab', 'scripts/sdfv'])
