# Code for the Master thesis 'DaCe on GPU for Climate and Weather Models Using CLOUDSC as a Case Study' by Samuel Martin

This folder contains the required code to reproduce the SDFG and results and presented in my master thesis. It does not
contain all small and failed experiments/programs and so on, for these please take a look at the 
[repository used to develop it](https://github.com/Sajohn-CH/dace/tree/thesis_playground).

This folder contains the source code to generated the SDFG for the full CLOUDSC code and for the partial CLOUDSC
examples used as well as the small example kernels grouped into three classes.

## Required packages
My run scripts require the following additional python packages in order to collect data and plot:

- pandas
- seaborn

## Structure of this folder
This folder contains several subfolders. These are:

- `basic_sdfg`: Storage of basic SDFGs used
- `fortran_programs`: The source code of the Fortran programs used
- `src`: The python code used to generate the SDFGs and so on
- `results_v2`: Folder where the results of the smaller example codes are stored

In addition the scripts might create two additional folders:
- `sdfg_graphs`: Intermediate SDFGs will be stored here for further inspection
- `full_cloudsc_logs`: The logfiles created when running the scripts for the full cloudsc as well as the generated SDFGs

## SDFG generation process
In order to cut down the time to generate a SDFG, especially for the full CLOUDSC code I implemented a two phase model.
First a basic SDFG is generated without any optimisations applied, but all loops are converted to maps if possible as
this takes the longest amount of time (on my local machine 2-3 hours). These are then stored in the `basic_sdfg` folder.
This SDFG is then loaded to perform the further optimisations (these take on my local machine something from 2min up to
just under 30min). 

For the full cloudsc code the basic SDFG is already provided for your convenience and as some of my scripts have certain
state names hard coded which might change depending on the other of transformations applied which is not fixed (when
using `sdfg.apply_transformation_repeated`)

## How to run
All commands given here assume that you execute them for this directory.

### Full CLOUDSC

To generate the SDFG of the full CLOUDSC code as read from `fortran_programs/cloudscexp4.f90` execute:
```bash
python3 src/gen_full_cloudsc.py gen <opt-level>
```
`<opt-level>` can be one of the following:

- baseline: Do not apply any specific optimization
- k-caching: Apply K-caching only
- change-strides: Apply the change in strides / change in array order optimization
- all: Apply both optimizations

An optional `--device` with either `GPU` or `CPU` can be added to specify the device to compile it for. It defaults to
`GPU`. The generated SDFG is stored into the folder `full_cloudsc_logs`. To generate the C++/CUDA code execute (it takes
again optionally a `--device` flag, `<opt-level>` as before):
```bash
python3 src/gen_full_cloudsc.py compile <opt-level>
```

Generating the SDFG my be faster locally than on ault, in this case the generated SDFG needs to be copied to ault using
`scp`. The graph will be located in the folder `full_cloudsc_logs` together with a logfile. The provided bash script
`run_cloudsc.sh` can be run to execute the cloudsc code. Please adjust the paths inside it first to the location of your
CLOUDSC repository. In order to prepared the CLOUDSC repository please follow the instructions inside the
[cloudsc_instructions folder](cloudsc_instructions/README.md)

The python scripts `src/run_full_cloudsc.py` was used to run the cloudsc code multiple times and plot the results. You
will need to adapt the path going to the CLOUDSC repository inside `src/utils/full_cloudsc.py`. Use `--help` to
discover the option the scripts take. This works for any python run script in this folder.


### Smaller example code
The smaller example codes can be run using the script `src/run2.py` and `src/plot2.py`, use `--help` to discover their
options.

#### CLOUDSC example programs
There are 5 example programs used to develop K-caching and the change in strides. They represent the whole CLOUDSC
program structure but are much smaller. The first one was used to develop the DaCe transformations, the other four
implement them manually and were used to get a first feeling if it makes sense to implement the transformations.

- `cloudsc_vert_loop_10.f90`: This was used to develop the DaCe transformations
- `cloudsc_vert_loop_4_ZSOLQA.f90`: Baseline without any optimisation
- `clousdc_vert_loop_6_ZSOLQA.f90`: Change in stride applied
- `clousdc_vert_loop_6_1_ZSOLQA.f90`: Change in stride with fix for temporary array applied
- `cloudsc_vert_loop_7_3.f90`: Change in stride and K-caching

#### CLOUDSC kernels grouped into classes
There are 3-4 programs per class as described in the thesis. The source files are named after the following scheme:
`cloudsc_class<class-number>_<line>.f90` where `<class-number>` is the class number 1, 2 or 3 and `<line>` the name of
the program also indicating the line where the program starts in the [cloudsc source code](https://github.com/ecmwf-ifs/dwarf-p-cloudsc/blob/main/src/cloudsc_fortran/cloudsc.F90).
