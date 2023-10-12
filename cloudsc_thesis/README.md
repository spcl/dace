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
cloudsc repository. It also needs to be adapted.


