pipeline {
  agent any
  stages {
    stage('Setup') {
      steps {
        sh '''
                    echo "Reverting configuration to defaults"
                    rm -f ~/.dace.conf
                    echo "Clearing caches"
                    rm -rf .dacecache tests/.dacecache client_configs tests/client_configs
                    echo "Installing additional dependencies"
                    pip3 install --upgrade --user tensorflow-gpu==1.14.0
                    echo "Installing DaCe"
                    pip3 uninstall -y dace
                    pip3 install --upgrade --user ".[testing]" .
                    pip3 install --user cmake
                    pip3 install --user coverage
                    pip3 install --user mpi4py
                '''
      }
    }

    stage('Test') {
      parallel {
        stage('Test CUDA') {
          steps {
            sh '''export PYTHON_BINARY="python3 -m coverage run --source=`pwd`/dace --parallel-mode"
            export COVERAGE_RCFILE=`pwd`/.coveragerc
            export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export CUDA_ROOT=/usr/local/cuda
export DACE_debugprint=1
tests/cuda_test.sh
                '''
          }
        }

        stage('Test Xilinx') {
          steps {
            sh '''export PYTHON_BINARY="python3 -m coverage run --source=`pwd`/dace --parallel-mode"
            export COVERAGE_RCFILE=`pwd`/.coveragerc
            export PATH=/opt/Xilinx/SDx/2018.2/bin:$PATH
export DACE_compiler_xilinx_executable=xocc
export DACE_compiler_xilinx_platform=xilinx_vcu1525_dynamic_5_1
export XILINXD_LICENSE_FILE=2100@sgv-license-01
export DACE_debugprint=1
tests/xilinx_test.sh 0
'''
          }
        }
        
        

        stage('Test MPI') {
          steps {
            sh '''export PYTHON_BINARY="python3 -m coverage run --source=`pwd`/dace --parallel-mode"
            export COVERAGE_RCFILE=`pwd`/.coveragerc
            export PATH=/opt/mpich3.2.11/bin:$PATH
export DACE_debugprint=1
tests/mpi_test.sh'''
          }
        }

      }
    }
    stage('Test Intel FPGA') {
          steps {
            sh '''export PYTHON_BINARY="python3 -m coverage run --source=`pwd`/dace --parallel-mode"
            export COVERAGE_RCFILE=`pwd`/.coveragerc
            source /opt/intelFPGA_pro/19.1/hld/init_opencl.sh
export DACE_debugprint=1
tests/intel_fpga_test.sh 
'''
          }
    }
    stage('Report') {
      steps {
        sh '''python3 -m coverage combine . tests/ tests/*/
        python3 -m coverage report
        python3 -m coverage xml
        curl -s https://codecov.io/bash | bash
        '''
      }
    }
  }
}
