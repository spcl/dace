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
                    pip3 install --ignore-installed --upgrade --user .
                    pip3 install --user cmake
                '''
      }
    }

    stage('Test') {
      parallel {
        stage('Test CUDA') {
          steps {
            sh '''export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
export CUDA_ROOT=/usr/local/cuda
export DACE_debugprint=1
tests/cuda_test.sh
                '''
          }
        }

        stage('Test Xilinx') {
          steps {
            sh '''export PATH=/opt/Xilinx/SDx/2018.2/bin:$PATH
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
            sh '''export PATH=/opt/mpich3.2.11/bin:$PATH
export DACE_debugprint=1
tests/mpi_test.sh'''
          }
        }

      }
    }

  }
}