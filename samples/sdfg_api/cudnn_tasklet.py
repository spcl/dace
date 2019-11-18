''' Example that defines a CUDNN C++ tasklet. '''
import dace as dp
import numpy as np
import os

# First, add libraries to link (CUDNN) to configuration
if os.name == 'nt':
    dp.Config.append('compiler', 'cpu', 'libs', value='cudnn.lib;')
    dp.Config.append('compiler','cpu','libs', value='cuda.lib')
else:
    dp.Config.append('compiler', 'cpu', 'libs', value='libcudnn.so')
#####################################################################

# Create symbols
H = dp.symbol('H')
R = dp.symbol('R')
S = dp.symbol('S')
W = dp.symbol('W')
H.set(10)
R.set(3)
S.set(3)
W.set(10)

# Create a GPU SDFG with a custom C++ tasklet
sdfg = dp.SDFG('cudnntest')
state = sdfg.add_state()

# Add arrays
sdfg.add_array('X', [H, W], dtype=dp.float32)
sdfg.add_array('F', [R, S], dtype=dp.float32)
sdfg.add_array('Y', [H-R+1, W-S+1], dtype=dp.float32)

# Add transient GPU arrays
sdfg.add_transient('gX', [H, W], dp.float32, dp.StorageType.GPU_Global)
sdfg.add_transient('gF', [R, S], dp.float32, dp.StorageType.GPU_Global)
sdfg.add_transient('gY', [H-R+1, W-S+1], dp.float32, dp.StorageType.GPU_Global)

# Add custom C++ tasklet to graph
tasklet = state.add_tasklet(
    # Tasklet name (can be arbitrary)
    name='convfor',
    # Inputs and output names (will be obtained as raw pointers)
    inputs={'x', 'f'},
    outputs={'y'},
    # Custom code (on invocation)
    code='''
    // Set the current stream to match DaCe (for correct synchronization)
    cudnnSetStream(handle, __dace_current_stream);          
    #define checkCUDNN(expression)                           \\
    {{                                                        \\
      cudnnStatus_t status = (expression);                   \\
      if (status != CUDNN_STATUS_SUCCESS) {{                  \\
        printf(\"%s \\n \\n \", cudnnGetErrorString(status));                \\
      }}                                                      \\
    }}    
    
    cudnnTensorDescriptor_t xDesc;
    checkCUDNN(cudnnCreateTensorDescriptor(&xDesc));
    checkCUDNN(cudnnSetTensor4dDescriptor(xDesc,
                               /*format=*/CUDNN_TENSOR_NCHW,
                               /*dataType=*/CUDNN_DATA_FLOAT,
                               /*batch_size=*/1,
                               /*channels=*/1,
                               /*height=*/H,
                               /*width=*/W));

    cudnnTensorDescriptor_t yDesc;
    cudnnCreateTensorDescriptor(&yDesc);
    checkCUDNN(cudnnSetTensor4dDescriptor(yDesc,
                               /*format=*/CUDNN_TENSOR_NCHW,
                               /*dataType=*/CUDNN_DATA_FLOAT,
                               /*batch_size=*/1,
                               /*channels=*/1,
                               /*image_height=*/H-R+1,
                               /*image_width=*/W-S+1));

    cudnnFilterDescriptor_t fDesc;
    cudnnCreateFilterDescriptor(&fDesc);
    checkCUDNN(cudnnSetFilter4dDescriptor(fDesc,
                               /*dataType=*/CUDNN_DATA_FLOAT,
                               /*format=*/CUDNN_TENSOR_NCHW,
                               /*out_channels=*/1,
                               /*in_channels=*/1,
                               /*kernel_height=*/R,
                               /*kernel_width=*/S));

    cudnnConvolutionDescriptor_t convDesc;
    cudnnCreateConvolutionDescriptor(&convDesc);
    checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc,
                                   /*pad_height=*/0,
                                   /*pad_width=*/0,
                                   /*vertical_stride=*/1,
                                   /*horizontal_stride=*/1,
                                   /*dilation_height=*/1,
                                   /*dilation_width=*/1,
                                   /*mode=*/CUDNN_CROSS_CORRELATION,
                                   /*computeType=*/CUDNN_DATA_FLOAT));
    cudnnConvolutionFwdAlgo_t algo;
    checkCUDNN(cudnnGetConvolutionForwardAlgorithm(handle,
                                    xDesc,
                                    fDesc,
                                    convDesc,
                                    yDesc,
                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                    /*memoryLimitInBytes=*/0,
                                    &algo));
    size_t workSpaceSizeInBytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(handle,
                                            xDesc,
                                            fDesc,
                                            convDesc,
                                            yDesc,
                                            algo,
                                            &workSpaceSizeInBytes));
    void* workSpace{nullptr};
    cudaMalloc(&workSpace, workSpaceSizeInBytes);
    
    float alpha = 1.0, beta = 0.0;
    cudnnConvolutionForward(handle, &alpha, xDesc, x,
                            fDesc, f,
                            convDesc, algo,
                            workSpace, workSpaceSizeInBytes,
                            &beta,
                            yDesc, y);
                            
    ''',
    # Global code (top of file, can be used for includes and global variables)
    code_global='''
    #include <cudnn.h>
    cudnnHandle_t handle;
    ''',
    # Initialization code (called in __dace_init())
    code_init='''
    cudnnCreate(&handle);
    ''',
    # Teardown code (called in __dace_exit())
    code_exit='''
    cudnnDestroy(handle);
    ''',
    # Language (C++ in this case)
    language=dp.Language.CPP)

# Add CPU arrays, GPU arrays, and connect to tasklet
X = state.add_read('X')
F = state.add_read('F')
Y = state.add_write('Y')
gX = state.add_access('gX')
gF = state.add_access('gF')
gY = state.add_access('gY')

# Memlets cover all data
state.add_edge(gX, None, tasklet, 'x', dp.Memlet.simple('gX', '0:H, 0:W'))
state.add_edge(gF, None, tasklet, 'f', dp.Memlet.simple('gF', '0:R, 0:S'))
state.add_edge(tasklet, 'y', gY, None, dp.Memlet.simple('gY', '0:(H-R+1), 0:(W-S+1)'))


# Between two arrays we use a convenience function, `add_nedge`, which is
# short for "no-connector edge", i.e., `add_edge(u, None, v, None, memlet)`.
state.add_nedge(X, gX, dp.Memlet.simple('gX', '0:H, 0:W'))
state.add_nedge(F, gF, dp.Memlet.simple('gF', '0:R, 0:S'))
state.add_nedge(gY, Y, dp.Memlet.simple('Y', '0:(H-R+1), 0:(W-S+1)'))

######################################################################

# Validate GPU SDFG
sdfg.validate()

# Draw SDFG to file
sdfg.draw_to_file()

######################################################################

if __name__ == '__main__':
    # Initialize arrays. We are using column-major order to support CUBLAS!
    X = np.ndarray([H.get(), W.get()], dtype=np.float32)
    F = np.ndarray([R.get(), S.get()], dtype=np.float32)
    Y = np.ndarray([H.get()-R.get()+1, W.get()-S.get()+1], dtype=np.float32)

    X[:] = np.random.rand(H.get(), W.get())
    F[:] = [[0,0,0],[0,10,0],[0,0,0]]

    from scipy import signal

    Y_ref = signal.convolve2d(X, F, mode='valid')
    # We can safely call numpy with arrays allocated on the CPU, since they
    # will be copied.
    sdfg(X=X, F=F, Y=Y, H=H, W=W, R=R, S=S)
    diff = np.linalg.norm(Y - Y_ref) / (H.get() * W.get())
    print('Difference: ', diff)

    if(diff < 1e-5):
        print("Success")
