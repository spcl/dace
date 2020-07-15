import dace.library
from dace.transformation.pattern_matching import ExpandTransformation
from dace.libraries.cudnn import cudnn


@dace.library.expansion
class ExpandConv2dBackpropInputCudnn(ExpandTransformation):
    environments = [cudnn.cuDNN]

    @staticmethod
    def expansion(node, state, sdfg: dace.SDFG):
        name = sdfg._find_new_name(node.name)
        [N, H, W, C] = [node._data_format.find(x) for x in ['N', 'H', 'W', 'C']]
        code = '''
                    cudnnSetStream(cudnn_handle_{i}, __dace_current_stream);          
                    float alpha = 1.0, beta = 0.0;
                    checkCUDNN(cudnnConvolutionBackwardData(cudnn_handle_{i}, &alpha, fDesc_{i}, w,
                                                            yDesc_{i}, dy, convDesc_{i}, algo_{i},
                                                            workSpace_{i}, workSpaceSizeInBytes_{i},
                                                            &beta, xDesc_{i}, dx));
                '''.format(i=name)
        tasklet = dace.sdfg.nodes.Tasklet(name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language=dace.dtypes.Language.CPP)
        sdfg.append_init_code('''

                cudnnCreate(&cudnn_handle_{i});
                
                cudnnCreateFilterDescriptor(&fDesc_{i});
                checkCUDNN(cudnnSetFilter4dDescriptor(fDesc_{i},
                                           /*dataType=*/CUDNN_DATA_FLOAT,
                                           /*format=*/CUDNN_TENSOR_{format},
                                           /*out_channels=*/{K},
                                           /*in_channels=*/{C},
                                           /*kernel_height=*/{R},
                                           /*kernel_width=*/{S}));

                checkCUDNN(cudnnCreateTensorDescriptor(&xDesc_{i}));
                checkCUDNN(cudnnSetTensor4dDescriptor(xDesc_{i},
                                           /*format=*/CUDNN_TENSOR_{format},
                                           /*dataType=*/CUDNN_DATA_FLOAT,
                                           /*batch_size=*/{N},
                                           /*channels=*/{C},
                                           /*height=*/{H},
                                           /*width=*/{W}));

                cudnnCreateConvolutionDescriptor(&convDesc_{i});
                checkCUDNN(cudnnSetConvolution2dDescriptor(convDesc_{i},
                                               /*pad_height=*/{padh},
                                               /*pad_width=*/{padw},
                                               /*vertical_stride=*/{vstr},
                                               /*horizontal_stride=*/{hstr},
                                               /*dilation_height=*/{dilh},
                                               /*dilation_width=*/{dilw},
                                               /*mode=*/CUDNN_CROSS_CORRELATION,
                                               /*computeType=*/CUDNN_DATA_FLOAT));
                
                int out_k = 0;
                int out_c = 0;
                int out_h = 0;
                int out_w = 0;                                               
                checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convDesc_{i}, xDesc_{i}, fDesc_{i}, 
                                                                 &out_k , &out_c, &out_h, &out_w));

                checkCUDNN(cudnnCreateTensorDescriptor(&yDesc_{i}));
                checkCUDNN(cudnnSetTensor4dDescriptor(yDesc_{i},
                                           /*format=*/CUDNN_TENSOR_{format},
                                           /*dataType=*/CUDNN_DATA_FLOAT,
                                           /*batch_size=*/out_k ,
                                           /*channels=*/out_c,
                                           /*height=*/out_h,
                                           /*width=*/out_w));

                checkCUDNN(cudnnGetConvolutionBackwardDataAlgorithm(
                            cudnn_handle_{i}, fDesc_{i}, yDesc_{i}, convDesc_{i}, xDesc_{i},
                            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0, &algo_{i}));

                checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                    cudnn_handle_{i}, fDesc_{i}, yDesc_{i}, convDesc_{i}, xDesc_{i}, algo_{i}, &workSpaceSizeInBytes_{i}));

                cudaMalloc(&workSpace_{i}, workSpaceSizeInBytes_{i});
        '''.format(N=node._output_dims_list[N], C=node._output_dims_list[C], H=node._output_dims_list[H],
                   W=node._output_dims_list[W],
                   K=node._filter_dims_list[3], R=node._filter_dims_list[0], S=node._filter_dims_list[1],
                   padh=node._padh, padw=node._padw, vstr=node._strides[H], hstr=node._strides[W],
                   dilh=node._dilations[H], dilw=node._dilations[W], i=name, format=node._data_format))
        sdfg.append_exit_code('''
            cudnnDestroy(cudnn_handle_{i});
            cudnnDestroyTensorDescriptor(xDesc_{i});
            cudnnDestroyTensorDescriptor(yDesc_{i});
            cudnnDestroyFilterDescriptor(fDesc_{i});
            cudnnDestroyConvolutionDescriptor(convDesc_{i});
            cudaFree(workSpace_{i});
        '''.format(i=name))
        sdfg.append_global_code('''
            cudnnHandle_t cudnn_handle_{i};
            cudnnFilterDescriptor_t fDesc_{i};
            cudnnTensorDescriptor_t xDesc_{i};
            cudnnConvolutionDescriptor_t convDesc_{i};
            cudnnTensorDescriptor_t yDesc_{i};
            cudnnConvolutionBwdDataAlgo_t algo_{i};
            size_t workSpaceSizeInBytes_{i} = 0;
            void* workSpace_{i}{{nullptr}};

            #define checkCUDNN(expression)                                   \\
            {{                                                                \\
              cudnnStatus_t status = (expression);                           \\
              if (status != CUDNN_STATUS_SUCCESS) {{                          \\
                printf(\"%d: %s\\n\", __LINE__, cudnnGetErrorString(status));\\
              }}                                                              \\
            }}

            '''.format(i=name))
        return tasklet


@dace.library.node
class Conv2DBackpropInput(dace.sdfg.nodes.LibraryNode):
    implementations = {"cudnn": ExpandConv2dBackpropInputCudnn}
    default_implementation = None

    def __init__(self,
                 name,
                 output_dims_list=None,
                 filter_dims_list=None,
                 padh=0,
                 padw=0,
                 strides=None,
                 dilations=None,
                 data_format=None,
                 *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs={"w", "dy"},
                         outputs={"dx"},
                         **kwargs)
        self._output_dims_list = output_dims_list
        self._filter_dims_list = filter_dims_list
        self._padh = padh
        self._padw = padw
        self._strides = strides
        self._dilations = dilations
        self._data_format = data_format
