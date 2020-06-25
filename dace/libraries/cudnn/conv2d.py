import dace.library
from dace.transformation.pattern_matching import ExpandTransformation
from dace.libraries.cudnn import cudnn

@dace.library.expansion
class ExpandConv2DCuDNN(ExpandTransformation):

    environments = [cudnn.cuDNN]
    @staticmethod
    def expansion(node, state, sdfg: dace.SDFG):
        code = '''
                     cudnnSetStream(cudnn_handle, __dace_current_stream);
                     float alpha = 1.0, beta = 0.0;
                     checkCUDNN(cudnnConvolutionForward(cudnn_handle_{i}, &alpha, xDesc_{i}, x,
                                             fDesc_{i}, f,
                                             convDesc_{i}, algo_{i},
                                             workSpace_{i}, workSpaceSizeInBytes_{i},
                                             &beta,
                                             yDesc_{i}, y));
                '''
        tasklet = dace.sdfg.nodes.Tasklet(node.name,
                                          node.in_connectors,
                                          node.out_connectors,
                                          code,
                                          language= dace.dtypes.Language.CPP)
        sdfg.append_init_code('')
        sdfg.append_exit_code('')
        return tasklet


@dace.library.node
class Conv2D(dace.sdfg.nodes.LibraryNode):
    implementations = {"cudnn": ExpandConv2DCuDNN}
    default_implementation = None

    def __init__(self, name, *args, **kwargs):
        super().__init__(name,
                         *args,
                         inputs ={"x", "f"},
                         outputs={"y"},
                         **kwargs)
