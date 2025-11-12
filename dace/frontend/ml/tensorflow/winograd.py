# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import numpy as np
import re
from dace.memlet import Memlet
from dace import dtypes
from math import ceil


def add_cublas_cusolver(sdfg: dace.SDFG):
    """ Add CUBLAS and CUSOLVER handles to the SDFG. """
    sdfg.set_global_code("""
        #include <iostream>
        #include <cublas_v2.h>
        #include <cusolverDn.h>
        #include <cusparse.h>
        #include <thrust/complex.h>
        cublasHandle_t handle;
        float* const_zero;
        float* const_pone;
        """)
    sdfg.set_init_code("""
            cublasCreate(&handle);
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
            cudaMalloc<float>(&const_zero, sizeof(float) *
            1);
            float zero = 0.0;
            cudaMemcpy(const_zero, &zero, sizeof(float) * 1,
            cudaMemcpyHostToDevice);
            cudaMalloc<float>(&const_pone, sizeof(float) *
            1);
            float pone = 1.0;
            cudaMemcpy(const_pone, &pone, sizeof(float) * 1,
            cudaMemcpyHostToDevice);
        """)
    sdfg.set_exit_code("""
            cublasDestroy(handle);
            cudaFree(const_zero);
            cudaFree(const_pone);
        """)


# TODO: Use library nodes


# Make sure C_memlet has a wcr if map_exit is used
# Matrix multiplication with one matrix as constant
def mm_small(
    state,
    A_node,
    B_node,
    C_node,
    A_subset=None,
    B_subset=None,
    C_subset=None,
    A_memlet=None,
    B_memlet=None,
    C_memlet=None,
    map_entry=None,
    map_exit=None,
    A_direct=True,
    B_direct=True,
):
    # C = A@B
    sdfg = state.parent
    Cshape = C_node.desc(sdfg).shape if C_subset is None else C_subset
    if isinstance(B_node, str):
        someshape = list(A_node.desc(sdfg).shape) if A_subset is None else A_subset
        someshape = [someshape[0]] + [someshape[1]] + [Cshape[1]]
    elif isinstance(A_node, str):
        someshape = list(B_node.desc(sdfg).shape) if B_subset is None else B_subset
        someshape = [Cshape[0]] + [someshape[0]] + [someshape[1]]
    else:
        raise AssertionError("one of these have to be an str, else don't call this function")

    mapRange = ["0:" + str(_s) for _s in someshape]
    mapParams = ["i2", "i3", "i4"]
    mmapEntry, mmapExit = state.add_map(
        "matmul_sequential",
        dict(zip(mapParams, mapRange)),
        schedule=dace.ScheduleType.Sequential,
    )
    if isinstance(A_node, str):
        tasklet = state.add_tasklet("matmul_sequential", {"j1"}, {"out"}, "out=" + A_node + "[i2, i3]" + "*j1")
        if B_memlet:
            state.add_edge(map_entry, None, mmapEntry, None, B_memlet)
            b_memlet_trailing = [str(_t[0]) for _t in B_memlet.subset[-2:]]
            state.add_edge(
                mmapEntry,
                None,
                tasklet,
                "j1",
                Memlet.simple(B_node, ",".join(["i3", "i4"] + b_memlet_trailing)),
            )
        else:
            state.add_edge(
                B_node if B_direct else map_entry,
                None,
                mmapEntry,
                None,
                Memlet.from_array(B_node, B_node.desc(sdfg)),
            )
            state.add_edge(
                mmapEntry,
                None,
                tasklet,
                "j1",
                Memlet.simple(B_node, ",".join(["i3", "i4"])),
            )
    else:
        tasklet = state.add_tasklet("matmul_sequential", {"j0"}, {"out"}, "out=j0*" + B_node + "[i3, i4]")
        if A_memlet:
            state.add_edge(map_entry, None, mmapEntry, None, A_memlet)
            a_memlet_trailing = [str(_t[0]) for _t in A_memlet.subset[-2:]]
            state.add_edge(
                mmapEntry,
                None,
                tasklet,
                "j0",
                Memlet.simple(A_node, ",".join(["i2", "i3"] + a_memlet_trailing)),
            )
        else:
            state.add_edge(
                A_node if A_direct else map_entry,
                None,
                mmapEntry,
                None,
                Memlet.from_array(A_node, A_node.desc(sdfg)),
            )
            state.add_edge(
                mmapEntry,
                None,
                tasklet,
                "j0",
                Memlet.simple(A_node, ",".join(["i2", "i3"])),
            )

    if C_memlet:
        c_memlet_trailing = [str(_t[0]) for _t in C_memlet.subset[-2:]]
        state.add_edge(
            tasklet,
            "out",
            mmapExit,
            None,
            Memlet.simple(
                C_node,
                ",".join(["i2", "i4"] + c_memlet_trailing),
                wcr_str="lambda a,b: a+b",
                wcr_conflict=False,
            ),
        )
        state.add_edge(mmapExit, None, map_exit, None, C_memlet)
    else:
        state.add_edge(
            tasklet,
            "out",
            mmapExit,
            None,
            Memlet.simple(
                C_node,
                ",".join(["i2", "i4"]),
                wcr_str="lambda a,b:a+b",
                wcr_conflict=False,
            ),
        )
        state.add_edge(
            mmapExit,
            None,
            C_node if map_exit is None else map_exit,
            None,
            Memlet.simple(
                C_node,
                ",".join(["0:" + str(_s) for _s in C_node.desc(sdfg).shape]),
                wcr_str="lambda a,b:a+b",
                wcr_conflict=False,
            ),
        )


# takes input and stores output in column major order. give swapped input (B, A)
# instead of (A, B)
def mm(
    state,
    A_node,
    B_node,
    C_node,
    A_mode: str = "N",
    B_mode: str = "N",
    label: str = None,
    A_subset=None,
    B_subset=None,
    C_subset=None,
    A_memlet=None,
    B_memlet=None,
    C_memlet=None,
    map_entry=None,
    map_exit=None,
    shadow_a=False,
    shadow_b=False,
    buffer_a=False,
    buffer_c=False,
):
    sdfg = state.parent
    Adesc = A_node.desc(sdfg)
    Bdesc = B_node.desc(sdfg)
    Cdesc = C_node.desc(sdfg)
    Ashape = Adesc.shape if A_subset is None else A_subset
    Bshape = Bdesc.shape if B_subset is None else B_subset
    Cshape = Cdesc.shape if C_subset is None else C_subset

    # TODO change here for subset
    kdim_A = 0 if A_mode == "N" else 1
    lda = Ashape[1]
    ldb = Bshape[1]
    ldc = Cshape[1]

    # Set label
    if label is None:
        label = state.label

    # Create tasklet
    tasklet = state.add_tasklet(
        name=label + "_" + "mm_tasklet",
        inputs={"a", "b"},
        outputs={"c"},
        code="""
        cublasSetStream(handle, __dace_current_stream);
        cublasStatus_t status = cublasSgemm(
            handle,
            CUBLAS_OP_{amode}, CUBLAS_OP_{bmode},
            {m}, {n}, {k},
            const_pone,
            (float*)a, {lda},
            (float*)b, {ldb},
            const_zero,
            (float*)c, {ldc}
        );
        if (status)
            printf("Multiplication {a}*{b}->{c} failed (status %d)\\n", status);
        """.format(
            a=A_node.data,
            b=B_node.data,
            c=C_node.data,
            amode=A_mode,
            bmode=B_mode,
            m=Cshape[1],
            n=Cshape[0],
            k=Ashape[kdim_A],
            lda=lda,
            ldb=ldb,
            ldc=ldc,
        ),
        language=dace.dtypes.Language.CPP,
    )

    if not buffer_a:
        state.add_edge(
            A_node if not shadow_a else map_entry,
            None,
            tasklet,
            "a",
            A_memlet or dace.Memlet.from_array(A_node.data, Adesc),
        )
    else:
        # Assume that map_entry is to be used
        memcopy_tasklet = state.add_tasklet(label + "_" + "buffer_copy", {"j0"}, {"out"}, "out=j0")
        bufMapEntry, bufMapExit = state.add_map(
            label + "_bufmap",
            dict(zip(["i3", "i4"], ["0:" + str(_s) for _s in A_subset])),
            schedule=dace.ScheduleType.GPU_Device,
        )
        state.add_edge(map_entry, None, bufMapEntry, None, A_memlet)
        state.add_edge(
            bufMapEntry,
            None,
            memcopy_tasklet,
            "j0",
            Memlet.simple(
                A_node,
                ",".join(["i3", "i4"] + [str(_t[0]) for _t in A_memlet.subset[-2:]]),
            ),
        )
        bufferNode = state.add_transient(
            A_node.data + "_buffercopy",
            A_subset,
            Adesc.dtype,
            storage=dace.StorageType.GPU_Global,
        )
        state.add_edge(memcopy_tasklet, "out", bufMapExit, None, Memlet.simple(bufferNode, "i3,i4"))
        state.add_edge(
            bufMapExit,
            None,
            bufferNode,
            None,
            dace.Memlet.from_array(bufferNode, bufferNode.desc(sdfg)),
        )
        state.add_edge(
            bufferNode,
            None,
            tasklet,
            "a",
            dace.Memlet.from_array(bufferNode, bufferNode.desc(sdfg)),
        )
    state.add_edge(
        B_node if not shadow_b else map_entry,
        None,
        tasklet,
        "b",
        B_memlet or dace.Memlet.from_array(B_node.data, Bdesc),
    )
    if not buffer_c:
        state.add_edge(
            tasklet,
            "c",
            C_node if map_exit is None else map_exit,
            None,
            C_memlet or dace.Memlet.from_array(C_node.data, Cdesc),
        )
    else:
        memcopy_tasklet = state.add_tasklet(label + "_" + "buffer_copy_output", {"j0"}, {"out"}, "out=j0")
        bufMapEntry, bufMapExit = state.add_map(
            label + "_bufmap",
            dict(zip(["i3", "i4"], ["0:" + str(_s) for _s in C_subset])),
            schedule=dace.ScheduleType.GPU_Device,
        )
        bufferNode = state.add_transient(
            C_node.data + "_buffercopy_output",
            C_subset,
            Cdesc.dtype,
            storage=dace.StorageType.GPU_Global,
        )
        state.add_edge(
            tasklet,
            "c",
            bufferNode,
            None,
            dace.Memlet.from_array(bufferNode, bufferNode.desc(sdfg)),
        )
        state.add_edge(
            bufferNode,
            None,
            bufMapEntry,
            None,
            dace.Memlet.from_array(bufferNode, bufferNode.desc(sdfg)),
        )
        state.add_edge(bufMapEntry, None, memcopy_tasklet, "j0", Memlet.simple(bufferNode, "i3,i4"))

        state.add_edge(
            memcopy_tasklet,
            "out",
            bufMapExit,
            None,
            Memlet.simple(
                C_node,
                ",".join(["i3", "i4"] + [str(_t[0]) for _t in C_memlet.subset[-2:]]),
            ),
        )
        state.add_edge(bufMapExit, None, map_exit, None, C_memlet)


IMAGE_TILE_SIZE = 4
OUTPUT_TILE_SIZE = 2

bt = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]).astype(np.float32)
b = np.array([[1, 0, 0, 0], [0, 1, -1, 1], [-1, 1, 1, 0], [0, 0, 0, -1]]).astype(np.float32)
g = np.array([[1.0, 0, 0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0, 0, 1]]).astype(np.float32)
gt = np.array([[1, 0.5, 0.5, 0], [0, 0.5, -0.5, 0], [0, 0.5, 0.5, 1]]).astype(np.float32)
at = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).astype(np.float32)
a = np.array([[1, 0], [1, 1], [1, -1], [0, -1]]).astype(np.float32)


def printer(*inp):
    print(inp)


def string_builder(string):
    """ To match DaCe variable naming conventions, replaces all undesired
        characters with "_".
    """
    newstring = string
    if string[0].isdigit():
        newstring = "_" + string
    out = re.sub("[^a-zA-Z0-9_]", "_", newstring)
    return out


def winograd_convolution(dace_session, tf_node):
    debugNodes = []
    state = dace_session.state
    add_cublas_cusolver(dace_session.graph)
    #############Add constants for transformation matrices###############
    dace_session.graph.add_constant('Btrans', bt)
    dace_session.graph.add_constant('B', b)
    bNode = 'B'
    bTransposeNode = 'Btrans'

    dace_session.graph.add_constant('G', g)
    dace_session.graph.add_constant('Gtrans', gt)
    gNode = 'G'
    gTransposeNode = 'Gtrans'

    dace_session.graph.add_constant('Atrans', at)
    dace_session.graph.add_constant('A', a)
    aNode = 'A'
    aTransposeNode = 'Atrans'

    inputNodes = []
    inputParams = []
    inputDims = []
    for _inp in tf_node.inputs:
        _node, _params, _dims = dace_session.create_and_add_input_node(_inp)
        inputNodes.append(_node)
        inputParams.append(_params)
        inputDims.append(_dims)
    # Manually add copy for kernel from CPU to GPU
    kernel_desc = inputNodes[1].desc(dace_session.graph)
    kernelGPU = state.add_transient(
        inputNodes[1].data + "GPU",
        shape=kernel_desc.shape,
        dtype=kernel_desc.dtype,
        lifetime=dtypes.AllocationLifetime.SDFG,
        storage=dace.StorageType.GPU_Global,
    )
    state.add_edge(
        inputNodes[1],
        None,
        kernelGPU,
        None,
        Memlet.from_array(inputNodes[1], inputNodes[1].desc(dace_session.graph)),
    )
    inputNodes[1] = kernelGPU
    outputList = dace_session.create_and_add_output_node(tf_node)
    outputDims = dace_session.get_default_dims(tf_node.outputs[0])
    if str(tf_node.get_attr("padding"))[2:-1] == "SAME":
        paddedInput, paddedDims = dace_session.inputPadding(
            tf_node,
            inputNodes[0],
            inputNodes[0].desc(dace_session.graph),
            outputList[0].desc(dace_session.graph).shape[1],
            inputNodes[1].desc(dace_session.graph).shape[0],
            tf_node.get_attr("strides")[1],
            inputDims[0],
        )
        inputDims[0] = paddedDims
        inputNodes[0] = paddedInput
    outputShape = [int(_s) for _s in tf_node.outputs[0].shape]
    inputViewShape = [
        IMAGE_TILE_SIZE,
        IMAGE_TILE_SIZE,
        tf_node.inputs[0].shape[-1],
        outputShape[0] * ceil(outputShape[1] / OUTPUT_TILE_SIZE) * ceil(outputShape[2] / OUTPUT_TILE_SIZE),
    ]
    inputViewDims = ["0:" + str(_x) for _x in inputViewShape]
    ########Tiling the image#################################
    inputViewParams = [
        "i3%" + str(outputShape[0]),
        "(i3/" + str(outputShape[0]) + ")%"
        # + str(output_shape[0] * ceil(output_shape[1] / OUTPUT_TILE_SIZE))
        + str(ceil(outputShape[2] / OUTPUT_TILE_SIZE)) + "*" + str(OUTPUT_TILE_SIZE) + "+i0",
        # + str(
        #    ceil(output_shape[1] / OUTPUT_TILE_SIZE)
        #    * ceil(output_shape[2] / OUTPUT_TILE_SIZE)
        # ),
        "int_floor(i3,"
        # + str(ceil(output_shape[1] / OUTPUT_TILE_SIZE))
        + str(outputShape[0] * ceil(outputShape[2] / OUTPUT_TILE_SIZE)) + ")*" + str(OUTPUT_TILE_SIZE) + "+i1",
        "i2",
    ]
    inputView = state.add_transient(
        "V" + "_".join([str(_s) for _s in inputViewShape]),
        inputViewShape,
        dace.float32,
        dace.StorageType.GPU_Global,
    )
    mapEntry, mapExit = state.add_map(
        string_builder(tf_node.name) + "_input_tile",
        dict(zip(inputParams[0], inputViewDims)),
    )
    tasklet = state.add_tasklet(string_builder(tf_node.name) + "_input_tile", {"j0"}, {"out"}, "out = j0")
    dace_session.add_in_memlets([inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputViewParams])
    dace_session.add_out_memlets([inputView], mapExit, tasklet, [inputViewDims], [inputParams[0]])
    ##################Transforming all input tiles#########################
    #[TODO] try to re-use memory
    vNode = state.add_transient(
        "V_output" + "_".join([str(_s) for _s in inputViewShape]),
        inputViewShape,
        dace.float32,
        dace.StorageType.GPU_Global,
    )
    vNode.setzero = True
    mapEntry, mapExit = state.add_map(
        string_builder(tf_node.name) + "_input_txform",
        dict(zip(inputParams[0][0:2], inputViewDims[2:4])),
        dace.ScheduleType.GPU_Device,
    )
    intermediateResultNode = state.add_transient("BtI", bt.shape, dace.float32, dace.StorageType.Register)
    intermediateResultNode.setzero = True
    state.add_edge(
        inputView,
        None,
        mapEntry,
        None,
        Memlet.simple(inputView, ",".join(inputViewDims)),
    )
    mm_small(
        state,
        bTransposeNode,
        inputView,
        intermediateResultNode,
        B_subset=[IMAGE_TILE_SIZE, IMAGE_TILE_SIZE],
        B_memlet=Memlet.simple(inputView, ",".join(inputViewDims[0:2] + inputParams[0][0:2])),
        map_entry=mapEntry,
        B_direct=False,
    )
    mm_small(
        state,
        intermediateResultNode,
        bNode,
        vNode,
        map_exit=mapExit,
        C_subset=[IMAGE_TILE_SIZE, IMAGE_TILE_SIZE],
        C_memlet=Memlet.simple(
            vNode,
            ",".join(inputViewDims[0:2] + inputParams[0][0:2]),
            wcr_str="lambda a,b: a+b",
            wcr_conflict=False,
        ),
        map_entry=mapEntry,
        A_direct=True,
    )
    state.add_edge(
        mapExit,
        None,
        vNode,
        None,
        Memlet.simple(
            vNode,
            ",".join(inputViewDims),
            wcr_str="lambda a,b: a+b",
            wcr_conflict=False,
        ),
    )
    #############Transforming the kernel###############################
    mapEntry, mapExit = state.add_map(
        string_builder(tf_node.name) + "_kernel_txform",
        dict(zip(inputParams[1][0:2], inputDims[1][2:4])),
        dace.ScheduleType.GPU_Device,
    )
    intermediateResultNode = state.add_transient("GF", g.shape, dace.float32, dace.StorageType.Register)
    intermediateResultNode.setzero = True
    processedKernelNode = state.add_transient(
        "U" + "_".join([str(_s) for _s in inputViewShape[0:2] + list(tf_node.inputs[1].shape[-1:-3:-1])]),
        inputViewShape[0:2] + list(tf_node.inputs[1].shape[-1:-3:-1]),
        dace.float32,
        dace.StorageType.GPU_Global,
    )
    processedKernelNode.setzero = True
    state.add_edge(
        inputNodes[1],
        None,
        mapEntry,
        None,
        dace.Memlet.from_array(inputNodes[1].data, inputNodes[1].desc(dace_session.graph)),
    )

    mm_small(
        state,
        gNode,
        inputNodes[1],
        intermediateResultNode,
        map_entry=mapEntry,
        B_subset=tf_node.inputs[1].shape[0:2],
        B_memlet=Memlet.simple(inputNodes[1], ",".join(inputDims[1][0:2] + inputParams[1][0:2])),
        B_direct=False,
    )
    mm_small(
        state,
        intermediateResultNode,
        gTransposeNode,
        processedKernelNode,
        C_subset=[IMAGE_TILE_SIZE, IMAGE_TILE_SIZE],
        C_memlet=Memlet.simple(
            processedKernelNode,
            ",".join(inputViewDims[0:2] + [inputParams[0][1]] + [inputParams[0][0]]),
            wcr_str="lambda a,b: a+b",
            wcr_conflict=False,
        ),
        map_entry=mapEntry,
        map_exit=mapExit,
        A_direct=True,
    )
    state.add_edge(
        mapExit,
        None,
        processedKernelNode,
        None,
        Memlet.simple(
            processedKernelNode.data,
            ",".join(["0:" + str(_s) for _s in processedKernelNode.desc(dace_session.graph).shape]),
            wcr_str="lambda a,b: a+b",
            wcr_conflict=False,
        ),
    )
    mNode = state.add_transient(
        "m" + "_".join([str(_s) for _s in inputViewShape[0:2] + [tf_node.inputs[1].shape[-1], inputViewShape[-1]]]),
        inputViewShape[0:2] + [tf_node.inputs[1].shape[-1], inputViewShape[-1]],
        dace.float32,
        dace.StorageType.GPU_Global,
    )
    mNodeDims = ["0:" + str(_d) for _d in mNode.desc(dace_session.graph).shape]
    mapEntry, mapExit = state.add_map(
        string_builder(tf_node.name) + "_eltwise_product",
        dict(zip(inputParams[0][0:2], inputViewDims[0:2])),
        dace.ScheduleType.Sequential,
    )
    state.add_edge(
        vNode,
        None,
        mapEntry,
        None,
        Memlet.from_array(vNode.data, vNode.desc(dace_session.graph)),
    )
    state.add_edge(
        processedKernelNode,
        None,
        mapEntry,
        None,
        Memlet.from_array(processedKernelNode.data, processedKernelNode.desc(dace_session.graph)),
    )
    mm(
        state,
        vNode,
        processedKernelNode,
        mNode,
        A_subset=inputViewShape[2:4],
        A_memlet=Memlet.simple(vNode, ",".join(inputParams[0][0:2] + inputViewDims[-2:])),
        B_subset=tf_node.inputs[1].shape[-1:-3:-1],
        B_memlet=Memlet.simple(
            processedKernelNode,
            ",".join(inputParams[0][0:2] + ["0:" + str(_s) for _s in tf_node.inputs[1].shape[-1:-3:-1]]),
        ),
        C_subset=[tf_node.inputs[1].shape[-1], inputViewShape[-1]],
        C_memlet=Memlet.simple(mNode, ",".join(inputParams[0][0:2] + mNodeDims[-2:])),
        map_entry=mapEntry,
        map_exit=mapExit,
        shadow_a=True,
        shadow_b=True,
    )
    state.add_edge(mapExit, None, mNode, None, Memlet.simple(mNode, ",".join(mNodeDims)))
    #################OUTPUT TRANSFORMATION################################
    mapRange = [inputDims[1][-1]] + [inputViewDims[-1]]
    mapEntry, mapExit = state.add_map(
        string_builder(tf_node.name) + "_output_txform",
        dict(zip(inputParams[0][0:2], mapRange)),
        dace.ScheduleType.GPU_Device,
    )
    intermediateResultNode = state.add_transient("AtM", at.shape, dace.float32, dace.StorageType.Register)
    intermediateResultNode.setzero = True
    transformedOutputNode = state.add_transient(
        "inv_txformed_output" + "_".join([str(tf_node.inputs[1].shape[-1])] + [str(inputViewShape[-1])]),
        [OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE] + [tf_node.inputs[1].shape[-1]] + [inputViewShape[-1]],
        dace.float32,
        dace.StorageType.GPU_Global,
    )
    transformedOutputNode.setzero = True
    state.add_edge(mNode, None, mapEntry, None, Memlet.simple(mNode, ",".join(mNodeDims)))
    mm_small(
        state,
        aTransposeNode,
        mNode,
        intermediateResultNode,
        B_subset=inputViewShape[0:2],
        B_memlet=Memlet.simple(mNode, ",".join(inputViewDims[0:2] + inputParams[0][0:2])),
        map_entry=mapEntry,
        B_direct=False,
    )
    mm_small(
        state,
        intermediateResultNode,
        aNode,
        transformedOutputNode,
        C_subset=[OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE],
        C_memlet=Memlet.simple(
            transformedOutputNode,
            ",".join(["0:" + str(OUTPUT_TILE_SIZE), "0:" + str(OUTPUT_TILE_SIZE)] + inputParams[0][0:2]),
            wcr_str="lambda a,b:a+b",
            wcr_conflict=False,
        ),
        map_entry=mapEntry,
        map_exit=mapExit,
        A_direct=True,
    )
    state.add_edge(
        mapExit,
        None,
        transformedOutputNode,
        None,
        Memlet.simple(
            transformedOutputNode.data,
            ",".join(["0:" + str(_s) for _s in transformedOutputNode.desc(dace_session.graph).shape]),
            wcr_str="lambda a,b: a+b",
            wcr_conflict=False,
        ),
    )
    ###################Un-Tile the output to NHWC format###################
    outputParams = [
        "i3%" + str(outputShape[0]),
        "(i3/" + str(outputShape[0]) + ")%" + str(ceil(outputShape[2] / OUTPUT_TILE_SIZE)) + "*" +
        str(OUTPUT_TILE_SIZE) + "+i0",
        "int_floor(i3," + str(outputShape[0] * ceil(outputShape[2] / OUTPUT_TILE_SIZE)) + ")*" + str(OUTPUT_TILE_SIZE) +
        "+i1",
        "i2",
    ]
    mapRange = ["0:" + str(_s) for _s in transformedOutputNode.desc(dace_session.graph).shape]
    mapEntry, mapExit = state.add_map(
        string_builder(tf_node.name) + "_output_untile",
        dict(zip(inputParams[0], mapRange)),
    )
    tasklet = state.add_tasklet(string_builder(tf_node.name) + "_output_untile", {"j0"}, {"out"}, "out = j0")
    dace_session.add_in_memlets([transformedOutputNode], mapEntry, tasklet, [mapRange], [inputParams[0]])
    dace_session.add_out_memlets(outputList, mapExit, tasklet, [outputDims], [outputParams])
    ################# Debugging with callbacks #############
    taskletInputs = ["i" + str(index) for index in range(len(debugNodes))]
    callback_tasklet = state.add_tasklet(
        string_builder(tf_node.name) + "_printer",
        {*taskletInputs},
        {},
        string_builder(tf_node.name) + "_printer" + "(" + ",".join(taskletInputs) + ");",
        language=dace.dtypes.Language.CPP,
    )
    for _n, _conn in zip(debugNodes, taskletInputs):
        _n_cpu = state.add_transient(_n.data + "_cpucopy",
                                     _n.desc(dace_session.graph).shape,
                                     _n.desc(dace_session.graph).dtype,
                                     storage=dace.StorageType.CPU_Heap,
                                     lifetime=dtypes.AllocationLifetime.SDFG)
        state.add_edge(_n, None, _n_cpu, None, Memlet.from_array(_n, _n.desc(dace_session.graph)))
        state.add_edge(
            _n_cpu,
            None,
            callback_tasklet,
            _conn,
            Memlet.from_array(_n_cpu, _n_cpu.desc(dace_session.graph)),
        )
    callback_input_types = []
    for somenode in debugNodes:
        callback_input_types.append(somenode.desc(dace_session.graph))
    dace_session.callbackFunctionDict[string_builder(tf_node.name) + "_printer"] = printer
    dace_session.callbackTypeDict[string_builder(tf_node.name) + "_printer"] = dace.data.Scalar(
        dace.callback(None, *callback_input_types))
