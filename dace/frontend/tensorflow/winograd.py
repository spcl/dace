import dace
import numpy as np
from dace.memlet import Memlet
from math import ceil

# takes input and stores output in column major order. give swapped input, B, A in place of A, B
def mm(
    state,
    A_node,
    B_node,
    C_node,
    A_mode: str = "N",
    B_mode: str = "N",
    label: str = None,
    m=None,
    n=None,
    k=None,
    A_memlet=None,
    B_memlet=None,
    C_memlet=None,
):
    sdfg = state.parent
    Adesc = A_node.desc(sdfg)
    Bdesc = B_node.desc(sdfg)
    Cdesc = C_node.desc(sdfg)
    Ashape = Adesc.shape
    Bshape = Bdesc.shape
    Cshape = Cdesc.shape

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
            m=m or Cshape[1],
            n=n or Cshape[0],
            k=k or Ashape[kdim_A],
            lda=lda,
            ldb=ldb,
            ldc=ldc,
        ),
        language=dace.types.Language.CPP,
    )

    state.add_edge(
        A_node,
        None,
        tasklet,
        "a",
        A_memlet or dace.Memlet.from_array(A_node.data, Adesc),
    )
    state.add_edge(
        B_node,
        None,
        tasklet,
        "b",
        B_memlet or dace.Memlet.from_array(B_node.data, Bdesc),
    )
    state.add_edge(
        tasklet,
        "c",
        C_node,
        None,
        C_memlet or dace.Memlet.from_array(C_node.data, Cdesc),
    )


IMAGE_TILE_SIZE = 4
OUTPUT_TILE_SIZE = 2

bt = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]).astype(
    np.float32
)
b = np.transpose(bt)
g = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]])
gt = np.transpose(g)
at = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).astype(np.float32)
a = np.transpose(at)


def winograd_convolution(dace_session, tf_node):
    state = dace_session.state
    #############Add nodes and constants for transformation matrices###############
    if "Btrans" in dace_session.constDict:
        bTransposeNode = state.find_node("Btrans")
        bNode = state.find_node("B")
    else:
        dace_session.constDict["Btrans"] = bt
        dace_session.constDict["B"] = b
        bTransposeNode = state.add_array(
            "Btrans", bt.shape, dace.float32, transient=False, toplevel=True
        )
        bNode = state.add_array(
            "B", b.shape, dace.float32, transient=False, toplevel=True
        )

    if "G" in dace_session.constDict:
        gNode = state.find_node("G")
        gTransposeNode = state.find_node("Gtrans")
    else:
        dace_session.constDict["G"] = g
        dace_session.constDict["Gtrans"] = gt
        gNode = state.add_array(
            "G", g.shape, dace.float32, transient=False, toplevel=True
        )
        gTransposeNode = state.add_array(
            "Gtrans", gt.shape, dace.float32, transient=False, toplevel=True
        )

    if "Atrans" in dace_session.constDict:
        aTransposeNode = state.find_node("Atrans")
        aNode = state.find_node("A")
    else:
        dace_session.constDict["Atrans"] = at
        dace_session.constDict["A"] = a
        aTransposeNode = state.add_array(
            "Atrans", at.shape, dace.float32, transient=False, toplevel=True
        )
        aNode = state.add_array(
            "A", a.shape, dace.float32, transient=False, toplevel=True
        )
    inputNodes = []
    inputParams = []
    inputDims = []
    for _inp in tf_node.inputs:
        _node, _params, _dims = dace_session.create_and_add_input_node(_inp)
        inputNodes.append(_node)
        inputParams.append(_params)
        inputDims.append(_dims)
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
        outputShape[0]
        * ceil(outputShape[1] / OUTPUT_TILE_SIZE)
        * ceil(outputShape[2] / OUTPUT_TILE_SIZE),
    ]
    inputViewDims = ["0:" + str(_x) for _x in inputViewShape]
    ########Tiling the image#################################
    inputViewParams = [
        "i3%" + str(outputShape[0]),
        "(i3/" + str(outputShape[0]) + ")%"
        # + str(output_shape[0] * ceil(output_shape[1] / OUTPUT_TILE_SIZE))
        + str(ceil(outputShape[2] / OUTPUT_TILE_SIZE))
        + "*"
        + str(OUTPUT_TILE_SIZE)
        + "+i0",
        # + str(
        #    ceil(output_shape[1] / OUTPUT_TILE_SIZE)
        #    * ceil(output_shape[2] / OUTPUT_TILE_SIZE)
        # ),
        "floor(i3/"
        # + str(ceil(output_shape[1] / OUTPUT_TILE_SIZE))
        + str(outputShape[0] * ceil(outputShape[2] / OUTPUT_TILE_SIZE))
        + ")*"
        + str(OUTPUT_TILE_SIZE)
        + "+i1",
        "i2",
    ]
    inputView = state.add_transient("V", inputViewShape, dace.float32)
    mapEntry, mapExit = state.add_map(
        tf_node.name + "_input_tile", dict(zip(inputParams[0], inputViewDims))
    )
    tasklet = state.add_tasklet(
        tf_node.name + "_input_tile", {"j0"}, {"out"}, "out = j0"
    )
    dace_session.add_in_memlets(
        [inputNodes[0]], mapEntry, tasklet, [inputDims[0]], [inputViewParams]
    )
    dace_session.add_out_memlets(
        [inputView], mapExit, tasklet, [inputViewDims], [inputParams[0]]
    )
    ##################Transforming all input tiles#########################
    # Re-use memory
    vNode = state.add_write(inputView.data)
    mapEntry, mapExit = state.add_map(
        tf_node.name + "_input_txform",
        dict(zip(inputParams[0][0:2], inputViewDims[2:4])),
    )
    tileNode = state.add_transient(
        "image_tile", [IMAGE_TILE_SIZE, IMAGE_TILE_SIZE], dace.float32
    )
    state.add_edge(
        inputView,
        None,
        mapEntry,
        None,
        Memlet.simple(inputView, ",".join(inputViewDims)),
    )
    state.add_edge(
        mapEntry,
        None,
        tileNode,
        None,
        Memlet.simple(inputView, ",".join(inputViewDims[0:2] + inputParams[0][0:2])),
    )
    # TODO Figure out how to not re-allocate for each winograd instance
    intermediateResultNode = state.add_transient("BtI", bt.shape, dace.float32)
    processedTileNode = state.add_write(tileNode.data)
    mm(state, tileNode, bTransposeNode, intermediateResultNode)
    mm(state, bNode, intermediateResultNode, processedTileNode)
    state.add_edge(
        processedTileNode,
        None,
        mapExit,
        None,
        Memlet.simple(vNode, ",".join(inputViewDims[0:2] + inputParams[0][0:2])),
    )
    state.add_edge(
        mapExit, None, vNode, None, Memlet.simple(vNode, ",".join(inputViewDims))
    )
    #############Transforming the kernel###############################
    mapEntry, mapExit = state.add_map(
        tf_node.name + "_kernel_txform",
        dict(zip(inputParams[1][0:2], inputDims[1][2:4])),
    )
    tileNode = state.add_transient(
        "single_filter", tf_node.inputs[1].shape[0:2], dace.float32
    )
    intermediateResultNode = state.add_transient("GF", g.shape, dace.float32)
    processedFilterNode = state.add_transient(
        "transformed_single_filter", [IMAGE_TILE_SIZE, IMAGE_TILE_SIZE], dace.float32
    )
    processedKernelNode = state.add_transient(
        "U", inputViewShape[0:2] + list(tf_node.inputs[1].shape[-1:-3:-1]), dace.float32
    )
    state.add_edge(
        inputNodes[1],
        None,
        mapEntry,
        None,
        dace.Memlet.from_array(
            inputNodes[1].data, inputNodes[1].desc(dace_session.graph)
        ),
    )
    state.add_edge(
        mapEntry,
        None,
        tileNode,
        None,
        Memlet.simple(inputNodes[1], ",".join(inputDims[1][0:2] + inputParams[1][0:2])),
    )
    mm(state, tileNode, gNode, intermediateResultNode)
    mm(state, gTransposeNode, intermediateResultNode, processedFilterNode)
    state.add_edge(
        processedFilterNode,
        None,
        mapExit,
        None,
        Memlet.simple(
            processedKernelNode,
            ",".join(inputViewDims[0:2] + [inputParams[0][1]] + [inputParams[0][0]]),
        ),
    )
    state.add_edge(
        mapExit,
        None,
        processedKernelNode,
        None,
        Memlet.from_array(
            processedKernelNode.data, processedKernelNode.desc(dace_session.graph)
        ),
    )
    ###############U/V product############################################
    vSliceNode = state.add_transient("v_slice", inputViewShape[2:4], dace.float32)
    uSliceNode = state.add_transient(
        "u_slice", tf_node.inputs[1].shape[-1:-3:-1], dace.float32
    )
    mSliceNode = state.add_transient(
        "m_slice", [tf_node.inputs[1].shape[-1], inputViewShape[-1]], dace.float32
    )
    mNode = state.add_transient(
        "m",
        inputViewShape[0:2] + list(mSliceNode.desc(dace_session.graph).shape),
        dace.float32,
    )
    mNodeDims = ["0:" + str(_d) for _d in mNode.desc(dace_session.graph).shape]
    mapEntry, mapExit = state.add_map(
        tf_node.name + "_eltwise_product",
        dict(zip(inputParams[0][0:2], inputViewDims[0:2])),
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
        Memlet.from_array(
            processedKernelNode.data, processedKernelNode.desc(dace_session.graph)
        ),
    )
    state.add_edge(
        mapEntry,
        None,
        vSliceNode,
        None,
        Memlet.simple(vNode, ",".join(inputParams[0][0:2] + inputViewDims[-2:])),
    )
    state.add_edge(
        mapEntry,
        None,
        uSliceNode,
        None,
        Memlet.simple(
            processedKernelNode,
            ",".join(
                inputParams[0][0:2]
                + ["0:"+str(_s) for _s in tf_node.inputs[1].shape[-1:-3:-1]]
            ),
        ),
    )
    mm(state, vSliceNode, uSliceNode, mSliceNode)
    state.add_edge(
        mSliceNode,
        None,
        mapExit,
        None,
        Memlet.simple(mNode, ",".join(inputParams[0][0:2] + mNodeDims[-2:])),
    )
    state.add_edge(
        mapExit, None, mNode, None, Memlet.simple(mNode, ",".join(mNodeDims))
    )
    #################OUTPUT TRANSFORMATIION################################
    mapRange = [inputDims[1][-1]] + [inputViewDims[-1]]
    mapEntry, mapExit = state.add_map(
        tf_node.name + "_output_txform", dict(zip(inputParams[0][0:2], mapRange))
    )
    tileNode = state.add_transient("output_tile", inputViewShape[0:2], dace.float32)
    intermediateResultNode = state.add_transient("AtM", at.shape, dace.float32)
    transformedOutputTileNode = state.add_transient(
        "inv_txformed_output_tile", [OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE], dace.float32
    )
    transformedOutputNode = state.add_transient(
        "inv_txformed_output",
        [OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE]
        + [tf_node.inputs[1].shape[-1]]
        + [inputViewShape[-1]],
        dace.float32,
    )
    state.add_edge(
        mNode, None, mapEntry, None, Memlet.simple(mNode, ",".join(mNodeDims))
    )
    state.add_edge(
        mapEntry,
        None,
        tileNode,
        None,
        Memlet.simple(mNode, ",".join(inputViewDims[0:2] + inputParams[0][0:2])),
    )
    mm(state, tileNode, aTransposeNode, intermediateResultNode)
    mm(state, aNode, intermediateResultNode, transformedOutputTileNode)
    state.add_edge(
        transformedOutputTileNode,
        None,
        mapExit,
        None,
        Memlet.simple(
            transformedOutputNode,
            ",".join(
                ["0:" + str(OUTPUT_TILE_SIZE), "0:" + str(OUTPUT_TILE_SIZE)]
                + inputParams[0][0:2]
            ),
        ),
    )
    state.add_edge(
        mapExit,
        None,
        transformedOutputNode,
        None,
        Memlet.from_array(
            transformedOutputNode.data, transformedOutputNode.desc(dace_session.graph)
        ),
    )
    ###################Un-Tile the output to NHWC format###################
    outputParams = [
        "i3%" + str(outputShape[0]),
        "(i3/" + str(outputShape[0]) + ")%"
        # + str(output_shape[0] * ceil(output_shape[1] / OUTPUT_TILE_SIZE))
        + str(ceil(outputShape[2] / OUTPUT_TILE_SIZE))
        + "*"
        + str(OUTPUT_TILE_SIZE)
        + "+i0",
        # + str(
        #    ceil(output_shape[1] / OUTPUT_TILE_SIZE)
        #    * ceil(output_shape[2] / OUTPUT_TILE_SIZE)
        # ),
        "int_floor(i3,"
        # + str(ceil(output_shape[1] / OUTPUT_TILE_SIZE))
        + str(outputShape[0] * ceil(outputShape[2] / OUTPUT_TILE_SIZE))
        + ")*"
        + str(OUTPUT_TILE_SIZE)
        + "+i1",
        "i2",
    ]
    mapRange = [
        "0:" + str(_s) for _s in transformedOutputNode.desc(dace_session.graph).shape
    ]
    mapEntry, mapExit = state.add_map(
        tf_node.name + "_output_untile", dict(zip(inputParams[0], mapRange))
    )
    tasklet = state.add_tasklet(
        tf_node.name + "_output_untile", {"j0"}, {"out"}, "out = j0"
    )
    dace_session.add_in_memlets(
        [transformedOutputNode], mapEntry, tasklet, [mapRange], [inputParams[0]]
    )
    dace_session.add_out_memlets(
        outputList, mapExit, tasklet, [outputDims], [outputParams]
    )
