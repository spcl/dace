import dace

def write_vecin_from_global(callsite_stream, dst_name, src_name, beg, width, nodedesc):
    callsite_stream.write(
        f"""// Global -> VECIN: Alloc Local, DataCopy, EnQue
        {dst_name} = queue_{dst_name}.AllocTensor<{nodedesc.dtype.ctype}>();
        {src_name}_GM.SetGlobalBuffer(&{src_name}_typed[{beg}], {width});
        AscendC::DataCopyParams {dst_name}Params;
        AscendC::DataCopy({dst_name}, {src_name}_GM, {width});
        queue_{dst_name}.EnQue({dst_name});\n"""
    )

def write_vecout_from_vecin(callsite_stream, dst_name, src_name, beg, width, nodedesc):
    callsite_stream.write(
        f"""
        // Should be handled by a tasklet
        // VECIN -> VECOUT: DeQue, Enque, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {dst_name} = queue_{dst_name}.AllocTensor<{nodedesc.dtype.ctype}>();
        {dst_name} = {src_name};"
        queue_{dst_name}.EnQue<{nodedesc.dtype.ctype}>({dst_name});
        queue_{src_name}.FreeTensor({src_name});\n"""
    )

def write_global_from_vecout(callsite_stream, dst_name, src_name, beg, width, nodedesc):
    callsite_stream.write(
        f"""// VECOUT -> Global: DeQue, DataCopy, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {dst_name}_GM.SetGlobalBuffer(&{dst_name}_typed[{beg}], {width});
        AscendC::DataCopy({dst_name}_GM, {src_name}, {width});
        queue_{src_name}.FreeTensor({src_name});"""
    )


def write_vecin_from_global_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc):
    # Coupling Architecture
    #* @param [in] intriParams.blockCount number of blocks
    #* @param [in] intriParams.blockLen Length of blocks
    #* @param [in] intriParams.srcStride src block stride
    #* @param [in] intriParams.dstStride dst block stride
    callsite_stream.write(
        f"""// Global -> VECIN: Alloc Local, DataCopy, EnQue
        {dst_name} = queue_{dst_name}.AllocTensor<{nodedesc.dtype.ctype}>();
        {src_name}_GM.SetGlobalBuffer(&{src_name}_typed[{beg1} * {nodedesc.strides[0]} + {beg2}], {height} * {width});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = static_cast<uint16_t>(({width} / 16) - 1);
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}, {src_name}_GM, {dst_name}Params);
        queue_{dst_name}.EnQue({dst_name});\n"""
    )


def write_global_from_vecout_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc):
    callsite_stream.write(
        f"""// VECOUT -> Global: DeQue, DataCopy, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {src_name}_GM.SetGlobalBuffer(&{src_name}_typed[{beg2} * {nodedesc.strides[0]} + {beg1}], {height} * {width});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = static_cast<uint16_t>(({width} / 16) - 1);
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}_GM, {src_name}, {dst_name}Params);
        queue_{src_name}.FreeTensor({src_name});\n
        """
    )

def write_l1_from_global_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc, storage_type):
    assert storage_type == "A1" or storage_type == "B1"
    dst_storage_name = storage_type
    copy_str = f"// Global -> {dst_storage_name}: Alloc Local, DataCopy, EnQue"
    callsite_stream.write(
        f"""
        {copy_str}
        {src_name}_GM.SetGlobalBuffer(&{src_name}_typed[{beg2} * {nodedesc.strides[0]} + {beg1}], {width} * {height});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = static_cast<uint16_t>(({nodedesc.strides[0]} / 16) - 1);
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}, {src_name}_GM, {dst_name}Params);
        queue_{dst_name}.EnQue({dst_name});\n
        """
    )

def write_l0_from_l1_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc, storage_type):
    """
    repeatTimes:
    The number of iterations. Each iteration can process 512B of data. Value range: repeatTimes∈[1, 255].
    srcStride:
    The interval between the starting addresses of the previous fractal and the next fractal of the source operand between adjacent iterations, unit: 512B (16x16 block). Value range: src_stride∈[0, 65535]. The default value is 0.
    When the data type of the source operand/destination operand is uint16_t/int16_t/half/bfloat16_t, the size of the fractal matrix on A1/B1/A2/B2 is 16*16
    """
    assert storage_type == "A2" or storage_type == "B2"
    dst_storage_name = storage_type
    src_storage_name = storage_type.replace("2", "1")
    copy_str = f"// {src_storage_name} -> {dst_storage_name}: Alloc Local, DataLoad, EnQue"
    assert((width * height) % 256 == 0)

    callsite_stream.write(
        f"""
        {copy_str}
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        AscendC::LoadData2dParams {dst_name}LoadDataParams;
        {dst_name}LoadDataParams.repeatTimes = {(width * height) // 256};
        {dst_name}LoadDataParams.srcStride = 1;
        {dst_name}LoadDataParams.ifTranspose = {'false' if storage_type == 'A2' else 'true'};
        AscendC::LoadData({dst_name}, {src_name}, {dst_name}LoadDataParams);
        queue_{dst_name}.EnQue({dst_name});
        queue_{src_name}.FreeTensor({src_name});\n
        """
    )

def write_co2_from_co1_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc):
    src_storage_name = "CO1"
    dst_storage_name = "CO2"
    callsite_stream.write(
        f"""
        // {src_storage_name} -> {dst_storage_name}: DeQue, DataCopy, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = 0;
        {dst_name}Params.dstStride = static_cast<uint16_t>(({nodedesc.strides[0]} / 16) - 1);
        AscendC::DataCopyEnhancedParams  {dst_name}EnhancedParams;
        {dst_name}EnhancedParams.blockMode = AscendC::BlockMode::BLOCK_MODE_MATRIX;
        AscendC::DataCopy({dst_name}, {src_name}, {dst_name}Params, {dst_name}EnhancedParams);
        queue_{dst_name}.EnQue<{nodedesc.dtype.ctype}>({dst_name});
        queue_{src_name}.FreeTensor({src_name});
        """
    )

def write_vecin_from_co2_2d(callsite_stream,  src_name, dst_name, beg1, beg2, width, height, nodedesc):
    src_storage_name = "CO1"
    dst_storage_name = "VECIN"
    callsite_stream.write(
        f"""
        // {src_storage_name} -> {dst_storage_name}: DeQue, DataCopy, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = 0;
        {dst_name}Params.dstStride = static_cast<uint16_t>(({nodedesc.strides[0]} / 16) - 1);
        AscendC::DataCopy({dst_name}, {src_name}, {dst_name}Params);
        queue_{dst_name}.EnQue<{nodedesc.dtype.ctype}>({dst_name});
        queue_{src_name}.FreeTensor({src_name});
        """
    )

def write_global_from_co2_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc):
    src_storage_name = "CO2"
    dst_storage_name = "Glb"
    callsite_stream.write(
        f"""
        // {src_storage_name} -> {dst_storage_name}: DeQue, DataCopy, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {dst_name}_GM.SetGlobalBuffer(&{dst_name}_typed[{beg2} * {nodedesc.strides[0]} + {beg1}], {height} * {width});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = 0;
        {dst_name}Params.dstStride = static_cast<uint16_t>(({nodedesc.strides[0]} / 16) - 1);
        AscendC::DataCopy({dst_name}_GM, {src_name}, {dst_name}Params);
        queue_{src_name}.FreeTensor({src_name});
        """
    )

def write_tensor_copy(callsite_stream, memlet, src_name, dst_name, src_storage, dst_storage, nodedesc):
    subset = memlet.subset
    if len(subset.ranges) == 1:
        beg, end, step = subset.ranges[0]
        assert step == 1
        length = (end + 1) - beg
        width = length
        if src_storage.name == "Ascend_Global" and dst_storage.name == "Ascend_VECIN":
            write_vecin_from_global(callsite_stream, dst_name, src_name, beg, width, nodedesc)
        elif src_storage.name == "Ascend_VECIN" and dst_storage.name == "Ascend_VECOUT":
            write_vecout_from_vecin(callsite_stream, dst_name, src_name, nodedesc)
        elif src_storage.name == "Ascend_VECOUT" and dst_storage.name == "Ascend_Global":
            write_global_from_vecout(callsite_stream, dst_name, src_name, beg, width, nodedesc)
        else:
            raise NotImplementedError(f"1D copy not implemented for {src_storage} -> {dst_storage}")
    elif len(subset.ranges) == 2:
        beg1, end1, step1 = subset.ranges[0]
        beg2, end2, step2 = subset.ranges[1]
        assert step1 == 1 and step2 == 1
        assert nodedesc.strides[1] == 1
        width = (end2 + 1) - beg2
        height = (end1 + 1) - beg1
        if src_storage.name == "Ascend_Global" and dst_storage.name == "Ascend_VECIN":
            write_vecin_from_global_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc)
        elif src_storage.name == "Ascend_Global" and dst_storage.name == "Ascend_A1":
            write_l1_from_global_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc, "A1")
        elif src_storage.name == "Ascend_Global" and dst_storage.name == "Ascend_B1":
            write_l1_from_global_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc, "B1")
        elif src_storage.name == "Ascend_A1" and dst_storage.name == "Ascend_A2":
            write_l0_from_l1_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc, "A2")
        elif src_storage.name == "Ascend_B1" and dst_storage.name == "Ascend_B2":
            write_l0_from_l1_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc, "B2")
        elif src_storage.name == "Ascend_CO1" and dst_storage.name == "Ascend_CO2":
            write_co2_from_co1_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc)
        elif src_storage.name == "Ascend_CO2" and dst_storage.name == "Ascend_VECIN":
            write_vecin_from_co2_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc)
        elif src_storage.name == "Ascend_CO2" and dst_storage.name == "Ascend_Global":
            write_global_from_co2_2d(callsite_stream, dst_name, src_name, beg1, beg2, width, height, nodedesc)
        else:
            if src_storage == dst_storage and memlet == dace.memlet.Memlet.from_array(src_name, nodedesc) and src_name == dst_name:
                print("Warning!: Trivial copy (same storage, complete array shape), {src_storage} -> {dst_storage}, Memlet({memlet}) == {nodedesc.shape}")
            else:
                raise NotImplementedError(f"2D copy not implemented for {src_storage} ({src_name}) -> {dst_storage} ({dst_name}), Memlet({memlet}), {nodedesc.shape}")
    else:
        raise NotImplementedError("Only 1D and 2D copies are supported.")

