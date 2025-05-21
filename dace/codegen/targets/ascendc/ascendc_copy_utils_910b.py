def write_vecin_from_global(callsite_stream, name, src_name, dst_name, beg, width, nodedesc):
    callsite_stream.write(
        f"""// Global -> VECIN: Alloc Local, DataCopy, EnQue
        {dst_name} = queue_{dst_name}.AllocTensor<{nodedesc.dtype.ctype}>();
        {name}_GM.SetGlobalBuffer(&{name}_typed[{beg}], {width});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = 1;
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = 0;
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}, {name}_GM, {dst_name}Params);
        queue_{dst_name}.EnQue({dst_name});\n"""
    )

def write_vecout_form_vecin(callsite_stream, name, src_name, dst_name, beg, width, nodedesc):
    callsite_stream.write(
        f"""// VECIN -> VECOUT: DeQue, Enque, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {dst_name} = queue_{dst_name}.AllocTensor<{nodedesc.dtype.ctype}>();
        {dst_name} = {src_name};"
        queue_{dst_name}.EnQue<{nodedesc.dtype.ctype}>({dst_name});
        queue_{src_name}.FreeTensor({src_name});\n"""
    )

def write_global_from_vecout(callsite_stream, name, src_name, dst_name, beg, width, nodedesc):
    callsite_stream.write(
        f"""// VECOUT -> Global: DeQue, DataCopy, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {name}_GM.SetGlobalBuffer(&{name}_typed[{beg}], {width});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = 1;
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = 0;
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({name}_GM, {src_name}, {dst_name}Params);
        queue_{src_name}.FreeTensor({src_name});"""
    )


def write_vecin_from_global_2d(callsite_stream, name, src_name, dst_name, beg1, beg2, width, height, nodedesc):
    # Coupling Architecture
    #* @param [in] intriParams.blockCount number of blocks
    #* @param [in] intriParams.blockLen Length of blocks
    #* @param [in] intriParams.srcStride src block stride
    #* @param [in] intriParams.dstStride dst block stride
    callsite_stream.write(
        f"""// Global -> VECIN: Alloc Local, DataCopy, EnQue
        {dst_name} = queue_{dst_name}.AllocTensor<{nodedesc.dtype.ctype}>();
        {name}_GM.SetGlobalBuffer(&{name}_typed[{beg1} * {nodedesc.strides[0]} + {beg2}], {height} * {width});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = static_cast<uint16_t>(({width} / 16) - 1);
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}, {src_name}_GM, {dst_name}Params);
        queue_{dst_name}.EnQue({dst_name});\n"""
    )

def write_vecout_from_vecin(callsite_stream, name, src_name, dst_name, beg1, beg2, width, height, nodedesc):
    callsite_stream.write(f"""// VECIN -> VECOUT: DeQue, Enque, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {dst_name} = queue_{dst_name}.AllocTensor<{nodedesc.dtype.ctype}>();
        {dst_name} = {src_name};
        queue_{dst_name}.EnQue<{nodedesc.dtype.ctype}>({dst_name});
        queue_{src_name}.FreeTensor({src_name});\n"""
    )

def write_global_from_vecout(callsite_stream, name, src_name, dst_name, beg1, beg2, width, height, nodedesc):
    callsite_stream.write(
        f"""// VECOUT -> Global: DeQue, DataCopy, Free Prev.
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        {name}_GM.SetGlobalBuffer(&{name}_typed[{beg2} * {nodedesc.strides[0]} + {beg1}], {height} * {width});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = static_cast<uint16_t>(({width} / 16) - 1);
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}_GM, {src_name}, {dst_name}Params);
        queue_{src_name}.FreeTensor({src_name});\n
        """
    )

def write_l1_from_global(callsite_stream, name, src_name, dst_name, beg1, beg2, width, height, nodedesc, storage_type):
    assert storage_type == "A1" or storage_type == "B1"
    dst_storage_name = storage_type
    copy_str = f"// Global -> {dst_storage_name}: Alloc Local, DataCopy, EnQue"

    """ # Seperation Architecture but compile error
    callsite_stream.write(
        {copy_str}
        {name}_GM.SetGlobalBuffer(&{name}_typed[{beg2} * {nodedesc.strides[1]} + {beg1}], {width} * {height});
        AscendC::Nd2NzParams  nd2nz{dst_name}Params;
        nd2nz{dst_name}Params.ndNum = 1;
        nd2nz{dst_name}Params.nValue = {height};
        nd2nz{dst_name}Params.dValue = {width};
        nd2nz{dst_name}Params.srcNdMatrixStride = 0;
        nd2nz{dst_name}Params.srcDValue = {width};
        nd2nz{dst_name}Params.dstNzC0Stride = CeilCubeBlock({height}) * CUBE_BLOCK;
        nd2nz{dst_name}Params.dstNzNStride = 1;
        nd2nz{dst_name}Params.dstNzMatrixStride = 0;
        AscendC::DataCopy({dst_name}, {src_name}, nd2nz{dst_name}Params);
        queue_{dst_name}.EnQue<{nodedesc.dtype.ctype}>({dst_name});
    )
    """
    callsite_stream.write(
        f"""
        {copy_str}
        {name}_GM.SetGlobalBuffer(&{name}_typed[{beg2} * {nodedesc.strides[0]} + {beg1}], {width} * {height});
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = static_cast<uint16_t>(({nodedesc.strides[0]} / 16) - 1);
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}, {src_name}_GM, {dst_name}Params);
        queue_{dst_name}.EnQue({dst_name});\n
        """
    )

def write_l0_from_l1(callsite_stream, name, src_name, dst_name, beg1, beg2, width, height, nodedesc, storage_type):
    assert storage_type == "A2" or storage_type == "B2"
    dst_storage_name = storage_type
    src_storage_name = storage_type.replace("2", "1")
    copy_str = f"// {src_storage_name} -> {dst_storage_name}: Alloc Local, DataLoad, EnQue"
    """
    callsite_stream.write(
        {copy_str}
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        uint32_t {dst_name}_dstOffset = CeilCubeBlock({width}) * CUBE_BLOCK_SIZE;
        uint32_t {src_name}_srcOffset = CUBE_BLOCK_SIZE;
        AscendC::LoadData2DParams loadData{dst_name}Params;
        loadData{dst_name}Params.repeatTimes  =  CeilCubeBlock({width});
        loadData{dst_name}Params.srcStride  =  CeilCubeBlock({height});
        loadData{dst_name}Params.dstGap  =  0;
        loadData{dst_name}Params.ifTranspose  =  false;
        for  (int  i  =  0;  i  <  CeilCubeBlock({height});  ++i)  {{
            AscendC::LoadData({dst_name}[i * {dst_name}_dstOffset], {src_name}[i * {src_name}_srcOffset], loadData{dst_name}Params);
        }}
        queue_{dst_name}.EnQue<{nodedesc.dtype.ctype}>({dst_name});
        queue_{src_name}.FreeTensor({src_name});
    )
    """
    callsite_stream.write(
        f"""
        {copy_str}
        {src_name} = queue_{src_name}.DeQue<{nodedesc.dtype.ctype}>();
        AscendC::DataCopyParams {dst_name}Params;
        {dst_name}Params.blockCount = {height};
        {dst_name}Params.blockLen = {width} / 16;
        {dst_name}Params.srcStride = static_cast<uint16_t>(({nodedesc.strides[0]} / 16) - 1);
        {dst_name}Params.dstStride = 0;
        AscendC::DataCopy({dst_name}, {src_name}, {dst_name}Params);
        queue_{dst_name}.EnQue({dst_name});
        queue_{src_name}.FreeTensor({src_name});\n
        """
    )

def write_global_from_co1(callsite_stream, name, src_name, dst_name, beg1, beg2, width, height, nodedesc):
    src_storage_name = "CO1"
    dst_storage_name = "Global"
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


def write_tensor_copy(callsite_stream, memlet, src_name, dst_name, src_storage, dst_storage, nodedesc):
    subset = memlet.subset
    if len(subset.ranges) == 1:
        beg, end, step = subset.ranges[0]
        assert step == 1
        length = (end + 1) - beg
        width = length
        if src_storage.name == "Ascend_Global" and dst_storage.name == "Ascend_VECIN":
            write_vecin_from_global(callsite_stream, memlet.data, dst_name, beg, length, width, nodedesc)
        elif src_storage.name == "Ascend_VECIN" and dst_storage.name == "Ascend_VECOUT":
            write_vecout_from_vecin(callsite_stream, src_name, dst_name, nodedesc)
        elif src_storage.name == "Ascend_VECOUT" and dst_storage.name == "Ascend_Global":
            write_global_from_vecout(callsite_stream, memlet.data, src_name, beg, length, width, nodedesc)
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
            write_vecin_from_global_2d(callsite_stream, memlet.data, dst_name, src_name, beg1, beg2, width, height, nodedesc)
        elif src_storage.name == "Ascend_Global" and dst_storage.name == "Ascend_A1":
            write_l1_from_global(callsite_stream, memlet.data, dst_name, src_name, beg1, beg2, width, height, nodedesc, "A1")
        elif src_storage.name == "Ascend_Global" and dst_storage.name == "Ascend_B1":
            write_l1_from_global(callsite_stream, memlet.data, dst_name, src_name, beg1, beg2, width, height, nodedesc, "B1")
        elif src_storage.name == "Ascend_A1" and dst_storage.name == "Ascend_A2":
            write_l0_from_l1(callsite_stream, memlet.data, dst_name, src_name, beg1, beg2, width, height, nodedesc, "A2")
        elif src_storage.name == "Ascend_B1" and dst_storage.name == "Ascend_B2":
            write_l0_from_l1(callsite_stream, memlet.data, dst_name, src_name, beg1, beg2, width, height, nodedesc, "B2")
        elif src_storage.name == "Ascend_CO1" and dst_storage.name == "Ascend_CO2":
            write_global_from_co1(callsite_stream, memlet.data, dst_name, src_name, beg1, beg2, width, height, nodedesc)
        else:
            raise NotImplementedError(f"2D copy not implemented for {src_storage} -> {dst_storage}")
    else:
        raise NotImplementedError("Only 1D and 2D copies are supported.")

