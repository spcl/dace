
def _check_combo(combo, tcdm_usage_func=None):
    '''
    Check if the combination is valid
    '''
    (
        M_val,
        N_val,
        K_val,
        hwM,
        hwN,
        hwK,
        thread_group_dims,
        tcdm_size
    ) = combo
    # print(f"Checking combo: {combo}")
    dim_x = thread_group_dims[0]
    dim_y = thread_group_dims[1]
    
    if M_val % hwM != 0:
        return False
    if N_val % hwN != 0:
        return False
    if K_val % hwK != 0:
        return False
    if M_val % dim_x != 0:
        return False
    if N_val % dim_y != 0:
        return False
    if M_val % (hwM * dim_x) != 0:
        return False
    if N_val % (hwN * dim_y) != 0:
        return False
    if (hwM * dim_x) > M_val:
        return False
    if (hwN * dim_y) > N_val:
        return False
    
    # Use provided function or default calculation
    if tcdm_usage_func is None:
        tcdm_usage_func = lambda m, n, k: 2 * (2 * m * k + 2 * k * n + m * n)
    
    tcdm_usage = tcdm_usage_func(hwM, hwN, hwK)
    if tcdm_usage >= tcdm_size:
        return False
    # if tcdm_usage < (tcdm_size // 2):
    #     if (hwM * dim_x >= M_val // 2) and (hwN * dim_y >= N_val // 2):
    #         pass
    #     else:   
    #         return False
    return True


def generate_tiling(M_val, N_val, K_val, thread_group_dims, tcdm_size, 
                    input_elem_size,
                    output_elem_size,
                    tcdm_usage_func=None, 
                    OI_func_type=1, 
                    min_tiling_size=64, 
                    redmule_h=64, redmule_w=16, hbm_bw=512):
    dim_m, dim_n = thread_group_dims
    num_clusters = dim_m * dim_n
    peak_performance = 2 * dim_m * dim_n * redmule_h * redmule_w

    ridge_OI = peak_performance / hbm_bw
    print(f"Ridge OI: {ridge_OI}")

    #TODO: whether to remap the thread group dimensions
    if min_tiling_size * dim_m > M_val and min_tiling_size * dim_n > N_val:
        raise ValueError("M and N must be divisible by the thread group dimensions")

    # Generate lists of thread group dimensions
    dim_x_list  = [dim_m]
    dim_y_list  = [dim_n]
    if min_tiling_size * dim_m > M_val:
        dim_x_list = [2**i for i in range(0, 6) if 2**i <= M_val // min_tiling_size]
        dim_y_list = [num_clusters // dim_x for dim_x in dim_x_list]
    elif min_tiling_size * dim_n > N_val:
        dim_y_list = [2**i for i in range(0, 6) if 2**i <= N_val // min_tiling_size]
        dim_x_list = [num_clusters // dim_y for dim_y in dim_y_list]


    print(f"dim_x_list: {dim_x_list}")
    print(f"dim_y_list: {dim_y_list}")

    all_combos = []

    for (dim_x, dim_y) in zip(dim_x_list, dim_y_list):
        # Default TCDM usage calculation if none provided
        if tcdm_usage_func is None:
            tcdm_usage_func = lambda m, n, k: (2 * m * k * input_elem_size + 
                                               2 * k * n * input_elem_size + 
                                               m * n * output_elem_size)
        
        if OI_func_type == 0:
            OI_func = lambda m, n, k: 2 * m * n * k / ((m * n * output_elem_size + 
                                                        m * k * input_elem_size + 
                                                        n * k * input_elem_size))
        elif OI_func_type == 1:
            OI_func = lambda m, n, k: 2 * dim_x * dim_y * m * n * k / (
                                                                (dim_x * dim_y * m * n * output_elem_size + 
                                                                dim_x * m * k * input_elem_size + 
                                                                dim_y * n * k * input_elem_size ))

        max_OI_func = lambda m, n, k: 2 * m * n * k / ((m * n * output_elem_size + 
                                                        m * k * input_elem_size + 
                                                        n * k * input_elem_size))
        # Unpack thread group dimensions
        
        combos = []

        def factors(n):
            result = []
            # iterate up to the square root of n
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    result.append(i)
                    if i != n // i:
                        result.append(n // i)
            return sorted(result)
        
        if (M_val % dim_x != 0) or (N_val % dim_y != 0):
            raise ValueError("M and N must be divisible by the thread group dimensions")

        # Generate lists of hardware tile sizes
        hw_M_list = [i for i in factors(M_val // dim_x) if i >= min_tiling_size]
        hw_N_list = [i for i in factors(N_val // dim_y) if i >= min_tiling_size]
        if len(hw_N_list) == 0:
            hw_N_list = [max([i for i in factors(N_val // dim_y) if i < min_tiling_size])]
        
        hw_K_list = [i for i in factors(K_val) if i >= min_tiling_size]

        print(f"hw_M_list: {hw_M_list}")
        print(f"hw_N_list: {hw_N_list}")
        print(f"hw_K_list: {hw_K_list}")

        # Iterate over all combinations of hardware tile sizes.
        for hw_M in hw_M_list:
            for hw_N in hw_N_list:
                for hw_K in hw_K_list:
                    # If the estimated memory footprint exceeds TCDM size,
                    # break out of the innermost loop (assuming hw_K_list is in ascending order).
                    if tcdm_usage_func(hw_M, hw_N, hw_K) > tcdm_size:
                        break
                    OI = OI_func(hw_M, hw_N, K_val)
                    max_OI = max_OI_func(M_val, N_val, K_val)
                    print(f"OI: {OI}, max_OI: {max_OI}")
                    # If the OI is less than the ridge OI, skip this combination
                    if OI < ridge_OI and max_OI > ridge_OI:
                        continue
                    # Build the configuration tuple.
                    combo = (M_val, N_val, K_val, hw_M, hw_N, hw_K,
                            (dim_x, dim_y), tcdm_size)
                    if _check_combo(combo, tcdm_usage_func):
                        combos.append(combo)

        # Filter combinations based on the maximum product of hw_M and hw_N.
        if combos:
            max_OI = max([OI_func(combo[3], combo[4], combo[2]) for combo in combos
                        if _check_combo(combo, tcdm_usage_func)])
            filtered_combos = [combo for combo in combos if (OI_func(combo[3], combo[4], combo[2])) >= max_OI/2]
            all_combos.extend(filtered_combos)

    return all_combos

def generate_remap_split_k_tiling(M_val, N_val, K_val, thread_group_dims, tcdm_size, 
                    input_elem_size,
                    output_elem_size,
                    tcdm_usage_func=None, 
                    OI_func_type=1, 
                    min_tiling_size=64, 
                    redmule_h=64, redmule_w=16, hbm_bw=512):
    dim_m, dim_n = thread_group_dims
    num_clusters = dim_m * dim_n
    peak_performance = 2 * dim_m * dim_n * redmule_h * redmule_w

    ridge_OI = peak_performance / hbm_bw
    print(f"Ridge OI: {ridge_OI}")

    #TODO: whether to remap the thread group dimensions
    if min_tiling_size * dim_m > M_val and min_tiling_size * dim_n > N_val:
        raise ValueError("M and N must be divisible by the thread group dimensions")

    # Generate lists of thread group dimensions
    dim_x_list  = [dim_m]
    dim_y_list  = [dim_n]
    if min_tiling_size * dim_m > M_val:
        dim_x_list = [2**i for i in range(0, 6) if 2**i <= M_val // min_tiling_size]
        dim_y_list = [num_clusters // dim_x for dim_x in dim_x_list]
    elif min_tiling_size * dim_n > N_val:
        dim_y_list = [2**i for i in range(0, 6) if 2**i <= N_val // min_tiling_size]
        dim_x_list = [num_clusters // dim_y for dim_y in dim_y_list]


    print(f"dim_x_list: {dim_x_list}")
    print(f"dim_y_list: {dim_y_list}")

    all_combos = []

    for (dim_x, dim_y) in zip(dim_x_list, dim_y_list):
        if dim_x != 1:
            continue
        
        k_groups = [(1, 2**i) for i in range(0, 10) if 2**i <= dim_y]
        print(f"k_groups: {k_groups}")
        # Default TCDM usage calculation if none provided
        if tcdm_usage_func is None:
            tcdm_usage_func = lambda m, n, k: (2 * m * k * input_elem_size + 
                                               2 * k * n * input_elem_size + 
                                               m * n * output_elem_size)
        
        if OI_func_type == 0:
            OI_func = lambda m, n, k: 2 * m * n * k / ((m * n * output_elem_size + 
                                                        m * k * input_elem_size + 
                                                        n * k * input_elem_size))
        elif OI_func_type == 1:
            OI_func = lambda m, n, k: 2 * dim_x * dim_y * m * n * k / (
                                                                (dim_x * dim_y * m * n * output_elem_size + 
                                                                dim_x * m * k * input_elem_size + 
                                                                dim_y * n * k * input_elem_size ))

        max_OI_func = lambda m, n, k: 2 * m * n * k / ((m * n * output_elem_size + 
                                                        m * k * input_elem_size + 
                                                        n * k * input_elem_size))
        # Unpack thread group dimensions
        
        combos = []

        def factors(n):
            result = []
            # iterate up to the square root of n
            for i in range(1, int(n ** 0.5) + 1):
                if n % i == 0:
                    result.append(i)
                    if i != n // i:
                        result.append(n // i)
            return sorted(result)
        
        for (kg_m, kg_n) in k_groups:
            # Generate lists of hardware tile sizes
            hw_M_list = [i for i in factors(M_val*kg_m // dim_x) if i >= min_tiling_size]
            hw_N_list = [i for i in factors(N_val*kg_n // dim_y) if i >= min_tiling_size]
            if len(hw_N_list) == 0 or len(hw_M_list) == 0:
                continue
            
            hw_K_list = [i for i in factors(K_val//(kg_m*kg_n)) if i >= min_tiling_size]

            print(f"hw_M_list: {hw_M_list}")
            print(f"hw_N_list: {hw_N_list}")
            print(f"hw_K_list: {hw_K_list}")

            # Iterate over all combinations of hardware tile sizes.
            for hw_M in hw_M_list:
                for hw_N in hw_N_list:
                    for hw_K in hw_K_list:
                        # If the estimated memory footprint exceeds TCDM size,
                        # break out of the innermost loop (assuming hw_K_list is in ascending order).
                        if tcdm_usage_func(hw_M, hw_N, hw_K) > tcdm_size:
                            break
                        OI = OI_func(hw_M/kg_m, hw_N/kg_n, K_val)
                        max_OI = max_OI_func(M_val, N_val, K_val)
                        print(f"OI: {OI}, max_OI: {max_OI}")
                        # If the OI is less than the ridge OI, skip this combination
                        if OI < ridge_OI and max_OI > ridge_OI:
                            continue
                        # Build the configuration tuple.
                        combo = (M_val, N_val, K_val, hw_M, hw_N, hw_K,
                                (dim_x, dim_y), tcdm_size)
                        combos.append(combo)

            # Filter combinations based on the maximum product of hw_M and hw_N.
            if combos:
                max_OI = max([OI_func(combo[3]/kg_m, combo[4]/kg_n, combo[2]) for combo in combos])
                filtered_combos = [combo for combo in combos if (OI_func(combo[3]/kg_m, combo[4]/kg_n, combo[2])) >= max_OI/2]
                filtered_combos = [(combo[:7] + ((kg_m, kg_n),) + combo[7:]) for combo in filtered_combos]
                all_combos.extend(filtered_combos)

    return all_combos

