void* flex_tcdm_malloc(uint32_t size)
{
    void* ptr;
    DACE_ACL_CHECK(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HBM_ONLY));
    return ptr;
}




void copy_map_outer_0_0_4(const dace::float16* __restrict__ soft_hier_A,
                          dace::float16* __restrict__ soft_hier_B) {
  {
    // TEST KERNEL SCOPE
    flex_global_barrier_xy();  // Global barrier

    // TEST KERNEL SCOPE
    uint32_t cluster_id = flex_get_cluster_id();
    uint32_t core_id = flex_get_core_id();
    int i = 0;
    {
      // TEST DEVICE SCOPE
      dace::float16* frag_A;
      DACE_ACL_CHECK(aclrtMalloc((void**)&frag_A, 256 * sizeof(dace::float16)));
      dace::float16* frag_B;
      DACE_ACL_CHECK(aclrtMalloc((void**)&frag_B, 256 * sizeof(dace::float16)));
      int ii = (256 * cluster_id);
      {
        // SoftHier: Emitting copy from soft_hier_A to frag_A

        dace::CopyND<dace::float16, 1, false, 256>::template ConstDst<1>::Copy(
            soft_hier_A + (i + ii), frag_A, 1);
        // SoftHier: Emitting copy from frag_A to frag_B

        dace::CopyND<dace::float16, 1, false, 256>::template ConstDst<1>::Copy(
            frag_A, frag_B, 1);
        // SoftHier: Emitting copy from frag_B to soft_hier_B

        dace::CopyND<dace::float16, 1, false, 256>::template ConstDst<1>::Copy(
            frag_B, soft_hier_B + (i + ii), 1);
      }
    }
  }
}