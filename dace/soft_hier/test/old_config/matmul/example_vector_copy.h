#define FRAG_A_SIZE (512)
#define FRAG_A_ADDR (0)
#define FRAG_B_SIZE (512)
#define FRAG_B_ADDR ((0 + FRAG_A_SIZE))
void copy_map_outer_0_0_4(uint32_t soft_hier_A, uint32_t soft_hier_B) {
    {
        // TEST KERNEL SCOPE
        flex_global_barrier_xy();
        uint32_t cluster_id = flex_get_cluster_id();
        uint32_t core_id = flex_get_core_id();
        int i = 0;
        flex_global_barrier_xy();
        {
            // TEST DEVICE SCOPE
            uint32_t frag_A = FRAG_A_ADDR;
            uint32_t frag_B = FRAG_B_ADDR;
            int ii = (256 * cluster_id);
            if (ii < 32) {
                // Minels: [0], Maxels: [8191]
                // SoftHier_HBM -> SoftHier_TCDM
                if(flex_is_dm_core())
                {
                    flex_dma_async_1d(hbm_addr(soft_hier_A + (i + ii)), local(frag_A), 256*2);
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                // SoftHier_TCDM -> SoftHier_TCDM
                if(flex_is_dm_core())
                {
                    flex_dma_async_1d(local(frag_A), local(frag_B), 256*2);
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
                // SoftHier_TCDM -> SoftHier_HBM
                if(flex_is_dm_core())
                {
                    flex_dma_async_1d(local(frag_B), hbm_addr(soft_hier_B + (i + ii)), 256*2);
                    flex_dma_async_wait_all();
                }
                flex_intra_cluster_sync();
            }
        }
        flex_global_barrier_xy();
    }
}

