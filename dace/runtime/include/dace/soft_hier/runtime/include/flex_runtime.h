#ifndef _FLEX_RUNTIME_H_
#define _FLEX_RUNTIME_H_
#include <stdint.h>
#include "flex_cluster_arch.h"

#define ARCH_NUM_CLUSTER            (ARCH_NUM_CLUSTER_X*ARCH_NUM_CLUSTER_Y)
#define cluster_index(x,y)          ((y)*ARCH_NUM_CLUSTER_X+(x))
#define local(offset)               (ARCH_CLUSTER_TCDM_BASE+offset)
#define zomem(offset)               (ARCH_CLUSTER_ZOMEM_BASE+offset)
#define remote_cid(cid,offset)      (ARCH_CLUSTER_TCDM_REMOTE+cid*ARCH_CLUSTER_TCDM_SIZE+offset)
#define remote_xy(x,y,offset)       (ARCH_CLUSTER_TCDM_REMOTE+cluster_index(x,y)*ARCH_CLUSTER_TCDM_SIZE+offset)
#define remote_pos(pos,offset)      (ARCH_CLUSTER_TCDM_REMOTE+cluster_index(pos.x,pos.y)*ARCH_CLUSTER_TCDM_SIZE+offset)
#define hbm_addr(offset)            (ARCH_HBM_START_BASE+offset)
#define hbm_west(hid,offset)        (ARCH_HBM_START_BASE+(hid)*ARCH_HBM_NODE_ADDR_SPACE+offset)
#define hbm_north(hid,offset)       (ARCH_HBM_START_BASE+(hid)*ARCH_HBM_NODE_ADDR_SPACE+ARCH_HBM_NODE_ADDR_SPACE*ARCH_NUM_CLUSTER_Y+offset)
#define hbm_east(hid,offset)        (ARCH_HBM_START_BASE+(hid)*ARCH_HBM_NODE_ADDR_SPACE+ARCH_HBM_NODE_ADDR_SPACE*(ARCH_NUM_CLUSTER_Y+ARCH_NUM_CLUSTER_X)+offset)
#define hbm_south(hid,offset)       (ARCH_HBM_START_BASE+(hid)*ARCH_HBM_NODE_ADDR_SPACE+ARCH_HBM_NODE_ADDR_SPACE*2*ARCH_NUM_CLUSTER_Y+ARCH_HBM_NODE_ADDR_SPACE*ARCH_NUM_CLUSTER_X+offset)

/*******************
* Cluster Position *
*******************/

typedef struct FlexPosition
{
    uint32_t x;
    uint32_t y;
}FlexPosition;

FlexPosition get_pos(uint32_t cluster_id) {
    FlexPosition pos;
    pos.x = cluster_id % ARCH_NUM_CLUSTER_X;
    pos.y = cluster_id / ARCH_NUM_CLUSTER_X;
    return pos;
}

//Methods
FlexPosition right_pos(FlexPosition pos) {
    uint32_t new_x = (pos.x + 1) % ARCH_NUM_CLUSTER_X;
    uint32_t new_y = pos.y;
    FlexPosition new_pos;
    new_pos.x = new_x;
    new_pos.y = new_y;
    return new_pos;
}

FlexPosition left_pos(FlexPosition pos) {
    uint32_t new_x = (pos.x + ARCH_NUM_CLUSTER_X - 1) % ARCH_NUM_CLUSTER_X;
    uint32_t new_y = pos.y;
    FlexPosition new_pos;
    new_pos.x = new_x;
    new_pos.y = new_y;
    return new_pos;
}

FlexPosition top_pos(FlexPosition pos) {
    uint32_t new_x = pos.x;
    uint32_t new_y = (pos.y + 1) % ARCH_NUM_CLUSTER_Y;
    FlexPosition new_pos;
    new_pos.x = new_x;
    new_pos.y = new_y;
    return new_pos;
}

FlexPosition bottom_pos(FlexPosition pos) {
    uint32_t new_x = pos.x;
    uint32_t new_y = (pos.y + ARCH_NUM_CLUSTER_Y - 1) % ARCH_NUM_CLUSTER_Y;
    FlexPosition new_pos;
    new_pos.x = new_x;
    new_pos.y = new_y;
    return new_pos;
}

uint32_t flex_get_cluster_id(){
    uint32_t * cluster_reg      = ARCH_CLUSTER_REG_BASE;
    return *cluster_reg;
}

/*******************
*  Core Position   *
*******************/

uint32_t flex_get_core_id(){
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return hartid;
}

uint32_t flex_is_dm_core(){
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return (hartid == ARCH_NUM_CORE_PER_CLUSTER-1);
}

uint32_t flex_is_first_core(){
    uint32_t hartid;
    asm volatile("csrr %0, mhartid" : "=r"(hartid));
    return (hartid == 0);
}

/*******************
*  Global Barrier  *
*******************/

uint32_t flex_get_barrier_amo_value(){
    uint32_t * amo_reg      = ARCH_CLUSTER_REG_BASE+4;
    return *amo_reg;
}

uint32_t flex_get_barrier_num_cluster(){
    uint32_t * info_reg      = ARCH_CLUSTER_REG_BASE+8;
    return *info_reg;
}

uint32_t flex_get_barrier_num_cluster_x(){
    uint32_t * info_reg      = ARCH_CLUSTER_REG_BASE+12;
    return *info_reg;
}

uint32_t flex_get_barrier_num_cluster_y(){
    uint32_t * info_reg      = ARCH_CLUSTER_REG_BASE+16;
    return *info_reg;
}

void flex_reset_barrier(uint32_t* barrier){
    *barrier = 0;
}

uint32_t flex_amo_fetch_add(uint32_t* barrier){
    return __atomic_fetch_add(barrier, flex_get_barrier_amo_value(), __ATOMIC_RELAXED);
}

void flex_intra_cluster_sync(){
    asm volatile("csrr x0, 0x7C2" ::: "memory");
}

void flex_barrier_init(){
    uint32_t * barrier      = ARCH_SYNC_BASE;
    uint32_t * wakeup_reg   = ARCH_SOC_REGISTER_WAKEUP;
    uint32_t * cluster_reg  = ARCH_CLUSTER_REG_BASE;

    if (flex_is_dm_core()){
        if (flex_get_cluster_id() == 0)
        {
            // __atomic_store_n(barrier, 0, __ATOMIC_RELAXED);
            flex_reset_barrier(barrier);
            *wakeup_reg = 1;
        }
        *cluster_reg = 1;
    }

    flex_intra_cluster_sync();
}

void flex_global_barrier(){
    uint32_t * barrier      = ARCH_SYNC_BASE;
    uint32_t * wakeup_reg   = ARCH_SOC_REGISTER_WAKEUP;
    uint32_t * cluster_reg  = ARCH_CLUSTER_REG_BASE;

    flex_intra_cluster_sync();

    if (flex_is_dm_core()){
        if ((flex_get_barrier_num_cluster() - 1) == flex_amo_fetch_add(barrier)) {
            flex_reset_barrier(barrier);
            *wakeup_reg = 1;
        }
        *cluster_reg = 1;
    }

    flex_intra_cluster_sync();
}

void flex_barrier_xy_init(){
    FlexPosition pos        = get_pos(flex_get_cluster_id());
    uint32_t   pos_x_middel = (ARCH_NUM_CLUSTER_X)/2;
    uint32_t   pos_y_middel = (ARCH_NUM_CLUSTER_Y)/2;
    uint32_t * barrier_y    = ARCH_SYNC_BASE+(cluster_index(pos_x_middel,pos_y_middel)*ARCH_SYNC_INTERLEAVE)+16;
    uint32_t * wakeup_reg   = ARCH_SOC_REGISTER_WAKEUP;
    uint32_t * cluster_reg  = ARCH_CLUSTER_REG_BASE;

    if (flex_is_dm_core()){
        if (flex_get_cluster_id() == 0)
        {
            flex_reset_barrier(barrier_y);
            for (int i = 0; i < ARCH_NUM_CLUSTER_Y; ++i)
            {
                uint32_t * barrier_x = ARCH_SYNC_BASE+(cluster_index(pos_x_middel,i)*ARCH_SYNC_INTERLEAVE)+8;
                flex_reset_barrier(barrier_x);
            }
            *wakeup_reg = 1;
        }
        *cluster_reg = 1;
    }

    flex_intra_cluster_sync();
}

void flex_global_barrier_xy(){

    flex_intra_cluster_sync();

    if (flex_is_dm_core()){

        FlexPosition pos        = get_pos(flex_get_cluster_id());
        uint32_t   pos_x_middel = (flex_get_barrier_num_cluster_x())/2;
        uint32_t   pos_y_middel = (flex_get_barrier_num_cluster_y())/2;
        uint32_t * barrier_x    = ARCH_SYNC_BASE+(cluster_index(pos_x_middel,pos.y       )*ARCH_SYNC_INTERLEAVE)+8;
        uint32_t * barrier_y    = ARCH_SYNC_BASE+(cluster_index(pos_x_middel,pos_y_middel)*ARCH_SYNC_INTERLEAVE)+16;
        uint32_t * wakeup_reg   = ARCH_SOC_REGISTER_WAKEUP;
        uint32_t * cluster_reg  = ARCH_CLUSTER_REG_BASE;

        //First Barrier X
        if ((flex_get_barrier_num_cluster_x() - 1) == flex_amo_fetch_add(barrier_x)) {
            flex_reset_barrier(barrier_x);

            //For cluster synced X, then sync Y
            if ((flex_get_barrier_num_cluster_y() - 1) == flex_amo_fetch_add(barrier_y))
            {
                flex_reset_barrier(barrier_y);
                *wakeup_reg = 1;
            }
        }
        *cluster_reg = 1;
    }

    flex_intra_cluster_sync();
}

/*******************
*        EoC       *
*******************/

void flex_eoc(uint32_t val){
    volatile uint32_t * eoc_reg = ARCH_SOC_REGISTER_EOC;
    *eoc_reg = val;
}

/*******************
*   Perf Counter   *
*******************/

void flex_timer_start(){
    volatile uint32_t * start_reg    = ARCH_SOC_REGISTER_EOC + 8;
    volatile uint32_t * wakeup_reg   = ARCH_SOC_REGISTER_WAKEUP;
    volatile uint32_t * cluster_reg  = ARCH_CLUSTER_REG_BASE;

    if (flex_is_dm_core()){
        if (flex_get_cluster_id() == 0)
        {
            *start_reg = 1;
            *wakeup_reg = 1;
        }
        *cluster_reg = 1;
    }

    flex_intra_cluster_sync();
}

void flex_timer_end(){
    volatile uint32_t * end_reg = ARCH_SOC_REGISTER_EOC + 12;
    volatile uint32_t * wakeup_reg   = ARCH_SOC_REGISTER_WAKEUP;
    volatile uint32_t * cluster_reg  = ARCH_CLUSTER_REG_BASE;

    if (flex_is_dm_core()){
        if (flex_get_cluster_id() == 0)
        {
            *end_reg = 1;
            *wakeup_reg = 1;
        }
        *cluster_reg = 1;
    }

    flex_intra_cluster_sync();
}

/*******************
*      Logging     *
*******************/

void flex_log_char(char c){
    uint32_t data = (uint32_t) c;
    volatile uint32_t * log_reg = (volatile uint32_t *)(ARCH_SOC_REGISTER_EOC + 16);
    *log_reg = data;
}

void flex_print(char * str){
    for (int i = 0; str[i] != '\0'; i++) {
        flex_log_char(str[i]);
    }
}

void flex_print_int(uint32_t data){
    volatile uint32_t * log_reg = (volatile uint32_t *)(ARCH_SOC_REGISTER_EOC + 20);
    *log_reg = data;
}

/****************************
*      Stack Operations     *
****************************/


inline void flex_push_stack(){
    // asm volatile (
    //     "addi sp, sp, -96 \n"    // Adjust stack pointer (allocate 96 bytes on stack)
    //     "sw ra, 92(sp) \n"       // Save return address (ra) on stack
    //     "sw s0, 88(sp) \n"       // Save frame pointer (s0) on stack
    //     "sw s1, 84(sp) \n"       // Save s1 on stack
    //     "sw s2, 80(sp) \n"       // Save s2 on stack
    //     "sw s3, 76(sp) \n"       // Save s3 on stack
    //     "sw s4, 72(sp) \n"       // Save s4 on stack
    //     "sw s5, 68(sp) \n"       // Save s5 on stack
    //     "sw s6, 64(sp) \n"       // Save s6 on stack
    //     "sw s7, 60(sp) \n"       // Save s7 on stack
    //     "sw s8, 56(sp) \n"       // Save s8 on stack
    //     "sw s9, 52(sp) \n"       // Save s9 on stack
    //     "sw s10, 48(sp) \n"      // Save s10 on stack
    //     "sw s11, 44(sp) \n"      // Save s11 on stack
    //     "sw t0, 40(sp) \n"       // Save t0 on stack
    //     "sw t1, 36(sp) \n"       // Save t1 on stack
    //     "sw t2, 32(sp) \n"       // Save t2 on stack
    //     "sw t3, 28(sp) \n"       // Save t3 on stack
    //     "sw t4, 24(sp) \n"       // Save t4 on stack
    //     "sw t5, 20(sp) \n"       // Save t5 on stack
    //     "sw t6, 16(sp) \n"       // Save t6 on stack
    // );
}

inline void flex_pull_stack(){
    // asm volatile (
    //     "lw ra, 92(sp) \n"       // Restore return address (ra)
    //     "lw s0, 88(sp) \n"       // Restore frame pointer (s0)
    //     "lw s1, 84(sp) \n"       // Restore s1
    //     "lw s2, 80(sp) \n"       // Restore s2
    //     "lw s3, 76(sp) \n"       // Restore s3
    //     "lw s4, 72(sp) \n"       // Restore s4
    //     "lw s5, 68(sp) \n"       // Restore s5
    //     "lw s6, 64(sp) \n"       // Restore s6
    //     "lw s7, 60(sp) \n"       // Restore s7
    //     "lw s8, 56(sp) \n"       // Restore s8
    //     "lw s9, 52(sp) \n"       // Restore s9
    //     "lw s10, 48(sp) \n"      // Restore s10
    //     "lw s11, 44(sp) \n"      // Restore s11
    //     "lw t0, 40(sp) \n"       // Restore t0
    //     "lw t1, 36(sp) \n"       // Restore t1
    //     "lw t2, 32(sp) \n"       // Restore t2
    //     "lw t3, 28(sp) \n"       // Restore t3
    //     "lw t4, 24(sp) \n"       // Restore t4
    //     "lw t5, 20(sp) \n"       // Restore t5
    //     "lw t6, 16(sp) \n"       // Restore t6
    //     "addi sp, sp, 96 \n"     // Adjust stack pointer back (deallocate 96 bytes from stack)
    // );
}

#endif