#ifndef _FLEX_REDMULE_H_
#define _FLEX_REDMULE_H_
#include "flex_cluster_arch.h"

typedef enum {
    REDMULE_NONE_16,
    REDMULE_UINT_16,
    REDMULE_INT_16,
    REDMULE_FP_16
} redmule_compute_format_t;


void flex_redmule_config(uint16_t m_size, uint16_t n_size, uint16_t k_size){
    flex_push_stack();
    uint32_t cfg_reg0 = ((k_size << 16) | (m_size << 0));
    uint32_t cfg_reg1 = (n_size << 0);

    asm volatile ("addi t3, %0, 0" :: "r"(cfg_reg0): "t3");
    asm volatile ("addi t4, %0, 0" :: "r"(cfg_reg1): "t4");

    /* mcnfig instruction */
    // asm volatile(
    //      ".word (0x0       << 25) | \     /* Empty         */
    //             (0b11101   << 20) | \     /* Rs2 (t4 here) */
    //             (0b11100   << 15) | \     /* Rs1 (t3 here) */
    //             (0x00      <<  7) | \     /* Empty         */
    //             (0b0001010 <<  0)   \n"); /* OpCode        */
      
    asm volatile(
         ".word (0x0       << 25) | \
                (0b11101   << 20) | \
                (0b11100   << 15) | \
                (0x00      <<  7) | \
                (0b0001010 <<  0)   \n");
    flex_pull_stack();
}

void flex_redmule_trigger(uint32_t x_addr, uint32_t w_addr, uint32_t y_addr, redmule_compute_format_t format){
    flex_push_stack();
    // if (flex_is_first_core() && flex_get_cluster_id() == 0)
    // {
    //     flex_print("x_addr: ");flex_print_int(x_addr);
    //     flex_print(" w_addr: ");flex_print_int(w_addr);
    //     flex_print(" y_addr: ");flex_print_int(y_addr);
    //     flex_print("\n");
    // }
    asm volatile ("addi t0, %0, 0" :: "r"(x_addr): "t0");
    asm volatile ("addi t1, %0, 0" :: "r"(w_addr): "t1");
    asm volatile ("addi t2, %0, 0" :: "r"(y_addr): "t2");

    /* arith instruction */
    // sm volatile(
    //     ".word (0b00111   << 27) | \     /* Rs3 (t2 here)                */
    //            (0b00      << 25) | \     /* Empty                        */
    //            (0b00110   << 20) | \     /* Rs2 (t1 here)                */
    //            (0b00101   << 15) | \     /* Rs1 (t0 here)                */
    //            (0b0       << 14) | \     /* Custom format enable/disable */
    //            (0b0       << 13) | \     /* Widening enable/disable      */
    //            (0b001     << 10) | \     /* Operation selection          */
    //            (0b001     <<  7) | \     /* Data format                  */
    //            (0b0101010 <<  0)   \n"); /* OpCode                       */

    switch (format) {
        case REDMULE_NONE_16:
            asm volatile(
                 ".word (0b00111   << 27) | \
                        (0b00      << 25) | \
                        (0b00110   << 20) | \
                        (0b00101   << 15) | \
                        (0b0       << 14) | \
                        (0b0       << 13) | \
                        (0b000     << 10) | \
                        (0b000     <<  7) | \
                        (0b0101010 <<  0)   \n");
            break;
        case REDMULE_UINT_16:
            asm volatile(
                 ".word (0b00111   << 27) | \
                        (0b00      << 25) | \
                        (0b00110   << 20) | \
                        (0b00101   << 15) | \
                        (0b0       << 14) | \
                        (0b0       << 13) | \
                        (0b000     << 10) | \
                        (0b001     <<  7) | \
                        (0b0101010 <<  0)   \n");
            break;
        case REDMULE_INT_16:
            asm volatile(
                 ".word (0b00111   << 27) | \
                        (0b00      << 25) | \
                        (0b00110   << 20) | \
                        (0b00101   << 15) | \
                        (0b0       << 14) | \
                        (0b0       << 13) | \
                        (0b000     << 10) | \
                        (0b010     <<  7) | \
                        (0b0101010 <<  0)   \n");
            break;
        case REDMULE_FP_16:
            asm volatile(
                 ".word (0b00111   << 27) | \
                        (0b00      << 25) | \
                        (0b00110   << 20) | \
                        (0b00101   << 15) | \
                        (0b0       << 14) | \
                        (0b0       << 13) | \
                        (0b000     << 10) | \
                        (0b011     <<  7) | \
                        (0b0101010 <<  0)   \n");
            break;
    }
            
    flex_pull_stack();
}

uint32_t flex_redmule_wait(){
    volatile uint32_t * redmule_reg  = ARCH_REDMULE_REG_BASE + 40;
    return (*redmule_reg);
}

#endif