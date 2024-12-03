# Copyright 2020 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

.include "flex_cluster_arch.inc"

.section .init
_start:
    .globl _start

init_int_registers:
    # Clear integer registers
    mv t0, x0
    mv t1, x0
    mv t2, x0
    mv t3, x0
    mv t4, x0
    mv t5, x0
    mv t6, x0
    mv a0, x0
    mv a1, x0
    mv a2, x0
    mv a3, x0
    mv a4, x0
    mv a5, x0
    mv a6, x0
    mv a7, x0
    mv s0, x0
    mv s1, x0
    mv s2, x0
    mv s3, x0
    mv s4, x0
    mv s5, x0
    mv s6, x0
    mv s7, x0
    mv s8, x0
    mv s9, x0
    mv s10, x0
    mv s11, x0

init_fp_registers:
    # Check if core has FP registers otherwise skip
    csrr    t0, misa
    andi    t0, t0, (1 << 3) | (1 << 5) # D/F - single/double precision float extension
    beqz    t0, 3f
    # Clear FP registers
    fcvt.d.w f0, zero
    fcvt.d.w f1, zero
    fcvt.d.w f2, zero
    fcvt.d.w f3, zero
    fcvt.d.w f4, zero
    fcvt.d.w f5, zero
    fcvt.d.w f6, zero
    fcvt.d.w f7, zero
    fcvt.d.w f8, zero
    fcvt.d.w f9, zero
    fcvt.d.w f10, zero
    fcvt.d.w f11, zero
    fcvt.d.w f12, zero
    fcvt.d.w f13, zero
    fcvt.d.w f14, zero
    fcvt.d.w f15, zero
    fcvt.d.w f16, zero
    fcvt.d.w f17, zero
    fcvt.d.w f18, zero
    fcvt.d.w f19, zero
    fcvt.d.w f20, zero
    fcvt.d.w f21, zero
    fcvt.d.w f22, zero
    fcvt.d.w f23, zero
    fcvt.d.w f24, zero
    fcvt.d.w f25, zero
    fcvt.d.w f26, zero
    fcvt.d.w f27, zero
    fcvt.d.w f28, zero
    fcvt.d.w f29, zero
    fcvt.d.w f30, zero
    fcvt.d.w f31, zero
3:

init_global_pointer:
    # Initialize global pointer
    .option push
    .option norelax
1:  auipc   gp, %pcrel_hi(__global_pointer$)
    addi    gp, gp, %pcrel_lo(1b)
    .option pop


init_stack:
    # Get core id
    csrr a0, mhartid

    # Calculate cluster's Stack start address
    lui a2, %hi(ARCH_CLUSTER_STACK_BASE)
    addi a2, a2, %lo(ARCH_CLUSTER_STACK_BASE)

    lui t1, %hi(ARCH_CLUSTER_STACK_SIZE)   # Load upper 20 bits into t1
    addi t1, t1, %lo(ARCH_CLUSTER_STACK_SIZE)  # Add lower 12 bits

    add  a2, a2, t1

    # Compute the relative start address of the stack for each hart.
    # The stack for hart N starts at the end of the stack of hart N-1.
    sll  t0, a0, 0xa
    
    # Initialize the stack pointer to the start of the stack
    sub  sp, a2, t0

softhier.main:
    call main

softhier.end:
1:
    wfi
    j       1b


