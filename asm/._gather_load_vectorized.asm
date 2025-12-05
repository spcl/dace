
.dacecache/gather_load_vectorized/build/libgather_load_vectorized.so:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64
    1004:	48 83 ec 08          	sub    $0x8,%rsp
    1008:	48 8b 05 c9 2f 00 00 	mov    0x2fc9(%rip),%rax        # 3fd8 <__gmon_start__@Base>
    100f:	48 85 c0             	test   %rax,%rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	call   *%rax
    1016:	48 83 c4 08          	add    $0x8,%rsp
    101a:	c3                   	ret

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 ca 2f 00 00    	push   0x2fca(%rip)        # 3ff0 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	ff 25 cc 2f 00 00    	jmp    *0x2fcc(%rip)        # 3ff8 <_GLOBAL_OFFSET_TABLE_+0x10>
    102c:	0f 1f 40 00          	nopl   0x0(%rax)
    1030:	f3 0f 1e fa          	endbr64
    1034:	68 00 00 00 00       	push   $0x0
    1039:	e9 e2 ff ff ff       	jmp    1020 <_init+0x20>
    103e:	66 90                	xchg   %ax,%ax
    1040:	f3 0f 1e fa          	endbr64
    1044:	68 01 00 00 00       	push   $0x1
    1049:	e9 d2 ff ff ff       	jmp    1020 <_init+0x20>
    104e:	66 90                	xchg   %ax,%ax
    1050:	f3 0f 1e fa          	endbr64
    1054:	68 02 00 00 00       	push   $0x2
    1059:	e9 c2 ff ff ff       	jmp    1020 <_init+0x20>
    105e:	66 90                	xchg   %ax,%ax
    1060:	f3 0f 1e fa          	endbr64
    1064:	68 03 00 00 00       	push   $0x3
    1069:	e9 b2 ff ff ff       	jmp    1020 <_init+0x20>
    106e:	66 90                	xchg   %ax,%ax
    1070:	f3 0f 1e fa          	endbr64
    1074:	68 04 00 00 00       	push   $0x4
    1079:	e9 a2 ff ff ff       	jmp    1020 <_init+0x20>
    107e:	66 90                	xchg   %ax,%ax
    1080:	f3 0f 1e fa          	endbr64
    1084:	68 05 00 00 00       	push   $0x5
    1089:	e9 92 ff ff ff       	jmp    1020 <_init+0x20>
    108e:	66 90                	xchg   %ax,%ax
    1090:	f3 0f 1e fa          	endbr64
    1094:	68 06 00 00 00       	push   $0x6
    1099:	e9 82 ff ff ff       	jmp    1020 <_init+0x20>
    109e:	66 90                	xchg   %ax,%ax
    10a0:	f3 0f 1e fa          	endbr64
    10a4:	68 07 00 00 00       	push   $0x7
    10a9:	e9 72 ff ff ff       	jmp    1020 <_init+0x20>
    10ae:	66 90                	xchg   %ax,%ax

Disassembly of section .plt.got:

00000000000010b0 <__cxa_finalize@plt>:
    10b0:	f3 0f 1e fa          	endbr64
    10b4:	ff 25 0e 2f 00 00    	jmp    *0x2f0e(%rip)        # 3fc8 <__cxa_finalize@GLIBC_2.2.5>
    10ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

00000000000010c0 <_Znwm@plt>:
    10c0:	f3 0f 1e fa          	endbr64
    10c4:	ff 25 36 2f 00 00    	jmp    *0x2f36(%rip)        # 4000 <_Znwm@GLIBCXX_3.4>
    10ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000010d0 <_ZdlPvm@plt>:
    10d0:	f3 0f 1e fa          	endbr64
    10d4:	ff 25 2e 2f 00 00    	jmp    *0x2f2e(%rip)        # 4008 <_ZdlPvm@CXXABI_1.3.9>
    10da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000010e0 <__stack_chk_fail@plt>:
    10e0:	f3 0f 1e fa          	endbr64
    10e4:	ff 25 26 2f 00 00    	jmp    *0x2f26(%rip)        # 4010 <__stack_chk_fail@GLIBC_2.4>
    10ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000010f0 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id@plt>:
    10f0:	f3 0f 1e fa          	endbr64
    10f4:	ff 25 1e 2f 00 00    	jmp    *0x2f1e(%rip)        # 4018 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id@@Base+0x2c98>
    10fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001100 <_Z20gather_double_avx512PKdPKlPd@plt>:
    1100:	f3 0f 1e fa          	endbr64
    1104:	ff 25 16 2f 00 00    	jmp    *0x2f16(%rip)        # 4020 <_Z20gather_double_avx512PKdPKlPd@@Base+0x2e20>
    110a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001110 <GOMP_parallel@plt>:
    1110:	f3 0f 1e fa          	endbr64
    1114:	ff 25 0e 2f 00 00    	jmp    *0x2f0e(%rip)        # 4028 <GOMP_parallel@GOMP_4.0>
    111a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001120 <omp_get_thread_num@plt>:
    1120:	f3 0f 1e fa          	endbr64
    1124:	ff 25 06 2f 00 00    	jmp    *0x2f06(%rip)        # 4030 <omp_get_thread_num@OMP_1.0>
    112a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001130 <omp_get_num_threads@plt>:
    1130:	f3 0f 1e fa          	endbr64
    1134:	ff 25 fe 2e 00 00    	jmp    *0x2efe(%rip)        # 4038 <omp_get_num_threads@OMP_1.0>
    113a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

Disassembly of section .text:

0000000000001140 <deregister_tm_clones>:
    1140:	48 8d 3d 09 2f 00 00 	lea    0x2f09(%rip),%rdi        # 4050 <completed.0>
    1147:	48 8d 05 02 2f 00 00 	lea    0x2f02(%rip),%rax        # 4050 <completed.0>
    114e:	48 39 f8             	cmp    %rdi,%rax
    1151:	74 15                	je     1168 <deregister_tm_clones+0x28>
    1153:	48 8b 05 76 2e 00 00 	mov    0x2e76(%rip),%rax        # 3fd0 <_ITM_deregisterTMCloneTable@Base>
    115a:	48 85 c0             	test   %rax,%rax
    115d:	74 09                	je     1168 <deregister_tm_clones+0x28>
    115f:	ff e0                	jmp    *%rax
    1161:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1168:	c3                   	ret
    1169:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001170 <register_tm_clones>:
    1170:	48 8d 3d d9 2e 00 00 	lea    0x2ed9(%rip),%rdi        # 4050 <completed.0>
    1177:	48 8d 35 d2 2e 00 00 	lea    0x2ed2(%rip),%rsi        # 4050 <completed.0>
    117e:	48 29 fe             	sub    %rdi,%rsi
    1181:	48 89 f0             	mov    %rsi,%rax
    1184:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1188:	48 c1 f8 03          	sar    $0x3,%rax
    118c:	48 01 c6             	add    %rax,%rsi
    118f:	48 d1 fe             	sar    $1,%rsi
    1192:	74 14                	je     11a8 <register_tm_clones+0x38>
    1194:	48 8b 05 45 2e 00 00 	mov    0x2e45(%rip),%rax        # 3fe0 <_ITM_registerTMCloneTable@Base>
    119b:	48 85 c0             	test   %rax,%rax
    119e:	74 08                	je     11a8 <register_tm_clones+0x38>
    11a0:	ff e0                	jmp    *%rax
    11a2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    11a8:	c3                   	ret
    11a9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000011b0 <__do_global_dtors_aux>:
    11b0:	f3 0f 1e fa          	endbr64
    11b4:	80 3d 95 2e 00 00 00 	cmpb   $0x0,0x2e95(%rip)        # 4050 <completed.0>
    11bb:	75 2b                	jne    11e8 <__do_global_dtors_aux+0x38>
    11bd:	55                   	push   %rbp
    11be:	48 83 3d 02 2e 00 00 	cmpq   $0x0,0x2e02(%rip)        # 3fc8 <__cxa_finalize@GLIBC_2.2.5>
    11c5:	00 
    11c6:	48 89 e5             	mov    %rsp,%rbp
    11c9:	74 0c                	je     11d7 <__do_global_dtors_aux+0x27>
    11cb:	48 8b 3d 6e 2e 00 00 	mov    0x2e6e(%rip),%rdi        # 4040 <__dso_handle>
    11d2:	e8 d9 fe ff ff       	call   10b0 <__cxa_finalize@plt>
    11d7:	e8 64 ff ff ff       	call   1140 <deregister_tm_clones>
    11dc:	c6 05 6d 2e 00 00 01 	movb   $0x1,0x2e6d(%rip)        # 4050 <completed.0>
    11e3:	5d                   	pop    %rbp
    11e4:	c3                   	ret
    11e5:	0f 1f 00             	nopl   (%rax)
    11e8:	c3                   	ret
    11e9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000011f0 <frame_dummy>:
    11f0:	f3 0f 1e fa          	endbr64
    11f4:	e9 77 ff ff ff       	jmp    1170 <register_tm_clones>
    11f9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001200 <_Z20gather_double_avx512PKdPKlPd>:
    1200:	f3 0f 1e fa          	endbr64
    1204:	62 f1 fe 48 6f 0e    	vmovdqu64 (%rsi),%zmm1
    120a:	c5 f9 90 0d ee 0d 00 	kmovb  0xdee(%rip),%k1        # 2000 <_fini+0xbac>
    1211:	00 
    1212:	62 f2 fd 49 93 04 cf 	vgatherqpd (%rdi,%zmm1,8),%zmm0{%k1}
    1219:	62 f1 fd 48 11 02    	vmovupd %zmm0,(%rdx)
    121f:	c5 f8 77             	vzeroupper
    1222:	c3                   	ret
    1223:	66 90                	xchg   %ax,%ax
    1225:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    122c:	00 00 00 00 

0000000000001230 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id._omp_fn.0>:
    1230:	f3 0f 1e fa          	endbr64
    1234:	55                   	push   %rbp
    1235:	48 89 e5             	mov    %rsp,%rbp
    1238:	41 57                	push   %r15
    123a:	41 56                	push   %r14
    123c:	41 55                	push   %r13
    123e:	41 54                	push   %r12
    1240:	53                   	push   %rbx
    1241:	48 83 e4 c0          	and    $0xffffffffffffffc0,%rsp
    1245:	48 81 ec 00 01 00 00 	sub    $0x100,%rsp
    124c:	64 4c 8b 24 25 28 00 	mov    %fs:0x28,%r12
    1253:	00 00 
    1255:	4c 89 a4 24 f8 00 00 	mov    %r12,0xf8(%rsp)
    125c:	00 
    125d:	49 89 fc             	mov    %rdi,%r12
    1260:	e8 cb fe ff ff       	call   1130 <omp_get_num_threads@plt>
    1265:	89 c3                	mov    %eax,%ebx
    1267:	e8 b4 fe ff ff       	call   1120 <omp_get_thread_num@plt>
    126c:	41 8b 54 24 28       	mov    0x28(%r12),%edx
    1271:	89 c1                	mov    %eax,%ecx
    1273:	8d 42 0e             	lea    0xe(%rdx),%eax
    1276:	83 c2 07             	add    $0x7,%edx
    1279:	0f 49 c2             	cmovns %edx,%eax
    127c:	c1 f8 03             	sar    $0x3,%eax
    127f:	99                   	cltd
    1280:	f7 fb                	idiv   %ebx
    1282:	39 d1                	cmp    %edx,%ecx
    1284:	8d 58 01             	lea    0x1(%rax),%ebx
    1287:	0f 4d d8             	cmovge %eax,%ebx
    128a:	b8 00 00 00 00       	mov    $0x0,%eax
    128f:	0f 4c d0             	cmovl  %eax,%edx
    1292:	0f af cb             	imul   %ebx,%ecx
    1295:	01 ca                	add    %ecx,%edx
    1297:	01 d3                	add    %edx,%ebx
    1299:	39 da                	cmp    %ebx,%edx
    129b:	0f 8d b1 00 00 00    	jge    1352 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id._omp_fn.0+0x122>
    12a1:	49 8b 44 24 18       	mov    0x18(%r12),%rax
    12a6:	4d 8b 7c 24 10       	mov    0x10(%r12),%r15
    12ab:	48 89 44 24 38       	mov    %rax,0x38(%rsp)
    12b0:	8d 04 dd 00 00 00 00 	lea    0x0(,%rbx,8),%eax
    12b7:	89 44 24 34          	mov    %eax,0x34(%rsp)
    12bb:	8d 04 d5 00 00 00 00 	lea    0x0(,%rdx,8),%eax
    12c2:	48 63 d2             	movslq %edx,%rdx
    12c5:	4c 63 f0             	movslq %eax,%r14
    12c8:	48 8d 44 24 40       	lea    0x40(%rsp),%rax
    12cd:	48 c1 e2 06          	shl    $0x6,%rdx
    12d1:	4c 89 f6             	mov    %r14,%rsi
    12d4:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
    12d9:	48 8d 84 24 80 00 00 	lea    0x80(%rsp),%rax
    12e0:	00 
    12e1:	49 01 d7             	add    %rdx,%r15
    12e4:	48 f7 de             	neg    %rsi
    12e7:	48 89 44 24 28       	mov    %rax,0x28(%rsp)
    12ec:	4c 8d 2c f2          	lea    (%rdx,%rsi,8),%r13
    12f0:	4d 03 6c 24 08       	add    0x8(%r12),%r13
    12f5:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    12fc:	00 00 00 00 
    1300:	62 d1 fe 48 6f 07    	vmovdqu64 (%r15),%zmm0
    1306:	49 8b 5c 24 20       	mov    0x20(%r12),%rbx
    130b:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
    1310:	48 8b 74 24 28       	mov    0x28(%rsp),%rsi
    1315:	48 8b 7c 24 38       	mov    0x38(%rsp),%rdi
    131a:	62 f1 fd 48 7f 44 24 	vmovdqa64 %zmm0,0x80(%rsp)
    1321:	02 
    1322:	c5 f8 77             	vzeroupper
    1325:	e8 d6 fd ff ff       	call   1100 <_Z20gather_double_avx512PKdPKlPd@plt>
    132a:	62 f2 fd 48 19 03    	vbroadcastsd (%rbx),%zmm0
    1330:	62 f1 fd 48 59 44 24 	vmulpd 0x40(%rsp),%zmm0,%zmm0
    1337:	01 
    1338:	49 83 c7 40          	add    $0x40,%r15
    133c:	62 91 7f 48 7f 44 f5 	vmovdqu8 %zmm0,0x0(%r13,%r14,8)
    1343:	00 
    1344:	49 83 c6 08          	add    $0x8,%r14
    1348:	44 39 74 24 34       	cmp    %r14d,0x34(%rsp)
    134d:	7f b1                	jg     1300 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id._omp_fn.0+0xd0>
    134f:	c5 f8 77             	vzeroupper
    1352:	48 8b 84 24 f8 00 00 	mov    0xf8(%rsp),%rax
    1359:	00 
    135a:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    1361:	00 00 
    1363:	75 0f                	jne    1374 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id._omp_fn.0+0x144>
    1365:	48 8d 65 d8          	lea    -0x28(%rbp),%rsp
    1369:	5b                   	pop    %rbx
    136a:	41 5c                	pop    %r12
    136c:	41 5d                	pop    %r13
    136e:	41 5e                	pop    %r14
    1370:	41 5f                	pop    %r15
    1372:	5d                   	pop    %rbp
    1373:	c3                   	ret
    1374:	e8 67 fd ff ff       	call   10e0 <__stack_chk_fail@plt>
    1379:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001380 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id>:
    1380:	f3 0f 1e fa          	endbr64
    1384:	55                   	push   %rbp
    1385:	c4 e1 f9 6e cf       	vmovq  %rdi,%xmm1
    138a:	c4 e1 f9 6e d2       	vmovq  %rdx,%xmm2
    138f:	48 8d 3d 9a fe ff ff 	lea    -0x166(%rip),%rdi        # 1230 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id._omp_fn.0>
    1396:	48 89 e5             	mov    %rsp,%rbp
    1399:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    139d:	31 d2                	xor    %edx,%edx
    139f:	48 83 ec 60          	sub    $0x60,%rsp
    13a3:	c5 fb 11 44 24 18    	vmovsd %xmm0,0x18(%rsp)
    13a9:	c4 e3 f1 22 c6 01    	vpinsrq $0x1,%rsi,%xmm1,%xmm0
    13af:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
    13b4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    13bb:	00 00 
    13bd:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    13c2:	31 c0                	xor    %eax,%eax
    13c4:	c5 f9 7f 44 24 20    	vmovdqa %xmm0,0x20(%rsp)
    13ca:	c4 e3 e9 22 c1 01    	vpinsrq $0x1,%rcx,%xmm2,%xmm0
    13d0:	c5 f9 7f 44 24 30    	vmovdqa %xmm0,0x30(%rsp)
    13d6:	48 8d 44 24 18       	lea    0x18(%rsp),%rax
    13db:	31 c9                	xor    %ecx,%ecx
    13dd:	44 89 44 24 48       	mov    %r8d,0x48(%rsp)
    13e2:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    13e7:	e8 24 fd ff ff       	call   1110 <GOMP_parallel@plt>
    13ec:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    13f1:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    13f8:	00 00 
    13fa:	75 02                	jne    13fe <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id+0x7e>
    13fc:	c9                   	leave
    13fd:	c3                   	ret
    13fe:	e8 dd fc ff ff       	call   10e0 <__stack_chk_fail@plt>
    1403:	66 90                	xchg   %ax,%ax
    1405:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    140c:	00 00 00 00 

0000000000001410 <__program_gather_load_vectorized>:
    1410:	f3 0f 1e fa          	endbr64
    1414:	e9 d7 fc ff ff       	jmp    10f0 <_Z41__program_gather_load_vectorized_internalP30gather_load_vectorized_state_tPdPlS1_id@plt>
    1419:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001420 <__dace_init_gather_load_vectorized>:
    1420:	f3 0f 1e fa          	endbr64
    1424:	bf 01 00 00 00       	mov    $0x1,%edi
    1429:	e9 92 fc ff ff       	jmp    10c0 <_Znwm@plt>
    142e:	66 90                	xchg   %ax,%ax

0000000000001430 <__dace_exit_gather_load_vectorized>:
    1430:	f3 0f 1e fa          	endbr64
    1434:	48 85 ff             	test   %rdi,%rdi
    1437:	74 17                	je     1450 <__dace_exit_gather_load_vectorized+0x20>
    1439:	48 83 ec 08          	sub    $0x8,%rsp
    143d:	be 01 00 00 00       	mov    $0x1,%esi
    1442:	e8 89 fc ff ff       	call   10d0 <_ZdlPvm@plt>
    1447:	31 c0                	xor    %eax,%eax
    1449:	48 83 c4 08          	add    $0x8,%rsp
    144d:	c3                   	ret
    144e:	66 90                	xchg   %ax,%ax
    1450:	31 c0                	xor    %eax,%eax
    1452:	c3                   	ret

Disassembly of section .fini:

0000000000001454 <_fini>:
    1454:	f3 0f 1e fa          	endbr64
    1458:	48 83 ec 08          	sub    $0x8,%rsp
    145c:	48 83 c4 08          	add    $0x8,%rsp
    1460:	c3                   	ret
