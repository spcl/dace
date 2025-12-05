
.dacecache/gather_load/build/libgather_load.so:     file format elf64-x86-64


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

Disassembly of section .plt.got:

00000000000010a0 <__cxa_finalize@plt>:
    10a0:	f3 0f 1e fa          	endbr64
    10a4:	ff 25 1e 2f 00 00    	jmp    *0x2f1e(%rip)        # 3fc8 <__cxa_finalize@GLIBC_2.2.5>
    10aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

00000000000010b0 <_Znwm@plt>:
    10b0:	f3 0f 1e fa          	endbr64
    10b4:	ff 25 46 2f 00 00    	jmp    *0x2f46(%rip)        # 4000 <_Znwm@GLIBCXX_3.4>
    10ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000010c0 <_ZdlPvm@plt>:
    10c0:	f3 0f 1e fa          	endbr64
    10c4:	ff 25 3e 2f 00 00    	jmp    *0x2f3e(%rip)        # 4008 <_ZdlPvm@CXXABI_1.3.9>
    10ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000010d0 <__stack_chk_fail@plt>:
    10d0:	f3 0f 1e fa          	endbr64
    10d4:	ff 25 36 2f 00 00    	jmp    *0x2f36(%rip)        # 4010 <__stack_chk_fail@GLIBC_2.4>
    10da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000010e0 <GOMP_parallel@plt>:
    10e0:	f3 0f 1e fa          	endbr64
    10e4:	ff 25 2e 2f 00 00    	jmp    *0x2f2e(%rip)        # 4018 <GOMP_parallel@GOMP_4.0>
    10ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

00000000000010f0 <omp_get_thread_num@plt>:
    10f0:	f3 0f 1e fa          	endbr64
    10f4:	ff 25 26 2f 00 00    	jmp    *0x2f26(%rip)        # 4020 <omp_get_thread_num@OMP_1.0>
    10fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001100 <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id@plt>:
    1100:	f3 0f 1e fa          	endbr64
    1104:	ff 25 1e 2f 00 00    	jmp    *0x2f1e(%rip)        # 4028 <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id@@Base+0x2db8>
    110a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001110 <omp_get_num_threads@plt>:
    1110:	f3 0f 1e fa          	endbr64
    1114:	ff 25 16 2f 00 00    	jmp    *0x2f16(%rip)        # 4030 <omp_get_num_threads@OMP_1.0>
    111a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

Disassembly of section .text:

0000000000001120 <deregister_tm_clones>:
    1120:	48 8d 3d 19 2f 00 00 	lea    0x2f19(%rip),%rdi        # 4040 <completed.0>
    1127:	48 8d 05 12 2f 00 00 	lea    0x2f12(%rip),%rax        # 4040 <completed.0>
    112e:	48 39 f8             	cmp    %rdi,%rax
    1131:	74 15                	je     1148 <deregister_tm_clones+0x28>
    1133:	48 8b 05 96 2e 00 00 	mov    0x2e96(%rip),%rax        # 3fd0 <_ITM_deregisterTMCloneTable@Base>
    113a:	48 85 c0             	test   %rax,%rax
    113d:	74 09                	je     1148 <deregister_tm_clones+0x28>
    113f:	ff e0                	jmp    *%rax
    1141:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1148:	c3                   	ret
    1149:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001150 <register_tm_clones>:
    1150:	48 8d 3d e9 2e 00 00 	lea    0x2ee9(%rip),%rdi        # 4040 <completed.0>
    1157:	48 8d 35 e2 2e 00 00 	lea    0x2ee2(%rip),%rsi        # 4040 <completed.0>
    115e:	48 29 fe             	sub    %rdi,%rsi
    1161:	48 89 f0             	mov    %rsi,%rax
    1164:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1168:	48 c1 f8 03          	sar    $0x3,%rax
    116c:	48 01 c6             	add    %rax,%rsi
    116f:	48 d1 fe             	sar    $1,%rsi
    1172:	74 14                	je     1188 <register_tm_clones+0x38>
    1174:	48 8b 05 65 2e 00 00 	mov    0x2e65(%rip),%rax        # 3fe0 <_ITM_registerTMCloneTable@Base>
    117b:	48 85 c0             	test   %rax,%rax
    117e:	74 08                	je     1188 <register_tm_clones+0x38>
    1180:	ff e0                	jmp    *%rax
    1182:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1188:	c3                   	ret
    1189:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001190 <__do_global_dtors_aux>:
    1190:	f3 0f 1e fa          	endbr64
    1194:	80 3d a5 2e 00 00 00 	cmpb   $0x0,0x2ea5(%rip)        # 4040 <completed.0>
    119b:	75 2b                	jne    11c8 <__do_global_dtors_aux+0x38>
    119d:	55                   	push   %rbp
    119e:	48 83 3d 22 2e 00 00 	cmpq   $0x0,0x2e22(%rip)        # 3fc8 <__cxa_finalize@GLIBC_2.2.5>
    11a5:	00 
    11a6:	48 89 e5             	mov    %rsp,%rbp
    11a9:	74 0c                	je     11b7 <__do_global_dtors_aux+0x27>
    11ab:	48 8b 3d 86 2e 00 00 	mov    0x2e86(%rip),%rdi        # 4038 <__dso_handle>
    11b2:	e8 e9 fe ff ff       	call   10a0 <__cxa_finalize@plt>
    11b7:	e8 64 ff ff ff       	call   1120 <deregister_tm_clones>
    11bc:	c6 05 7d 2e 00 00 01 	movb   $0x1,0x2e7d(%rip)        # 4040 <completed.0>
    11c3:	5d                   	pop    %rbp
    11c4:	c3                   	ret
    11c5:	0f 1f 00             	nopl   (%rax)
    11c8:	c3                   	ret
    11c9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000011d0 <frame_dummy>:
    11d0:	f3 0f 1e fa          	endbr64
    11d4:	e9 77 ff ff ff       	jmp    1150 <register_tm_clones>
    11d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000011e0 <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id._omp_fn.0>:
    11e0:	f3 0f 1e fa          	endbr64
    11e4:	55                   	push   %rbp
    11e5:	53                   	push   %rbx
    11e6:	48 89 fb             	mov    %rdi,%rbx
    11e9:	48 83 ec 08          	sub    $0x8,%rsp
    11ed:	e8 1e ff ff ff       	call   1110 <omp_get_num_threads@plt>
    11f2:	89 c5                	mov    %eax,%ebp
    11f4:	e8 f7 fe ff ff       	call   10f0 <omp_get_thread_num@plt>
    11f9:	89 c1                	mov    %eax,%ecx
    11fb:	8b 43 28             	mov    0x28(%rbx),%eax
    11fe:	99                   	cltd
    11ff:	f7 fd                	idiv   %ebp
    1201:	39 d1                	cmp    %edx,%ecx
    1203:	8d 70 01             	lea    0x1(%rax),%esi
    1206:	0f 4c c6             	cmovl  %esi,%eax
    1209:	be 00 00 00 00       	mov    $0x0,%esi
    120e:	0f 4c d6             	cmovl  %esi,%edx
    1211:	0f af c8             	imul   %eax,%ecx
    1214:	01 ca                	add    %ecx,%edx
    1216:	01 d0                	add    %edx,%eax
    1218:	39 c2                	cmp    %eax,%edx
    121a:	7d 3e                	jge    125a <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id._omp_fn.0+0x7a>
    121c:	4c 8b 4b 18          	mov    0x18(%rbx),%r9
    1220:	4c 8b 43 10          	mov    0x10(%rbx),%r8
    1224:	48 8b 7b 08          	mov    0x8(%rbx),%rdi
    1228:	48 63 d2             	movslq %edx,%rdx
    122b:	48 8b 73 20          	mov    0x20(%rbx),%rsi
    122f:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1235:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    123c:	00 00 00 00 
    1240:	49 8b 0c d0          	mov    (%r8,%rdx,8),%rcx
    1244:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1248:	c4 c1 7b 59 04 c9    	vmulsd (%r9,%rcx,8),%xmm0,%xmm0
    124e:	c5 fb 11 04 d7       	vmovsd %xmm0,(%rdi,%rdx,8)
    1253:	48 ff c2             	inc    %rdx
    1256:	39 d0                	cmp    %edx,%eax
    1258:	7f e6                	jg     1240 <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id._omp_fn.0+0x60>
    125a:	48 83 c4 08          	add    $0x8,%rsp
    125e:	5b                   	pop    %rbx
    125f:	5d                   	pop    %rbp
    1260:	c3                   	ret
    1261:	0f 1f 40 00          	nopl   0x0(%rax)
    1265:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    126c:	00 00 00 00 

0000000000001270 <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id>:
    1270:	f3 0f 1e fa          	endbr64
    1274:	55                   	push   %rbp
    1275:	c4 e1 f9 6e cf       	vmovq  %rdi,%xmm1
    127a:	c4 e1 f9 6e d2       	vmovq  %rdx,%xmm2
    127f:	48 8d 3d 5a ff ff ff 	lea    -0xa6(%rip),%rdi        # 11e0 <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id._omp_fn.0>
    1286:	48 89 e5             	mov    %rsp,%rbp
    1289:	48 83 e4 e0          	and    $0xffffffffffffffe0,%rsp
    128d:	31 d2                	xor    %edx,%edx
    128f:	48 83 ec 60          	sub    $0x60,%rsp
    1293:	c5 fb 11 44 24 18    	vmovsd %xmm0,0x18(%rsp)
    1299:	c4 e3 f1 22 c6 01    	vpinsrq $0x1,%rsi,%xmm1,%xmm0
    129f:	48 8d 74 24 20       	lea    0x20(%rsp),%rsi
    12a4:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    12ab:	00 00 
    12ad:	48 89 44 24 58       	mov    %rax,0x58(%rsp)
    12b2:	31 c0                	xor    %eax,%eax
    12b4:	c5 f9 7f 44 24 20    	vmovdqa %xmm0,0x20(%rsp)
    12ba:	c4 e3 e9 22 c1 01    	vpinsrq $0x1,%rcx,%xmm2,%xmm0
    12c0:	c5 f9 7f 44 24 30    	vmovdqa %xmm0,0x30(%rsp)
    12c6:	48 8d 44 24 18       	lea    0x18(%rsp),%rax
    12cb:	31 c9                	xor    %ecx,%ecx
    12cd:	44 89 44 24 48       	mov    %r8d,0x48(%rsp)
    12d2:	48 89 44 24 40       	mov    %rax,0x40(%rsp)
    12d7:	e8 04 fe ff ff       	call   10e0 <GOMP_parallel@plt>
    12dc:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
    12e1:	64 48 2b 04 25 28 00 	sub    %fs:0x28,%rax
    12e8:	00 00 
    12ea:	75 02                	jne    12ee <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id+0x7e>
    12ec:	c9                   	leave
    12ed:	c3                   	ret
    12ee:	e8 dd fd ff ff       	call   10d0 <__stack_chk_fail@plt>
    12f3:	66 90                	xchg   %ax,%ax
    12f5:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
    12fc:	00 00 00 00 

0000000000001300 <__program_gather_load>:
    1300:	f3 0f 1e fa          	endbr64
    1304:	e9 f7 fd ff ff       	jmp    1100 <_Z30__program_gather_load_internalP19gather_load_state_tPdPlS1_id@plt>
    1309:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001310 <__dace_init_gather_load>:
    1310:	f3 0f 1e fa          	endbr64
    1314:	bf 01 00 00 00       	mov    $0x1,%edi
    1319:	e9 92 fd ff ff       	jmp    10b0 <_Znwm@plt>
    131e:	66 90                	xchg   %ax,%ax

0000000000001320 <__dace_exit_gather_load>:
    1320:	f3 0f 1e fa          	endbr64
    1324:	48 85 ff             	test   %rdi,%rdi
    1327:	74 17                	je     1340 <__dace_exit_gather_load+0x20>
    1329:	48 83 ec 08          	sub    $0x8,%rsp
    132d:	be 01 00 00 00       	mov    $0x1,%esi
    1332:	e8 89 fd ff ff       	call   10c0 <_ZdlPvm@plt>
    1337:	31 c0                	xor    %eax,%eax
    1339:	48 83 c4 08          	add    $0x8,%rsp
    133d:	c3                   	ret
    133e:	66 90                	xchg   %ax,%ax
    1340:	31 c0                	xor    %eax,%eax
    1342:	c3                   	ret

Disassembly of section .fini:

0000000000001344 <_fini>:
    1344:	f3 0f 1e fa          	endbr64
    1348:	48 83 ec 08          	sub    $0x8,%rsp
    134c:	48 83 c4 08          	add    $0x8,%rsp
    1350:	c3                   	ret
