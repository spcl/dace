# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The MPI thread level ScaLAPACK runs under: asked for when it is ours to ask, checked when it is not."""

#: C++ for the head of a ScaLAPACK environment's ``init_code``, before the BLACS grid is built.
#:
#: BLACS brings MPI up itself when nobody else has, and it does so with a bare ``MPI_Init`` -- which
#: promises ``MPI_THREAD_SINGLE``, that the process has exactly one thread. No DaCe process keeps
#: that promise: its maps are OpenMP regions and its BLAS calls spawn a pool of their own. So bring
#: MPI up here at a level this process can keep, and when someone else got there first and asked
#: for less, refuse to run rather than proceed under a promise nothing honours.
#:
#: ``MPI_THREAD_FUNNELED`` is the floor because it is what this code actually does: PBLAS is called
#: from one thread, and the other threads in the process never touch MPI. It is NOT a fix for
#: wrong numbers -- the pgemm miscompare on the heterogeneous runner happened with MPI at
#: ``MPI_THREAD_MULTIPLE`` (measured), and its trigger was an OpenMP runtime dlopened into the
#: interpreter at collection time, not the thread level. This guard turns one silent illegal
#: configuration into a loud one; it does not claim to be the only way to get a wrong product.
MPI_THREAD_LEVEL_GUARD = """
    int __scalapack_thread_level = MPI_THREAD_SINGLE;
    int __scalapack_mpi_up = 0;
    MPI_Initialized(&__scalapack_mpi_up);
    if (!__scalapack_mpi_up) {
        MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &__scalapack_thread_level);
    }
    MPI_Query_thread(&__scalapack_thread_level);
    if (__scalapack_thread_level < MPI_THREAD_FUNNELED) {
        fprintf(stderr,
                "ScaLAPACK was entered with MPI at thread level %d, below MPI_THREAD_FUNNELED "
                "(%d). This process runs OpenMP maps and a threaded BLAS, so that level is a "
                "promise it cannot keep, and PBLAS answers it with wrong numbers rather than an "
                "error. Bring MPI up with MPI_Init_thread, or import mpi4py before dace so its "
                "own bring-up runs, instead of a bare MPI_Init.\\n",
                __scalapack_thread_level, MPI_THREAD_FUNNELED);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
"""
