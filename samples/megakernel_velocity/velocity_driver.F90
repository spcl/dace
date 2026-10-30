! Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
! Runs ICON velocity_tendencies on the R02B05 dumps through the generated Fortran binding.
!
! The dumps hold the original derived types, so they load into those types; the binding takes them with the kernel's
! own signature and does the AoS to SoA unpacking. The untouched Fortran kernel is linked next to it: both run on the
! same t0 state and each is compared to the t1 dump and to the other.
!
! Usage: velocity_driver <data dir> <timesteps csv> <reps> <kernel: dace|fortran|both>
#define LOAD1(x, name) call header(u, name, lo, hi); allocate (x(lo(1):hi(1))); read (u, *) x
#define LOAD2(x, name) call header(u, name, lo, hi); allocate (x(lo(1):hi(1), lo(2):hi(2))); read (u, *) x
#define LOAD3(x, name) call header(u, name, lo, hi); allocate (x(lo(1):hi(1), lo(2):hi(2), lo(3):hi(3))); read (u, *) x
#define LOAD4(x, name) call header(u, name, lo, hi); allocate (x(lo(1):hi(1), lo(2):hi(2), lo(3):hi(3), lo(4):hi(4))); read (u, *) x
module velocity_dumps
  use mo_model_domain, only: t_patch
  use mo_intp_data_strc, only: t_int_state
  use mo_nonhydro_types, only: t_nh_diag, t_nh_metrics, t_nh_prog
  implicit none
  integer, parameter :: wp = 8, line_length = 128

  !> Everything the kernel writes, as dumped after a step
  type :: t_outputs
    real(wp), allocatable :: vt(:, :, :), vn_ie(:, :, :), w_concorr_c(:, :, :)
    real(wp), allocatable :: ddt_vn_apc_pc(:, :, :, :), ddt_w_adv_pc(:, :, :, :)
    real(wp), allocatable :: z_kin_hor_e(:, :, :), z_vt_ie(:, :, :), z_w_concorr_me(:, :, :)
    real(wp) :: max_vcfl_dyn
  end type t_outputs
contains
  subroutine open_dump(u, root, name, timestep)
    integer, intent(out) :: u
    character(*), intent(in) :: root, name
    integer, intent(in) :: timestep
    character(line_length * 4) :: path
    write (path, '(a,"/",a,".",i0,".data")') root, name, timestep
    open (newunit=u, file=path, status='old', action='read')
  end subroutine open_dump

  !> Moves past the next line that reads ``# <name>``, over section titles and blank lines
  subroutine seek(u, name)
    integer, intent(in) :: u
    character(*), intent(in) :: name
    character(line_length) :: line
    do
      read (u, '(a)') line
      if (trim(line) == '# '//name) return
    end do
  end subroutine seek

  !> Reads the header of field ``name`` (empty: a dump of one bare array), leaving ``u`` at its entries; ``lo``/``hi``
  !> are its bounds
  subroutine header(u, name, lo, hi)
    integer, intent(in) :: u
    character(*), intent(in) :: name
    integer, intent(out) :: lo(4), hi(4)
    character(line_length) :: line
    integer :: rank, flag, extent(4)
    if (name /= '') call seek(u, name)
    read (u, '(a)') line
    if (trim(line) /= '# rank') then  ! Allocatables and pointers say whether they are set first
      read (u, *) flag
      if (flag /= 1) error stop 'unset field in a dump: '//name
      read (u, '(a)') line
      if (trim(line) == '# missing') then
        read (u, *) flag
        call seek(u, 'rank')
      end if
    end if
    read (u, *) rank
    lo = 1
    extent = 1
    call seek(u, 'size')
    read (u, *) extent(1:rank)
    call seek(u, 'lbound')
    read (u, *) lo(1:rank)
    call seek(u, 'entries')
    hi = lo + extent - 1
  end subroutine header

  integer function read_int(root, name, timestep) result(value)
    character(*), intent(in) :: root, name
    integer, intent(in) :: timestep
    integer :: u
    call open_dump(u, root, name, timestep)
    read (u, *) value
    close (u)
  end function read_int

  real(wp) function read_real(root, name, timestep) result(value)
    character(*), intent(in) :: root, name
    integer, intent(in) :: timestep
    integer :: u
    call open_dump(u, root, name, timestep)
    read (u, *) value
    close (u)
  end function read_real

  subroutine read_patch(root, timestep, p)
    character(*), intent(in) :: root
    integer, intent(in) :: timestep
    type(t_patch), intent(out) :: p
    integer :: u, lo(4), hi(4)
    integer, allocatable :: mask(:, :)
    call open_dump(u, root, 'p_patch', timestep)
    call seek(u, 'nblks_c'); read (u, *) p%nblks_c
    call seek(u, 'nblks_e'); read (u, *) p%nblks_e
    call seek(u, 'nblks_v'); read (u, *) p%nblks_v
    p%id = 1
    p%nshift = 0
    LOAD3(p%cells%neighbor_idx, 'neighbor_idx')
    LOAD3(p%cells%neighbor_blk, 'neighbor_blk')
    LOAD3(p%cells%edge_idx, 'edge_idx')
    LOAD3(p%cells%edge_blk, 'edge_blk')
    LOAD2(p%cells%area, 'area')
    LOAD1(p%cells%start_index, 'start_index')
    LOAD1(p%cells%end_index, 'end_index')
    LOAD1(p%cells%start_block, 'start_block')
    LOAD1(p%cells%end_block, 'end_block')
    LOAD2(mask, 'owner_mask')
    allocate (p%cells%decomp_info%owner_mask(lo(1):hi(1), lo(2):hi(2)))
    p%cells%decomp_info%owner_mask = mask /= 0
    LOAD3(p%edges%cell_idx, 'cell_idx')
    LOAD3(p%edges%cell_blk, 'cell_blk')
    LOAD3(p%edges%vertex_idx, 'vertex_idx')
    LOAD3(p%edges%vertex_blk, 'vertex_blk')
    LOAD2(p%edges%tangent_orientation, 'tangent_orientation')
    LOAD3(p%edges%quad_idx, 'quad_idx')
    LOAD3(p%edges%quad_blk, 'quad_blk')
    LOAD2(p%edges%inv_primal_edge_length, 'inv_primal_edge_length')
    LOAD2(p%edges%inv_dual_edge_length, 'inv_dual_edge_length')
    LOAD2(p%edges%area_edge, 'area_edge')
    LOAD2(p%edges%f_e, 'f_e')
    LOAD2(p%edges%fn_e, 'fn_e')
    LOAD2(p%edges%ft_e, 'ft_e')
    LOAD1(p%edges%start_index, 'start_index')
    LOAD1(p%edges%end_index, 'end_index')
    LOAD1(p%edges%start_block, 'start_block')
    LOAD1(p%edges%end_block, 'end_block')
    LOAD3(p%verts%cell_idx, 'cell_idx')
    LOAD3(p%verts%cell_blk, 'cell_blk')
    LOAD3(p%verts%edge_idx, 'edge_idx')
    LOAD3(p%verts%edge_blk, 'edge_blk')
    LOAD1(p%verts%start_index, 'start_index')
    LOAD1(p%verts%end_index, 'end_index')
    LOAD1(p%verts%start_block, 'start_block')
    LOAD1(p%verts%end_block, 'end_block')
    close (u)
  end subroutine read_patch

  subroutine read_interpolation(root, timestep, p)
    character(*), intent(in) :: root
    integer, intent(in) :: timestep
    type(t_int_state), intent(out) :: p
    integer :: u, lo(4), hi(4)
    call open_dump(u, root, 'p_int', timestep)
    LOAD3(p%c_lin_e, 'c_lin_e')
    LOAD3(p%e_bln_c_s, 'e_bln_c_s')
    LOAD3(p%cells_aw_verts, 'cells_aw_verts')
    LOAD3(p%rbf_vec_coeff_e, 'rbf_vec_coeff_e')
    LOAD3(p%geofac_grdiv, 'geofac_grdiv')
    LOAD3(p%geofac_rot, 'geofac_rot')
    LOAD3(p%geofac_n2s, 'geofac_n2s')
    close (u)
  end subroutine read_interpolation

  subroutine read_metrics(root, timestep, p)
    character(*), intent(in) :: root
    integer, intent(in) :: timestep
    type(t_nh_metrics), intent(out) :: p
    integer :: u, lo(4), hi(4)
    call open_dump(u, root, 'p_metrics.t0', timestep)
    LOAD3(p%ddxn_z_full, 'ddxn_z_full')
    LOAD3(p%ddxt_z_full, 'ddxt_z_full')
    LOAD3(p%ddqz_z_full_e, 'ddqz_z_full_e')
    LOAD3(p%ddqz_z_half, 'ddqz_z_half')
    LOAD3(p%wgtfac_c, 'wgtfac_c')
    LOAD3(p%wgtfac_e, 'wgtfac_e')
    LOAD3(p%wgtfacq_e, 'wgtfacq_e')
    LOAD3(p%coeff_gradekin, 'coeff_gradekin')
    LOAD3(p%coeff1_dwdz, 'coeff1_dwdz')
    LOAD3(p%coeff2_dwdz, 'coeff2_dwdz')
    LOAD1(p%deepatmo_gradh_mc, 'deepatmo_gradh_mc')
    LOAD1(p%deepatmo_invr_mc, 'deepatmo_invr_mc')
    LOAD1(p%deepatmo_gradh_ifc, 'deepatmo_gradh_ifc')
    LOAD1(p%deepatmo_invr_ifc, 'deepatmo_invr_ifc')
    close (u)
  end subroutine read_metrics

  subroutine read_prognostics(root, timestep, p)
    character(*), intent(in) :: root
    integer, intent(in) :: timestep
    type(t_nh_prog), intent(out) :: p
    integer :: u, lo(4), hi(4)
    call open_dump(u, root, 'p_prog.t0', timestep)
    LOAD3(p%w, 'w')
    LOAD3(p%vn, 'vn')
    close (u)
  end subroutine read_prognostics

  !> Reads the diagnostics and the ``z_`` fields of ``stamp`` (``t0`` or ``t1``)
  subroutine read_outputs(root, stamp, timestep, o)
    character(*), intent(in) :: root, stamp
    integer, intent(in) :: timestep
    type(t_outputs), intent(out) :: o
    integer :: u, lo(4), hi(4)
    call open_dump(u, root, 'p_diag.'//stamp, timestep)
    LOAD3(o%vt, 'vt')
    LOAD3(o%vn_ie, 'vn_ie')
    LOAD3(o%w_concorr_c, 'w_concorr_c')
    LOAD4(o%ddt_vn_apc_pc, 'ddt_vn_apc_pc')
    LOAD4(o%ddt_w_adv_pc, 'ddt_w_adv_pc')
    call seek(u, 'max_vcfl_dyn'); read (u, *) o%max_vcfl_dyn
    close (u)
    call open_dump(u, root, 'z_kin_hor_e.'//stamp, timestep)
    LOAD3(o%z_kin_hor_e, '')
    close (u)
    call open_dump(u, root, 'z_vt_ie.'//stamp, timestep)
    LOAD3(o%z_vt_ie, '')
    close (u)
    call open_dump(u, root, 'z_w_concorr_me.'//stamp, timestep)
    LOAD3(o%z_w_concorr_me, '')
    close (u)
  end subroutine read_outputs

  !> Points ``p_diag`` at the buffers of ``o``: the kernel fills them, and ``o`` is what a check reads
  subroutine bind_diagnostics(o, p_diag)
    type(t_outputs), target, intent(inout) :: o
    type(t_nh_diag), intent(inout) :: p_diag
    p_diag%vt => o%vt
    p_diag%vn_ie => o%vn_ie
    p_diag%w_concorr_c => o%w_concorr_c
    p_diag%ddt_vn_apc_pc => o%ddt_vn_apc_pc
    p_diag%ddt_w_adv_pc => o%ddt_w_adv_pc
    p_diag%max_vcfl_dyn = o%max_vcfl_dyn
  end subroutine bind_diagnostics

  subroutine report(label, name, got, want)
    character(*), intent(in) :: label, name
    real(wp), intent(in) :: got(:), want(:)
    real(wp) :: scale, diff
    diff = maxval(abs(got - want))
    scale = max(maxval(abs(want)), tiny(scale))
    print '("DIFF ",a," ",a," max_abs=",es10.3," max_rel=",es10.3," n_differ=",i0)', &
      label, name, diff, diff / scale, count(got /= want)
  end subroutine report

  subroutine compare(label, got, want)
    character(*), intent(in) :: label
    type(t_outputs), intent(in) :: got, want
    call report(label, 'vt', reshape(got%vt, [size(got%vt)]), reshape(want%vt, [size(want%vt)]))
    call report(label, 'vn_ie', reshape(got%vn_ie, [size(got%vn_ie)]), reshape(want%vn_ie, [size(want%vn_ie)]))
    call report(label, 'w_concorr_c', reshape(got%w_concorr_c, [size(got%w_concorr_c)]), &
                reshape(want%w_concorr_c, [size(want%w_concorr_c)]))
    call report(label, 'ddt_vn_apc_pc', reshape(got%ddt_vn_apc_pc, [size(got%ddt_vn_apc_pc)]), &
                reshape(want%ddt_vn_apc_pc, [size(want%ddt_vn_apc_pc)]))
    call report(label, 'ddt_w_adv_pc', reshape(got%ddt_w_adv_pc, [size(got%ddt_w_adv_pc)]), &
                reshape(want%ddt_w_adv_pc, [size(want%ddt_w_adv_pc)]))
    call report(label, 'z_kin_hor_e', reshape(got%z_kin_hor_e, [size(got%z_kin_hor_e)]), &
                reshape(want%z_kin_hor_e, [size(want%z_kin_hor_e)]))
    call report(label, 'z_vt_ie', reshape(got%z_vt_ie, [size(got%z_vt_ie)]), reshape(want%z_vt_ie, [size(want%z_vt_ie)]))
    call report(label, 'z_w_concorr_me', reshape(got%z_w_concorr_me, [size(got%z_w_concorr_me)]), &
                reshape(want%z_w_concorr_me, [size(want%z_w_concorr_me)]))
    call report(label, 'max_vcfl_dyn', [got%max_vcfl_dyn], [want%max_vcfl_dyn])
  end subroutine compare
end module velocity_dumps

program velocity_driver
  use mo_init_vgrid, only: nflatlev
  use mo_model_domain, only: t_patch
  use mo_intp_data_strc, only: t_int_state
  use mo_nonhydro_types, only: t_nh_diag, t_nh_metrics, t_nh_prog
  use mo_nonhydrostatic_config, only: lextra_diffu
  use mo_parallel_config, only: nproma
  use mo_run_config, only: lvert_nest, timers_level
  use mo_velocity_advection, only: velocity_tendencies
  use mo_vertical_grid, only: nrdmax
  use velocity_dumps
#ifndef FORTRAN_ONLY
  use velocity_tendencies_dace_bindings, only: velocity_tendencies_dace, velocity_tendencies_dace_finalize
#endif
  implicit none
  character(1024) :: data_dir, argument, kernel
  integer :: nargs, nreps, n_timesteps, step
  integer, allocatable :: timesteps(:)

  call get_command_argument(1, data_dir)
  call get_command_argument(2, argument)
  nargs = 1 + count([(argument(step:step) == ',', step=1, len_trim(argument))])
  allocate (timesteps(nargs))
  read (argument, *) timesteps
  n_timesteps = nargs
  call get_command_argument(3, argument)
  read (argument, *) nreps
  call get_command_argument(4, kernel)

  do step = 1, n_timesteps
    call run_timestep(trim(data_dir), timesteps(step), nreps, trim(kernel))
  end do
#ifndef FORTRAN_ONLY
  call velocity_tendencies_dace_finalize()
#endif

contains
  !> The configuration the dumps of ``timestep`` read from ICON's namelists and modules
  subroutine read_module_state(root, timestep)
    character(*), intent(in) :: root
    integer, intent(in) :: timestep
    integer :: u, lo(4), hi(4), value
    integer, allocatable :: levels(:)
    call open_dump(u, root, 'global_data.t0', timestep)
    call header(u, 'nflatlev', lo, hi)
    allocate (levels(lo(1):hi(1)))
    read (u, *) levels
    nflatlev = levels(1:size(nflatlev))
    call seek(u, 'lextra_diffu'); read (u, *) value
    lextra_diffu = value /= 0
    call seek(u, 'nproma'); read (u, *) nproma
    call header(u, 'nrdmax', lo, hi)
    read (u, *) levels
    nrdmax = levels(1:size(nrdmax))
    close (u)
    timers_level = 0  ! The dumps say 10: the kernel would start ICON's timers, which this build stubs out
    lvert_nest = .false.
  end subroutine read_module_state

  !> Runs ``kernel`` from the t0 state of ``timestep``, ``nreps`` times, and compares what the last run wrote
  subroutine run_one(label, is_dace, t0, want, p_prog, p_patch, p_int, p_metrics, p_diag, got, ntnd, istep, lvn_only, &
                     dtime, dt_linintp_ubc, ldeepatmo, timestep, nreps)
    character(*), intent(in) :: label
    logical, intent(in) :: is_dace, lvn_only, ldeepatmo
    type(t_outputs), intent(in) :: t0, want
    type(t_outputs), target, intent(inout) :: got
    type(t_nh_prog), intent(inout) :: p_prog
    type(t_patch), intent(in) :: p_patch
    type(t_int_state), intent(in) :: p_int
    type(t_nh_metrics), intent(inout) :: p_metrics
    type(t_nh_diag), intent(inout) :: p_diag
    integer, intent(in) :: ntnd, istep, timestep, nreps
    real(wp), intent(in) :: dtime, dt_linintp_ubc
    integer :: rep
    integer(8) :: start, stop, rate

    do rep = 1, nreps
      call restore(t0, got)
      p_diag%max_vcfl_dyn = t0%max_vcfl_dyn
      call system_clock(start, rate)
      if (is_dace) then
#ifndef FORTRAN_ONLY
        call velocity_tendencies_dace(p_prog, p_patch, p_int, p_metrics, p_diag, got%z_w_concorr_me, got%z_kin_hor_e, &
                                      got%z_vt_ie, ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
#endif
      else
        call velocity_tendencies(p_prog, p_patch, p_int, p_metrics, p_diag, got%z_w_concorr_me, got%z_kin_hor_e, &
                                 got%z_vt_ie, ntnd, istep, lvn_only, dtime, dt_linintp_ubc, ldeepatmo)
      end if
      call system_clock(stop)
      print '("TIME ",a," timestep=",i0," rep=",i0," seconds=",f12.6)', label, timestep, rep, real(stop - start, wp) / rate
    end do
    got%max_vcfl_dyn = p_diag%max_vcfl_dyn
    call compare(label//'_vs_t1', got, want)
  end subroutine run_one

  subroutine restore(from, to)
    type(t_outputs), intent(in) :: from
    type(t_outputs), intent(inout) :: to
    to%vt = from%vt
    to%vn_ie = from%vn_ie
    to%w_concorr_c = from%w_concorr_c
    to%ddt_vn_apc_pc = from%ddt_vn_apc_pc
    to%ddt_w_adv_pc = from%ddt_w_adv_pc
    to%z_kin_hor_e = from%z_kin_hor_e
    to%z_vt_ie = from%z_vt_ie
    to%z_w_concorr_me = from%z_w_concorr_me
  end subroutine restore

  subroutine run_timestep(root, timestep, nreps, kernel)
    character(*), intent(in) :: root, kernel
    integer, intent(in) :: timestep, nreps
    type(t_patch), target :: p_patch
    type(t_int_state), target :: p_int
    type(t_nh_metrics) :: p_metrics
    type(t_nh_prog) :: p_prog
    type(t_nh_diag) :: p_diag
    type(t_outputs), target :: t0, want, got, reference
    integer :: ntnd, istep, nlev, nblks_e
    logical :: lvn_only, ldeepatmo
    real(wp) :: dtime, dt_linintp_ubc

    call read_module_state(root, timestep)
    istep = read_int(root, 'istep', timestep)
    ntnd = read_int(root, 'ntnd', timestep)
    lvn_only = read_int(root, 'lvn_only', timestep) /= 0
    ldeepatmo = read_int(root, 'ldeepatmo', timestep) /= 0
    dtime = read_real(root, 'dtime', timestep)
    dt_linintp_ubc = read_real(root, 'dt_linintp_ubc', timestep)
    call read_patch(root, timestep, p_patch)
    call read_interpolation(root, timestep, p_int)
    call read_metrics(root, timestep, p_metrics)
    call read_prognostics(root, timestep, p_prog)
    call read_outputs(root, 't0', timestep, t0)
    call read_outputs(root, 't1', timestep, want)
    nlev = size(p_prog%w, 2) - 1
    p_patch%nlev = nlev
    p_patch%nlevp1 = nlev + 1
    nblks_e = p_patch%nblks_e
    ! Fields the kernel's signature holds but this configuration never touches, so the dumps left them out
    allocate (p_diag%vn_ie_ubc(nproma, 2, nblks_e), p_diag%ddt_vn_cor_pc(nproma, nlev, nblks_e, 3))
    p_diag%vn_ie_ubc = 0
    p_diag%ddt_vn_cor_pc = 0
    print '("STEP timestep=",i0," istep=",i0," lvn_only=",l1," ntnd=",i0," nlev=",i0," nproma=",i0)', &
      timestep, istep, lvn_only, ntnd, nlev, nproma

    if (kernel == 'fortran' .or. kernel == 'both') then
      reference = t0
      call bind_diagnostics(reference, p_diag)
      call run_one('fortran', .false., t0, want, p_prog, p_patch, p_int, p_metrics, p_diag, reference, ntnd, istep, &
                   lvn_only, dtime, dt_linintp_ubc, ldeepatmo, timestep, nreps)
    end if
    if (kernel == 'dace' .or. kernel == 'both') then
      got = t0
      call bind_diagnostics(got, p_diag)
      call run_one('dace', .true., t0, want, p_prog, p_patch, p_int, p_metrics, p_diag, got, ntnd, istep, &
                   lvn_only, dtime, dt_linintp_ubc, ldeepatmo, timestep, nreps)
    end if
    if (kernel == 'both') call compare('dace_vs_fortran', got, reference)
  end subroutine run_timestep
end program velocity_driver
