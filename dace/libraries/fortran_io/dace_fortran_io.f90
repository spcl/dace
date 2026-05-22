! Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
!
! iso_c_binding wrappers around Fortran external-file I/O, shipped with the
! ``dace.libraries.fortran_io`` library.  The library nodes lower a Fortran
! READ / WRITE statement to a C++ tasklet that calls these ``bind(c)`` entry
! points, so the real Fortran runtime performs the transfer (exact
! list-directed semantics) while the call interface stays the standardized
! C-interop ABI.  Each ``open`` returns a fresh ``newunit`` the matching
! read/write/close calls reuse, so concurrent I/O nodes never clash on a unit.
module dace_fortran_io
   use, intrinsic :: iso_c_binding
   implicit none

contains

   !> Open ``path`` (``path_len`` C chars) and return the allocated unit.
   !> ``for_write`` /= 0 opens a fresh formatted file for writing
   !> (status ``replace``); otherwise an existing file for reading.
   integer(c_int) function dace_fio_open(path, path_len, for_write) result(unit) &
         bind(c, name="dace_fio_open")
      character(kind=c_char), intent(in) :: path(*)
      integer(c_int), value :: path_len
      integer(c_int), value :: for_write
      character(len=path_len) :: fname
      integer :: i, u
      do i = 1, path_len
         fname(i:i) = path(i)
      end do
      if (for_write /= 0_c_int) then
         open (newunit=u, file=fname, status='replace', action='write', form='formatted')
      else
         open (newunit=u, file=fname, status='old', action='read', form='formatted')
      end if
      unit = int(u, c_int)
   end function dace_fio_open

   subroutine dace_fio_close(unit) bind(c, name="dace_fio_close")
      integer(c_int), value :: unit
      close (int(unit))
   end subroutine dace_fio_close

   ! --- list-directed reads: one scalar / one whole array per call ----------

   subroutine dace_fio_read_f64(unit, x) bind(c, name="dace_fio_read_f64")
      integer(c_int), value :: unit
      real(c_double), intent(out) :: x
      read (int(unit), *) x
   end subroutine dace_fio_read_f64

   subroutine dace_fio_read_f64_arr(unit, x, n) bind(c, name="dace_fio_read_f64_arr")
      integer(c_int), value :: unit, n
      real(c_double), intent(out) :: x(n)
      read (int(unit), *) x
   end subroutine dace_fio_read_f64_arr

   subroutine dace_fio_read_f32(unit, x) bind(c, name="dace_fio_read_f32")
      integer(c_int), value :: unit
      real(c_float), intent(out) :: x
      read (int(unit), *) x
   end subroutine dace_fio_read_f32

   subroutine dace_fio_read_f32_arr(unit, x, n) bind(c, name="dace_fio_read_f32_arr")
      integer(c_int), value :: unit, n
      real(c_float), intent(out) :: x(n)
      read (int(unit), *) x
   end subroutine dace_fio_read_f32_arr

   subroutine dace_fio_read_i32(unit, x) bind(c, name="dace_fio_read_i32")
      integer(c_int), value :: unit
      integer(c_int), intent(out) :: x
      read (int(unit), *) x
   end subroutine dace_fio_read_i32

   subroutine dace_fio_read_i32_arr(unit, x, n) bind(c, name="dace_fio_read_i32_arr")
      integer(c_int), value :: unit, n
      integer(c_int), intent(out) :: x(n)
      read (int(unit), *) x
   end subroutine dace_fio_read_i32_arr

   subroutine dace_fio_read_i64(unit, x) bind(c, name="dace_fio_read_i64")
      integer(c_int), value :: unit
      integer(c_int64_t), intent(out) :: x
      read (int(unit), *) x
   end subroutine dace_fio_read_i64

   subroutine dace_fio_read_i64_arr(unit, x, n) bind(c, name="dace_fio_read_i64_arr")
      integer(c_int), value :: unit, n
      integer(c_int64_t), intent(out) :: x(n)
      read (int(unit), *) x
   end subroutine dace_fio_read_i64_arr

   ! --- list-directed writes ------------------------------------------------

   subroutine dace_fio_write_f64(unit, x) bind(c, name="dace_fio_write_f64")
      integer(c_int), value :: unit
      real(c_double), intent(in) :: x
      write (int(unit), *) x
   end subroutine dace_fio_write_f64

   subroutine dace_fio_write_f64_arr(unit, x, n) bind(c, name="dace_fio_write_f64_arr")
      integer(c_int), value :: unit, n
      real(c_double), intent(in) :: x(n)
      write (int(unit), *) x
   end subroutine dace_fio_write_f64_arr

   subroutine dace_fio_write_f32(unit, x) bind(c, name="dace_fio_write_f32")
      integer(c_int), value :: unit
      real(c_float), intent(in) :: x
      write (int(unit), *) x
   end subroutine dace_fio_write_f32

   subroutine dace_fio_write_f32_arr(unit, x, n) bind(c, name="dace_fio_write_f32_arr")
      integer(c_int), value :: unit, n
      real(c_float), intent(in) :: x(n)
      write (int(unit), *) x
   end subroutine dace_fio_write_f32_arr

   subroutine dace_fio_write_i32(unit, x) bind(c, name="dace_fio_write_i32")
      integer(c_int), value :: unit
      integer(c_int), intent(in) :: x
      write (int(unit), *) x
   end subroutine dace_fio_write_i32

   subroutine dace_fio_write_i32_arr(unit, x, n) bind(c, name="dace_fio_write_i32_arr")
      integer(c_int), value :: unit, n
      integer(c_int), intent(in) :: x(n)
      write (int(unit), *) x
   end subroutine dace_fio_write_i32_arr

   subroutine dace_fio_write_i64(unit, x) bind(c, name="dace_fio_write_i64")
      integer(c_int), value :: unit
      integer(c_int64_t), intent(in) :: x
      write (int(unit), *) x
   end subroutine dace_fio_write_i64

   subroutine dace_fio_write_i64_arr(unit, x, n) bind(c, name="dace_fio_write_i64_arr")
      integer(c_int), value :: unit, n
      integer(c_int64_t), intent(in) :: x(n)
      write (int(unit), *) x
   end subroutine dace_fio_write_i64_arr

end module dace_fortran_io
