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

   ! Generic namelist support (see the namelist section below): a small pool of
   ! slots, each holding one opened group's normalised text.  ``dace_nml_open``
   ! claims a slot; the typed ``dace_nml_get_*`` calls read members out of it;
   ! ``dace_nml_close`` releases it.  The pool size caps concurrently-open
   ! groups (one per live ``NamelistRead`` tasklet); the buffer caps a group's
   ! text length.
   integer, parameter :: DACE_NML_SLOTS = 8
   integer, parameter :: DACE_NML_MAXLEN = 65536
   character(len=DACE_NML_MAXLEN), save :: dace_nml_text(DACE_NML_SLOTS)
   logical, save :: dace_nml_busy(DACE_NML_SLOTS) = .false.

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

   ! --- generic namelist support (called from pure C++ tasklets) -----------
   !
   ! A fixed Fortran NAMELIST read needs its group declared at compile time, so
   ! a single shipped routine cannot serve an arbitrary caller's group.  The
   ! namelist *input* grammar is standardized, though, so these routines parse
   ! the ``&group name=value /`` text directly: the group's body is lowercased
   ! and comma-normalised once at open, then each typed getter locates its
   ! member by name and consumes the value(s) with a list-directed internal
   ! read (which stops after the requested count, ignoring later members).

   !> Lowercase ``s`` in place (NAMELIST names are case-insensitive).
   subroutine dace_nml_lower(s)
      character(len=*), intent(inout) :: s
      integer :: i, code
      do i = 1, len(s)
         code = iachar(s(i:i))
         if (code >= iachar('A') .and. code <= iachar('Z')) s(i:i) = achar(code + 32)
      end do
   end subroutine dace_nml_lower

   !> Position in ``buf`` just past the ``=`` of the assignment to ``name``
   !> (a whole token followed, after optional blanks, by ``=``), or 0 if absent.
   integer function dace_nml_find(buf, name) result(p)
      character(len=*), intent(in) :: buf, name
      integer :: i, j, n, blen
      n = len_trim(name)
      blen = len_trim(buf)
      p = 0
      do i = 1, blen - n + 1
         if (i > 1) then
            if (buf(i - 1:i - 1) /= ' ') cycle
         end if
         if (buf(i:i + n - 1) /= name(1:n)) cycle
         j = i + n
         do while (j <= blen)
            if (buf(j:j) /= ' ') exit
            j = j + 1
         end do
         if (j <= blen .and. buf(j:j) == '=') then
            p = j + 1
            return
         end if
      end do
   end function dace_nml_find

   !> Open ``path`` and stash the ``&group ... /`` body (lowercased, commas ->
   !> spaces) in a free slot; return its 0-based handle, or -1 if none free.
   integer(c_int) function dace_nml_open(path, path_len, group, group_len) result(handle) &
         bind(c, name="dace_nml_open")
      character(kind=c_char), intent(in) :: path(*), group(*)
      integer(c_int), value :: path_len, group_len
      character(len=path_len) :: fname
      character(len=group_len) :: gname
      character(len=DACE_NML_MAXLEN) :: line
      integer :: u, i, slot, ios, gpos, term, start
      logical :: in_group

      slot = 0
      do i = 1, DACE_NML_SLOTS
         if (.not. dace_nml_busy(i)) then
            slot = i
            exit
         end if
      end do
      if (slot == 0) then
         handle = -1_c_int
         return
      end if

      do i = 1, path_len
         fname(i:i) = path(i)
      end do
      do i = 1, group_len
         gname(i:i) = group(i)
      end do
      call dace_nml_lower(gname)

      dace_nml_text(slot) = ''
      in_group = .false.
      open (newunit=u, file=fname, status='old', action='read')
      do
         read (u, '(A)', iostat=ios) line
         if (ios /= 0) exit
         call dace_nml_lower(line)
         if (.not. in_group) then
            gpos = index(line, '&'//gname(1:group_len))
            if (gpos == 0) cycle
            in_group = .true.
            start = gpos + 1 + group_len
            if (start <= len(line)) then
               line = line(start:)
            else
               line = ''
            end if
         end if
         term = index(line, '/')
         if (term > 0) then
            dace_nml_text(slot) = trim(dace_nml_text(slot))//' '//line(1:term - 1)
            exit
         end if
         dace_nml_text(slot) = trim(dace_nml_text(slot))//' '//trim(line)
      end do
      close (u)

      do i = 1, len_trim(dace_nml_text(slot))
         if (dace_nml_text(slot) (i:i) == ',') dace_nml_text(slot) (i:i) = ' '
      end do

      dace_nml_busy(slot) = .true.
      handle = int(slot - 1, c_int)
   end function dace_nml_open

   subroutine dace_nml_close(handle) bind(c, name="dace_nml_close")
      integer(c_int), value :: handle
      if (handle >= 0_c_int .and. handle < int(DACE_NML_SLOTS, c_int)) dace_nml_busy(handle + 1) = .false.
   end subroutine dace_nml_close

   subroutine dace_nml_get_f64(handle, name, name_len, x) bind(c, name="dace_nml_get_f64")
      integer(c_int), value :: handle, name_len
      character(kind=c_char), intent(in) :: name(*)
      real(c_double), intent(inout) :: x
      character(len=name_len) :: nm
      integer :: i, p
      do i = 1, name_len
         nm(i:i) = name(i)
      end do
      call dace_nml_lower(nm)
      p = dace_nml_find(dace_nml_text(handle + 1), nm)
      if (p > 0) read (dace_nml_text(handle + 1) (p:), *) x
   end subroutine dace_nml_get_f64

   subroutine dace_nml_get_f64_arr(handle, name, name_len, x, n) bind(c, name="dace_nml_get_f64_arr")
      integer(c_int), value :: handle, name_len, n
      character(kind=c_char), intent(in) :: name(*)
      real(c_double), intent(inout) :: x(n)
      character(len=name_len) :: nm
      integer :: i, p
      do i = 1, name_len
         nm(i:i) = name(i)
      end do
      call dace_nml_lower(nm)
      p = dace_nml_find(dace_nml_text(handle + 1), nm)
      if (p > 0) read (dace_nml_text(handle + 1) (p:), *) x(1:n)
   end subroutine dace_nml_get_f64_arr

   subroutine dace_nml_get_f32(handle, name, name_len, x) bind(c, name="dace_nml_get_f32")
      integer(c_int), value :: handle, name_len
      character(kind=c_char), intent(in) :: name(*)
      real(c_float), intent(inout) :: x
      character(len=name_len) :: nm
      integer :: i, p
      do i = 1, name_len
         nm(i:i) = name(i)
      end do
      call dace_nml_lower(nm)
      p = dace_nml_find(dace_nml_text(handle + 1), nm)
      if (p > 0) read (dace_nml_text(handle + 1) (p:), *) x
   end subroutine dace_nml_get_f32

   subroutine dace_nml_get_f32_arr(handle, name, name_len, x, n) bind(c, name="dace_nml_get_f32_arr")
      integer(c_int), value :: handle, name_len, n
      character(kind=c_char), intent(in) :: name(*)
      real(c_float), intent(inout) :: x(n)
      character(len=name_len) :: nm
      integer :: i, p
      do i = 1, name_len
         nm(i:i) = name(i)
      end do
      call dace_nml_lower(nm)
      p = dace_nml_find(dace_nml_text(handle + 1), nm)
      if (p > 0) read (dace_nml_text(handle + 1) (p:), *) x(1:n)
   end subroutine dace_nml_get_f32_arr

   subroutine dace_nml_get_i32(handle, name, name_len, x) bind(c, name="dace_nml_get_i32")
      integer(c_int), value :: handle, name_len
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), intent(inout) :: x
      character(len=name_len) :: nm
      integer :: i, p
      do i = 1, name_len
         nm(i:i) = name(i)
      end do
      call dace_nml_lower(nm)
      p = dace_nml_find(dace_nml_text(handle + 1), nm)
      if (p > 0) read (dace_nml_text(handle + 1) (p:), *) x
   end subroutine dace_nml_get_i32

   subroutine dace_nml_get_i32_arr(handle, name, name_len, x, n) bind(c, name="dace_nml_get_i32_arr")
      integer(c_int), value :: handle, name_len, n
      character(kind=c_char), intent(in) :: name(*)
      integer(c_int), intent(inout) :: x(n)
      character(len=name_len) :: nm
      integer :: i, p
      do i = 1, name_len
         nm(i:i) = name(i)
      end do
      call dace_nml_lower(nm)
      p = dace_nml_find(dace_nml_text(handle + 1), nm)
      if (p > 0) read (dace_nml_text(handle + 1) (p:), *) x(1:n)
   end subroutine dace_nml_get_i32_arr

end module dace_fortran_io
