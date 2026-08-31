! The engine behind the BLAS compatible layer.
!
! Two families live here. The `_d` procedures take device pointers and leave
! residency to the caller. The `_host_ptr` procedures take host addresses,
! allocate device buffers, move the data, run the same device path and move the
! result back; they are what the standard mangled names in ckl_blas.f90 call,
! because that is what an existing Fortran program passes.
!
! Everything is column major, which is what Fortran and the reference BLAS both
! are, so the calls go straight to the CKL column major path with each operand
! keeping its own op flag. No transposes are synthesised anywhere in this file.
!
! SGEMV is a GEMM with n = 1. That is not a shortcut, it is the same arithmetic:
! y = alpha * op(A) * x + beta * y is exactly C = alpha * op(A) * B + beta * C
! with B and C single columns. Non unit increments are packed on the host first,
! which is O(len) work outside the multiply.
!
! STRSM is blocked. The trailing updates, which are the bulk of the work, are
! GEMMs and run on the device. The diagonal blocks are small, inherently
! sequential and are solved on the host, so each block step pulls a bs by bs
! corner of A and the matching panel of B down and pushes the solved panel back.
! That is the hybrid split, and docs/fortran.md says so.

module ckl_blas_core
    use, intrinsic :: iso_c_binding
    use, intrinsic :: iso_fortran_env, only: error_unit
    use ckl
    implicit none
    private

    public :: ckl_blas_handle, ckl_blas_shutdown, ckl_blas_abort, ckl_blas_op_code
    public :: ckl_sgemm_d, ckl_sgemv_d, ckl_strsm_d
    public :: ckl_sgemm_host_ptr, ckl_sgemv_host_ptr, ckl_strsm_host_ptr

    ! Diagonal block edge for the blocked triangular solve. 64 keeps the host
    ! side solve small while leaving the trailing update large enough to be worth
    ! a device launch.
    integer, parameter :: TRSM_BLOCK = 64

    ! One process wide handle, created on first use. The standard BLAS names
    ! carry no handle argument, so there is nowhere else to put it. This is not
    ! thread safe; a threaded caller should use the `_d` entry points with its
    ! own handle.
    type(c_ptr), save :: g_handle = c_null_ptr

contains

    ! -------------------------------------------------------------------------
    ! Handle and error reporting
    ! -------------------------------------------------------------------------

    function ckl_blas_handle() result(h)
        type(c_ptr) :: h
        integer(c_int) :: rc

        if (.not. c_associated(g_handle)) then
            rc = ckl_create(g_handle)
            if (rc /= CKL_STATUS_SUCCESS) then
                write (error_unit, '(a)') 'CKL BLAS: ckl_create failed: '//ckl_error_text(rc)
                error stop 1
            end if
        end if
        h = g_handle
    end function ckl_blas_handle

    subroutine ckl_blas_shutdown()
        integer(c_int) :: rc

        if (c_associated(g_handle)) then
            rc = ckl_destroy(g_handle)
            g_handle = c_null_ptr
        end if
    end subroutine ckl_blas_shutdown

    ! The reference BLAS reports a bad argument through XERBLA, which prints and
    ! stops. A BLAS signature has no status to return, so a CKL failure takes the
    ! same exit: loud, and never a wrong answer travelling on.
    subroutine ckl_blas_abort(routine, info)
        character(len=*), intent(in) :: routine
        integer, intent(in) :: info
        character(len=16) :: num

        write (num, '(i0)') info
        write (error_unit, '(a)') ' ** On entry to '//trim(routine)// &
            ' parameter number '//trim(num)//' had an illegal value'
        error stop 1
    end subroutine ckl_blas_abort

    subroutine ckl_blas_require(routine, stat)
        character(len=*), intent(in) :: routine
        integer(c_int), intent(in) :: stat

        if (stat == CKL_STATUS_SUCCESS) return
        write (error_unit, '(a)') ' ** '//trim(routine)//' failed inside CKL: '// &
            ckl_error_text(stat)
        error stop 1
    end subroutine ckl_blas_require

    ! 'N'/'n' -> CKL_OP_N, 'T'/'t' -> CKL_OP_T, 'C'/'c' -> CKL_OP_C. ok is false
    ! for anything else, which is an XERBLA case at the call site.
    subroutine ckl_blas_op_code(ch, op, ok)
        character(kind=c_char), intent(in) :: ch
        integer(c_int), intent(out) :: op
        logical, intent(out) :: ok

        ok = .true.
        select case (ch)
        case ('N', 'n')
            op = CKL_OP_N
        case ('T', 't')
            op = CKL_OP_T
        case ('C', 'c')
            op = CKL_OP_C
        case default
            op = CKL_OP_N
            ok = .false.
        end select
    end subroutine ckl_blas_op_code

    ! -------------------------------------------------------------------------
    ! Strided device block transfers
    ! -------------------------------------------------------------------------

    ! Pulls an nrows by ncols column major block, device leading dimension ld,
    ! into a packed host array. dptr already points at the block origin.
    subroutine block_to_host(host, dptr, ld, nrows, ncols, stat)
        real(c_float), intent(out), target, contiguous :: host(:, :)
        type(c_ptr), intent(in) :: dptr
        integer, intent(in) :: ld, nrows, ncols
        integer(c_int), intent(out) :: stat
        integer :: j

        stat = CKL_STATUS_SUCCESS
        do j = 1, ncols
            stat = ckl_memcpy(c_loc(host(1, j)), &
                              ckl_ptr_offset(dptr, int(j - 1, c_int64_t)*int(ld, c_int64_t)), &
                              int(nrows, c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_D2H)
            if (stat /= CKL_STATUS_SUCCESS) return
        end do
    end subroutine block_to_host

    subroutine block_to_device(dptr, host, ld, nrows, ncols, stat)
        type(c_ptr), intent(in) :: dptr
        real(c_float), intent(in), target, contiguous :: host(:, :)
        integer, intent(in) :: ld, nrows, ncols
        integer(c_int), intent(out) :: stat
        integer :: j

        stat = CKL_STATUS_SUCCESS
        do j = 1, ncols
            stat = ckl_memcpy(ckl_ptr_offset(dptr, int(j - 1, c_int64_t)*int(ld, c_int64_t)), &
                              c_loc(host(1, j)), &
                              int(nrows, c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_H2D)
            if (stat /= CKL_STATUS_SUCCESS) return
        end do
    end subroutine block_to_device

    ! -------------------------------------------------------------------------
    ! Device pointer entry points
    ! -------------------------------------------------------------------------

    ! C := alpha * op(A) * op(B) + beta * C, everything already resident.
    subroutine ckl_sgemm_d(h, opa, opb, m, n, k, alpha, da, lda, db, ldb, beta, dc, ldc, info)
        type(c_ptr), intent(in) :: h
        integer(c_int), intent(in) :: opa, opb
        integer, intent(in) :: m, n, k, lda, ldb, ldc
        real(c_float), intent(in) :: alpha, beta
        type(c_ptr), intent(in) :: da, db, dc
        integer(c_int), intent(out) :: info

        info = ckl_sgemm(h, CKL_COL_MAJOR, opa, opb, &
                         int(m, c_int64_t), int(n, c_int64_t), int(k, c_int64_t), alpha, &
                         da, int(lda, c_int64_t), db, int(ldb, c_int64_t), beta, &
                         dc, int(ldc, c_int64_t))
        if (info /= CKL_STATUS_SUCCESS) return
        info = ckl_device_synchronize()
    end subroutine ckl_sgemm_d

    ! y := alpha * op(A) * x + beta * y with A m by n, everything resident. Unit
    ! increments only: packing a strided vector needs a host round trip, and
    ! doing that silently under a device pointer name is the fallback ground rule
    ! 6 bans, so a strided call returns CKL_STATUS_NOT_SUPPORTED instead.
    subroutine ckl_sgemv_d(h, opa, m, n, alpha, da, lda, dx, incx, beta, dy, incy, info)
        type(c_ptr), intent(in) :: h
        integer(c_int), intent(in) :: opa
        integer, intent(in) :: m, n, lda, incx, incy
        real(c_float), intent(in) :: alpha, beta
        type(c_ptr), intent(in) :: da, dx, dy
        integer(c_int), intent(out) :: info
        integer :: leny, lenx

        if (incx /= 1 .or. incy /= 1) then
            info = CKL_STATUS_NOT_SUPPORTED
            return
        end if
        if (opa == CKL_OP_N) then
            leny = m
            lenx = n
        else
            leny = n
            lenx = m
        end if
        call ckl_sgemm_d(h, opa, CKL_OP_N, leny, 1, lenx, alpha, da, lda, &
                         dx, max(lenx, 1), beta, dy, max(leny, 1), info)
    end subroutine ckl_sgemv_d

    ! Solves op(A) X = alpha B (lside) or X op(A) = alpha B, B m by n, in place
    ! over B. A is k by k with k = m or n. Blocked: the trailing updates are
    ! GEMMs on the device, the diagonal blocks are solved on the host.
    subroutine ckl_strsm_d(h, lside, lower, notrans, unitdiag, m, n, alpha, da, lda, db, ldb, info)
        type(c_ptr), intent(in) :: h
        logical, intent(in) :: lside, lower, notrans, unitdiag
        integer, intent(in) :: m, n, lda, ldb
        real(c_float), intent(in) :: alpha
        type(c_ptr), intent(in) :: da, db
        integer(c_int), intent(out) :: info

        logical :: tlower
        integer :: k, nb, i0, i1, bs, p0, p1, nprev, step, ii, jj
        integer(c_int64_t) :: aoff, boff, coff
        integer(c_int) :: opa
        real(c_float), allocatable, target :: dblk(:, :), araw(:, :), panel(:, :)
        type(c_ptr) :: aptr, bptr, cptr

        info = CKL_STATUS_SUCCESS
        if (m <= 0 .or. n <= 0) return

        k = m
        if (.not. lside) k = n

        ! op(A) is lower triangular exactly when A is lower and the op is the
        ! identity, or A is upper and the op transposes.
        tlower = (lower .eqv. notrans)
        opa = CKL_OP_N
        if (.not. notrans) opa = CKL_OP_T

        nb = min(TRSM_BLOCK, k)
        allocate (dblk(nb, nb), araw(nb, nb))
        if (lside) then
            allocate (panel(nb, n))
        else
            allocate (panel(m, nb))
        end if

        ! Sweep direction. Left side: a lower op(A) is a forward substitution
        ! over block rows. Right side: X op(A) = B with an upper op(A) resolves
        ! left to right over block columns.
        if (lside .eqv. tlower) then
            step = 1
        else
            step = -1
        end if

        if (step == 1) then
            i0 = 1
        else
            i0 = ((k - 1)/nb)*nb + 1
        end if

        do
            if (step == 1) then
                if (i0 > k) exit
            else
                if (i0 < 1) exit
            end if
            i1 = min(i0 + nb - 1, k)
            bs = i1 - i0 + 1

            ! The already solved part of X, always one contiguous range.
            if (step == 1) then
                p0 = 1
                p1 = i0 - 1
            else
                p0 = i1 + 1
                p1 = k
            end if
            nprev = p1 - p0 + 1
            if (nprev < 0) nprev = 0

            if (nprev > 0) then
                ! The alpha scaling rides in as the GEMM beta, so B is never
                ! scaled in a separate pass.
                if (lside) then
                    if (notrans) then
                        aoff = int(p0 - 1, c_int64_t)*int(lda, c_int64_t) + int(i0 - 1, c_int64_t)
                    else
                        aoff = int(i0 - 1, c_int64_t)*int(lda, c_int64_t) + int(p0 - 1, c_int64_t)
                    end if
                    aptr = ckl_ptr_offset(da, aoff)
                    bptr = ckl_ptr_offset(db, int(p0 - 1, c_int64_t))
                    cptr = ckl_ptr_offset(db, int(i0 - 1, c_int64_t))
                    call ckl_sgemm_d(h, opa, CKL_OP_N, bs, n, nprev, -1.0_c_float, &
                                     aptr, lda, bptr, ldb, alpha, cptr, ldb, info)
                else
                    if (notrans) then
                        aoff = int(i0 - 1, c_int64_t)*int(lda, c_int64_t) + int(p0 - 1, c_int64_t)
                    else
                        aoff = int(p0 - 1, c_int64_t)*int(lda, c_int64_t) + int(i0 - 1, c_int64_t)
                    end if
                    aptr = ckl_ptr_offset(da, aoff)
                    boff = int(p0 - 1, c_int64_t)*int(ldb, c_int64_t)
                    coff = int(i0 - 1, c_int64_t)*int(ldb, c_int64_t)
                    bptr = ckl_ptr_offset(db, boff)
                    cptr = ckl_ptr_offset(db, coff)
                    call ckl_sgemm_d(h, CKL_OP_N, opa, m, bs, nprev, -1.0_c_float, &
                                     bptr, ldb, aptr, lda, alpha, cptr, ldb, info)
                end if
                if (info /= CKL_STATUS_SUCCESS) return
            end if

            ! Diagonal block of A, transposed on the way in when the op asks for
            ! it, so the host solve always sees a plain triangle.
            aoff = int(i0 - 1, c_int64_t)*int(lda, c_int64_t) + int(i0 - 1, c_int64_t)
            call block_to_host(araw, ckl_ptr_offset(da, aoff), lda, bs, bs, info)
            if (info /= CKL_STATUS_SUCCESS) return
            do jj = 1, bs
                do ii = 1, bs
                    if (notrans) then
                        dblk(ii, jj) = araw(ii, jj)
                    else
                        dblk(ii, jj) = araw(jj, ii)
                    end if
                end do
            end do
            if (unitdiag) then
                do ii = 1, bs
                    dblk(ii, ii) = 1.0_c_float
                end do
            end if

            if (lside) then
                call block_to_host(panel, ckl_ptr_offset(db, int(i0 - 1, c_int64_t)), &
                                   ldb, bs, n, info)
                if (info /= CKL_STATUS_SUCCESS) return
                if (nprev == 0) panel(1:bs, 1:n) = alpha*panel(1:bs, 1:n)
                call solve_left(tlower, bs, n, dblk, panel)
                call block_to_device(ckl_ptr_offset(db, int(i0 - 1, c_int64_t)), panel, &
                                     ldb, bs, n, info)
            else
                boff = int(i0 - 1, c_int64_t)*int(ldb, c_int64_t)
                call block_to_host(panel, ckl_ptr_offset(db, boff), ldb, m, bs, info)
                if (info /= CKL_STATUS_SUCCESS) return
                if (nprev == 0) panel(1:m, 1:bs) = alpha*panel(1:m, 1:bs)
                call solve_right(tlower, m, bs, dblk, panel)
                call block_to_device(ckl_ptr_offset(db, boff), panel, ldb, m, bs, info)
            end if
            if (info /= CKL_STATUS_SUCCESS) return

            i0 = i0 + step*nb
        end do
    end subroutine ckl_strsm_d

    ! Solves d X = w in place over w, d bs by bs triangular with an explicit
    ! diagonal, w bs by ncol.
    subroutine solve_left(tlower, bs, ncol, d, w)
        logical, intent(in) :: tlower
        integer, intent(in) :: bs, ncol
        real(c_float), intent(in) :: d(:, :)
        real(c_float), intent(inout) :: w(:, :)
        integer :: i

        if (tlower) then
            do i = 1, bs
                w(i, 1:ncol) = (w(i, 1:ncol) - matmul(d(i, 1:i - 1), w(1:i - 1, 1:ncol)))/d(i, i)
            end do
        else
            do i = bs, 1, -1
                w(i, 1:ncol) = (w(i, 1:ncol) - matmul(d(i, i + 1:bs), w(i + 1:bs, 1:ncol)))/d(i, i)
            end do
        end if
    end subroutine solve_left

    ! Solves X d = w in place over w, w nrow by bs.
    subroutine solve_right(tlower, nrow, bs, d, w)
        logical, intent(in) :: tlower
        integer, intent(in) :: nrow, bs
        real(c_float), intent(in) :: d(:, :)
        real(c_float), intent(inout) :: w(:, :)
        integer :: j

        if (tlower) then
            do j = bs, 1, -1
                w(1:nrow, j) = (w(1:nrow, j) - matmul(w(1:nrow, j + 1:bs), d(j + 1:bs, j)))/d(j, j)
            end do
        else
            do j = 1, bs
                w(1:nrow, j) = (w(1:nrow, j) - matmul(w(1:nrow, 1:j - 1), d(1:j - 1, j)))/d(j, j)
            end do
        end if
    end subroutine solve_right

    ! -------------------------------------------------------------------------
    ! Host pointer entry points: allocate, transfer, run, transfer back
    ! -------------------------------------------------------------------------

    ! Elements spanned by an nrows by ncols column major block with leading
    ! dimension ld, counted from its first element. Copying exactly this many
    ! never reads past what the caller's declaration guarantees.
    pure function span(ld, nrows, ncols) result(s)
        integer, intent(in) :: ld, nrows, ncols
        integer(c_int64_t) :: s

        s = int(ld, c_int64_t)*int(ncols - 1, c_int64_t) + int(nrows, c_int64_t)
    end function span

    subroutine host_to_new_device(dptr, hptr, nelem, stat)
        type(c_ptr), intent(out) :: dptr
        type(c_ptr), intent(in) :: hptr
        integer(c_int64_t), intent(in) :: nelem
        integer(c_int), intent(out) :: stat

        call ckl_alloc_floats(dptr, nelem, stat)
        if (stat /= CKL_STATUS_SUCCESS) return
        stat = ckl_memcpy(dptr, hptr, int(nelem, c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_H2D)
    end subroutine host_to_new_device

    subroutine ckl_sgemm_host_ptr(opa, opb, m, n, k, alpha, ha, lda, hb, ldb, beta, hc, ldc)
        integer(c_int), intent(in) :: opa, opb
        integer, intent(in) :: m, n, k, lda, ldb, ldc
        real(c_float), intent(in) :: alpha, beta
        type(c_ptr), intent(in) :: ha, hb, hc

        type(c_ptr) :: h, da, db, dc
        integer(c_int) :: info, rc
        integer :: rows_a, cols_a, rows_b, cols_b
        integer(c_int64_t) :: nc

        if (m == 0 .or. n == 0) return
        if (alpha == 0.0_c_float .and. beta == 1.0_c_float) return

        h = ckl_blas_handle()

        if (opa == CKL_OP_N) then
            rows_a = m
            cols_a = k
        else
            rows_a = k
            cols_a = m
        end if
        if (opb == CKL_OP_N) then
            rows_b = k
            cols_b = n
        else
            rows_b = n
            cols_b = k
        end if
        nc = span(ldc, m, n)

        da = c_null_ptr
        db = c_null_ptr
        dc = c_null_ptr

        call host_to_new_device(da, ha, max(span(lda, rows_a, cols_a), 1_c_int64_t), info)
        call ckl_blas_require('SGEMM', info)
        call host_to_new_device(db, hb, max(span(ldb, rows_b, cols_b), 1_c_int64_t), info)
        call ckl_blas_require('SGEMM', info)
        if (beta /= 0.0_c_float) then
            call host_to_new_device(dc, hc, nc, info)
        else
            call ckl_alloc_floats(dc, nc, info)
        end if
        call ckl_blas_require('SGEMM', info)

        call ckl_sgemm_d(h, opa, opb, m, n, k, alpha, da, lda, db, ldb, beta, dc, ldc, info)
        call ckl_blas_require('SGEMM', info)

        info = ckl_memcpy(hc, dc, int(nc, c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_D2H)
        call ckl_blas_require('SGEMM', info)

        call ckl_free(da, rc)
        call ckl_free(db, rc)
        call ckl_free(dc, rc)
    end subroutine ckl_sgemm_host_ptr

    subroutine ckl_sgemv_host_ptr(opa, m, n, alpha, ha, lda, hx, incx, beta, hy, incy)
        integer(c_int), intent(in) :: opa
        integer, intent(in) :: m, n, lda, incx, incy
        real(c_float), intent(in) :: alpha, beta
        type(c_ptr), intent(in) :: ha, hx, hy

        type(c_ptr) :: h, da, dx, dy
        integer(c_int) :: info, rc
        integer :: lenx, leny, i, kx, ky
        real(c_float), pointer :: xraw(:), yraw(:)
        real(c_float), allocatable, target :: xp(:), yp(:)

        if (m == 0 .or. n == 0) return
        if (alpha == 0.0_c_float .and. beta == 1.0_c_float) return

        h = ckl_blas_handle()

        if (opa == CKL_OP_N) then
            leny = m
            lenx = n
        else
            leny = n
            lenx = m
        end if

        ! Pack x and y into unit stride buffers. The reference BLAS increment
        ! convention: a negative increment walks the vector backwards from its
        ! last element.
        call c_f_pointer(hx, xraw, [1 + (lenx - 1)*abs(incx)])
        call c_f_pointer(hy, yraw, [1 + (leny - 1)*abs(incy)])
        allocate (xp(lenx), yp(leny))
        if (incx > 0) then
            kx = 1
        else
            kx = 1 - (lenx - 1)*incx
        end if
        if (incy > 0) then
            ky = 1
        else
            ky = 1 - (leny - 1)*incy
        end if
        do i = 1, lenx
            xp(i) = xraw(kx + (i - 1)*incx)
        end do
        if (beta /= 0.0_c_float) then
            do i = 1, leny
                yp(i) = yraw(ky + (i - 1)*incy)
            end do
        else
            yp = 0.0_c_float
        end if

        da = c_null_ptr
        dx = c_null_ptr
        dy = c_null_ptr

        call host_to_new_device(da, ha, span(lda, m, n), info)
        call ckl_blas_require('SGEMV', info)
        call host_to_new_device(dx, c_loc(xp(1)), int(lenx, c_int64_t), info)
        call ckl_blas_require('SGEMV', info)
        call host_to_new_device(dy, c_loc(yp(1)), int(leny, c_int64_t), info)
        call ckl_blas_require('SGEMV', info)

        call ckl_sgemv_d(h, opa, m, n, alpha, da, lda, dx, 1, beta, dy, 1, info)
        call ckl_blas_require('SGEMV', info)

        info = ckl_memcpy(c_loc(yp(1)), dy, int(leny, c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_D2H)
        call ckl_blas_require('SGEMV', info)

        do i = 1, leny
            yraw(ky + (i - 1)*incy) = yp(i)
        end do

        call ckl_free(da, rc)
        call ckl_free(dx, rc)
        call ckl_free(dy, rc)
    end subroutine ckl_sgemv_host_ptr

    subroutine ckl_strsm_host_ptr(lside, lower, notrans, unitdiag, m, n, alpha, ha, lda, hb, ldb)
        logical, intent(in) :: lside, lower, notrans, unitdiag
        integer, intent(in) :: m, n, lda, ldb
        real(c_float), intent(in) :: alpha
        type(c_ptr), intent(in) :: ha, hb

        type(c_ptr) :: h, da, db
        integer(c_int) :: info, rc
        integer :: k, j
        integer(c_int64_t) :: nb_elems
        real(c_float), pointer :: braw(:)

        if (m == 0 .or. n == 0) return

        k = m
        if (.not. lside) k = n

        nb_elems = span(ldb, m, n)

        if (alpha == 0.0_c_float) then
            call c_f_pointer(hb, braw, [nb_elems])
            do j = 1, n
                braw((j - 1)*ldb + 1:(j - 1)*ldb + m) = 0.0_c_float
            end do
            return
        end if

        h = ckl_blas_handle()

        da = c_null_ptr
        db = c_null_ptr
        call host_to_new_device(da, ha, span(lda, k, k), info)
        call ckl_blas_require('STRSM', info)
        call host_to_new_device(db, hb, nb_elems, info)
        call ckl_blas_require('STRSM', info)

        call ckl_strsm_d(h, lside, lower, notrans, unitdiag, m, n, alpha, da, lda, db, ldb, info)
        call ckl_blas_require('STRSM', info)

        info = ckl_memcpy(hb, db, int(nb_elems, c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_D2H)
        call ckl_blas_require('STRSM', info)

        call ckl_free(da, rc)
        call ckl_free(db, rc)
    end subroutine ckl_strsm_host_ptr

end module ckl_blas_core
