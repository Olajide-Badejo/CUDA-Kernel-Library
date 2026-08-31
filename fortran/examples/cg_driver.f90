! Conjugate gradient on a synthetic SPD system, driven through CKL.
!
! The matrix is built with a CKL SGEMM (A = G^T G / n + I, which is symmetric
! positive definite by construction and lands at a condition number near 5), and
! every CG iteration's matrix vector product is a CKL SGEMV on the device with A
! resident across iterations.
!
! Two things are checked afterwards. LAPACK factors an untouched copy with
! SPOTRF and solves with SPOTRS, and the CG answer is compared against that. The
! Gate E residual bound
!     norm(A x - b, inf) / (norm(A, inf) * norm(x, inf) * n * eps) <= 10
! is computed in double precision from the single precision answer and printed.
!
! The last section times the same matrix vector product two ways: CKL with A
! already resident, and the CPU reference BLAS SGEMV. Both numbers are measured
! here at run time and printed as measured. Nothing is copied into a document.
!
! Exit code 77 means no CUDA device answered.

program cg_driver
    use, intrinsic :: iso_c_binding
    use ckl
    use ckl_blas_core, only: ckl_sgemv_d
    implicit none

    integer, parameter :: n = 2048
    integer, parameter :: max_iter = 400
    integer, parameter :: timing_reps = 200

    type(c_ptr) :: h, da, dp, dap
    integer(c_int) :: rc, stat
    real(c_float), allocatable, target :: a(:, :), afac(:, :), b(:), x(:), xref(:)
    real(c_float), allocatable, target :: p(:), r(:), ap(:)
    double precision :: resid_bound, cg_vs_lapack
    integer :: iters

    rc = ckl_create(h)
    if (rc /= CKL_STATUS_SUCCESS) then
        write (*, '(a)') 'skip: no CUDA device ('//ckl_error_text(rc)//')'
        stop 77
    end if

    write (*, '(a)') 'CKL conjugate gradient driver, version '//ckl_version_text()
    write (*, '(a,i0)') 'system size n = ', n

    allocate (a(n, n), afac(n, n), b(n), x(n), xref(n), p(n), r(n), ap(n))

    call build_spd(h, a)
    call build_rhs(b)

    ! A stays on the device for the whole solve and the whole timing section.
    call ckl_alloc_floats(da, int(n, c_int64_t)*int(n, c_int64_t), stat)
    call must(stat, 'alloc A')
    call ckl_put2(da, a, stat)
    call must(stat, 'copy A')
    call ckl_alloc_floats(dp, int(n, c_int64_t), stat)
    call must(stat, 'alloc p')
    call ckl_alloc_floats(dap, int(n, c_int64_t), stat)
    call must(stat, 'alloc Ap')

    call cg_solve(h, da, dp, dap, b, x, iters)
    write (*, '(a,i0,a)') 'conjugate gradient converged in ', iters, ' iterations'

    ! LAPACK reference solve on an untouched copy.
    afac = a
    xref = b
    call lapack_solve(afac, xref)

    cg_vs_lapack = maxval(abs(dble(x) - dble(xref)))/max(maxval(abs(dble(xref))), 1.0d0)
    write (*, '(a,es11.4)') 'max relative difference from the LAPACK solve: ', cg_vs_lapack

    resid_bound = gate_e_residual(a, b, x)
    write (*, '(a,es11.4,a)') &
        'Gate E residual norm(Ax-b,inf)/(norm(A,inf)*norm(x,inf)*n*eps) = ', &
        resid_bound, '  (bound 10)'

    call time_gemv(h, da, dp, dap, a, p)

    call ckl_free(da, rc)
    call ckl_free(dp, rc)
    call ckl_free(dap, rc)
    rc = ckl_destroy(h)

    if (.not. (resid_bound <= 10.0d0)) then
        write (*, '(a)') 'FAILED: Gate E residual bound exceeded'
        error stop 1
    end if
    write (*, '(a)') 'Gate E residual bound satisfied'

contains

    subroutine must(stat_, where)
        integer(c_int), intent(in) :: stat_
        character(len=*), intent(in) :: where

        if (stat_ == CKL_STATUS_SUCCESS) return
        write (*, '(a)') 'FATAL '//where//': '//ckl_error_text(stat_)
        error stop 1
    end subroutine must

    subroutine fill(v, seed)
        real(c_float), intent(out) :: v(:, :)
        integer(c_int64_t), intent(inout) :: seed
        integer :: i, j

        do j = 1, size(v, 2)
            do i = 1, size(v, 1)
                seed = mod(1103515245_c_int64_t*seed + 12345_c_int64_t, 2147483648_c_int64_t)
                v(i, j) = real(seed, c_float)/1073741824.0_c_float - 1.0_c_float
            end do
        end do
    end subroutine fill

    ! A = G^T G / n + I through the CKL column major GEMM. G^T G is positive
    ! semidefinite for any G, and the shift makes it definite with room to spare.
    subroutine build_spd(hh, out)
        type(c_ptr), intent(in) :: hh
        real(c_float), intent(out) :: out(:, :)
        real(c_float), allocatable :: g(:, :)
        integer(c_int64_t) :: seed
        integer(c_int) :: st
        integer :: i, sz

        sz = size(out, 1)
        allocate (g(sz, sz))
        seed = 20260901_c_int64_t
        call fill(g, seed)
        out = 0.0_c_float

        call ckl_sgemm_arrays(hh, CKL_OP_T, CKL_OP_N, 1.0_c_float/real(sz, c_float), &
                              g, g, 0.0_c_float, out, st)
        call must(st, 'build_spd gemm')

        ! Kill the last bit of asymmetry the float rounding leaves behind, then
        ! shift. SPOTRF reads one triangle, but CG needs a genuinely symmetric
        ! operator.
        out = 0.5_c_float*(out + transpose(out))
        do i = 1, sz
            out(i, i) = out(i, i) + 1.0_c_float
        end do
    end subroutine build_spd

    subroutine build_rhs(rhs)
        real(c_float), intent(out) :: rhs(:)
        real(c_float), allocatable :: tmp(:, :)
        integer(c_int64_t) :: seed

        allocate (tmp(size(rhs), 1))
        seed = 55555_c_int64_t
        call fill(tmp, seed)
        rhs = tmp(:, 1)
    end subroutine build_rhs

    ! y := A * v with A resident. One CKL SGEMV, one vector up, one vector back.
    subroutine device_matvec(hh, da_, dv, dy, v, y)
        type(c_ptr), intent(in) :: hh, da_, dv, dy
        real(c_float), intent(in) :: v(:)
        real(c_float), intent(out) :: y(:)
        integer(c_int) :: st

        call ckl_put(dv, v, st)
        call must(st, 'matvec put')
        call ckl_sgemv_d(hh, CKL_OP_N, n, n, 1.0_c_float, da_, n, dv, 1, 0.0_c_float, dy, 1, st)
        call must(st, 'matvec sgemv')
        call ckl_get(y, dy, st)
        call must(st, 'matvec get')
    end subroutine device_matvec

    subroutine cg_solve(hh, da_, dv, dy, rhs, sol, it)
        type(c_ptr), intent(in) :: hh, da_, dv, dy
        real(c_float), intent(in) :: rhs(:)
        real(c_float), intent(out) :: sol(:)
        integer, intent(out) :: it

        double precision :: rr, rr_new, pap, alpha_, beta_, rhs_norm, tol
        integer :: i

        sol = 0.0_c_float
        r = rhs
        p = r
        rr = dot_product(dble(r), dble(r))
        rhs_norm = sqrt(dot_product(dble(rhs), dble(rhs)))
        ! Single precision residuals stop improving near eps times the right hand
        ! side, so the stopping tolerance is derived from that rather than picked.
        tol = 8.0d0*dble(epsilon(1.0_c_float))*rhs_norm
        tol = tol*tol

        it = 0
        do i = 1, max_iter
            if (rr <= tol) exit
            call device_matvec(hh, da_, dv, dy, p, ap)
            pap = dot_product(dble(p), dble(ap))
            if (pap <= 0.0d0) then
                write (*, '(a)') 'FAILED: the operator is not positive definite'
                error stop 1
            end if
            alpha_ = rr/pap
            sol = sol + real(alpha_, c_float)*p
            r = r - real(alpha_, c_float)*ap
            rr_new = dot_product(dble(r), dble(r))
            beta_ = rr_new/rr
            p = r + real(beta_, c_float)*p
            rr = rr_new
            it = i
        end do
    end subroutine cg_solve

    subroutine lapack_solve(afac_, rhs)
        real(c_float), intent(inout) :: afac_(n, n)
        real(c_float), intent(inout) :: rhs(n)
        integer :: info
        external :: spotrf, spotrs

        call spotrf('L', n, afac_, n, info)
        if (info /= 0) then
            write (*, '(a,i0)') 'FAILED: SPOTRF returned info = ', info
            error stop 1
        end if
        call spotrs('L', n, 1, afac_, n, rhs, n, info)
        if (info /= 0) then
            write (*, '(a,i0)') 'FAILED: SPOTRS returned info = ', info
            error stop 1
        end if
    end subroutine lapack_solve

    function gate_e_residual(amat, rhs, sol) result(v)
        real(c_float), intent(in) :: amat(:, :), rhs(:), sol(:)
        double precision :: v
        double precision, allocatable :: res(:)
        double precision :: anorm, xnorm, rnorm
        integer :: i

        allocate (res(n))
        do i = 1, n
            res(i) = sum(dble(amat(i, :))*dble(sol)) - dble(rhs(i))
        end do
        rnorm = maxval(abs(res))
        anorm = 0.0d0
        do i = 1, n
            anorm = max(anorm, sum(abs(dble(amat(i, :)))))
        end do
        xnorm = maxval(abs(dble(sol)))
        v = rnorm/(anorm*xnorm*dble(n)*dble(epsilon(1.0_c_float)))
    end function gate_e_residual

    ! Same operation, two implementations, both measured now. The CKL number is
    ! the device call with A already resident plus the vector round trip, which
    ! is what the CG loop actually pays; the transfer is inside the timed region
    ! because the caller cannot avoid it.
    subroutine time_gemv(hh, da_, dv, dy, amat, vec)
        type(c_ptr), intent(in) :: hh, da_, dv, dy
        real(c_float), intent(in) :: amat(n, n)
        real(c_float), intent(inout) :: vec(n)
        real(c_float), allocatable :: y_ckl(:), y_cpu(:)
        integer(c_int64_t) :: t0, t1, rate
        integer(c_int64_t) :: seed
        real(c_float), allocatable :: tmp(:, :)
        double precision :: ckl_ms, cpu_ms, gflop
        integer :: i
        external :: sgemv

        allocate (y_ckl(n), y_cpu(n), tmp(n, 1))
        seed = 31337_c_int64_t
        call fill(tmp, seed)
        vec = tmp(:, 1)

        ! Warm up both paths outside the timed region: first call costs a device
        ! context and a cuBLAS handle, and neither is part of the measurement.
        call device_matvec(hh, da_, dv, dy, vec, y_ckl)
        call sgemv('N', n, n, 1.0_c_float, amat, n, vec, 1, 0.0_c_float, y_cpu, 1)

        call system_clock(count_rate=rate)

        call system_clock(t0)
        do i = 1, timing_reps
            call device_matvec(hh, da_, dv, dy, vec, y_ckl)
        end do
        call system_clock(t1)
        ckl_ms = 1000.0d0*dble(t1 - t0)/dble(rate)/dble(timing_reps)

        call system_clock(t0)
        do i = 1, timing_reps
            call sgemv('N', n, n, 1.0_c_float, amat, n, vec, 1, 0.0_c_float, y_cpu, 1)
        end do
        call system_clock(t1)
        cpu_ms = 1000.0d0*dble(t1 - t0)/dble(rate)/dble(timing_reps)

        gflop = 2.0d0*dble(n)*dble(n)/1.0d9
        write (*, '(a,i0,a)') 'SGEMV timing, n = ', n, ', mean over the repeat count'
        write (*, '(a,i0)') '  repeats            ', timing_reps
        write (*, '(a,f10.4,a,f8.2,a)') '  CKL device + copy ', ckl_ms, ' ms  ', &
            gflop/(ckl_ms/1000.0d0), ' GFLOP/s'
        write (*, '(a,f10.4,a,f8.2,a)') '  CPU reference BLAS', cpu_ms, ' ms  ', &
            gflop/(cpu_ms/1000.0d0), ' GFLOP/s'
        write (*, '(a,es11.4)') '  max abs difference ', &
            dble(maxval(abs(y_ckl - y_cpu)))
    end subroutine time_gemv

end program cg_driver
