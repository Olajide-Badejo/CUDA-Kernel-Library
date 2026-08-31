! Gate E's alias test: call SGEMM, SGEMV and STRSM through the standard
! reference BLAS signatures, with host arrays and character option arguments,
! and match a Fortran computed oracle.
!
! Nothing here says CKL except the device probe at the top. The calls below are
! the ones an existing Fortran program already contains; what makes them run on
! the GPU is that libckl_blas is on the link line instead of a CPU BLAS.
!
! Exit code 77 means no CUDA device answered.

program test_ckl_blas
    use, intrinsic :: iso_c_binding
    use ckl, only: ckl_create, ckl_destroy, ckl_error_text, CKL_STATUS_SUCCESS
    use ckl_blas_core, only: ckl_blas_shutdown
    implicit none

    type(c_ptr) :: probe
    integer(c_int) :: rc
    integer :: failures

    rc = ckl_create(probe)
    if (rc /= CKL_STATUS_SUCCESS) then
        write (*, '(a)') 'skip: no CUDA device ('//ckl_error_text(rc)//')'
        stop 77
    end if
    rc = ckl_destroy(probe)

    failures = 0
    call check_sgemm(failures)
    call check_sgemv(failures)
    call check_strsm('L', 'L', 'N', 'N', 200, 40, failures)
    call check_strsm('L', 'U', 'T', 'U', 200, 40, failures)
    call check_strsm('R', 'U', 'N', 'N', 40, 200, failures)
    call check_strsm('R', 'L', 'T', 'U', 40, 200, failures)

    call ckl_blas_shutdown()

    if (failures /= 0) then
        write (*, '(a,i0,a)') 'FAILED: ', failures, ' check(s)'
        error stop 1
    end if
    write (*, '(a)') 'all BLAS alias checks passed'

contains

    subroutine fill(a, seed)
        real, intent(out) :: a(:, :)
        integer(c_int64_t), intent(inout) :: seed
        integer :: i, j

        do j = 1, size(a, 2)
            do i = 1, size(a, 1)
                seed = mod(1103515245_c_int64_t*seed + 12345_c_int64_t, 2147483648_c_int64_t)
                a(i, j) = real(seed)/1073741824.0 - 1.0
            end do
        end do
    end subroutine fill

    function tol_for(k) result(t)
        integer, intent(in) :: k
        double precision :: t

        t = 8.0d0*sqrt(dble(k))*dble(epsilon(1.0))
    end function tol_for

    subroutine report(name, err, tol, nfail)
        character(len=*), intent(in) :: name
        double precision, intent(in) :: err, tol
        integer, intent(inout) :: nfail

        write (*, '(a,a,es11.4,a,es11.4)') name, ' rel err ', err, '  tol ', tol
        if (.not. (err <= tol)) then
            write (*, '(a)') 'FAIL: '//name//' above tolerance'
            nfail = nfail + 1
        end if
    end subroutine report

    ! The leading dimensions are deliberately larger than the extents, because a
    ! wrapper that ignores lda still passes a packed test.
    subroutine check_sgemm(nfail)
        integer, intent(inout) :: nfail
        integer, parameter :: m = 77, n = 53, k = 64
        integer, parameter :: lda = m + 5, ldb = k + 3, ldc = m + 7
        real :: a(lda, k), b(ldb, n), c(ldc, n), c0(ldc, n)
        double precision :: ref(m, n), err
        integer(c_int64_t) :: seed
        external :: sgemm

        seed = 12345_c_int64_t
        call fill(a, seed)
        call fill(b, seed)
        call fill(c, seed)
        c0 = c

        ref = 1.5d0*matmul(dble(a(1:m, 1:k)), dble(b(1:k, 1:n))) - 0.25d0*dble(c0(1:m, 1:n))

        call sgemm('N', 'N', m, n, k, 1.5, a, lda, b, ldb, -0.25, c, ldc)

        err = maxval(abs(dble(c(1:m, 1:n)) - ref))/max(maxval(abs(ref)), 1.0d0)
        call report('SGEMM N N', err, tol_for(k), nfail)

        ! Transposed A, so the shim has to keep the op with its own operand.
        block
            real :: at(k + 4, m), c2(ldc, n)
            integer :: ldat
            ldat = k + 4
            seed = 777_c_int64_t
            call fill(at, seed)
            call fill(c2, seed)
            ref = 1.0d0*matmul(transpose(dble(at(1:k, 1:m))), dble(b(1:k, 1:n)))
            call sgemm('T', 'N', m, n, k, 1.0, at, ldat, b, ldb, 0.0, c2, ldc)
            err = maxval(abs(dble(c2(1:m, 1:n)) - ref))/max(maxval(abs(ref)), 1.0d0)
            call report('SGEMM T N', err, tol_for(k), nfail)
        end block
    end subroutine check_sgemm

    subroutine check_sgemv(nfail)
        integer, intent(inout) :: nfail
        integer, parameter :: m = 129, n = 96, lda = m + 3
        real :: a(lda, n), x(2*n), y(2*m), y0(2*m)
        double precision :: ref(m), err
        integer(c_int64_t) :: seed
        integer :: i
        external :: sgemv

        seed = 4242_c_int64_t
        block
            real :: ax(lda, n), xx(2*n, 1), yy(2*m, 1)
            call fill(ax, seed)
            call fill(xx, seed)
            call fill(yy, seed)
            a = ax
            x = xx(:, 1)
            y = yy(:, 1)
        end block
        y0 = y

        ! Unit increments, no transpose.
        ref = 2.0d0*matmul(dble(a(1:m, 1:n)), dble(x(1:n))) + 0.5d0*dble(y0(1:m))
        call sgemv('N', m, n, 2.0, a, lda, x, 1, 0.5, y, 1)
        err = maxval(abs(dble(y(1:m)) - ref))/max(maxval(abs(ref)), 1.0d0)
        call report('SGEMV N inc1', err, tol_for(n), nfail)

        ! Strided x and y, which the shim packs on the host before the multiply.
        y = y0
        ref = 0.0d0
        do i = 1, m
            ref(i) = sum(dble(a(i, 1:n))*dble(x(1:2*n:2)))
        end do
        ref = ref - 1.0d0*dble(y0(1:2*m:2))
        call sgemv('N', m, n, 1.0, a, lda, x, 2, -1.0, y, 2)
        err = maxval(abs(dble(y(1:2*m:2)) - ref))/max(maxval(abs(ref)), 1.0d0)
        call report('SGEMV N inc2', err, tol_for(n), nfail)

        ! Transposed, so the output is length n.
        block
            real :: yt(n)
            double precision :: reft(n)
            yt = 0.0
            reft = matmul(transpose(dble(a(1:m, 1:n))), dble(x(1:m)))
            call sgemv('T', m, n, 1.0, a, lda, x, 1, 0.0, yt, 1)
            err = maxval(abs(dble(yt) - reft))/max(maxval(abs(reft)), 1.0d0)
            call report('SGEMV T inc1', err, tol_for(m), nfail)
        end block
    end subroutine check_sgemv

    ! Solves through the standard STRSM signature, then multiplies the answer
    ! back by the dense form of op(A) and checks it reproduces alpha * B.
    subroutine check_strsm(side, uplo, transa, diag, m, n, nfail)
        character(len=1), intent(in) :: side, uplo, transa, diag
        integer, intent(in) :: m, n
        integer, intent(inout) :: nfail

        integer :: k, lda, ldb, i, j
        real, allocatable :: a(:, :), b(:, :), b0(:, :)
        double precision, allocatable :: t(:, :), resid(:, :)
        double precision :: err, tol
        real, parameter :: alpha = 0.75
        integer(c_int64_t) :: seed
        logical :: left, lower, notrans, unit
        external :: strsm

        left = (side == 'L')
        lower = (uplo == 'L')
        notrans = (transa == 'N')
        unit = (diag == 'U')

        k = n
        if (left) k = m
        lda = k + 2
        ldb = m + 3

        allocate (a(lda, k), b(ldb, n), b0(ldb, n))
        seed = int(1000 + m*3 + n, c_int64_t)
        call fill(a, seed)
        call fill(b, seed)

        ! Off diagonals scaled by 1/k, so each row's off diagonal mass is about
        ! half the unit diagonal used when diag is 'U' and a quarter of the
        ! explicit diagonal otherwise. That keeps the solve well conditioned in
        ! single precision while leaving the off diagonal part large enough that
        ! a wrong trailing update would show up in the residual.
        do j = 1, k
            do i = 1, k
                a(i, j) = a(i, j)/real(k)
            end do
            a(j, j) = 2.0 + 0.5*real(j)/real(k)
        end do
        b0 = b

        call strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)

        ! Dense op(A) with the referenced triangle only, unit diagonal applied.
        allocate (t(k, k))
        t = 0.0d0
        do j = 1, k
            do i = 1, k
                if (lower .and. i > j) t(i, j) = dble(a(i, j))
                if ((.not. lower) .and. i < j) t(i, j) = dble(a(i, j))
            end do
            if (unit) then
                t(j, j) = 1.0d0
            else
                t(j, j) = dble(a(j, j))
            end if
        end do
        if (.not. notrans) t = transpose(t)

        allocate (resid(m, n))
        if (left) then
            resid = matmul(t, dble(b(1:m, 1:n))) - dble(alpha)*dble(b0(1:m, 1:n))
        else
            resid = matmul(dble(b(1:m, 1:n)), t) - dble(alpha)*dble(b0(1:m, 1:n))
        end if

        err = maxval(abs(resid))/max(maxval(abs(dble(alpha)*dble(b0(1:m, 1:n)))), 1.0d0)
        tol = tol_for(k)
        call report('STRSM '//side//uplo//transa//diag, err, tol, nfail)
    end subroutine check_strsm

end program test_ckl_blas
