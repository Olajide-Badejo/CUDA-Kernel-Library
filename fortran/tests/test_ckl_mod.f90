! Fortran driver for the module layer, device pointer path.
!
! It allocates through ckl_device_malloc, moves operands with ckl_memcpy, calls
! ckl_sgemm on the column major path and compares against a reference computed
! in Fortran in double precision. Exit code 77 means no CUDA device answered,
! which ctest reads as a skip.

program test_ckl_mod
    use, intrinsic :: iso_c_binding
    use ckl
    implicit none

    type(c_ptr) :: h
    integer(c_int) :: rc
    integer :: failures

    rc = ckl_create(h)
    if (rc /= CKL_STATUS_SUCCESS) then
        write (*, '(a)') 'skip: no CUDA device ('//ckl_error_text(rc)//')'
        stop 77
    end if

    write (*, '(a)') 'ckl version: '//ckl_version_text()

    failures = 0
    call check_version(failures)
    call check_query(h, failures)
    call check_gemm(h, 64, 48, 80, CKL_OP_N, CKL_OP_N, 1.25_c_float, -0.5_c_float, failures)
    call check_gemm(h, 33, 17, 96, CKL_OP_T, CKL_OP_N, 1.0_c_float, 0.0_c_float, failures)
    call check_gemm(h, 40, 40, 40, CKL_OP_N, CKL_OP_T, -2.0_c_float, 1.0_c_float, failures)
    call check_arrays_layer(h, failures)

    rc = ckl_destroy(h)
    if (rc /= CKL_STATUS_SUCCESS) then
        write (*, '(a)') 'FAIL ckl_destroy: '//ckl_error_text(rc)
        failures = failures + 1
    end if

    if (failures /= 0) then
        write (*, '(a,i0,a)') 'FAILED: ', failures, ' check(s)'
        error stop 1
    end if
    write (*, '(a)') 'all module layer checks passed'

contains

    ! Deterministic fill, so a failure reproduces exactly.
    subroutine fill(a, seed)
        real(c_float), intent(out) :: a(:, :)
        integer(c_int64_t), intent(inout) :: seed
        integer :: i, j

        do j = 1, size(a, 2)
            do i = 1, size(a, 1)
                seed = mod(1103515245_c_int64_t*seed + 12345_c_int64_t, 2147483648_c_int64_t)
                a(i, j) = real(seed, c_float)/1073741824.0_c_float - 1.0_c_float
            end do
        end do
    end subroutine fill

    subroutine check_version(nfail)
        integer, intent(inout) :: nfail
        integer(c_int) :: v

        v = ckl_get_version()
        if (v /= 10100) then
            write (*, '(a,i0)') 'FAIL ckl_get_version expected 10100, got ', v
            nfail = nfail + 1
        end if
    end subroutine check_version

    ! gemm_query launches nothing; it only has to answer and answer in range.
    subroutine check_query(hh, nfail)
        type(c_ptr), intent(in) :: hh
        integer, intent(inout) :: nfail
        integer(c_int) :: stat, chosen
        integer(c_size_t) :: ws

        chosen = -1
        stat = ckl_gemm_query(hh, CKL_COL_MAJOR, CKL_OP_N, CKL_OP_N, &
                              512_c_int64_t, 512_c_int64_t, 512_c_int64_t, &
                              CKL_R_32F, 512_c_int64_t, CKL_R_32F, 512_c_int64_t, &
                              CKL_R_32F, 512_c_int64_t, chosen)
        if (stat /= CKL_STATUS_SUCCESS) then
            write (*, '(a)') 'FAIL ckl_gemm_query: '//ckl_error_text(stat)
            nfail = nfail + 1
        else if (chosen < CKL_ALGO_AUTO .or. chosen > CKL_ALGO_CUBLAS) then
            write (*, '(a,i0)') 'FAIL ckl_gemm_query chosen out of range: ', chosen
            nfail = nfail + 1
        else
            write (*, '(a,i0)') 'gemm_query chose algo ', chosen
        end if

        ws = huge(ws)
        stat = ckl_gemm_workspace_size(hh, CKL_COL_MAJOR, CKL_OP_N, CKL_OP_N, &
                                       512_c_int64_t, 512_c_int64_t, 512_c_int64_t, &
                                       CKL_R_32F, 512_c_int64_t, CKL_R_32F, 512_c_int64_t, &
                                       CKL_R_32F, 512_c_int64_t, CKL_ALGO_AUTO, ws)
        if (stat /= CKL_STATUS_SUCCESS) then
            write (*, '(a)') 'FAIL ckl_gemm_workspace_size: '//ckl_error_text(stat)
            nfail = nfail + 1
        end if
    end subroutine check_query

    ! C := alpha * op(A) * op(B) + beta * C through the raw device pointer path.
    subroutine check_gemm(hh, m, n, k, opa, opb, alpha, beta, nfail)
        type(c_ptr), intent(in) :: hh
        integer, intent(in) :: m, n, k
        integer(c_int), intent(in) :: opa, opb
        real(c_float), intent(in) :: alpha, beta
        integer, intent(inout) :: nfail

        real(c_float), allocatable :: a(:, :), b(:, :), c(:, :), c0(:, :)
        real(c_double), allocatable :: ref(:, :), oa(:, :), ob(:, :)
        type(c_ptr) :: da, db, dc
        integer(c_int) :: stat, rc2
        integer :: rows_a, cols_a, rows_b, cols_b
        integer(c_int64_t) :: seed
        real(c_double) :: tol, err, scale

        if (opa == CKL_OP_N) then
            rows_a = m; cols_a = k
        else
            rows_a = k; cols_a = m
        end if
        if (opb == CKL_OP_N) then
            rows_b = k; cols_b = n
        else
            rows_b = n; cols_b = k
        end if

        allocate (a(rows_a, cols_a), b(rows_b, cols_b), c(m, n), c0(m, n))
        seed = int(20260831 + m*7 + n*13 + k, c_int64_t)
        call fill(a, seed)
        call fill(b, seed)
        call fill(c, seed)
        c0 = c

        if (opa == CKL_OP_N) then
            oa = real(a, c_double)
        else
            oa = transpose(real(a, c_double))
        end if
        if (opb == CKL_OP_N) then
            ob = real(b, c_double)
        else
            ob = transpose(real(b, c_double))
        end if
        ref = real(alpha, c_double)*matmul(oa, ob) + real(beta, c_double)*real(c0, c_double)

        da = c_null_ptr; db = c_null_ptr; dc = c_null_ptr
        call ckl_alloc_floats(da, int(size(a), c_int64_t), stat)
        call fatal(stat, 'ckl_alloc_floats(a)')
        call ckl_alloc_floats(db, int(size(b), c_int64_t), stat)
        call fatal(stat, 'ckl_alloc_floats(b)')
        call ckl_alloc_floats(dc, int(size(c), c_int64_t), stat)
        call fatal(stat, 'ckl_alloc_floats(c)')

        call ckl_put2(da, a, stat)
        call fatal(stat, 'ckl_put2(a)')
        call ckl_put2(db, b, stat)
        call fatal(stat, 'ckl_put2(b)')
        call ckl_put2(dc, c, stat)
        call fatal(stat, 'ckl_put2(c)')

        stat = ckl_sgemm(hh, CKL_COL_MAJOR, opa, opb, &
                         int(m, c_int64_t), int(n, c_int64_t), int(k, c_int64_t), alpha, &
                         da, int(rows_a, c_int64_t), db, int(rows_b, c_int64_t), beta, &
                         dc, int(m, c_int64_t))
        if (stat /= CKL_STATUS_SUCCESS) then
            write (*, '(a)') 'FAIL ckl_sgemm: '//ckl_error_text(stat)
            nfail = nfail + 1
            return
        end if
        stat = ckl_device_synchronize()
        call fatal(stat, 'ckl_device_synchronize')
        call ckl_get2(c, dc, stat)
        call fatal(stat, 'ckl_get2(c)')

        call ckl_free(da, rc2)
        call ckl_free(db, rc2)
        call ckl_free(dc, rc2)

        scale = max(maxval(abs(ref)), 1.0_c_double)
        err = maxval(abs(real(c, c_double) - ref))/scale
        tol = 8.0_c_double*sqrt(real(k, c_double))*real(epsilon(1.0_c_float), c_double)
        write (*, '(a,i0,a,i0,a,i0,a,i0,a,i0,a,es10.3,a,es10.3)') &
            'gemm m=', m, ' n=', n, ' k=', k, ' opa=', opa, ' opb=', opb, &
            '  rel err ', err, '  tol ', tol
        if (.not. (err <= tol)) then
            write (*, '(a)') 'FAIL: relative error above tolerance'
            nfail = nfail + 1
        end if
    end subroutine check_gemm

    ! The convenience layer does the same arithmetic with no device pointer in
    ! sight, so it gets the same oracle.
    subroutine check_arrays_layer(hh, nfail)
        type(c_ptr), intent(in) :: hh
        integer, intent(inout) :: nfail
        integer, parameter :: m = 51, n = 37, k = 64
        real(c_float) :: a(m, k), b(k, n), c(m, n), c0(m, n)
        real(c_double) :: ref(m, n), err, tol
        integer(c_int) :: stat
        integer(c_int64_t) :: seed

        seed = 991_c_int64_t
        call fill(a, seed)
        call fill(b, seed)
        call fill(c, seed)
        c0 = c
        ref = 0.75_c_double*matmul(real(a, c_double), real(b, c_double)) &
              + 0.25_c_double*real(c0, c_double)

        call ckl_sgemm_arrays(hh, CKL_OP_N, CKL_OP_N, 0.75_c_float, a, b, 0.25_c_float, c, stat)
        if (stat /= CKL_STATUS_SUCCESS) then
            write (*, '(a)') 'FAIL ckl_sgemm_arrays: '//ckl_error_text(stat)
            nfail = nfail + 1
            return
        end if

        err = maxval(abs(real(c, c_double) - ref))/max(maxval(abs(ref)), 1.0_c_double)
        tol = 8.0_c_double*sqrt(real(k, c_double))*real(epsilon(1.0_c_float), c_double)
        write (*, '(a,es10.3,a,es10.3)') 'ckl_sgemm_arrays rel err ', err, '  tol ', tol
        if (.not. (err <= tol)) then
            write (*, '(a)') 'FAIL: convenience layer above tolerance'
            nfail = nfail + 1
        end if
    end subroutine check_arrays_layer

    subroutine fatal(stat, where)
        integer(c_int), intent(in) :: stat
        character(len=*), intent(in) :: where

        if (stat == CKL_STATUS_SUCCESS) return
        write (*, '(a)') 'FATAL '//where//': '//ckl_error_text(stat)
        error stop 1
    end subroutine fatal

end program test_ckl_mod
