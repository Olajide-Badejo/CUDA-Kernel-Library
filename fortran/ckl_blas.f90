! Reference BLAS compatible aliases: SGEMM, SGEMV and STRSM under the symbol
! names a Fortran program's existing calls already resolve to.
!
! These are the only symbols in the ckl_blas archive, and the archive is behind
! option(CKL_BLAS_ALIASES OFF). Linking it alongside a CPU BLAS is a duplicate
! symbol error by construction, which is the point: it goes in *instead of* the
! CPU BLAS, never next to it. docs/fortran.md spells that out.
!
! ABI, gfortran and nvfortran convention:
!   - the symbol name is the lower case Fortran name with one trailing
!     underscore, hence bind(C, name="sgemm_");
!   - every argument except the hidden character lengths is passed by reference,
!     so the scalars are plain non VALUE dummies;
!   - a CHARACTER argument is passed as an address, and its length follows the
!     whole argument list as an extra by value integer. Those trailing dummies
!     are declared here so the ABI matches exactly. They are never read: a BLAS
!     option character is length 1, and the value only has to occupy the right
!     register slot. That is also why the gfortran size_t width and the
!     nvfortran width can differ without breaking the call.
!
! The arrays arrive as host addresses, so these entry points move the data.
! A caller that manages residency itself wants ckl_sgemm_d, ckl_sgemv_d or
! ckl_strsm_d from module ckl_blas_core instead.

module ckl_blas
    use, intrinsic :: iso_c_binding
    use ckl, only: CKL_OP_N
    use ckl_blas_core
    implicit none
    private

    public :: sgemm_alias, sgemv_alias, strsm_alias

contains

    subroutine sgemm_alias(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, &
                           ltransa, ltransb) bind(C, name="sgemm_")
        character(kind=c_char), intent(in) :: transa
        character(kind=c_char), intent(in) :: transb
        integer(c_int), intent(in) :: m, n, k, lda, ldb, ldc
        real(c_float), intent(in) :: alpha, beta
        type(c_ptr), value :: a, b, c
        integer(c_size_t), value :: ltransa, ltransb

        integer(c_int) :: opa, opb
        logical :: ok
        integer :: info, nrowa, nrowb


        call ckl_blas_op_code(transa, opa, ok)
        if (.not. ok) call ckl_blas_abort('SGEMM ', 1)
        call ckl_blas_op_code(transb, opb, ok)
        if (.not. ok) call ckl_blas_abort('SGEMM ', 2)

        if (opa == CKL_OP_N) then
            nrowa = int(m)
        else
            nrowa = int(k)
        end if
        if (opb == CKL_OP_N) then
            nrowb = int(k)
        else
            nrowb = int(n)
        end if

        info = 0
        if (m < 0) then
            info = 3
        else if (n < 0) then
            info = 4
        else if (k < 0) then
            info = 5
        else if (lda < max(1, nrowa)) then
            info = 8
        else if (ldb < max(1, nrowb)) then
            info = 10
        else if (ldc < max(1, int(m))) then
            info = 13
        end if
        if (info /= 0) call ckl_blas_abort('SGEMM ', info)

        call ckl_sgemm_host_ptr(opa, opb, int(m), int(n), int(k), alpha, &
                                a, int(lda), b, int(ldb), beta, c, int(ldc))
    end subroutine sgemm_alias

    subroutine sgemv_alias(trans, m, n, alpha, a, lda, x, incx, beta, y, incy, ltrans) &
        bind(C, name="sgemv_")
        character(kind=c_char), intent(in) :: trans
        integer(c_int), intent(in) :: m, n, lda, incx, incy
        real(c_float), intent(in) :: alpha, beta
        type(c_ptr), value :: a, x, y
        integer(c_size_t), value :: ltrans

        integer(c_int) :: opa
        logical :: ok
        integer :: info


        call ckl_blas_op_code(trans, opa, ok)
        if (.not. ok) call ckl_blas_abort('SGEMV ', 1)

        info = 0
        if (m < 0) then
            info = 2
        else if (n < 0) then
            info = 3
        else if (lda < max(1, int(m))) then
            info = 6
        else if (incx == 0) then
            info = 8
        else if (incy == 0) then
            info = 11
        end if
        if (info /= 0) call ckl_blas_abort('SGEMV ', info)

        call ckl_sgemv_host_ptr(opa, int(m), int(n), alpha, a, int(lda), &
                                x, int(incx), beta, y, int(incy))
    end subroutine sgemv_alias

    subroutine strsm_alias(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb, &
                           lside_, luplo_, ltransa_, ldiag_) bind(C, name="strsm_")
        character(kind=c_char), intent(in) :: side, uplo, transa, diag
        integer(c_int), intent(in) :: m, n, lda, ldb
        real(c_float), intent(in) :: alpha
        type(c_ptr), value :: a, b
        integer(c_size_t), value :: lside_, luplo_, ltransa_, ldiag_

        logical :: left, lower, notrans, unitdiag, ok
        integer(c_int) :: opa
        integer :: info, nrowa


        info = 0
        left = (side == 'L' .or. side == 'l')
        if (.not. (left .or. side == 'R' .or. side == 'r')) info = 1

        lower = (uplo == 'L' .or. uplo == 'l')
        if (info == 0 .and. .not. (lower .or. uplo == 'U' .or. uplo == 'u')) info = 2

        call ckl_blas_op_code(transa, opa, ok)
        if (info == 0 .and. .not. ok) info = 3
        notrans = (opa == CKL_OP_N)

        unitdiag = (diag == 'U' .or. diag == 'u')
        if (info == 0 .and. .not. (unitdiag .or. diag == 'N' .or. diag == 'n')) info = 4

        if (left) then
            nrowa = int(m)
        else
            nrowa = int(n)
        end if
        if (info == 0) then
            if (m < 0) then
                info = 5
            else if (n < 0) then
                info = 6
            else if (lda < max(1, nrowa)) then
                info = 9
            else if (ldb < max(1, int(m))) then
                info = 11
            end if
        end if
        if (info /= 0) call ckl_blas_abort('STRSM ', info)

        call ckl_strsm_host_ptr(left, lower, notrans, unitdiag, int(m), int(n), alpha, &
                                a, int(lda), b, int(ldb))
    end subroutine strsm_alias

end module ckl_blas
