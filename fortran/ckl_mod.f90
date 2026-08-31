! ISO_C_BINDING interfaces to the CUDA Kernel Library C ABI, plus a small
! Fortran-native layer on top of them.
!
! Two levels live here. The `interface` block below is a one for one binding of
! include/ckl/ckl.h: device pointers are type(c_ptr), statuses are
! integer(c_int), and the named constants mirror every C enum value. Nothing in
! it hides anything, so a caller that wants to manage device residency itself
! gets the C semantics unchanged.
!
! The procedures after `contains` are the convenience layer: they take
! assumed-shape contiguous real(c_float) arrays, work out the leading dimensions
! from the array shapes, and do the c_loc and the transfers. Fortran is column
! major and so is the reference BLAS, so everything here uses CKL_COL_MAJOR and
! no transpose gymnastics.

module ckl
    use, intrinsic :: iso_c_binding
    implicit none
    public

    ! ---------------------------------------------------------------------
    ! ckl_status_t
    ! ---------------------------------------------------------------------
    integer(c_int), parameter :: CKL_STATUS_SUCCESS = 0
    integer(c_int), parameter :: CKL_STATUS_NOT_INITIALIZED = 1
    integer(c_int), parameter :: CKL_STATUS_INVALID_VALUE = 2
    integer(c_int), parameter :: CKL_STATUS_ARCH_MISMATCH = 3
    integer(c_int), parameter :: CKL_STATUS_NOT_SUPPORTED = 4
    integer(c_int), parameter :: CKL_STATUS_ALLOC_FAILED = 5
    integer(c_int), parameter :: CKL_STATUS_EXECUTION_FAILED = 6
    integer(c_int), parameter :: CKL_STATUS_INTERNAL = 7

    ! ---------------------------------------------------------------------
    ! ckl_operation_t
    ! ---------------------------------------------------------------------
    integer(c_int), parameter :: CKL_OP_N = 0
    integer(c_int), parameter :: CKL_OP_T = 1
    integer(c_int), parameter :: CKL_OP_C = 2

    ! ---------------------------------------------------------------------
    ! ckl_layout_t
    ! ---------------------------------------------------------------------
    integer(c_int), parameter :: CKL_ROW_MAJOR = 0
    integer(c_int), parameter :: CKL_COL_MAJOR = 1

    ! ---------------------------------------------------------------------
    ! ckl_datatype_t
    ! ---------------------------------------------------------------------
    integer(c_int), parameter :: CKL_R_32F = 0
    integer(c_int), parameter :: CKL_R_16F = 1
    integer(c_int), parameter :: CKL_R_16BF = 2

    ! ---------------------------------------------------------------------
    ! ckl_memcpy_kind_t
    ! ---------------------------------------------------------------------
    integer(c_int), parameter :: CKL_MEMCPY_H2D = 0
    integer(c_int), parameter :: CKL_MEMCPY_D2H = 1
    integer(c_int), parameter :: CKL_MEMCPY_D2D = 2

    ! ---------------------------------------------------------------------
    ! ckl_algo_t
    ! ---------------------------------------------------------------------
    integer(c_int), parameter :: CKL_ALGO_AUTO = 0
    integer(c_int), parameter :: CKL_ALGO_NAIVE = 1
    integer(c_int), parameter :: CKL_ALGO_TILED = 2
    integer(c_int), parameter :: CKL_ALGO_REGISTER = 3
    integer(c_int), parameter :: CKL_ALGO_CP_ASYNC = 4
    integer(c_int), parameter :: CKL_ALGO_WMMA_FP16 = 5
    integer(c_int), parameter :: CKL_ALGO_WMMA_BF16 = 6
    integer(c_int), parameter :: CKL_ALGO_MMA_PTX = 7
    integer(c_int), parameter :: CKL_ALGO_MMA_LDM = 8
    integer(c_int), parameter :: CKL_ALGO_MMA_OPT = 9
    integer(c_int), parameter :: CKL_ALGO_TILE_FAMILY = 10
    integer(c_int), parameter :: CKL_ALGO_SPLITK = 11
    integer(c_int), parameter :: CKL_ALGO_STREAMK = 12
    integer(c_int), parameter :: CKL_ALGO_CUTLASS = 13
    integer(c_int), parameter :: CKL_ALGO_CUBLAS = 14

    ! Bytes in one real(c_float). Every device size in this module is a count of
    ! elements multiplied by this.
    integer(c_size_t), parameter :: CKL_FLOAT_BYTES = 4_c_size_t

    interface

        ! -----------------------------------------------------------------
        ! Version and diagnostics
        ! -----------------------------------------------------------------
        integer(c_int) function ckl_get_version() bind(C, name="ckl_get_version")
            import :: c_int
        end function ckl_get_version

        type(c_ptr) function ckl_get_version_string() bind(C, name="ckl_get_version_string")
            import :: c_ptr
        end function ckl_get_version_string

        type(c_ptr) function ckl_status_string(s) bind(C, name="ckl_status_string")
            import :: c_ptr, c_int
            integer(c_int), value :: s
        end function ckl_status_string

        integer(c_int) function ckl_last_error(buf, buflen) bind(C, name="ckl_last_error")
            import :: c_int, c_char, c_size_t
            character(kind=c_char), intent(inout) :: buf(*)
            integer(c_size_t), value :: buflen
        end function ckl_last_error

        ! -----------------------------------------------------------------
        ! Device memory
        ! -----------------------------------------------------------------
        integer(c_int) function ckl_device_malloc(dptr, nbytes) bind(C, name="ckl_device_malloc")
            import :: c_int, c_ptr, c_size_t
            type(c_ptr), intent(out) :: dptr
            integer(c_size_t), value :: nbytes
        end function ckl_device_malloc

        integer(c_int) function ckl_device_free(dptr) bind(C, name="ckl_device_free")
            import :: c_int, c_ptr
            type(c_ptr), value :: dptr
        end function ckl_device_free

        integer(c_int) function ckl_memcpy(dst, src, nbytes, kind_) bind(C, name="ckl_memcpy")
            import :: c_int, c_ptr, c_size_t
            type(c_ptr), value :: dst
            type(c_ptr), value :: src
            integer(c_size_t), value :: nbytes
            integer(c_int), value :: kind_
        end function ckl_memcpy

        integer(c_int) function ckl_device_synchronize() bind(C, name="ckl_device_synchronize")
            import :: c_int
        end function ckl_device_synchronize

        ! -----------------------------------------------------------------
        ! Handle
        ! -----------------------------------------------------------------
        integer(c_int) function ckl_create(h) bind(C, name="ckl_create")
            import :: c_int, c_ptr
            type(c_ptr), intent(out) :: h
        end function ckl_create

        integer(c_int) function ckl_destroy(h) bind(C, name="ckl_destroy")
            import :: c_int, c_ptr
            type(c_ptr), value :: h
        end function ckl_destroy

        integer(c_int) function ckl_set_stream(h, cuda_stream) bind(C, name="ckl_set_stream")
            import :: c_int, c_ptr
            type(c_ptr), value :: h
            type(c_ptr), value :: cuda_stream
        end function ckl_set_stream

        integer(c_int) function ckl_set_workspace(h, ws, nbytes) bind(C, name="ckl_set_workspace")
            import :: c_int, c_ptr, c_size_t
            type(c_ptr), value :: h
            type(c_ptr), value :: ws
            integer(c_size_t), value :: nbytes
        end function ckl_set_workspace

        ! -----------------------------------------------------------------
        ! GEMM
        ! -----------------------------------------------------------------
        integer(c_int) function ckl_gemm_workspace_size(h, layout, opa, opb, m, n, k, &
                                                        dta, lda, dtb, ldb, dtc, ldc, &
                                                        algo, nbytes) &
            bind(C, name="ckl_gemm_workspace_size")
            import :: c_int, c_int64_t, c_ptr, c_size_t
            type(c_ptr), value :: h
            integer(c_int), value :: layout, opa, opb
            integer(c_int64_t), value :: m, n, k
            integer(c_int), value :: dta
            integer(c_int64_t), value :: lda
            integer(c_int), value :: dtb
            integer(c_int64_t), value :: ldb
            integer(c_int), value :: dtc
            integer(c_int64_t), value :: ldc
            integer(c_int), value :: algo
            integer(c_size_t), intent(out) :: nbytes
        end function ckl_gemm_workspace_size

        integer(c_int) function ckl_gemm_query(h, layout, opa, opb, m, n, k, &
                                               dta, lda, dtb, ldb, dtc, ldc, chosen) &
            bind(C, name="ckl_gemm_query")
            import :: c_int, c_int64_t, c_ptr
            type(c_ptr), value :: h
            integer(c_int), value :: layout, opa, opb
            integer(c_int64_t), value :: m, n, k
            integer(c_int), value :: dta
            integer(c_int64_t), value :: lda
            integer(c_int), value :: dtb
            integer(c_int64_t), value :: ldb
            integer(c_int), value :: dtc
            integer(c_int64_t), value :: ldc
            integer(c_int), intent(out) :: chosen
        end function ckl_gemm_query

        integer(c_int) function ckl_sgemm(h, layout, opa, opb, m, n, k, alpha, &
                                          a, lda, b, ldb, beta, c, ldc) &
            bind(C, name="ckl_sgemm")
            import :: c_int, c_int64_t, c_ptr, c_float
            type(c_ptr), value :: h
            integer(c_int), value :: layout, opa, opb
            integer(c_int64_t), value :: m, n, k
            real(c_float), value :: alpha
            type(c_ptr), value :: a
            integer(c_int64_t), value :: lda
            type(c_ptr), value :: b
            integer(c_int64_t), value :: ldb
            real(c_float), value :: beta
            type(c_ptr), value :: c
            integer(c_int64_t), value :: ldc
        end function ckl_sgemm

        integer(c_int) function ckl_gemm_ex(h, layout, opa, opb, m, n, k, alpha, &
                                            a, dta, lda, b, dtb, ldb, beta, c, dtc, ldc, &
                                            algo, chosen) &
            bind(C, name="ckl_gemm_ex")
            import :: c_int, c_int64_t, c_ptr
            type(c_ptr), value :: h
            integer(c_int), value :: layout, opa, opb
            integer(c_int64_t), value :: m, n, k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: a
            integer(c_int), value :: dta
            integer(c_int64_t), value :: lda
            type(c_ptr), value :: b
            integer(c_int), value :: dtb
            integer(c_int64_t), value :: ldb
            type(c_ptr), value :: beta
            type(c_ptr), value :: c
            integer(c_int), value :: dtc
            integer(c_int64_t), value :: ldc
            integer(c_int), value :: algo
            integer(c_int), intent(out) :: chosen
        end function ckl_gemm_ex

        integer(c_int) function ckl_gemm_strided_batched_ex(h, layout, opa, opb, m, n, k, &
                                                            alpha, a, dta, lda, stride_a, &
                                                            b, dtb, ldb, stride_b, &
                                                            beta, c, dtc, ldc, stride_c, &
                                                            batch_count, algo) &
            bind(C, name="ckl_gemm_strided_batched_ex")
            import :: c_int, c_int32_t, c_int64_t, c_ptr
            type(c_ptr), value :: h
            integer(c_int), value :: layout, opa, opb
            integer(c_int64_t), value :: m, n, k
            type(c_ptr), value :: alpha
            type(c_ptr), value :: a
            integer(c_int), value :: dta
            integer(c_int64_t), value :: lda
            integer(c_int64_t), value :: stride_a
            type(c_ptr), value :: b
            integer(c_int), value :: dtb
            integer(c_int64_t), value :: ldb
            integer(c_int64_t), value :: stride_b
            type(c_ptr), value :: beta
            type(c_ptr), value :: c
            integer(c_int), value :: dtc
            integer(c_int64_t), value :: ldc
            integer(c_int64_t), value :: stride_c
            integer(c_int32_t), value :: batch_count
            integer(c_int), value :: algo
        end function ckl_gemm_strided_batched_ex

    end interface

contains

    ! -------------------------------------------------------------------------
    ! Diagnostics
    ! -------------------------------------------------------------------------

    ! Copies a NUL terminated C string into a Fortran allocatable. Only used on
    ! the version and status names, which are short and statically allocated.
    function ckl_c_string(p) result(s)
        type(c_ptr), intent(in) :: p
        character(len=:), allocatable :: s
        character(kind=c_char), pointer :: raw(:)
        integer, parameter :: cap = 256
        integer :: i, n

        if (.not. c_associated(p)) then
            s = ''
            return
        end if
        call c_f_pointer(p, raw, [cap])
        n = 0
        do i = 1, cap
            if (raw(i) == c_null_char) exit
            n = i
        end do
        allocate (character(len=n) :: s)
        do i = 1, n
            s(i:i) = raw(i)
        end do
    end function ckl_c_string

    function ckl_version_text() result(s)
        character(len=:), allocatable :: s
        s = ckl_c_string(ckl_get_version_string())
    end function ckl_version_text

    ! The status name plus whatever detail the last failing call left on this
    ! thread. This is what a Fortran caller prints; it is the only way to see
    ! the C side message.
    function ckl_error_text(stat) result(s)
        integer(c_int), intent(in) :: stat
        character(len=:), allocatable :: s
        character(kind=c_char) :: buf(512)
        integer(c_int) :: rc
        integer :: i, n

        s = ckl_c_string(ckl_status_string(stat))
        buf = c_null_char
        rc = ckl_last_error(buf, int(size(buf), c_size_t))
        if (rc /= CKL_STATUS_SUCCESS) return
        n = 0
        do i = 1, size(buf)
            if (buf(i) == c_null_char) exit
            n = i
        end do
        if (n == 0) return
        s = s//': '
        do i = 1, n
            s = s//buf(i)
        end do
    end function ckl_error_text

    ! A failed CKL call in a Fortran program has nowhere useful to go, so this
    ! prints the detail and stops rather than letting a wrong answer travel.
    subroutine ckl_check(stat, where)
        integer(c_int), intent(in) :: stat
        character(len=*), intent(in) :: where

        if (stat == CKL_STATUS_SUCCESS) return
        write (*, '(a)') 'CKL failure in '//trim(where)//': '//ckl_error_text(stat)
        error stop 1
    end subroutine ckl_check

    ! -------------------------------------------------------------------------
    ! Device pointer arithmetic
    ! -------------------------------------------------------------------------

    ! Advances a device pointer by nelem real(c_float) elements. Fortran has no
    ! pointer arithmetic, so the address goes through an integer.
    function ckl_ptr_offset(p, nelem) result(q)
        type(c_ptr), intent(in) :: p
        integer(c_int64_t), intent(in) :: nelem
        type(c_ptr) :: q
        integer(c_intptr_t) :: addr

        addr = transfer(p, addr)
        addr = addr + int(nelem, c_intptr_t)*int(CKL_FLOAT_BYTES, c_intptr_t)
        q = transfer(addr, q)
    end function ckl_ptr_offset

    ! -------------------------------------------------------------------------
    ! Allocation and transfer, in units of real(c_float)
    ! -------------------------------------------------------------------------

    subroutine ckl_alloc_floats(dptr, nelem, stat)
        type(c_ptr), intent(out) :: dptr
        integer(c_int64_t), intent(in) :: nelem
        integer(c_int), intent(out) :: stat

        stat = ckl_device_malloc(dptr, int(nelem, c_size_t)*CKL_FLOAT_BYTES)
    end subroutine ckl_alloc_floats

    subroutine ckl_free(dptr, stat)
        type(c_ptr), intent(inout) :: dptr
        integer(c_int), intent(out) :: stat

        stat = ckl_device_free(dptr)
        dptr = c_null_ptr
    end subroutine ckl_free

    ! host to device, rank 1. The dummy is contiguous, so a strided actual is
    ! packed by the compiler before the address is taken.
    subroutine ckl_put(dptr, host, stat)
        type(c_ptr), intent(in) :: dptr
        real(c_float), intent(in), target, contiguous :: host(:)
        integer(c_int), intent(out) :: stat

        if (size(host) == 0) then
            stat = CKL_STATUS_SUCCESS
            return
        end if
        stat = ckl_memcpy(dptr, c_loc(host(1)), &
                          int(size(host), c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_H2D)
    end subroutine ckl_put

    subroutine ckl_get(host, dptr, stat)
        real(c_float), intent(out), target, contiguous :: host(:)
        type(c_ptr), intent(in) :: dptr
        integer(c_int), intent(out) :: stat

        if (size(host) == 0) then
            stat = CKL_STATUS_SUCCESS
            return
        end if
        stat = ckl_memcpy(c_loc(host(1)), dptr, &
                          int(size(host), c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_D2H)
    end subroutine ckl_get

    ! host to device, rank 2. Column major and contiguous, so the whole array is
    ! one transfer and the device leading dimension is size(host, 1).
    subroutine ckl_put2(dptr, host, stat)
        type(c_ptr), intent(in) :: dptr
        real(c_float), intent(in), target, contiguous :: host(:, :)
        integer(c_int), intent(out) :: stat

        if (size(host) == 0) then
            stat = CKL_STATUS_SUCCESS
            return
        end if
        stat = ckl_memcpy(dptr, c_loc(host(1, 1)), &
                          int(size(host), c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_H2D)
    end subroutine ckl_put2

    subroutine ckl_get2(host, dptr, stat)
        real(c_float), intent(out), target, contiguous :: host(:, :)
        type(c_ptr), intent(in) :: dptr
        integer(c_int), intent(out) :: stat

        if (size(host) == 0) then
            stat = CKL_STATUS_SUCCESS
            return
        end if
        stat = ckl_memcpy(c_loc(host(1, 1)), dptr, &
                          int(size(host), c_size_t)*CKL_FLOAT_BYTES, CKL_MEMCPY_D2H)
    end subroutine ckl_get2

    ! -------------------------------------------------------------------------
    ! Fortran native GEMM on host arrays
    ! -------------------------------------------------------------------------

    ! C := alpha * op(A) * op(B) + beta * C with A, B and C plain Fortran arrays.
    ! The shapes carry the dimensions: op(A) is m by k, op(B) is k by n, C is m
    ! by n, and the leading dimensions are the first extents. Device buffers are
    ! allocated, filled, used and released inside; nothing survives the call.
    subroutine ckl_sgemm_arrays(h, opa, opb, alpha, a, b, beta, c, stat)
        type(c_ptr), intent(in) :: h
        integer(c_int), intent(in) :: opa, opb
        real(c_float), intent(in) :: alpha, beta
        real(c_float), intent(in), contiguous :: a(:, :), b(:, :)
        real(c_float), intent(inout), contiguous :: c(:, :)
        integer(c_int), intent(out) :: stat

        integer(c_int64_t) :: m, n, k, lda, ldb, ldc
        type(c_ptr) :: da, db, dc
        integer(c_int) :: rc

        lda = int(size(a, 1), c_int64_t)
        ldb = int(size(b, 1), c_int64_t)
        ldc = int(size(c, 1), c_int64_t)
        m = int(size(c, 1), c_int64_t)
        n = int(size(c, 2), c_int64_t)
        if (opa == CKL_OP_N) then
            k = int(size(a, 2), c_int64_t)
        else
            k = int(size(a, 1), c_int64_t)
        end if

        da = c_null_ptr
        db = c_null_ptr
        dc = c_null_ptr

        call ckl_alloc_floats(da, int(size(a), c_int64_t), stat)
        if (stat /= CKL_STATUS_SUCCESS) return
        call ckl_alloc_floats(db, int(size(b), c_int64_t), stat)
        if (stat /= CKL_STATUS_SUCCESS) then
            call ckl_free(da, rc)
            return
        end if
        call ckl_alloc_floats(dc, int(size(c), c_int64_t), stat)
        if (stat /= CKL_STATUS_SUCCESS) then
            call ckl_free(da, rc)
            call ckl_free(db, rc)
            return
        end if

        call ckl_put2(da, a, stat)
        if (stat == CKL_STATUS_SUCCESS) call ckl_put2(db, b, stat)
        if (stat == CKL_STATUS_SUCCESS .and. beta /= 0.0_c_float) call ckl_put2(dc, c, stat)
        if (stat == CKL_STATUS_SUCCESS) then
            stat = ckl_sgemm(h, CKL_COL_MAJOR, opa, opb, m, n, k, alpha, &
                             da, lda, db, ldb, beta, dc, ldc)
        end if
        if (stat == CKL_STATUS_SUCCESS) stat = ckl_device_synchronize()
        if (stat == CKL_STATUS_SUCCESS) call ckl_get2(c, dc, stat)

        call ckl_free(da, rc)
        call ckl_free(db, rc)
        call ckl_free(dc, rc)
    end subroutine ckl_sgemm_arrays

end module ckl
