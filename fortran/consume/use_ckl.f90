! Smallest program that proves the installed tree is usable from a plain
! gfortran command line. Nothing here knows where the build tree was.
!
!   pkg-config --cflags ckl   gives the module path
!   pkg-config --libs ckl     gives the link line
!
! See docs/fortran.md for the exact two line invocation.

program use_ckl
    use ckl
    implicit none
    type(c_ptr) :: h
    integer(c_int) :: rc

    rc = ckl_create(h)
    if (rc /= CKL_STATUS_SUCCESS) then
        write (*, '(a)') 'skip: no CUDA device ('//ckl_error_text(rc)//')'
        stop 77
    end if
    write (*, '(a)') 'installed CKL version '//ckl_version_text()
    rc = ckl_destroy(h)
end program use_ckl
