! Cranc nicolson approach to burgers equation
program funky

    implicit none
    integer, parameter    :: n = 500
    integer, parameter    :: inner_times = 10
    integer, parameter    :: n_steps = 30
    integer               :: i, i_time, inner_t, j
    real*8, dimension(n)  :: diag, outer_diag, const_vec, x, u
    real*8                :: alpha, beta, left, right, dx, dt, nu

    dx = 0.002
    dt = 0.001
    nu = 0.0006

    alpha   = 0.5*nu*dt/dx**2
    beta    = 0.5*dt/dx

    print *, "alpha", alpha, "beta", beta

    diag       = 1 + 2*alpha
    outer_diag = - alpha

    ! initialization
    do i=1, n
        u(i) = exp(-((dx*i - 0.5)/0.1)**2)
    end do
    
    do i_time=1, n_steps

        do i=1,n
            write(10, *) i*dx, i_time*dt, u(i)
        end do

        write(10, *) ""

        do inner_t=1, inner_times
            do i=1, n
                
                if (i == 1) then
                    left = u(n)
                else 
                    left = u(i-1)
                endif

                if (i == n) then
                    right = u(1)
                else
                    right = u(i+1)
                endif

                ! print *, left, right
                const_vec(i)  = (alpha+beta*u(i))*left + (1-2*alpha)*u(i) + (alpha-beta*u(i))*right
                
                ! PBC 
                if (i == 1) then
                    const_vec(i) = const_vec(i) + alpha*u(n)
                endif
                if (i == n) then
                    const_vec(i) = const_vec(i) + alpha*u(1)
                endif
            end do
            ! do j =1,n
            !     print *, "const_vec", j, const_vec(j)
            ! end do
            call tridiag(outer_diag, diag, outer_diag, const_vec, u, n)

        end do
        
    end do

contains
subroutine tridiag(lower_trid, diag_trid, upper_trid, const_vec_trid, x, n_in)

    real*8,     dimension(:),  intent(in)   :: lower_trid, diag_trid, upper_trid, const_vec_trid
    real*8,     dimension(:),  intent(out)  :: x
    integer,    intent(in)                  :: n_in
    real*8,     dimension(:),  allocatable  :: a, b, c, d

    integer :: i,j

    allocate(a(n_in), b(n_in), c(n_in), d(n_in))

    a = lower_trid
    b = diag_trid 
    c = upper_trid
    d = const_vec_trid

    do i =1, n_in - 1
        if (b(i) == 0.0) then
            print *, "Empty diagonal in tridiag before renorm in position", i
        endif
        b(i+1) = b(i+1) - a(i)/b(i)*c(i)
        d(i+1) = d(i+1) - a(i)/b(i)*d(i)
    end do

    do i =1, n_in - 1
        if (b(i) == 0.0) then
            print *, "Empty diagonal in tridiag after renorm"
        endif
    end do

    x(n_in) = d(n_in)/b(n_in)
    do i=n_in-1, 1, -1
        x(i) = (d(i) - c(i)*x(i+1))/b(i)
        if (isnan(x(i))) then
            print *, "Nan found in tridiag"
            do j=1,n_in
                print *, "a", a(j), "b" , b(j), "c", c(j), "d", d(j)
            end do
            stop
        endif
    end do

end subroutine tridiag

end program funky