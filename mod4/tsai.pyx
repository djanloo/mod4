# distutils: language = c++
from libcpp cimport bool

from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin

from utils import quad_int, get_tridiag
from utils cimport tridiag

from diffeq cimport a, sigma_squared_full


cpdef tsai_FV(double [:] p0, double x, dict physical_params, dict integration_params):
    ## Time
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)

    ## Space
    cdef double Lv,dv
    Lv,dv = map(integration_params.get, ["Lv", "dv"])

    cdef int  M = int(Lv/dv)
    cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

    cdef int time_index = 0, j = 0, k=0, m=0

    cdef double [:] p = p0.copy(), p_new = p0.copy()
    cdef double [:] P = np.zeros(len(p0)-1)

    # Computation of initial cell averages
    for j in range(M-1):
        P[j] = 0.5*(p[j] + p[j+1])

    cdef double theta = 0.5*dt/dv
    cdef double eta = 0.25*dt/dv**2

    # Declarations of the diagonals
    cdef double [:] lower, diagonal, upper, b
    lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

    cdef double [:, :] g = np.zeros((2,3)), s = np.zeros((2,3)), futures = np.zeros((2,3)), pasts = np.zeros((2,3))
    # NOTE:
    # NEVER call futures[0,1] or futures[1, 1]
    # central space does not exist

    #### Equivalence map ####
    # futures[0,0] is (a_-1)_{i-1}
    # futures[0,1] --------------
    # futures[0,2] is (a_1)_{i-1}
    # futures[1,0] is (a_-1)_{i}
    # futures[1,1] ------------
    # futures[1,2] is (a_1)_{i}

    for time_index in range(n_steps):
        time = t0 + time_index*dt

        ## Updates grid points
        for j in range(M):

            # Coefficients of equation 8
            for k in range(3):
                # k = (left, here, right)
                for m in range(2):
                    # m = (now, next)
                    s[m,k] = eta*sigma_squared_full(x, v[j] + (k-1)*dv, time + m*dt, physical_params)
                    g[m,k] = theta*a(x, v[j] + (k-1)*dv, time + m*dt, physical_params)

            for k in range(3):
                # K swithces coefficient (see equivalence map)
                for m in range(2):
                    # m is i-1 or i 
                    if k == 0:
                        futures[m,k] = (g[1, m] + 2*s[1, m+1] + 4*s[1, m])
                        pasts[m, k]  = (g[0, m] + 2*s[0, m+1] + 4*s[0, m])
                    if k == 1:
                        pasts[m,k] = 1 - 6*s[0, m+1] - 6*s[0, m]
                    if k == 2:
                        futures[m,k] =  (-g[1, m+1] + 4*s[1, m+1] + 2*s[1, m])
                        pasts[m,k]   =  (-g[0, m+1] + 4*s[0, m+1] + 2*s[0, m])

                    denom = 1 + 6*s[1, m+1] + 6*s[1, m]
                    pasts[m,k]   /= denom 
                    futures[m,k] /= denom

            # special case: remember lower definition
            # first coefficient should be A_minus = 1 - 3*futures[0,0]
            # but because of the lower definition i gets shifted forward by one
            lower[j]    = 1 - 3*futures[1,0]
            diagonal[j] = 4 - 3*futures[0, 2]-3*futures[1,0] 
            upper[j]    = 1 - 3*futures[1, 2]

            # Interfaces
            b[j]  = p[j]*(3*pasts[0,2] + 3*pasts[1,0])

            if j > 1:
                b[j] += p[j-1]*(3*pasts[0,0])
            if j < M-2:
                b[j] += p[j+1]*(3*pasts[1,2])

            #Averages
            if j!= M-1:
                b[j] += P[j]  *(3*pasts[1,1])
            if j!= 0:
                b[j] += P[j-1]*(3*pasts[0,1])

        # Solves for the values of p on the grid points
        p_new = tridiag(lower, diagonal, upper, b)

        # BC
        # p_new[0] = 0.0
        # p_new[M-1] = 0.0

        # Update of averages
        for j in range(M-1):
            # Coefficients of equation 8
            for k in range(3):
                # k = (left, here, right)
                for m in range(2):
                    # m = (now, next)
                    s[m,k] = eta*sigma_squared_full(x, v[j] + (k-1)*dv, time + m*dt, physical_params)
                    g[m,k] = theta*a(x, v[j] +(k-1)*dv, time + m*dt, physical_params)

            P[j] =  (1 - 6*s[0,2] - 6*s[0,1])*P[j] +\
                    (-g[1,2] + 4*s[1,2] + 2*s[1,1]) *p_new[j+1]  + (-g[0,2] + 4*s[0,2] + 2*s[0,1])  *p[j+1] +\
                    ( g[1,1] + 2*s[1,2] + 4*s[1,1]) *p_new[j]    + ( g[0,1] + 2*s[0,2] + 4*s[0,1])  *p[j]

            P[j] /= 1 + 6*s[1,2] + 6*s[1, 1]

        p = p_new.copy()
    return p

cpdef tsai_I(double [:] p0, double x, dict physical_params, dict integration_params):
    ## Time
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)

    ## Space
    cdef double Lv,dv
    Lv,dv = map(integration_params.get, ["Lv", "dv"])

    cdef int  M = int(Lv/dv)
    cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

    cdef int time_index = 0, j = 0

    cdef double [:] p = p0.copy(), p_new = p0.copy()
    cdef double [:] P = np.zeros(len(p0)-1)

    # Computation of initial cell averages
    for j in range(M-1):
        P[j] = 0.5*(p[j] + p[j+1])

    cdef double theta = dt/dv
    cdef double eta = dt/dv**2
    cdef double s_here, s_left, s_right, u_right, u_leftt

    # Declarations of the diagonals
    cdef double [:] lower, diagonal, upper, b
    lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

    for time_index in range(n_steps):
        time = t0 + time_index*dt

        # Evolution of values at the border
        # NOTE: only works for sigma constant in time
        for j in range(M):
            s_here  = eta*sigma_squared_full(x,v[j],      time, physical_params)
            s_left  = eta*sigma_squared_full(x,v[j] - dv, time, physical_params)
            s_right = eta*sigma_squared_full(x,v[j] + dv, time, physical_params)

            u_right = theta*a(x,v[j]+dv,time, physical_params)
            u_leftt = theta*a(x,v[j]-dv,time, physical_params)
            u_heree = theta*a(x, v[j],  time, physical_params)

            lower[j]    = 1.0/3.0 - u_heree - 2*s_right - 4*s_here
            diagonal[j] = 4.0/3.0 - 2*s_right -8*s_here - 2*s_left
            upper[j]    = 1.0/3.0 + u_right - 4*s_right + 2*s_here
            
            b[j] = 0.0
            if j !=  M-1:
                b[j] += P[j]   *(1-6*s_right - 6*s_here)
            if j != 0:
                b[j] += P[j-1]*(1-6*s_here - 6*s_left)

        p_new = tridiag(lower, diagonal, upper, b)

        # BC
        p_new[0] = 0.0
        p_new[M-1] = 0.0

        # Evolution of the mean values
        for j in range(M-1):
            s_here  = eta*sigma_squared_full(x,v[j],    time, physical_params)
            s_left  = eta*sigma_squared_full(x,v[j]-dv, time, physical_params)
            s_right = eta*sigma_squared_full(x,v[j]+dv, time, physical_params)

            P[j] =  P[j]*(1 - 6*s_right - 6*s_here) 
            P[j] += p_new[j+1]*(-theta*a(x,v[j] + dv, time,physical_params) + 4*s_right + 2*s_here)
            P[j] += p_new[j]  *( theta*a(x,v[j],      time,physical_params) + 2*s_right + 4*s_here)
        for j in range(M-1):
            P[j] = 0.5*(p[j] + p[j+1])
        p = p_new.copy()
    return p


cpdef tsai_E(double [:] p0, double x, dict physical_params, dict integration_params):
    ## Time
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)

    ## Space
    cdef double Lv,dv
    Lv,dv = map(integration_params.get, ["Lv", "dv"])

    cdef int  M = int(Lv/dv)
    cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

    cdef int time_index = 0, j = 0

    cdef double [:] p = p0.copy(), p_new = p0.copy()
    cdef double [:] P = np.zeros(len(p0)-1)

    # Computation of initial cell averages
    for j in range(M-1):
        P[j] = 0.5*(p[j] + p[j+1])

    cdef double theta = dt/dv
    cdef double eta = dt/dv**2
    cdef double s_here, s_left, s_right, u_right, u_leftt

    # Declarations of the diagonals
    cdef double [:] lower, diagonal, upper, b
    lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

    for time_index in range(n_steps):
        time = t0 + time_index*dt

        # Evolution of values at the border
        # NOTE: only works for sigma constant in time
        for j in range(M):
            s_here  = eta*sigma_squared_full(x,v[j],      time, physical_params)
            s_left  = eta*sigma_squared_full(x,v[j] - dv, time, physical_params)
            s_right = eta*sigma_squared_full(x,v[j] + dv, time, physical_params)

            u_right = theta*a(x,v[j]+dv,time, physical_params)
            u_leftt = theta*a(x,v[j]-dv,time, physical_params)
            u_heree = theta*a(x, v[j],  time, physical_params)

            lower[j]    = 1.0/3.0 
            diagonal[j] = 4.0/3.0
            upper[j]    = 1.0/3.0
            
            b[j] = p[j]*(2*s_right +8*s_here + 2*s_left)
            if j !=  M-1:
                b[j] += p[j+1]*(- u_right + 4*s_right + 2*s_here)
                b[j] += P[j]  *(  1-6*s_right - 6*s_here)
            if j != 0:
                b[j] += p[j-1]*( u_leftt + 2*s_here + 4*s_left)
                b[j] += P[j-1]*(1-6*s_here - 6*s_left)

        p_new = tridiag(lower, diagonal, upper, b)
        # BC
        p_new[0] = 0.0
        p_new[M-1] = 0.0

        # Evolution of the mean values
        for j in range(M-1):
            s_here  = eta*sigma_squared_full(x,v[j],    time, physical_params)
            s_left  = eta*sigma_squared_full(x,v[j]-dv, time, physical_params)
            s_right = eta*sigma_squared_full(x,v[j]+dv, time, physical_params)

            P[j] =  P[j]*(1 - 6*s_right - 6*s_here) 
            P[j] += p[j+1]*(-theta*a(x,v[j] + dv, time,physical_params) + 4*s_right + 2*s_here)
            P[j] += p[j]  *( theta*a(x,v[j],      time,physical_params) + 2*s_right + 4*s_here)
        # for j in range(M-1):
        #     P[j] = 0.5*(p[j] + p[j+1])
        p = p_new.copy()
    return p

cpdef tsai_2(double [:] p0, double x, dict physical_params, dict integration_params):
    ## Time
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)

    ## Space
    cdef double Lv,dv
    Lv,dv = map(integration_params.get, ["Lv", "dv"])

    cdef int  M = int(Lv/dv)
    cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

    cdef int time_index = 0, j = 0

    cdef double [:] p = p0.copy(), p_new = p0.copy()
    cdef double [:] P = np.zeros(len(p0)-1)

    # Computation of initial cell averages
    for j in range(M-1):
        P[j] = 0.5*(p[j] + p[j+1])

    cdef double theta = dt/dv
    cdef double eta = 0.5*dt/dv**2
    cdef double s_here, s_left, s_right, u_right, u_leftt

    # Declarations of the diagonals
    cdef double [:] lower, diagonal, upper, b
    lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)
    
    cdef double [:] which_prob

    for time_index in range(n_steps):
        time = t0 + time_index*dt

        # Evolution of values at the border
        # NOTE: only works for sigma constant in time
        for j in range(M):
            s_here  = eta*sigma_squared_full(x,v[j],      time, physical_params)
            s_left  = eta*sigma_squared_full(x,v[j] - dv, time, physical_params)
            s_right = eta*sigma_squared_full(x,v[j] + dv, time, physical_params)

            u_right = theta*a(x,v[j]+dv,time, physical_params)
            u_leftt = theta*a(x,v[j]-dv,time, physical_params)
            u_heree = theta*a(x, v[j],  time, physical_params)

            lower[j]    = 1.0/3.0 
            diagonal[j] = 4.0/3.0
            upper[j]    = 1.0/3.0
            
            b[j] = p[j]*(2*s_right +8*s_here + 2*s_left)
            if j !=  M-1:
                b[j] += p[j+1]*(- u_right + 4*s_right + 2*s_here)
                b[j] += P[j]  *(  1-6*s_right - 6*s_here)
            if j != 0:
                b[j] += p[j-1]*( u_leftt + 2*s_here + 4*s_left)
                b[j] += P[j-1]*(1-6*s_here - 6*s_left)

        p_new = tridiag(lower, diagonal, upper, b)
        # BC
        p_new[0] = 0.0
        p_new[M-1] = 0.0

        # Evolution of the mean values
        for j in range(M-1):

            for when in [0.0, 1.0]:
                s_here  = eta*sigma_squared_full(x,v[j],    time + when*dt, physical_params)
                # s_left  = eta*sigma_squared_full(x,v[j]-dv, time + when*dt, physical_params)
                s_right = eta*sigma_squared_full(x,v[j]+dv, time + when*dt, physical_params)

                u_right = theta*a(x,v[j]+dv,time + when*dt, physical_params)
                # u_leftt = theta*a(x,v[j]-dv,time + when*dt, physical_params)
                u_heree = theta*a(x,v[j],   time + when*dt, physical_params)
                
                if when == 0.0:
                    print("Defined P")
                    P[j] =  P[j] * (1 - 6*s_right - 6*s_here)

                if when == 0.0:
                    print("set prob to current")
                    which_prob = p 
                else:
                    print("set prob to future")
                    which_prob = p_new 
                
                P[j] += which_prob[j]  *(  u_heree + 2*s_right + 4*s_here)
                P[j] += which_prob[j+1]*( -u_right + 4*s_right + 2*s_here)

            P[j] /= 1.0 + 6*s_right + 6*s_here

        p = p_new.copy()
    return p