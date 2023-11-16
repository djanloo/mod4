# distutils: language = c++
from libcpp cimport bool

from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin

cimport utils
from utils import quad_int, get_tridiag
from utils cimport tridiag

cimport diffeq
from diffeq cimport a, sigma_squared

cdef double sigma_squared_full(double x, double v, double t, dict physical_params):
    return physical_params['sigma_squared']

cpdef tsai1d(double [:] p0, double x, dict physical_params, dict integration_params):
  ## Time
  cdef double dt   = integration_params['dt']
  cdef unsigned int n_steps = integration_params['n_steps']
  cdef double t0    = physical_params.get('t0', 0.0)

  ## Space
  cdef double Lv,dv
  Lv,dv = map(integration_params.get, ["Lv", "dv"])

  cdef unsigned int  M = int(Lv/dv)
  cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

  cdef unsigned int time_index = 0, j = 0

  cdef double [:] p = p0.copy()
  cdef double theta = dt/dv
  cdef double s = 0.0

  # Declarations of the diagonals
  cdef double [:] lower, diagonal, upper, b
  lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

  for time_index in range(n_steps):
    time = t0 + time_index*dt
    s = 0.5*dt/dv**2*sigma_squared(x, time, physical_params)

    for j in range(M):
        a_now_right  = theta*a(x, v[j] + dv,  time,       physical_params)
        a_now_left   = theta*a(x, v[j] - dv,  time,       physical_params)
        
        a_next_right = theta*a(x, v[j] + dv,  time+dt,    physical_params)
        a_next_left  = theta*a(x, v[j] - dv,  time+dt,    physical_params)

        a_next_here  = theta*a(x, v[j] ,      time+dt,    physical_params)

        lower[j]    = 1 - 1.5*a_next_here   - 3*s 
        diagonal[j] = 4 + 1.5*(a_next_left  - a_next_right) + 6*s
        upper[j]    = 1 + 1.5*a_next_right  - 3*s
      
        if j != 0 and j != M-1:
            b[j] =   (1 - 1.5*a_now_right + 3*s)* p[j+1] 
            b[j] +=  (4 - 1.5*(a_now_left - a_now_right) - 6*s)* p[j]
            b[j] +=  (1 + 1.5*a_now_left  + 3*s)* p[j-1]
        else:
            b[j] = 0

    p = tridiag(lower, diagonal, upper, b)
  return p

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

    cdef double theta = 2*dt/dv
    cdef double eta = dt/dv**2

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
                        futures[m,k] = 0.5*(g[1, m] + 2*s[1, m+1] + 4*s[1, m])
                        pasts[m, k]  = 0.5*(g[0, m] + 2*s[0, m+1] + 4*s[0, m])
                    if k == 1:
                        pasts[m,k] = 1 - 3*s[0, m+1] - 3*s[0, m]
                    if k == 2:
                        futures[m,k] = -0.5*(g[1, m+1] - 4*s[1, m+1] - 2*s[1, m])
                        pasts[m,k]   = 0.5*(-g[0, m+1] + 4*s[0, m+1] + 2*s[0, m])

                    denom = 1 + 3*s[1, m+1] + 3*s[1, m]
                    pasts[m,k]   /= denom 
                    futures[m,k] /= denom

            print(f"j = {j}")
            print(f"denom = {denom}")
            print(f"eta = {eta}")
            print(f"theta = {theta}")
            print(f"s = {np.array(s)}")
            print(f"g = {np.array(g)}")
            print(f"futures = {np.array(futures)}")
            print(f"pasts = {np.array(pasts)}")

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
        p_new[0] = 0.0
        p_new[M-1] = 0.0

        # # Update of averages
        # for j in range(M-1):
        #     # Coefficients of equation 8
        #     for k in range(3):
        #         # k = (left, here, right)
        #         for m in range(2):
        #             # m = (now, next)
        #             s[m,k] = eta*sigma_squared_full(x, v[j] + (k-1)*dv, time + m*dt, physical_params)
        #             g[m,k] = theta*a(x, v[j] +(k-1)*dv, time + m*dt, physical_params)

        #     for k in range(3):
        #         # K swithces coefficient
        #         # m is i-1 or i
        #         for m in range(2):
        #             if k == 0:
        #                 futures[m,k] = 0.5*(g[1, m] + 2*s[1, m+1] + 4*s[1, m])
        #                 pasts[m, k]  = 0.5*(g[0, m] + 2*s[0, m+1] + 4*s[0, m])
        #             if k==1:
        #                 pasts[m,k] = 1 - 3*s[0, m+1] - 3*s[0, m]
        #             if k == 2:
        #                 futures[m,k] = -0.5*(g[1, m+1] - 4*s[1, m+1] - 2*s[1, m])
        #                 pasts[m,k]   = 0.5*(-g[0, m+1] + 4*s[0, m+1] + 2*s[0, m])

        #             denom = 1 + 3*s[1, m+1] + 3*s[1, m]
        #             pasts[m,k]   /= denom 
        #             futures[m,k] /= denom

        #     P[j] = futures[1,0]*p_new[j] + futures[1,2]*p_new[j+1] + pasts[1,0]*p_new[j] + pasts[1,2]*p_new[j+1]            
        #     P[j] += pasts[1,1]*P[j]
        for j in range(M-1):
            P[j] = 0.5*(p_new[j] + p_new[j+1])
        p = p_new.copy()

    return p