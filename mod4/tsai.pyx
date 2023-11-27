# distutils: language = c++
from libcpp cimport bool

from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin

from mod4.utils import quad_int, get_tridiag, get_lin_mesh
from mod4.utils cimport tridiag

from mod4.diffeq cimport a, sigma_squared_full


cpdef tsai_1D_v(double [:] p0, double [:] P0, double x, dict physical_params, dict integration_params):
    ## Time
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)

    ## Space
    cdef double Lv,dv
    Lv,dv = map(integration_params.get, ["Lv", "dv"])

    cdef int  M = int(Lv/dv) + 1
    cdef double [:] v = get_lin_mesh(integration_params)

    cdef int time_index = 0, j = 0, k=0, m=0

    cdef double [:] p = p0.copy(), p_new = p0.copy()
    cdef double [:] P = P0.copy()

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
        p_new[0] = 0.0
        p_new[M-1] = 0.0

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
    return p, P

cpdef tsai_1D_x(double [:] p0, double [:] P0, double v, dict physical_params, dict integration_params):
    ## Time
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)

    ## Space
    cdef double Lx,dx
    Lx,dx = map(integration_params.get, ["Lx", "dx"])

    cdef int  N = int(Lx/dx) + 1    

    cdef int time_index = 0, i = 0, k=0, m=0

    # print(np.array(x))

    cdef double [:] p = p0.copy(), p_new = p0.copy()
    cdef double [:] P = P0.copy()

    cdef double theta = 0.5*dt/dx
    # cdef double eta = 0.25*dt/dx**2

    # Declarations of the diagonals
    cdef double [:] lower, diagonal, upper, b
    lower, diagonal, upper, b = np.ones(N), np.ones(N), np.ones(N), np.ones(N)

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
        for i in range(N):

            ## BULK
            # Coefficients of equation 8
            for k in range(3):
                # k = (left, here, right)
                for m in range(2):
                    # m = (now, next)
                    g[m,k] = theta*v

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
            lower[i]    = 1 - 3*futures[1,0]
            diagonal[i] = 4 - 3*futures[0, 2]-3*futures[1,0] 
            upper[i]    = 1 - 3*futures[1, 2]

            # Interfaces
            b[i]  = p[i]*(3*pasts[0,2] + 3*pasts[1,0])

            if i > 1:
                b[i] += p[i-1]*(3*pasts[0,0])
            if i < N-2:
                b[i] += p[i+1]*(3*pasts[1,2])

            #Averages
            if i!= N-1:
                b[i] += P[i]  *(3*pasts[1,1])
            if i!= 0:
                b[i] += P[i-1]*(3*pasts[0,1])

        ## BCs left
        lower[0] = 0.0
        upper[0] = 0.0
        diagonal[0] = 1.0
        b[0] = 0.0

        ## BCs right
        lower[N-2] = 0.0
        upper[N-2] = 0.0
        diagonal[N-1] = 1.0
        b[N-1] = 0.0


        # Solves for the values of p on the grid points
        p_new = tridiag(lower, diagonal, upper, b)
        # print(f"LEFT {p[0]}, RIGHT {p[N-1]}")

        # Update of averages
        for i in range(N-1):
            # Coefficients of equation 8
            for k in range(3):
                # k = (left, here, right)
                for m in range(2):
                    # m = (now, next)
                    g[m,k] = theta*v
            P[i] =  P[i] +\
                    (-g[1,2]) *p_new[i+1]  + (-g[0,2] )  *p[i+1] +\
                    ( g[1,1]) *p_new[i]    + ( g[0,1] )  *p[i]

        # Test on P- BCs
        P[0] = 0.0
        P[N-2] = 0.0
        
        p = p_new.copy()
    return p, P

# cpdef tsai_I(double [:] p0, double x, dict physical_params, dict integration_params):
#     ## Time
#     cdef double dt   = integration_params['dt']
#     cdef int n_steps = integration_params['n_steps']
#     cdef double t0    = physical_params.get('t0', 0.0)

#     ## Space
#     cdef double Lv,dv
#     Lv,dv = map(integration_params.get, ["Lv", "dv"])

#     cdef int  M = int(Lv/dv)
#     cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#     cdef int time_index = 0, j = 0

#     cdef double [:] p = p0.copy(), p_new = p0.copy()
#     cdef double [:] P = np.zeros(len(p0)-1)

#     # Computation of initial cell averages
#     for j in range(M-1):
#         P[j] = 0.5*(p[j] + p[j+1])

#     cdef double theta = dt/dv
#     cdef double eta = dt/dv**2
#     cdef double s_here, s_left, s_right, u_right, u_leftt

#     # Declarations of the diagonals
#     cdef double [:] lower, diagonal, upper, b
#     lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

#     for time_index in range(n_steps):
#         time = t0 + time_index*dt

#         # Evolution of values at the border
#         # NOTE: only works for sigma constant in time
#         for j in range(M):
#             s_here  = eta*sigma_squared_full(x,v[j],      time, physical_params)
#             s_left  = eta*sigma_squared_full(x,v[j] - dv, time, physical_params)
#             s_right = eta*sigma_squared_full(x,v[j] + dv, time, physical_params)

#             u_right = theta*a(x,v[j]+dv,time, physical_params)
#             u_leftt = theta*a(x,v[j]-dv,time, physical_params)
#             u_heree = theta*a(x, v[j],  time, physical_params)

#             lower[j]    = 1.0/3.0 - u_heree - 2*s_right - 4*s_here
#             diagonal[j] = 4.0/3.0 - 2*s_right -8*s_here - 2*s_left
#             upper[j]    = 1.0/3.0 + u_right - 4*s_right + 2*s_here
            
#             b[j] = 0.0
#             if j !=  M-1:
#                 b[j] += P[j]   *(1-6*s_right - 6*s_here)
#             if j != 0:
#                 b[j] += P[j-1]*(1-6*s_here - 6*s_left)

#         p_new = tridiag(lower, diagonal, upper, b)

#         # BC
#         p_new[0] = 0.0
#         p_new[M-1] = 0.0

#         # Evolution of the mean values
#         for j in range(M-1):
#             s_here  = eta*sigma_squared_full(x,v[j],    time, physical_params)
#             s_left  = eta*sigma_squared_full(x,v[j]-dv, time, physical_params)
#             s_right = eta*sigma_squared_full(x,v[j]+dv, time, physical_params)

#             P[j] =  P[j]*(1 - 6*s_right - 6*s_here) 
#             P[j] += p_new[j+1]*(-theta*a(x,v[j] + dv, time,physical_params) + 4*s_right + 2*s_here)
#             P[j] += p_new[j]  *( theta*a(x,v[j],      time,physical_params) + 2*s_right + 4*s_here)
#         for j in range(M-1):
#             P[j] = 0.5*(p[j] + p[j+1])
#         p = p_new.copy()
#     return p


# cpdef tsai_E(double [:] p0, double x, dict physical_params, dict integration_params):
#     ## Time
#     cdef double dt   = integration_params['dt']
#     cdef int n_steps = integration_params['n_steps']
#     cdef double t0    = physical_params.get('t0', 0.0)

#     ## Space
#     cdef double Lv,dv
#     Lv,dv = map(integration_params.get, ["Lv", "dv"])

#     cdef int  M = int(Lv/dv)
#     cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#     cdef int time_index = 0, j = 0

#     cdef double [:] p = p0.copy(), p_new = p0.copy()
#     cdef double [:] P = np.zeros(len(p0)-1)

#     # Computation of initial cell averages
#     for j in range(M-1):
#         P[j] = 0.5*(p[j] + p[j+1])

#     cdef double theta = dt/dv
#     cdef double eta = dt/dv**2
#     cdef double s_here, s_left, s_right, u_right, u_leftt

#     # Declarations of the diagonals
#     cdef double [:] lower, diagonal, upper, b
#     lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

#     for time_index in range(n_steps):
#         time = t0 + time_index*dt

#         # Evolution of values at the border
#         # NOTE: only works for sigma constant in time
#         for j in range(M):
#             s_here  = eta*sigma_squared_full(x,v[j],      time, physical_params)
#             s_left  = eta*sigma_squared_full(x,v[j] - dv, time, physical_params)
#             s_right = eta*sigma_squared_full(x,v[j] + dv, time, physical_params)

#             u_right = theta*a(x,v[j]+dv,time, physical_params)
#             u_leftt = theta*a(x,v[j]-dv,time, physical_params)
#             u_heree = theta*a(x, v[j],  time, physical_params)

#             lower[j]    = 1.0/3.0 
#             diagonal[j] = 4.0/3.0
#             upper[j]    = 1.0/3.0
            
#             b[j] = p[j]*(2*s_right +8*s_here + 2*s_left)
#             if j !=  M-1:
#                 b[j] += p[j+1]*(- u_right + 4*s_right + 2*s_here)
#                 b[j] += P[j]  *(  1-6*s_right - 6*s_here)
#             if j != 0:
#                 b[j] += p[j-1]*( u_leftt + 2*s_here + 4*s_left)
#                 b[j] += P[j-1]*(1-6*s_here - 6*s_left)

#         p_new = tridiag(lower, diagonal, upper, b)
#         # BC
#         p_new[0] = 0.0
#         p_new[M-1] = 0.0

#         # Evolution of the mean values
#         for j in range(M-1):
#             s_here  = eta*sigma_squared_full(x,v[j],    time, physical_params)
#             s_left  = eta*sigma_squared_full(x,v[j]-dv, time, physical_params)
#             s_right = eta*sigma_squared_full(x,v[j]+dv, time, physical_params)

#             P[j] =  P[j]*(1 - 6*s_right - 6*s_here) 
#             P[j] += p[j+1]*(-theta*a(x,v[j] + dv, time,physical_params) + 4*s_right + 2*s_here)
#             P[j] += p[j]  *( theta*a(x,v[j],      time,physical_params) + 2*s_right + 4*s_here)
#         # for j in range(M-1):
#         #     P[j] = 0.5*(p[j] + p[j+1])
#         p = p_new.copy()
#     return p

# cpdef tsai_2(double [:] p0, double x, dict physical_params, dict integration_params):

#     ## Time
#     cdef double dt   = integration_params['dt']
#     cdef int n_steps = integration_params['n_steps']
#     cdef double t0    = physical_params.get('t0', 0.0)

#     ## Space
#     cdef double Lv,dv
#     Lv,dv = map(integration_params.get, ["Lv", "dv"])

#     cdef int  M = int(Lv/dv)
#     cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#     cdef int time_index = 0, j = 0

#     cdef double [:] p = p0.copy(), p_new = p0.copy()
#     cdef double [:] P = np.zeros(len(p0)-1)

#     # Computation of initial cell averages
#     for j in range(M-1):
#         P[j] = 0.5*(p[j] + p[j+1])

#     cdef double theta = 0.5*dt/dv
#     cdef double eta   = 0.5*dt/dv**2

#     cdef double s_here, s_left, s_right
#     cdef double u_right, u_leftt, u_heree

#     # Declarations of the diagonals
#     cdef double [:] lower, diagonal, upper, b
#     lower, diagonal, upper, b = np.ones(M), np.ones(M), np.ones(M), np.ones(M)
    
#     cdef double [:] which_prob

#     for time_index in range(n_steps):
#         time = t0 + time_index*dt

#         # Evolution of values at the border
#         # First prepares the tridiagonal matrix 
#         # since factors depend only on future values
#         for j in range(M):
#             s_here  = eta*sigma_squared_full(x,v[j],      time+dt, physical_params)
#             s_left  = eta*sigma_squared_full(x,v[j] - dv, time+dt, physical_params)
#             s_right = eta*sigma_squared_full(x,v[j] + dv, time+dt, physical_params)

#             u_right = theta*a(x, v[j]+dv, time + dt, physical_params)
#             # u_leftt = theta*a(x, v[j]-dv, time + dt, physical_params)
#             u_heree = theta*a(x, v[j],    time + dt, physical_params)

#             lower[j]    = 1.0 - 3*( u_heree   + 2*s_right + 4*s_here)
#             diagonal[j] = 4.0 + 3*( 2*s_right + 8*s_here  + 2*s_left)
#             upper[j]    = 1.0 - 3*(-u_right   + 4*s_right + 2*s_here)
#         # print("lower\n",np.array(lower))
#         # print("diag\n", np.array(diagonal))
#         # print("upper\n",np.array(upper))
 
#         # Then prepares the constant vector
#         # Since factors depend only on present values
#         for j in range(M):
#             s_here  = eta*sigma_squared_full(x,v[j],      time, physical_params)
#             s_left  = eta*sigma_squared_full(x,v[j] - dv, time, physical_params)
#             s_right = eta*sigma_squared_full(x,v[j] + dv, time, physical_params)

#             u_right = theta*a(x, v[j]+dv, time, physical_params)
#             u_leftt = theta*a(x, v[j]-dv, time, physical_params)
#             u_heree = theta*a(x, v[j],    time, physical_params)

#             b[j] = p[j]*(2*s_right +8*s_here + 2*s_left)
#             if j !=  M-1:
#                 b[j] += p[j+1]*(- u_right + 4*s_right + 2*s_here)
#                 b[j] += P[j]  *(  1 - 6*s_right - 6*s_here)
#             if j != 0:
#                 b[j] += p[j-1]*( u_leftt + 2*s_here + 4*s_left)
#                 b[j] += P[j-1]*( 1 - 6*s_here - 6*s_left)

#             b[j] *= 3
#         print(np.array(b))
#         p_new = tridiag(lower, diagonal, upper, b)
#         # BC
#         p_new[0] = 0.0
#         p_new[M-1] = 0.0

#         # Evolution of the mean values
#         for j in range(M-1):

#             for when in [0.0, 1.0]:
#                 print(f"when {when}")
#                 s_here  = eta*sigma_squared_full(x,v[j],    time + when*dt, physical_params)
#                 s_right = eta*sigma_squared_full(x,v[j]+dv, time + when*dt, physical_params)

#                 u_right = theta*a(x,v[j]+dv,time + when*dt, physical_params)
#                 u_heree = theta*a(x,v[j],   time + when*dt, physical_params)
                
#                 if when == 0.0:
#                     P[j] =  P[j] * (1 - 6*s_right - 6*s_here)

#                 if when == 0.0:
#                     which_prob = p 
#                 else:
#                     which_prob = p_new 
                
#                 P[j] += which_prob[j]  *(  u_heree + 2*s_right + 4*s_here)
#                 P[j] += which_prob[j+1]*( -u_right + 4*s_right + 2*s_here)

#             P[j] /= 1.0 + 6*s_right + 6*s_here
#         # exit()
#         p = p_new.copy()
#     return p



def tsai_2D(double [:, :] p0, double [:,:] Px, double [:,:] Pv, dict physical_params, dict integration_params, switch_var=0):
    # return NotImplementedError("This does not work well")  
    ## Time  
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)
    cdef int time_index=0
    cdef double time 

    ## Space
    cdef double Lv,dv
    Lv,dv = map(integration_params.get, ["Lv", "dv"]) 
    cdef int  M = int(Lv/dv)
    cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

    cdef double Lx,dx
    Lx,dx = map(integration_params.get, ["Lx", "dx"])
    cdef int  N = int(Lx/dx)
    cdef double [:] x = np.arange(-int(N)//2, int(N)//2)*dv

    cdef int j=0, i=0, k=0, m=0

    cdef double [:,:] px = p0.copy(), pv = p0.copy()

    cdef double [:] p_new_x, p_new_v 
    p_new_x , p_new_v = np.zeros(N), np.zeros(M)

    cdef double theta = 0.5*dt/dv
    cdef double eta = 0.25*dt/dv**2

    # Declarations of the diagonals
    cdef double [:] lower_v, diagonal_v, upper_v, b_v
    lower_v, diagonal_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M), np.ones(M)

    cdef double [:] lower_x, diagonal_x, upper_x, b_x
    lower_x, diagonal_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N), np.ones(N)

    # Coefficients
    cdef double [:, :] g = np.zeros((2,3)), s = np.zeros((2,3)), futures = np.zeros((2,3)), pasts = np.zeros((2,3))
    cdef double denom
    # print("x",np.array(x))
    # print("v",np.array(v))
    # print("theta",theta)

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

        ########################### update: v ###################### 
    
        # For each value of x_i
        for i in range(1,N-1): # Extrema are Dirichlet-fixed

            for j in range(M):

                # Coefficients of equation 8
                for k in range(3):
                    # k = (left, here, right)
                    for m in range(2):
                        # m = (now, next)
                        s[m,k] = eta*sigma_squared_full(x[i], v[j] + (k-1)*dv, time + m*dt, physical_params)
                        g[m,k] = theta*a(x[i], v[j] + (k-1)*dv, time + m*dt, physical_params)

                for k in range(3):
                    # K swithces coefficient (see equivalence map)
                    for m in range(2):
                        # m is i-1 or i 
                        if k == 0:
                            futures[m,k] = (g[1, m] + 2*s[1, m+1] + 4*s[1, m])
                            pasts[m, k]  = (g[0, m] + 2*s[0, m+1] + 4*s[0, m])
                        if k == 1:
                            pasts[m,k] = 1.0 - 6*s[0, m+1] - 6*s[0, m]
                        if k == 2:
                            futures[m,k] =  (-g[1, m+1] + 4*s[1, m+1] + 2*s[1, m])
                            pasts[m,k]   =  (-g[0, m+1] + 4*s[0, m+1] + 2*s[0, m])

                        denom = 1 + 6*s[1, m+1] + 6*s[1, m]
                        pasts[m,k]   /= denom 
                        futures[m,k] /= denom

                # special case: remember lower definition
                # first coefficient should be A_minus = 1 - 3*futures[0,0]
                # but because of the lower definition i gets shifted forward by one
                lower_v[j]    = 1 - 3*futures[1,0]
                diagonal_v[j] = 4.0 - 3*futures[0, 2]-3*futures[1,0] 
                upper_v[j]    = 1 - 3*futures[1, 2]

                # Interfaces
                b_v[j]  = pv[j,i]*(3*pasts[0,2] + 3*pasts[1,0])

                if j > 1:
                    b_v[j] += pv[j-1,i]*(3*pasts[0,0])
                if j < M-2:
                    b_v[j] += pv[j+1,i]*(3*pasts[1,2])

                #Averages
                if j!= M-1:
                    b_v[j] += Pv[j,i]  *(3*pasts[1,1])
                if j!= 0:
                    b_v[j] += Pv[j-1,i]*(3*pasts[0,1])
                
            ## BCs left
            lower_v[0] = 0.0
            upper_v[0] = 0.0
            diagonal_v[0] = 1.0
            b_v[0] = 0.0

            ## BCs right
            lower_v[M-2] = 0.0
            upper_v[M-2] = 0.0
            diagonal_v[M-1] = 1.0
            b_v[M-1] = 0.0

            # Solves for the values of p on the grid points
            p_new_v = tridiag(lower_v, diagonal_v, upper_v, b_v)
            
            # Update of averages
            for j in range(M-1):
                # Coefficients of equation 8 
                for k in range(3):
                    # k = (left, here, right)
                    for m in range(2):
                        # m = (now, next)
                        s[m,k] = eta*sigma_squared_full(x[i], v[j] + (k-1)*dv, time + m*dt, physical_params)
                        g[m,k] = theta*a(x[i], v[j]+(k-1)*dv, time + m*dt, physical_params)

                Pv[j,i] =  (1 - 6*s[0,2] - 6*s[0,1])*Pv[j,i] +\
                        (-g[1,2] + 4*s[1,2] + 2*s[1,1]) *p_new_v[j+1]  + (-g[0,2] + 4*s[0,2] + 2*s[0,1])  *pv[j+1, i] +\
                        ( g[1,1] + 2*s[1,2] + 4*s[1,1]) *p_new_v[j]    + ( g[0,1] + 2*s[0,2] + 4*s[0,1])  *pv[j, i]

                Pv[j,i] /= 1 + 6*s[1,2] + 6*s[1, 1]

            for j in range(M):
                pv[j, i] = p_new_v[j]

        # # Gives the ball
        px = pv.copy()

        # ###################################### update: x ###################

        # For each value of v_j
        for j in range(1,M-1): # Extrema are Dirichlet-fixed: v=vmin and v=vmax are not evolved

            ## Generates tridiagonal coefficients
            for i in range(N):

                # Coefficients of equation 8
                for k in range(3):
                    # k = (left, here, right)
                    for m in range(2):
                        # m = (now, next)
                        s[m,k] = 0.0 # No diffusion
                        g[m,k] = theta*v[j]

                for k in range(3):
                    # K swithces coefficient (see equivalence map)
                    for m in range(2):
                        # m is i-1 or i 
                        if k == 0:
                            futures[m,k] = (g[1, m] + 2*s[1, m+1] + 4*s[1, m])
                            pasts[m, k]  = (g[0, m] + 2*s[0, m+1] + 4*s[0, m])
                        if k == 1:
                            pasts[m,k] = 1.0 - 6*s[0, m+1] - 6*s[0, m]
                        if k == 2:
                            futures[m,k] =  (-g[1, m+1] + 4*s[1, m+1] + 2*s[1, m])
                            pasts[m,k]   =  (-g[0, m+1] + 4*s[0, m+1] + 2*s[0, m])

                        denom = 1.0 + 6*s[1, m+1] + 6*s[1, m]
                        pasts[m,k]   /= denom 
                        futures[m,k] /= denom

                # special case: remember lower definition
                # first coefficient should be A_minus = 1 - 3*futures[0,0]
                # but because of the lower definition i gets shifted forward by one
                lower_x[i]    = 1.0 - 3*futures[1,0]
                diagonal_x[i] = 4.0 - 3*futures[0, 2]-3*futures[1,0] 
                upper_x[i]    = 1.0 - 3*futures[1, 2]

                # Interfaces
                b_x[i]  = px[j,i]*(3*pasts[0,2] + 3*pasts[1,0])

                if i > 1:
                    b_x[i] += px[j,i-1]*(3*pasts[0,0])
                if i < N-2:
                    b_x[i] += px[j,i+1]*(3*pasts[1,2])

                #Averages
                if i!= N-1:
                    b_x[i] += Px[j, i]  *(3*pasts[1,1])
                if i!= 0:
                    b_x[i] += Px[j, i-1]*(3*pasts[0,1])


            ## BCs left
            lower_x[0] = 0.0
            upper_x[0] = 0.0
            diagonal_x[0] = 1.0
            b_x[0] = 0.0

            ## BCs right
            lower_x[N-2] = 0.0
            upper_x[N-2] = 0.0
            diagonal_x[N-1] = 1.0
            b_x[N-1] = 0.0

            # Solves for the values of p on the grid points
            p_new_x = tridiag(lower_x, diagonal_x, upper_x, b_x) 

            # Update of averages
            for i in range(N-1):
                # Coefficients of equation 8
                for k in range(3):
                    # k = (left, here, right)
                    for m in range(2):
                        # m = (now, next)
                        s[m,k] = 0.0 # No diffusion
                        g[m,k] = theta*v[j]

                Px[j,i] =  (1 - 6*s[0,2] - 6*s[0,1])*Px[j,i] +\
                        (-g[1,2] + 4*s[1,2] + 2*s[1,1]) *p_new_x[i+1]  + (-g[0,2] + 4*s[0,2] + 2*s[0,1])  *px[j,i+1] +\
                        ( g[1,1] + 2*s[1,2] + 4*s[1,1]) *p_new_x[i]    + ( g[0,1] + 2*s[0,2] + 4*s[0,1])  *px[j,i]

                Px[j,i] /= 1 + 6*s[1,2] + 6*s[1, 1]

            for i in range(N):
                px[j, i] = p_new_x[i]


        # # Gives the ball
        pv = px.copy()
        

    return pv, Px, Pv