# distutils: language = c++
# from libcpp cimport bool
import numpy as np
cimport numpy as np

from mod4.utils cimport get_lin_mesh
from mod4.utils cimport tridiag

from mod4.diffeq cimport a, sigma_squared_full

DEF UNUSED_VAR = 1e20

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

cdef advection_diffusion_matrix(  
                        double [:] lower, double [:] diagonal, double [:] upper, double [:] b,
                        double [:] p, double [:] P,
                        double x, double v, double time,
                        dict physical_params, dict integration_params,
                        unsigned int type
                        ):
    cdef int N = len(diagonal), m = 0, k = 0, i = 0
    cdef double [:, :] g = np.zeros((2,3)), s = np.zeros((2,3)), futures = np.zeros((2,3)), pasts = np.zeros((2,3))
    cdef double theta = 0.0
    cdef double eta = 0.0

    cdef double dx = integration_params['dx'], dv = integration_params['dv'], dt = integration_params['dt']

    if type == 0:
        theta = 0.5*dt/dx

    cdef double [:] vvs = np.zeros(N)
    if type == 1:
        theta = 0.5*dt/dv
        vvs = get_lin_mesh(integration_params)
        eta = 0.5*dt/dv**2

    for i in range(N):
        # Coefficients of equation 8
        for k in range(3):
            # k = (left, here, right)
            for m in range(2):
                # m = (now, next)
                if type == 0:
                    s[m,k] = 0.0
                    g[m,k] = theta*v
                if type == 1:
                    s[m,k] = eta*sigma_squared_full(x, vvs[i] + (k-1)*dv, time + m*dt, physical_params)
                    g[m,k] = theta*a(x, vvs[i] + (k-1)*dv, time + m*dt, physical_params)

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
        lower[i]    = 1 - 3*futures[1,0]
        diagonal[i] = 4.0 - 3*futures[0, 2]-3*futures[1,0] 
        upper[i]    = 1 - 3*futures[1, 2]

        # Interfaces
        b[i]  = p[i]*(3*pasts[0,2] + 3*pasts[1,0])

        if i > 1:
            b[i] += p[i-1]*(3*pasts[0,0])
        if i < N-2:
            b[i] += p[i+1]*(3*pasts[1,2])

        #Averages
        if i != N-1:
            b[i] += P[i]  *(3*pasts[1,1])
        if i != 0:
            b[i] += P[i-1]*(3*pasts[0,1])

cdef update_averages(double [:] P, double [:] p_old, double [:] p_new,
                    double x, double v, double time,
                    dict physical_params, dict integration_params,
                    unsigned int type
                    ):
    cdef int N = len(p_old), m, k, i
    cdef double [:, :] g = np.zeros((2,3)), s = np.zeros((2,3))
    cdef double theta = 0.0
    cdef double eta = 0.0

    cdef double dx = integration_params['dx'], dv = integration_params['dv'], dt = integration_params['dt']

    if type == 0:
        theta = 0.5*dt/dx

    cdef double [:] vvs = np.zeros(N)

    if type == 1:
        theta = 0.5*dt/dv
        vvs = get_lin_mesh(integration_params)
        eta = 0.5*dt/dv**2
        
    for i in range(0, N-1):
        # Coefficients of equation 8 
        for k in range(3):
            # k = (left, here, right)
            for m in range(2):
                # m = (now, next)
                if type == 0:
                    s[m,k] = 0.0
                    g[m,k] = theta*v
                if type == 1:
                    s[m,k] = eta*sigma_squared_full(x, vvs[i] + (k-1)*dv, time + m*dt, physical_params)
                    g[m,k] = theta*a(x, vvs[i] + (k-1)*dv, time + m*dt, physical_params)

        P[i] =  (1.0 - 6*s[0,2] - 6*s[0,1])*P[i] +\
                (-g[1,2] + 4*s[1,2] + 2*s[1,1]) *p_new[i+1]  + (-g[0,2] + 4*s[0,2] + 2*s[0,1])  *p_old[i+1] +\
                ( g[1,1] + 2*s[1,2] + 4*s[1,1]) *p_new[i]    + ( g[0,1] + 2*s[0,2] + 4*s[0,1])  *p_old[i]

        P[i] /= 1.0 + 6*s[1,2] + 6*s[1, 1]
    
    return

cdef brutal_P_update(double [:] P, double [:] p):
    cdef  int N = len(P), i
    for i in range(N):
        P[i] = 0.5*(p[i] + p[i+1])

cdef compensative_P_update_implicit(double [:] P, 
                          x, v, time,
                          dict integration_params, dict physical_params, unsigned int type):
    cdef int B = len(P), k

    cdef double dt = integration_params.get('dt')
    cdef double dx = integration_params.get('dx')
    cdef double dv = integration_params.get('dv')

    cdef double thetino = dt/2/dx
    cdef double etino = physical_params['sigma_squared']*dt/dx**2
    cdef vvs = get_lin_mesh(dict(Lv=integration_params['Lv'], dv=dv))
    cdef double [:] P_old = P.copy(), P_new
    cdef double [:] lower, diagonal, upper

    lower, diagonal, upper = np.ones(B), np.ones(B), np.ones(B)

    if type == 1 and v != UNUSED_VAR:
        print("error, this should be unused")

    if type == 0 and x!= UNUSED_VAR:
        print('error, this should be unused')

    if type == 0:
        for k in range(B):

            lower[k] = - thetino*v
            diagonal[k] = 1.0
            upper[k] = thetino*v

    if type == 1: ## V-advection diffusion on values of Px
        for k in range(1,B-1):
            lower[k] = - thetino*a(x, vvs[k], time,physical_params) - etino
            diagonal[k] = 1 + 2*etino
            upper[k] = thetino*a(x, vvs[k] + dv, time,physical_params) - etino

    set_dirichlet(lower, diagonal, upper, P_old)
    P_new = tridiag(lower, diagonal, upper, P_old)
    for k in range(B):
        P[k] = P_new[k]

cdef compensative_P_update_explicit(double [:] P, 
                          x, v, time,
                          dict integration_params, dict physical_params, unsigned int type):
    cdef int B = len(P), k

    cdef double dt = integration_params.get('dt')
    cdef double dx = integration_params.get('dx')
    cdef double dv = integration_params.get('dv')

    cdef double thetino = dt/2/dx
    cdef double etino = physical_params['sigma_squared']*dt/dx**2
    cdef vvs = get_lin_mesh(dict(Lv=integration_params['Lv'], dv=dv))
    cdef double [:] P_old = P.copy()
    cdef double [:] lower, diagonal, upper, b

    lower, diagonal, upper = np.ones(B), np.ones(B), np.ones(B)

    if type == 1 and v != UNUSED_VAR:
        print("error, this should be unused")

    if type == 0 and x!= UNUSED_VAR:
        print('error, this should be unused')

    if type == 0:
        for k in range(1, B-1):
            P[k] = P_old[k]
            P[k] -= thetino*v*(P[k+1] - P[k-1])

    if type == 1: ## V-advection diffusion on values of Px
        for k in range(1,B-1):
            P[k] = P_old[k]
            P[k] -= thetino*(a(x, vvs[k]+dv, time,physical_params)*P[k+1] -  a(x, vvs[k]-dv, time,physical_params)*P[k-1])
            P[k] += etino*(P[k+1] - 2*P[k] + P[k-1])
     
cdef set_dirichlet( double [:] lower, 
                    double[:] diag, 
                    double[:] upper, 
                    double [:] b):
    cdef int M = len(diag)

    ## BCs left
    lower[0] = 0.0
    upper[0] = 0.0
    diag[0] = 1.0
    b[0] = 0.0

    ## BCs right
    lower[M-2] = 0.0
    upper[M-2] = 0.0
    diag[M-1] = 1.0
    b[M-1] = 0.0
    return

def tsai_2D_leapfrog(double [:, :] p0, double [:,:] P0x, double [:,:] P0v, 
                    dict physical_params, dict integration_params, switch=0):
    ## Time  
    cdef double dt   = integration_params['dt']
    cdef int n_steps = integration_params['n_steps']
    cdef double t0    = physical_params.get('t0', 0.0)
    cdef int time_index=0
    cdef double time 

    ## Space - v
    cdef double Lv,dv
    Lv,dv = map(integration_params.get, ["Lv", "dv"]) 
    cdef int  M = int(Lv/dv)
    cdef double [:] v = np.array(get_lin_mesh(dict(Lv=Lv, dv=dv)))
    cdef int j=0

    ## Space - x
    cdef double Lx,dx
    Lx,dx = map(integration_params.get, ["Lx", "dx"])
    cdef int  N = int(Lx/dx)
    cdef double [:] x = get_lin_mesh(dict(Lx=Lx, dx=dx))
    cdef int i=0

    ## Probabilities
    cdef double [:,:] p = p0.copy(), p_new = p0.copy(), Px = P0x.copy(), Pv = P0v.copy()

    # Declarations of the tridiagonal matrices
    cdef double [:] lower_v, diagonal_v, upper_v, b_v
    lower_v, diagonal_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M), np.ones(M)
    cdef double [:] lower_x, diagonal_x, upper_x, b_x
    lower_x, diagonal_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N), np.ones(N)

    print(f"v0 = {v[0]}, vM = {v[M]}")
    print(f"x0 = {x[0]}, xN = {x[N]}")

    # MAIN LOOP
    for time_index in range(n_steps):
        time = t0 + time_index*dt

        ######################### X - GRID UPDATE ####################
        # For each value of v_j
        if int(5*time) % 2 == 0:
            for j in range(1,M-1): # Extrema are Dirichlet-fixed

                advection_diffusion_matrix( lower_x, diagonal_x, upper_x, b_x, 
                                            p[j, :], Px[j, :], 
                                            UNUSED_VAR, v[j], time,
                                            physical_params, integration_params,
                                            0)
                set_dirichlet(lower_x, diagonal_x, upper_x, b_x)
                p_new[j, :] = tridiag(lower_x, diagonal_x, upper_x, b_x)
            
            ###################### X - AVG UPDATE ##################
            for j in range(M):
                update_averages(Px[j,:], p[j, :], p_new[j, :],
                                UNUSED_VAR, v[j], time,
                                physical_params, integration_params,
                                0)

            ####################### V - COMPENSATIVE AVG UPDATE ##################
            # for j in range(1, M-2):
            #     brutal_P_update(Pv[:, i], p_new[:,i])
            #     update_averages(Pv[j,:], p[j, :], p_new[j, :],
            #                     UNUSED_VAR, v[j], time,
            #                     physical_params, integration_params,
            #                     0)
            #     compensative_P_update_implicit(Pv[j,:], 
            #                         UNUSED_VAR, v[j], time,
            #                         integration_params, physical_params, 
            #                         0)
            
            p = p_new.copy() 
        else:
            ####################################################################################### SPLIT
            ######################### V - GRID UPDATE ####################
            # For each value of x_i
            for i in range(1,N-1): # Extrema are Dirichlet-fixed
            
                advection_diffusion_matrix(lower_v, diagonal_v, upper_v, b_v, 
                                            p[:, i], Pv[:, i], 
                                            x[i], UNUSED_VAR, time,
                                            physical_params, integration_params,
                                            1) 
                set_dirichlet(lower_v, diagonal_v, upper_v, b_v)
                p_new[:, i] = tridiag(lower_v, diagonal_v, upper_v, b_v)

            ####################### V - AVG UPDATE ##################
            for i in range(N):
                update_averages(Pv[:,i], p[:, i], p_new[:, i],
                                x[i],  UNUSED_VAR, time,
                                physical_params, integration_params,
                                1)

            ####################### X - COMPENSATIVE AVG UPDATE #################
            # for i in range(1,N-2):
            #     brutal_P_update(Px[j, :], p_new[j,:])
            #     update_averages(Px[:,i], p[:, i], p_new[:, i],
            #                     x[i], UNUSED_VAR, time,
            #                     physical_params, integration_params,
            #                     1)
            #     compensative_P_update_implicit(Px[:,i], 
            #                             x[i], UNUSED_VAR, time,
            #                             integration_params, physical_params, 
            #                             1)

            p = p_new.copy()

            # for i in range(N-2):
            #     for j in range(M-2):
            #         Px[j,i] = Pv[j,i]
        
    return p, Px, Pv

