# distutils: language = c++
from libcpp cimport bool

from cython.parallel import prange
import numpy as np

cimport cython
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin, fabs

cimport utils
from utils import quad_int, get_tridiag
from utils cimport tridiag

cdef double a(double x, double v, double time, dict physical_params):
  '''potential/ drag function'''
  return  -physical_params['omega_squared']*x - physical_params['gamma']*v*(x**2 - 1)

cdef double sigma_squared(double x, double t, dict physical_params):
  '''Noise function'''
  return physical_params['sigma_squared']#*(1 + fabs(x))



# def funker_plank( double [:,:] p0,
#                     physical_params,
#                     integration_params,
#                     save_norm = False,
#                     save_current=False
#                     ):
#   ## Time
#   cdef double dt   = integration_params['dt']
#   cdef unsigned int n_steps = integration_params['n_steps']
#   cdef double t0    = physical_params.get('t0', 0.0)

#   ## Space
#   cdef double Lx, Lv,dx,dv
#   Lx, Lv,dx,dv = map(integration_params.get, ["Lx", "Lv", "dx", "dv"])
#   cdef unsigned int N = int(Lx/dx), M = int(Lv/dv)
#   cdef double [:] x = np.arange(-int(N)//2, int(N)//2)*dx

#   cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#   cdef unsigned int time_index = 0, i = 0, j = 0

#   cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
#   cdef double [:] norm = np.zeros(n_steps)

#   cdef double theta = 0.5 * dt/dv
#   cdef double alpha = 0.5 * dt/dx 
#   cdef double eta =   0.5 * dt/dv**2
#   cdef double time = t0
  
#   # Declarations of the diagonals
#   cdef double [:] lower_x, diagonal_x, upper_x, b_x
#   cdef double [:] lower_v, diagonal_v, upper_v, b_v

#   lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
#   lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

#   diagonal_v = np.ones(M)
#   diagonal_x = np.ones(N)

#   cdef dict currents = dict(top=np.zeros(n_steps), 
#                             bottom=np.zeros(n_steps), 
#                             left=np.zeros(n_steps),
#                             right=np.zeros(n_steps))


#   for time_index in range(n_steps):
#     time = t0 + time_index*dt
#     # First evolution: differential wrt V
#     # For each value of x, a tridiagonal system is solved to find values of v
#     for i in range(N):

#       # Prepares tridiagonal matrix and the constant term
#       for j in range(M):
        
#         diagonal_v[j] = 1  + 0.5 * dt * d_a(x[i],v[j], time, physical_params)
#         diagonal_v[j] += 2 * eta * sigma_squared(x[i], time, physical_params)

#         upper_v[j]  = - eta * sigma_squared(x[i], time, physical_params)
#         upper_v[j] += theta * a(x[i], v[j], time, physical_params)
#         upper_v[j] += 0.25 * dt * d_a(x[i], v[j], time, physical_params)
   
#         lower_v[j] =  - eta * sigma_squared(x[i], time, physical_params)
#         lower_v[j] -= theta * a(x[i], v[j] + dv, time, physical_params)
#         lower_v[j] += 0.25 * dt * d_a(x[i], v[j] + dv, time, physical_params)

#         b_v[j] =  p[j, i]  

#       # Solves the tridiagonal system for the column
#       p_intermediate[:, i]= tridiag(lower_v, diagonal_v, upper_v, b_v)

#     # Second evolution: differential wrt x
#     # For each value of v, a tridiagonal system is solved to find values of x
#     for j in range(M):

#       # Prepares tridiagonal matrix and constant term
#       for i in range(N):
#         lower_x[i] = - alpha * v[j]
#         upper_x[i] =   alpha * v[j]

#         b_x[i] = p_intermediate[j, i]

#       # Solves the tridiagonal system for the row
#       p[j, :] =  tridiag(lower_x, diagonal_x, upper_x, b_x)
      
#     # Takes trace of normalization
#     if save_norm:
#       norm[time_index] = quad_int(p, integration_params)

#     if save_current: 
#       # Integral in v
#       for j in range(M):
#         currents['right'][time_index] += v[j]*p[j, N-2]*dv
#         currents['left'][time_index] -= v[j]*p[j, 1]*dv

#       # Integral in x
#       for i in range(N):
#         currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
#         currents['top'][time_index] -= 0.5*physical_params['sigma_squared']*( (p[M-1,i] - p[M-2,i])/dv )*dx

#         currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
#         currents['bottom'][time_index] += 0.5*physical_params['sigma_squared']**2*( (p[1,i] - p[0,i])/dv )*dx

#   return p, norm, currents

# def funker_plank_cn( double [:,:] p0,
#                     physical_params,
#                     integration_params,
#                     save_norm = False,
#                     save_current=False
#                     ):
#   ## Time
#   cdef double dt   = integration_params['dt']
#   cdef unsigned int n_steps = integration_params['n_steps']
#   cdef double t0    = physical_params.get('t0', 0.0)

#   ## Space
#   cdef double Lx, Lv,dx,dv
#   Lx, Lv,dx,dv = map(integration_params.get, ["Lx", "Lv", "dx", "dv"])
#   cdef unsigned int N = int(Lx/dx), M = int(Lv/dv)
#   cdef double [:] x = np.arange(-int(N)//2, int(N)//2)*dx

#   cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#   cdef unsigned int time_index = 0, i = 0, j = 0

#   cdef double [:,:] p = p0.copy(), p_intermediate = p0.copy()
#   cdef double [:] norm = np.zeros(n_steps)

#   cdef double theta = 0.25*dt/dv   ##CN
#   cdef double alpha = 0.25 * dt/dx ##CN
#   cdef double eta = 0.25*dt/dv**2  ## CN
#   cdef double time = t0
  
#   # Declarations of the diagonals
#   cdef double [:] lower_x, diagonal_x, upper_x, b_x
#   cdef double [:] lower_v, diagonal_v, upper_v, b_v

#   lower_v, upper_v, b_v = np.ones(M), np.ones(M), np.ones(M)
#   lower_x, upper_x, b_x = np.ones(N), np.ones(N), np.ones(N)

#   diagonal_v = np.ones(M)
#   diagonal_x = np.ones(N)

#   cdef dict currents = dict(top=np.zeros(n_steps), 
#                             bottom=np.zeros(n_steps), 
#                             left=np.zeros(n_steps),
#                             right=np.zeros(n_steps))

#   for time_index in range(n_steps):
#     time = t0 + time_index*dt
#     # First evolution: differential wrt V
#     # For each value of x, a tridiagonal system is solved to find values of v
#     for i in range(N):
      
#       # Prepares tridiagonal matrix and the constant term
#       for j in range(M):
        
#         diagonal_v[j] = 1  + 0.5* 0.5 * dt * d_a(x[i],v[j], time, physical_params)
#         diagonal_v[j] += 2 * eta * sigma_squared(x[i], time, physical_params)

#         upper_v[j]  = - eta * sigma_squared(x[i], time, physical_params)
#         upper_v[j] += theta * a(x[i], v[j], time, physical_params)
#         upper_v[j] += 0.5 * 0.25 * dt * d_a(x[i], v[j], time, physical_params)
   
#         lower_v[j] =  - eta * sigma_squared(x[i], time, physical_params)
#         lower_v[j] -= theta * a(x[i], v[j] + dv, time, physical_params)
#         lower_v[j] += 0.5 * 0.25* dt * d_a(x[i], v[j] + dv, time, physical_params)

#         b_v[j] =  p[j, i]  

#         ## CN
#         if j > 0 and j < M-1:
#           # Diffusion
#           b_v[j] +=   (eta *sigma_squared(x[i], time, physical_params) )   * p[j+1,i] 
#           b_v[j] +=   (-2*eta *sigma_squared(x[i], time, physical_params) )* p[j,i]
#           b_v[j] +=   (eta * sigma_squared(x[i], time, physical_params) )  * p[j-1, i]

#           # Drift
#           b_v[j] += (-0.25 * 0.5 * dt * d_a(x[i], v[j], time, physical_params) - theta * a(x[i], v[j], time, physical_params))*p[j+1, i]
#           b_v[j] += (-0.5  * 0.5 * dt * d_a(x[i], v[j], time, physical_params))*p[j, i]
#           b_v[j] += (-0.25 * 0.5 * dt * d_a(x[i], v[j], time, physical_params) + theta * a(x[i], v[j], time, physical_params))*p[j-1, i]

#       ## Raw conservation of norm
#       diagonal_v[0] = 1 - lower_v[0]
#       diagonal_v[M-1] = 1- upper_v[M-2]

#       # Solves the tridiagonal system for the column
#       p_intermediate[:, i] = tridiag(lower_v, diagonal_v, upper_v, b_v)

#     # Second evolution: differential wrt x
#     # For each value of v, a tridiagonal system is solved to find values of x
#     for j in range(M):

#       # Prepares tridiagonal matrix and constant term
#       for i in range(N):
#         lower_x[i] = - alpha * v[j]
#         upper_x[i] =   alpha * v[j] 

#         b_x[i] = p_intermediate[j, i] 
        
#         ## CN
#         if i != 0 and i != N-1:
#           b_x[i] += alpha * v[j] * (p_intermediate[j, i-1] - p_intermediate[j, i+1])

#       ## Raw conservation of norm
#       diagonal_x[0] = 1 - lower_x[0]
#       diagonal_x[M-1] = 1 - upper_x[M-2]
      
#       # Solves the tridiagonal system for the row
#       p[j, :] =  tridiag(lower_x, diagonal_x, upper_x, b_x)

#     # Takes trace of normalization
#     if save_norm:
#       norm[time_index] = quad_int(p, integration_params)

#     if save_current: 
#       # Integral in v
#       for j in range(M):
#         currents['right'][time_index] += v[j]*p[j, N-2]*dv
#         currents['left'][time_index]  -= v[j]*p[j, 1]*dv

#       # Integral in x
#       for i in range(N):
#         currents['top'][time_index] += a(x[i], v[M-2],  t0 + time_index*dt, physical_params)*p[M-2, i]*dx
#         currents['top'][time_index] -= 0.5*physical_params['sigma_squared']*( (p[M-1,i] - p[M-2,i])/dv )*dx

#         currents['bottom'][time_index] -= a(x[i], v[2],  t0 + time_index*dt, physical_params)*p[2, i]*dx
#         currents['bottom'][time_index] += 0.5*physical_params['sigma_squared']**2*( (p[1,i] - p[0,i])/dv )*dx

#   return p, norm, currents





# cpdef advect_BW(double [:] p0, double x, dict physical_params, dict integration_params):
#   ## Time
#   cdef double dt   = integration_params['dt']
#   cdef unsigned int n_steps = integration_params['n_steps']
#   cdef double t0    = physical_params.get('t0', 0.0)

#   ## Space
#   cdef double Lv,dv
#   Lv,dv = map(integration_params.get, ["Lv", "dv"])

#   cdef unsigned int  M = int(Lv/dv)
#   cdef double [:] v = np.arange(-int(M)//2, int(M)//2)*dv

#   cdef unsigned int time_index = 0, j = 0

#   cdef double [:] p = p0.copy(), p_half = p0.copy()
#   cdef double theta = 0.5 * dt/dv
  
#   for time_index in range(n_steps):
#     time = t0 + time_index*dt
      
#     # Half step LW
#     for j in range(M):

#       # Note tha theta is absorbed in working variable
#       a_plus  =  theta * a(x,v[j] + dv, time, physical_params)
#       a_here =  theta * a(x,v[j], time, physical_params)

#       if a_here > 0:
      
#       else:


#   return p