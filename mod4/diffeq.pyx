# distutils: language = c++
from libcpp cimport bool

from cython.parallel import prange  
import numpy as np 

cimport cython   
from cython.parallel import prange
cimport numpy as np

from time import perf_counter
from libc.math cimport sin, fabs

from mod4.utils import quad_int, get_tridiag
from mod4.utils cimport tridiag

cdef double a(double x, double v, double time, dict physical_params):
  '''potential/ drag function'''
  return  (-physical_params['omega_squared']*x - physical_params['gamma']*v)#*(x**2 - 1)

cdef double sigma_squared(double x, double t, dict physical_params):
  '''Noise function'''
  return physical_params['sigma_squared']

cdef double sigma_squared_full(double x, double v, double t, dict physical_params):
    return physical_params['sigma_squared']
