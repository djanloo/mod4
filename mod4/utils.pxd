cdef double [:] tridiag(double [:] lower, double [:] diag, double [:] upper, double [:] d)
cpdef get_tridiag(double [:] lower, double [:] diag, double [:] upper)
cpdef cyclic_tridiag(double [:] lower, double [:] diag, double [:] upper, double c_up_right, double c_down_left, double [:] d)
cdef complex det22(complex [:,:] A)
cdef complex [:,:] stupid_inverse(complex [:,:] A)
cpdef quad_int(double [:,:] f, integration_params)
