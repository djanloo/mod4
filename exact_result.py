import numpy as np
import matplotlib.pyplot as plt

def stupid_inverse(A):
    det = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    B = A.copy()
    B[0, 0] = A[1,1]/det 
    B[1, 1] = A[0,0]/det
    B[0,1] = -A[1,0]/det 
    B[1,0] = -A[0,1]/det
    return B

def analytic(x,v,t, omega=1.0, gamma=2.1, sigma=0.8):
    global x0, v0
    l1, l2 = 0.5 * gamma + np.sqrt( (0.5*gamma)**2 - omega**2), 0.5 * gamma - np.sqrt( (0.5*gamma)**2 - omega**2)
    exp1, exp2 = np.exp(-l1*t), np.exp(-l2*t)
    G = [[(l1*exp1 - l2*exp2)/(l1 - l2), (exp2 - exp1)/(l1 -l2)],
         [omega**2*(exp1 - exp2)/(l1-l2), (l1*exp1 - l2*exp2)/(l1 -l2)]]
    G = np.array(G)

    Sigm11 = sigma**2/(l1 -l2)**2*( (l1 + l2)/l1/l2 - 4*(1- exp1*exp2)/(l1+l1)- exp1**2 - exp2**2)
    Sigm12 = sigma**2/(l1 + l2)**2*(exp1 - exp2)**2
    Sigm22 = sigma**2/(l1-l2)**2

    S = np.array([  [Sigm11, Sigm12],
                    [Sigm12, Sigm22]])
    S_inv = stupid_inverse(S)
    detS = S[0,0]*S[1,1] - S[0,1]*S[1,0]

    extended_vect = np.array([x, v])
    extended_init = np.array([x0, v0])
    mu = extended_vect - np.dot(G, extended_init)
    exponent = -0.5*np.dot(mu, np.dot(S_inv, mu.T))
    return np.exp(exponent)/np.sqrt(detS)

Lx, Lv = 4, 4
x0, v0 = 1,  0.0

x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)
Z = X.copy()
physical_params = dict(omega=0.8, gamma=2.1, sigma= 0.8)

for i in range(len(x)):
    for j in range(len(v)):
        Z[i,j] = analytic(x[i], v[j], 0.0, **physical_params)


plt.contourf(X, V, Z)
plt.colorbar()
plt.show()