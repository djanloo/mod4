import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.rcParams['font.family'] = 'TeX Gyre Pagella'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (18/2.54, 10/2.54)

N = 20
M = 10
eps = 1e-3
u = np.ones(N)

factors_A = np.zeros(M)
factors_B = np.zeros(M)
epss = np.linspace(1e-3, 1e-2, M)

for samp in range(M):

    A = np.zeros((N,N))
    for i in range(N):
        A[i,i] = 1 - epss[samp]
        if i!= 0:
            A[i, i-1] = 0.5*epss[samp]
        if i!=N-1:
            A[i, i+1] = 0.5*epss[samp]
  
    eigval, eigvect = np.linalg.eig(A)
    print(np.sum(A, axis=0))


    print(eigval)
    B = np.linalg.inv(A)

    factors_A[samp] = np.mean(A.dot(u))
    factors_B[samp] = np.mean(B.dot(u))

plt.plot(epss, factors_A, ls="", marker=".")
plt.plot(epss, factors_B, ls="", marker=".")
plt.show()