import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from rich import print

plt.rcParams['font.family'] = 'TeX Gyre Pagella'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (18/2.54, 10/2.54)

N = 50
dv = 0.1

eps = 1e-2
u = np.ones(N, dtype=np.float128)
v = np.arange(-N//2,N//2)*dv

print("V", v)
def a(v):
    return -np.array(v).astype(np.float128)

V = np.zeros((N, N))
for i in range(N):
    V[i,i] = 1 + eps*(a(v[i] + 0.5*dv) - a(v[i] - 0.5*dv) )

for i in range(N-1):
    print(f"Upper: in ({i},{i+1}) placed {i+0.5}")
    V[i, i+1] = eps*a(v[i] + 0.5*dv)

for i in range(0, N-1):
    print(f"Lower: in ({i+1},{i}) placed {i+1-0.5}")
    V[i+1, i] = -eps*a(v[i+1] - 0.5*dv)
    

print(V)
print("sum of cols", np.sum(V, axis=0))
print("sum of rows", np.sum(V, axis=1))

u = np.exp(-(5*(np.linspace(-1,1,N) - 0.2))**2)
u /= np.sum(u)
colors = sns.color_palette("viridis", 200)
norm = []

# for i in range(N):
#     V[:,i] /= np.sum(V[:,i])
V_inv = np.linalg.inv(V)
for j in range(200):

    norm.append(np.sum(u))
    plt.plot(np.linspace(-1,1, N), u, color=colors[j])
    u = V_inv.dot(u)

plt.figure(2)
plt.plot(norm)
plt.show()
