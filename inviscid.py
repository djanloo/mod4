from mod4.tsai import tsai_1D_x, tsai_1D_v
from mod4.implicit import IMPL1D_x, IMPL1D_v
from mod4.utils import get_lin_mesh
from scipy.integrate import simpson
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'TeX Gyre Pagella'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (10/2.54, 7/2.54)


i_pars = dict(Lv=15 ,Lx=15, dx=0.1, dv=0.1, dt=1e-3, n_steps=1000)
phy_pars = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)

c = 3
v = np.array(get_lin_mesh(i_pars))
print(len(v))
rmss1 = []
ks1 = []

MM = 1
def coeff(f, base):
    global v
    return np.trapz(np.array(f)*base, v)

K_min = np.pi/i_pars['dv']
K_max = np.pi/i_pars['dv']

print(K_min, K_max)

a = 0.2
def base_func(k, x):
    f = np.cos(k*(x+3))*np.exp(-((x+3)/a)**2)
    f /= np.sqrt(coeff(f,f))
    return f

for m in np.arange(1, len(v)//2):
    k = m*(K_min/50)
    base = base_func(k, v)
    base /= np.sqrt( coeff(base, base))

    p = base.copy()

    P = np.zeros(len(p)-1)
    for i in range(len(P)):
        P[i] = 0.5*(p[i] + p[i+1])

    p, P = tsai_1D_x(p,P, c, phy_pars, i_pars)

    
    tt = i_pars['dt']*i_pars['n_steps']
    shifted_base =  base_func(k, v - c*tt)

    rmss1.append(coeff(p, shifted_base))
    ks1.append(k)

    if m == MM:
        # plt.plot(v, base, ls="-", color="k", label="base")
        fig, ax = plt.subplots(constrained_layout=True)
        # plt.plot(v, base, ls="-", color="k", label="base")
        mask = np.abs(v) < 3
        ax.plot(v[mask], shifted_base[mask],ls="-", color="k", label="esatta")
        ax.scatter(v[mask], np.array(p)[mask], facecolor='none',color="k", marker="o",label ="approssimata")
        ax.plot(v[mask], np.array(p)[mask], ls=":", color="k")
        ax.set_xlabel("z")
        ax.legend()
        plt.show()


rmss2 = []
ks2 = []
for m in np.arange(1, len(v)//2):
    k = m*(K_min/50)
    base = base_func(k, v)
    base /= np.sqrt(coeff(base, base))

    p = base.copy()


    p = IMPL1D_x(p, c, phy_pars, i_pars)

    tt = i_pars['dt']*i_pars['n_steps']
    shifted_base = base_func(k, v - c*tt)

    rmss2.append(coeff(p, shifted_base))
    ks2.append(k)

    if m == MM:
        fig, ax = plt.subplots(constrained_layout=True)
        # plt.plot(v, base, ls="-", color="k", label="base")
        mask = np.abs(v) < 3
        ax.plot(v[mask], shifted_base[mask],ls="-", color="k", label="esatta")
        ax.scatter(v[mask], np.array(p)[mask], facecolor='none',color="k", marker="o",label ="approssimata")
        ax.plot(v[mask], np.array(p)[mask], ls=":", color="k")
        ax.set_xlabel("z")
        ax.legend()
        plt.show()


rmss1 = np.array(rmss1)
rmss2 = np.array(rmss2)

plt.figure(2)
plt.plot(ks1,rmss1**(1/i_pars['n_steps']))
plt.plot(ks2,rmss2**(1/i_pars['n_steps']))





plt.xscale('log')
plt.yscale('log')

plt.show()

