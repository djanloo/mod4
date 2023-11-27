import numpy as np
import matplotlib.pyplot as plt
# from mod4.diffeq import advect_LW, diffuse_CN,  advect_diffuse_IMPL, advect_diffuse_LW, advectLW_diffuseCN
from mod4.tsai import tsai_1D_v
from mod4.implicit import IMPL1D_v 
from mod4.utils import get_lin_mesh
from scipy.integrate import simpson

import seaborn as sns; sns.set()
from matplotlib.animation import FuncAnimation
from scipy.special import erf
import matplotlib as mpl


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'TeX Gyre Pagella'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (18/2.54, 7/2.54)


fig1, (ax2, ax1)= plt.subplots(1,2,constrained_layout=True)

i_pars = dict(Lv=8, dv=0.1, dt=1e-3, n_steps=2000)
phy_pars = dict(omega_squared=1.0, gamma=1.0, sigma_squared=0.2**2)


v = np.array(get_lin_mesh(i_pars))
beta = phy_pars['gamma']/phy_pars['sigma_squared']

M = 20
x = 0
best_errs = np.zeros(M)

def err(x):
    return np.sqrt(np.mean((x**2)))

p = np.exp(-((v))**2)/np.sqrt(np.pi)

print(np.trapz(p, v))

steady = np.exp(-beta*(v+phy_pars['omega_squared']*x/phy_pars['gamma'])**2)
steady /= np.trapz(steady, v)

ax2.plot(v, steady, color="k")

P = np.zeros(len(p)-1)
for i in range(len(P)):
    xx = np.linspace(v[i], v[i]+i_pars['dv'], 100)
    ff = np.exp(-((xx))**2)/np.sqrt(np.pi)
    P[i] = simpson(ff,xx)/i_pars['dv']

err_old = 1e-1
err_new = 1e-2
c = 0
while abs(err_new - err_old)/err_old > 1e-5:
    p , P = tsai_1D_v(p, P, x, phy_pars, i_pars)
    err_old = err_new
    err_new  = err(steady - p)
    c+=1
    if c >10:
        break
    print(f"tsai {err_new:6.2e} -- {abs(err_new - err_old)/err_old:6.2e}")

# plt.scatter(v, np.array(p)-steady, facecolor="none", color="k", marker="o")
ax1.plot(v, np.array(p)-steady, color="k", ls="--", label="Tsai")
ax2.scatter(v, np.array(p), color="k", marker="o", facecolor="none")

################################### IMPL ##################

p = np.exp(-((v))**2)
p /= np.trapz(p, v)


err_old = 1e-1
err_new = 1e-2
c = 0
while abs(err_new - err_old)/err_old > 1e-5:
    p = IMPL1D_v(p, x, phy_pars, i_pars)
    err_old = err_new
    err_new  = err(steady - p)
    c+=1
    if c >10:
        break
    print(f"impl {err_new:6.2e} -- {abs(err_new - err_old)/err_old:6.2e}")

# plt.scatter(v, np.array(p)-steady, color="k", marker="x")
ax1.plot(v, np.array(p)-steady, color="k", label="implicito")
ax2.scatter(v, np.array(p), color="k", marker="x")

# scale_factor = 10**3
# fmt = mpl.ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x/scale_factor))
# ax1.yaxis.set_major_formatter(fmt)
ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1),
                       useOffset=False)

ax1.set_xlabel("z")
ax2.set_xlabel('z')
ax1.set_ylabel('Errore')
plt.legend()
plt.show()