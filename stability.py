import numpy as np
import matplotlib.pyplot as plt

from mod4.implicit import IMPL1D_x
from mod4.tsai import tsai_1D_x
from mod4.utils import get_lin_mesh
import seaborn as sns;sns.set()

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'TeX Gyre Pagella'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (18/2.54, 7/2.54)


figc, axc = plt.subplots()
fig, ax = plt.subplots(constrained_layout=True)
figp, axph = plt.subplots(constrained_layout=True)

i_pars = dict(Lx=10, dx=0.1, dt=1e-3, n_steps=500)
phy_pars = dict(omega_squared=1.0, gamma=1.0, sigma_squared=0.8**2)
v = 5
x = np.array(get_lin_mesh(i_pars))
p0 = np.exp(-x**2)
# p0[p0< 1e-16] = 0.0
# p0 *= np.random.normal(0, 1, size=len(p0))
# p += np.random.normal(0,0.1, size=len(p))

p0 /= np.trapz(p0, x)



p = p0.copy()
P = np.zeros(len(p)-1)
for i in range(len(P)):
    P[i] =0.5*( p[i] + p[i+1])

N = len(p)
print(N)

p, P = tsai_1D_x(p0.copy(), P,v, phy_pars, i_pars)
fft_before = np.fft.fft(p)[:N//2]
# ax.plot(fft_before)
freqs = np.fft.fftfreq(N, x[1] - x[0])[:N//2]
M = 3

g = np.zeros((M, len(fft_before)),dtype=complex)
for i in range(M):

    p, P = tsai_1D_x(p, P,v, phy_pars, i_pars)


    axc.plot(x, np.array(p))
    # ax.plot(freqs, (np.abs(np.fft.fft(p)[:N//2])/fft_before)**(1.0/(1+i)/i_pars['n_steps']), color="k", alpha=0.5)
    # ax.plot(np.abs(np.fft.fft(p)[:N//2]), ls=":", label=f"tsai {i}")
    g[i] = (np.fft.fft(p)[:N//2]/fft_before)**(1.0/(1+i)/i_pars['n_steps'])

ax.step(freqs, np.mean(g, axis=0), color="k", lw=1, where='mid')
ax.scatter(freqs, np.mean(g, axis=0), color="k", marker = "o",facecolor="none", label="Tsai")
avg_g = np.mean(g, axis=0)
axph.plot(freqs, np.arctan(np.imag(avg_g)/np.real(avg_g)))

mask = np.mean(np.abs(g), axis=0) > 1 + 1e-8
ax.scatter(freqs[mask], np.mean(g, axis=0)[mask], color="red", marker = "o",facecolor="none")

p = IMPL1D_x(p0.copy(),v, phy_pars, i_pars)
fft_before = np.fft.fft(p)[:N//2]

for i in range(M):
    # ax.plot(fft_before**1/i_pars['n_steps'])

    # p, P = tsai_1D_x(p, P,v, phy_pars, i_pars)
    p = IMPL1D_x(p,v, phy_pars, i_pars)


    axc.plot(x, np.array(p))
    # ax.plot(freqs, (np.abs(np.fft.fft(p)[:N//2])/fft_before)**(1.0/(1+i)/i_pars['n_steps']))
    # ax.plot(np.abs(np.fft.fft(p)[:N//2]),ls="-", label=f"impl {i}")
    g_here = np.fft.fft(p)[:N//2]/fft_before
    g[i] = (g_here)**(1.0/(1+i)/i_pars['n_steps'])

ax.step(freqs, np.mean(np.abs(g), axis=0), color="k",lw=1, where='mid')
ax.scatter(freqs, np.mean(np.abs(g), axis=0), color="k", marker="x", label="Implicito")

avg_g = np.mean(g, axis=0)
axph.plot(freqs, np.arctan(np.imag(avg_g)/np.real(avg_g)))

mask = np.mean(np.abs(g), axis=0) >1
ax.scatter(freqs[mask], np.mean(g, axis=0)[mask], color="red", marker="x")


ax.minorticks_on()
ax.legend()

# ax.grid(ls=":")
ax.set_xlabel('$k$')
ax.set_ylabel("$g(k)$")

ax.set_yscale('log')
ax.set_xscale('log')
plt.show()
