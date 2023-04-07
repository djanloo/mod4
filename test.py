import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;
from mod4.diffeq import FTCS, LAX, LAX_WENDROFF, burgers_lw, heat_cn

N = 300
Ntimes = 20
x = np.linspace(0, 1, N, endpoint=False)


u0 = np.sin(3*2*np.pi*x) + 5 + 0.5*np.sin(6*2*np.pi*x) + 0.2*np.sin(20*2*np.pi*x) 
u0 = np.exp(-((x-0.5)/0.01)**2)
dx = x[1]-x[0]

integration_args = dict(dx=dx, dt=5e-7, n_steps=3e2, nu=0.1)

fig, (axx, axk) = plt.subplots(2,1)
colors = sns.color_palette("flare", Ntimes)

u = u0.copy()
g = np.fft.fft(u)
k = np.fft.fftfreq(N, dx)
axk.step(k[:N//2], np.abs(g[:N//2])**2, where="mid", color='b')
axx.plot(x, u, color='b', marker='.')
# axx.plot(x + 1, u, color='b', marker=".")

for i in range(Ntimes):
    u = heat_cn(u, **integration_args)

    axx.plot(x, u, color=colors[i])
    # axx.plot(x + 1, u, color=colors[i])

    g = np.fft.fft(u)
    k = np.fft.fftfreq(N, dx)
    axk.step(k[:N//2], np.abs(g[:N//2])**2, where="mid", color=colors[i])
    

axk.set_xscale('log')
axk.set_yscale('log')

axk.set_ylabel(r"$|S(\omega)|^2$")
plt.show()