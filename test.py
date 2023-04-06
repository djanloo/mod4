import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;
from mod4.diffeq import FTCS, LAX, LAX_WENDROFF, burgers_lw

N = 5000
Ntimes = 11
x = np.linspace(0, 1, N, endpoint=False)


# u0 = 0.07*np.sin(100*2*np.pi*x)*np.exp(-((x-0.3)/0.02)**2) + 13.7
u0 = np.sin(2*np.pi*x)+1
dx = x[1]-x[0]

integration_args = dict(dx=dx, dt=1e-6, n_steps=8e3)

fig, (axx, axk) = plt.subplots(2,1)
colors = sns.color_palette("flare", Ntimes)
u = u0.copy()
for i in range(Ntimes):

    axx.plot(x, u, color=colors[i])

    g = np.fft.fft(u)

    k = np.fft.fftfreq(N, dx)

    axk.step(k[:N//2], np.abs(g[:N//2])**2, where="mid", color=colors[i])
    
    u = burgers_lw(u, **integration_args)

axk.set_xscale('log')
axk.set_yscale('log')

plt.show()