import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;
from mod4.diffeq import FTCS, LAX, LAX_WENDROFF

N = 2000
Ntimes = 10
x = np.linspace(0, 1, N, endpoint=False)

# u0 = np.zeros(N)
# for i in range(1,10):
#     u0 += np.sin(i*3*2*np.pi*x)

u0 = np.exp(-(x-0.5)**2/0.025)

v = 1.0
dx = x[1]-x[0]

integration_args = dict(dx=dx, dt=1e-4, v=v, n_steps=1e5)

fig, (axx, axk) = plt.subplots(2,1)
colors = sns.color_palette("flare", Ntimes)
u = u0.copy()
for i in range(Ntimes):

    axx.plot(x, u, color=colors[i])

    g = np.fft.fft(u)

    k = np.fft.fftfreq(N, dx)

    axk.step(k[:N//2], np.abs(g[:N//2]), where="mid", color=colors[i])
    
    u = LAX_WENDROFF(u, **integration_args)

axk.set_xscale('log')
axk.set_yscale('log')

plt.show()