import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns;
from mod4.diffeq import FTCS, LAX, LAX_WENDROFF, burgers_lw, heat_cn, diff_advec, burgers_cn

N = 500
Ntimes = 3
x = np.linspace(0, 3, N, endpoint=False)


# u0 = 1 + 0.05*np.sin(10*2*np.pi*x)*np.exp(-((x-0.8)/0.2)**2) #+ 0.5*np.sin(7*2*np.pi*x) + 0.2*np.sin(20*2*np.pi*x) + 0.5*np.sin(30*2*np.pi*x) + 0.5*np.sin(31*2*np.pi*x)
u0 = 1+np.exp(-((x-0.5)/0.1)**2)
# u0 = 2+np.sin(3*2*np.pi*x)

# u0 = 0.5*(u0 + u0[::-1])
dx = x[1]-x[0]

integration_args = dict(
                        dx=dx, dt=5e-5,
                        n_steps=5000, 
                        nu=0.003
                        )

fig, (axx, axk) = plt.subplots(2,1)
colors = sns.color_palette("flare", Ntimes)

start_args = integration_args.copy()
start_args["n_steps"] =10
u = burgers_cn(u0,  **start_args)

g = np.fft.fft(u)
k = np.fft.fftfreq(N, dx)
axk.step(k[:N//2], np.abs(g[:N//2])**2, where="mid", color='b')
axx.plot(x, u, color='b', marker='.', label="T=0")
# axx.plot(x + 1, u, color='b', marker=".")

for i in range(Ntimes):
    u = burgers_cn(u, **integration_args)
    axx.plot(x, u, color=colors[i], label = f"T = {i*integration_args['dt']*integration_args['n_steps']:.2f}")
    # axx.plot(x+1, u, color=colors[i])
    # axx.plot(x+2, u, color=colors[i])


    g = np.fft.fft(u)
    k = np.fft.fftfreq(N, dx)
    axk.step(k[:N//2], np.abs(g[:N//2])**2, where="mid", color=colors[i])
    
axx.legend(fontsize=8)
# axk.set_xscale('log')
axk.set_yscale('log')

axk.set_ylabel(r"$|S(\omega)|^2$")
plt.show()