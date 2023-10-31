import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from mod4.diffeq import funker_plank
from mod4.utils import analytic, quad_int


# # Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# print(np.diff(x)[0], np.diff(v)[0])

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)

# # Initial conditions
# x0, v0 = 0,0
# t0 = 0.1
# p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
# p0 /= quad_int(p0, x,v)

# # Test -1: moments after 1000 timesteps
# integration_params = dict(dt=np.pi/1000, n_steps=1000)

# p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params, save_norm=True, save_current=True)
# p_an = np.real(analytic(X,V, t0 + integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params)) 

# print(f"exact avg V= {quad_int(V*p_an, x,v):.7f}")
# print(f"exact avg X= {quad_int(X*p_an, x,v):.7f}")
# print(f"exact corr XV= {quad_int(X*V*p_an, x,v):.7}")
# print(f"exact avg V2= {quad_int(V**2*p_an, x,v):.7f}")
# print(f"exact avg X2= {quad_int(X**2*p_an, x,v):.7f}")

# print(f"num avg V= {quad_int(V*p_num, x,v):.7f}")
# print(f"num avg X= {quad_int(X*p_num, x,v):.7f}")
# print(f"num corr XV= {quad_int(X*V*p_num, x,v):.7}")
# print(f"num avg V2= {quad_int(V**2*p_num, x,v):.7f}")
# print(f"num avg X2= {quad_int(X**2*p_num, x,v):.7f}")

# # TEST 0: moments at stationarity
# integration_params = dict(dt=np.pi/1000, n_steps=100)
# x_old = quad_int(V**2*p0, x,v)
# x_new = 100
# p_num = p0
# c=0

# while abs((x_old - x_new)/x_old) > 1e-5:
    
#     x_old = x_new
#     p_num, norm, curr = funker_plank(p_num, x, v, physical_params, integration_params, save_norm=True, save_current=True)
#     x_new = quad_int(V**2*p_num, x,v) 
#     print(f"iter {c}, x_new = {x_new}, x_old = {x_old}")
#     c+=1
# print(c, x_new, x_new-x_old)

# p_an = np.real(analytic(X,V, t0 + c*integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params)) 

# print(f"At the same time: exact avg V= {quad_int(V*p_an, x,v):.5f}")
# print(f"At the same time: exact avg X= {quad_int(X*p_an, x,v):.5f}")
# print(f"At the same time: exact corr XV= {quad_int(X*V*p_an, x,v):.5}")
# print(f"At the same time: exact avg V2= {quad_int(V**2*p_an, x,v):.5f}")
# print(f"At the same time: exact avg X2= {quad_int(X**2*p_an, x,v):.5f}")

# print(f"At the same time: num avg V= {quad_int(V*p_num, x,v):.5f}")
# print(f"At the same time: num avg X= {quad_int(X*p_num, x,v):.5f}")
# print(f"At the same time: num corr XV= {quad_int(X*V*p_num, x,v):.5}")
# print(f"At the same time: num avg V2= {quad_int(V**2*p_num, x,v):.5f}")
# print(f"At the same time: num avg X2= {quad_int(X**2*p_num, x,v):.5f}")

# print(f"sigma^2/gamma/2 = {physical_params['sigma']**2/physical_params['gamma']/2}")

# print(f"RMSE on p: {np.sqrt(np.mean((p_num - p_an)**2))}")
# exit()
######################TEST 1: 800 timesteps ########################
# Environment setting
Lx, Lv = 4, 4
dx, dv = 0.1, 0.1
x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
X, V = np.meshgrid(x,v)

print(np.diff(x)[0], np.diff(v)[0])

# integration & physical parameters
physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)

# Initial conditions
x0, v0 = 0,0
t0 = 0.95
p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
p0 /= quad_int(p0, x,v)

integration_params = dict(dt=np.pi/1000, n_steps=5)
p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params, save_norm=True, save_current=True)
p_an = np.real(analytic(X,V, t0+ integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params))
p_num = np.array(p_num)
p_num[p_num<0] = 1e-40
error = np.sqrt(np.mean( (p_num - p_an)**2))
print(error)
print(np.array(norm)[-1])

plt.figure(1, figsize=(10/2.54, 10/2.54))
plt.contourf(X, V, (p_num-p_an)/error)
plt.colorbar(shrink=0.72)
plt.contour(X, V, p_an, colors="w", levels=np.logspace(-8, -2, 4))
plt.title(r"$\left(p_{numeric} - p_{analytic}\right)/\langle e \rangle$")
plt.ylabel("v")
plt.xlabel("x")
plt.gca().set_aspect('equal')
plt.tight_layout()

plt.figure(2, figsize=(10/2.54, 10/2.54))
plt.xlabel("t")
plt.plot(np.arange(len(norm)-1)*integration_params['dt'], np.diff(np.array(norm))/integration_params['dt'], label=r"$\partial_t N$")
total_current = np.zeros(integration_params['n_steps'])
for key in curr.keys():
    # plt.plot(np.array(curr[key]),alpha=0.5, label=f"{key} current")
    total_current += np.array(curr[key])
plt.plot(np.arange(len(total_current))*integration_params['dt'], -total_current, label=r"$- \int J \cdot n $")
plt.legend()
plt.tight_layout()


fig, axes = plt.subplot_mosaic([["n", "a", "e", "c"]],width_ratios=[1,1,1,0.1], figsize=(21/2.54, 10/2.54), constrained_layout=True)

for axname in ['n', 'e', 'a']:
    axes[axname].set_aspect('equal')
    axes[axname].set_xticklabels([])
    axes[axname].set_yticklabels([])

levels = np.arange(-20, 2, 2)
axes['n'].contourf(np.log(p_num),levels=levels)
img = axes['a'].contourf(np.log(p_an),levels=levels)
axes['e'].contourf(np.log(np.abs((p_num-p_an))),levels=levels)
plt.colorbar(img, cax = axes['c'], shrink=0.1)

axes['n'].set_title("numeric")
axes['a'].set_title("exact")
axes['e'].set_title("error")
plt.show()
exit()
################### TEST 2: time dependence ##################################
# # Environment setting
# Lx, Lv = 2, 2
# dx, dv = 0.05, 0.05
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# print(np.diff(x)[0], np.diff(v)[0])

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)

# # Initial conditions
# x0, v0 = 0,0
# t0 = 0.1
# p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
# p0 /= quad_int(p0, x,v)

# import matplotlib as mpl

# t0s = np.arange(1,8)*0.15
# cmap = mpl.colormaps["flare_r"]
# colors = cmap(t0s)
# print(colors)

# M = 30
# fig, ax = plt.subplot_mosaic([['plot', "leg"]], width_ratios=[1, 0.08], figsize=(20/2.54, 7/2.54))
# integration_params = dict(dt=np.pi/1000, n_steps=20)
# errors = np.zeros(M)

# for t0,col in zip(t0s, colors):
#     print(t0)
#     p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
#     p0 /= quad_int(p0, x,v)
#     p_num = p0
#     for i in range(M):
#         p_num, norm, curr = funker_plank(p_num, x, v, physical_params, integration_params, save_norm=True)
#         p_an = np.real(analytic(X,V, t0 + (i+1)*integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params))
#         p_num = np.array(p_num)
#         p_num /= quad_int(p_num, x,v)
#         # errors[i] = np.max(np.abs((p_num - p_an)))
#         # errors[i] = np.sqrt(np.mean((p_num - p_an)**2 ))
#         errors[i] = quad_int(np.abs(p_num - p_an), x, v)

#     ax['plot'].plot(np.arange(1,M+1)*integration_params['n_steps'], errors, label=rf"$t_0 = {t0:.2f}$", color=col)

# handles, labels = ax['plot'].get_legend_handles_labels()

# # ax['leg'].legend(handles, labels, loc="center")
# # ax['leg'].axis("off")
# norm = mpl.colors.Normalize(vmin=min(t0s),vmax=max(t0s))
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# boundaries = np.round(t0s - 0.5*np.diff(t0s)[0], 3)
# boundaries = np.concatenate((boundaries, [boundaries[-1] + np.diff(t0s)[0]]))
# print(boundaries)
# plt.colorbar(sm, ticks=t0s, 
#              boundaries=boundaries, cax=ax['leg'])
# ax['leg'].set_title(r"$t_0$")

# ax['plot'].set_xlabel("Steps")
# ax['plot'].set_ylabel(r"$||e||_{INT}$")
# ax['plot'].set_yscale('log')

# plt.show()
# exit()

################ TEST 2.5: matching time
# # Environment setting
# Lx, Lv = 2, 2
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# print(np.diff(x)[0], np.diff(v)[0])

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=1.1, sigma=0.8)
# # Initial conditions
# x0, v0 = 1,0
# t0 = 0.95
# p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
# p0 /= quad_int(p0, x,v)

# M = 15
# sim_times = np.linspace(0.0, 0.2 , M)


# for dt_mult in [1, 2,3]:
#     dt = dt_mult/1000
   

#     # print(f"dt = {dt}, steps = {steps}, second_per_phase = {dt*steps}")
#     best_match_deltas = np.zeros(M)

#     for i in range(M):
#         steps = int(sim_times[i]/dt)
#         integration_params = dict(dt=dt, n_steps=steps)

#         p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params)
#         p_num = np.array(p_num)

#         delta_t = np.linspace(-0.03, 0, 60)

#         print(f"p_num is simulated to {t0 + sim_times[i]:.2f} in {steps} steps --- scan is between {t0 + sim_times[i] + min(delta_t):.2f} and {t0 + sim_times[i] + max(delta_t):.2f}")
        
        
#         errors = np.zeros(len(delta_t))
#         for j, ddt in enumerate(delta_t):
#             p_an = np.real(analytic(X,V, t0 + sim_times[i] + ddt, x0, v0, physical_params))
#             errors[j] = quad_int((p_num - p_an)**2, x, v)

#         print(f"\tbest match is {delta_t[np.argmin(errors)]:.2f} with error {np.min(errors)} -- DELTA = {delta_t[np.argmin(errors)]}")
#         best_match_deltas[i] = delta_t[np.argmin(errors)]
#         # plt.plot(times, errors)
#         # plt.show()
#     plt.plot(t0 + sim_times,best_match_deltas, marker=".", label=f"dt = {dt:.2}")
# plt.axhline(0, ls=":", color="k")

# plt.ylabel(r"$\Delta$")
# plt.xlabel(r"$t$")
# plt.legend()
# plt.show()

################# TEST 3: space dimension dependence

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)
# integration_params = dict(dt=np.pi/1000, n_steps=800)

# # Grid dimensions and errors
# Ls = [0.5, 1, 1.5, 2, 2.5]
# ddds = np.linspace(0.01, 0.2, 5)
# rmses = np.zeros(len(ddds))
# sup = np.zeros(len(ddds))

# fig, axes = plt.subplot_mosaic([['R', 'leg', 'S']], width_ratios=[1, 0.2, 1],figsize=(20/2.54, 7/2.54), sharey=True, constrained_layout=True)

# axR, axS, axleg = map(axes.get, ['R', "S", "leg"])

# for L in Ls:
#     for i, ddd in enumerate(ddds):
#         print(L, ddd)
#         # Environment setting
#         Lx, Lv = L, L
#         dx, dv = ddd, ddd
#         x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
#         X, V = np.meshgrid(x,v)

#         # Initial conditions
#         x0, v0 = 0,0
#         t0 = 0.95
#         p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
#         p0 /= quad_int(p0, x,v)

#         # Solutions
#         p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params, save_norm=True, save_current=True)
#         p_an = np.real(analytic(X,V, t0 + integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params))

#         rmses[i] = np.sqrt(np.mean( (p_num - p_an)**2))
#         sup[i] = np.max(np.abs(p_num - p_an))
#     print("rmse", rmses)
#     print("sup", sup)
#     # plt.ylabel(r"$||e||$")
#     axR.plot(ddds, rmses, marker=".", ls="-", label=f"L={int(2*L)}")
#     axS.plot(ddds, sup, marker=".", ls="-", label=f"L = {int(2*L)}")

# axR.set_title("RMSE")
# axS.set_title("SUP")
# axR.set_xlabel(r"$\Delta$")
# axS.set_xlabel(r"$\Delta$")

# # Crea una legenda comune alla figura
# lines, labels = axR.get_legend_handles_labels()
# # lines2, labels2 = axS.get_legend_handles_labels()
# # lines += lines2
# # labels += labels2
# axleg.legend(lines, labels, loc='center')
# axleg.axis('off')

# plt.yscale('log')
# plt.show()