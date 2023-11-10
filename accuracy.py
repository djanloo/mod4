import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from scipy.optimize import curve_fit
import matplotlib as mpl

from mod4.diffeq import  funker_plank as funker_plank
from mod4.utils import analytic, quad_int


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'TeX Gyre Pagella'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.figsize'] = (18/2.54, 10/2.54)

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
################################## TEST -2: norm #########################
# # Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)
# fig,axes = plt.subplot_mosaic([['exp', 'norm', 'cbar']], width_ratios = [0.55, 0.4, 0.05], sharey=False, constrained_layout=True)

# # integration & physical parameters
# dts =  np.linspace(1e-3, 1e-2, 100)
# exps = []
# cmap = mpl.colormaps["flare_r"].resampled(5)
# colors = cmap(dts/max(dts))

# for i, dt in enumerate(dts):
#     nsteps = int(1.0/dt)
#     print(nsteps)
#     physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)
#     integration_params = dict(dt=dt, n_steps=nsteps)

#     # Initial conditions
#     x0, v0 = 0,0
#     t0 = 1.5
#     p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
#     p0 /= quad_int(p0, x,v)

#     p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params, save_norm=True, save_current=False)
#     lognorm = np.log(np.array(norm))
#     exps.append((lognorm[-1] - lognorm[0])/integration_params['dt']/integration_params['n_steps'])
#     axes['norm'].plot(integration_params['dt']*np.arange(integration_params['n_steps']), np.log(np.array(norm))*1e14, color=colors[i])

# axes['norm'].set_ylabel(r"$\log( N(t) ) \;\;\cdot 10^{14}$")
# axes['norm'].set_xlabel(r"$t$")

# axes['exp'].plot(dts, np.array(exps)*1e14, color="k")
# # axes['exp'].set_xscale('log')
# axes['exp'].set_xlabel(r'$dt$')
# axes['exp'].set_ylabel(r'$m \cdot 10 ^{14}$')


# norm = mpl.colors.Normalize(vmin=1,vmax=10)
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# sm.set_array([])
# # boundaries = np.round(1000*(dts - 0.5*np.diff(dts)[0]), 1)
# # boundaries = np.concatenate((boundaries, [boundaries[-1] + 1000*np.diff(dts)[0]]))
# # print(boundaries)
# plt.colorbar(sm, ticks=np.arange(1,10), boundaries=np.arange(1,10)+0.5, cax=axes['cbar'], label="dt")
# plt.show()

################################## Test -1: moments after 1000 timesteps #############################
# # Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# print(np.diff(x)[0], np.diff(v)[0])

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)

# # Initial conditions
# x0, v0 = 0,1
# t0 = 0.95
# p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
# p0 /= quad_int(p0, x,v)
# integration_params = dict(dt=np.pi/1000, n_steps=1000)

# p_num, norm, curr = funker_plank_original(p0, x, v, physical_params, integration_params, save_norm=True, save_current=True)
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

# TEST 0: moments at stationarity
# # Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# print(np.diff(x)[0], np.diff(v)[0])

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)

# # Initial conditions
# x0, v0 = 0,1
# t0 = 0.95
# p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
# p0 /= quad_int(p0, x,v)
# integration_params = dict(dt=np.pi/1000, n_steps=100)
# x_old = quad_int(V**2*p0, x,v)
# x_new = 100
# p_num = p0.copy()
# c=0

# while abs((x_old - x_new)/x_old) > 1e-5:
    
#     x_old = x_new
#     p_num, norm, curr = funker_plank_original(p_num, x, v, physical_params, integration_params, save_norm=True, save_current=True)
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
# ####################TEST 1: 800 timesteps ########################
# # #Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# print(np.diff(x)[0], np.diff(v)[0])

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)

# # Initial conditions
# x0, v0 = 0,0
# t0 = 1.5
# p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
# p0 /= quad_int(p0, x,v)

# integration_params = dict(dt=np.pi/1000, n_steps=1000)
# p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params, save_norm=True, save_current=True)
# p_an = np.real(analytic(X,V, t0+ integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params))
# p_num = np.array(p_num)
# print(p_num[:, 1])

# p_num[p_num<0] = 1e-40
# error_RMS = np.sqrt(np.mean( (p_num - p_an)**2))
# error_SUP = np.max(np.abs(p_num - p_an))
# print(f"RMS error is {error_RMS}")
# print(f"SUP error is {error_SUP}")

# print(np.array(norm)[-1])

# plt.figure(1, figsize=(10/2.54, 10/2.54))
# plt.contourf(X, V, (p_num-p_an)/error_RMS, cmap="rainbow")
# plt.colorbar(shrink=0.72)
# plt.contour(X, V, p_an, colors="w", levels=np.logspace(-8, -2, 4))
# plt.title(r"$\left(p_{numeric} - p_{analytic}\right)/\langle e \rangle$")
# plt.ylabel("v")
# plt.xlabel("x")
# plt.gca().set_aspect('equal')
# plt.tight_layout()

# plt.figure(2, figsize=(10/2.54, 10/2.54))
# colors = np.array(sns.color_palette('rainbow', 5))
# plt.xlabel("t")
# plt.plot(np.arange(len(norm)-1)*integration_params['dt'], np.diff(np.array(norm))/integration_params['dt'], label=r"$\partial_t N$",color=colors[0])
# total_current = np.zeros(integration_params['n_steps'])
# for key in curr.keys():
#     # plt.plot(np.arange(len(total_current))*integration_params['dt'],np.array(curr[key]),alpha=0.5, label=f"{key} current")
#     total_current += np.array(curr[key])
# plt.plot(np.arange(len(total_current))*integration_params['dt'], -0.5* total_current, label=r"$- \int J \cdot n $",  color=colors[4])
# plt.legend()
# plt.tight_layout()


# fig, axes = plt.subplot_mosaic([["n", "a", "e", "c"]],width_ratios=[1,1,1,0.1], figsize=(21/2.54, 10/2.54), constrained_layout=True)

# for axname in ['n', 'e', 'a']:
#     axes[axname].set_aspect('equal')
#     # axes[axname].set_xticklabels([])
#     # axes[axname].set_yticklabels([])

# axes['a'].set_yticklabels([])
# axes['e'].set_yticklabels([])

# levels = np.arange(-20, 3, 2)
# axes['n'].contourf(X, V, np.log(p_num),levels=levels, cmap="rainbow")
# img = axes['a'].contourf(X, V,np.log(p_an),levels=levels, cmap="rainbow")
# axes['e'].contourf(X, V,np.log(np.abs((p_num-p_an))),levels=levels, cmap="rainbow")
# plt.colorbar(img, cax = axes['c'], shrink=0.1)

# axes['n'].set_title("Numerica")
# axes['a'].set_title("Esatta")
# axes['e'].set_title("Errore")
# plt.show()
# exit()
################# TEST 2: time dependence ##################################
# # Environment setting
# Lx, Lv = 4, 4
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

############# TEST 2.5: matching time
# # Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# print(np.diff(x)[0], np.diff(v)[0])

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)
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

#         delta_t = np.linspace(-0.015, 0, 60)

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
# ############################# TEST 4: Time dependence ####################################
# # Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# def theor_mean(t, a, t0, tau1, tau2):
#     return a*np.exp(-(t-t0)/tau1) + (1-a)*np.exp(-(t-t0)/tau2)

# figm, axesmean = plt.subplot_mosaic([['mean', 'err']])
# figv, axesvar = plt.subplot_mosaic([['var', 'err']])

# tau1s = dict(num=[], an=[])
# tau2s = dict(num=[], an=[])

# tau1 = lambda gamma: 0.5*gamma + np.sqrt((0.5*gamma)**2 - 1)
# tau2 = lambda gamma: 0.5*gamma - np.sqrt((0.5*gamma)**2 - 1)

# gammas = np.linspace(0.7, 2.4, 5)
# colors = sns.color_palette("plasma", len(gammas))
# for j, gamma in enumerate(gammas):
#     print(gamma)
#     # integration & physical parameters
#     physical_params = dict(omega_squared=1.0, gamma=gamma, sigma_squared=0.8**2)

#     t0 = 0.95

#     # Initial conditions
#     x0, v0 = 1,0
#     # t0 = 0.95
#     p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
#     p0 /= quad_int(p0, x,v)
#     # plt.contourf(p0)
#     # plt.show()
#     integration_params = dict(dt=np.pi/1000, n_steps=75)

#     M = 40
#     tt = t0+np.arange(M)*integration_params['dt']*integration_params['n_steps']

#     p_num = p0.copy()
#     stats = dict()

#     for sol in ['num', 'an', 'diff', 'err']:
#         stats[sol] = dict()
#         stats[sol]['mean'] = np.zeros(M)
#         stats[sol]['var'] = np.zeros(M)
    

#     for i in range(M):
#         p_num, norm, curr = funker_plank(p_num, x, v, physical_params, integration_params, save_norm=True, save_current=False)
#         norm = np.array(norm)
#         stats['num']['mean'][i] = quad_int(p_num*X, x,v)/norm[-1]
#         stats['num']['var'][i] = quad_int(p_num*(X - stats['num']['mean'][i])**2, x,v)/norm[-1]
#         p_an = np.real(analytic(X,V, t0 + (i+1)*integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params))
#         stats['an']['mean'][i] = quad_int(p_an*X, x,v)
#         stats['an']['var'][i] = quad_int(p_an*(X - stats['an']['mean'][i])**2, x,v)
#         stats['diff']['mean'] = stats['num']['mean'] - stats['an']['mean']
#         stats['diff']['var'] = stats['num']['var'] - stats['an']['var']
#         stats['err']['mean'] = stats['diff']['mean']*1e4
#         stats['err']['var'] = stats['diff']['var']*1e4

    
#     axesmean['mean'].plot(tt, stats['num']['mean'], color=colors[j])
#     axesmean['err'].plot(tt, stats['err']['mean'], color=colors[j])
#     axesvar['var'].plot(tt, stats['num']['var'], color=colors[j])
#     axesvar['err'].plot(tt, stats['err']['var'], color=colors[j])



# axesmean['err'].set_ylabel(r"$(\langle x \rangle_{num} - \langle x \rangle_{an})\cdot10^{4}$")
# axesmean['mean'].set_ylabel(r"$\langle x\rangle_{num}$")
# axesmean['mean'].set_xlabel(r"t")
# axesmean['err'].set_xlabel(r"t")

# axesvar['err'].set_ylabel(r"$(var_{num} - var_{an})\cdot10^{4}$")
# axesvar['var'].set_ylabel(r"$var_{num}$")
# axesvar['var'].set_xlabel(r"t")
# axesvar['err'].set_xlabel(r"t")
# plt.show()

###################################3 TEST 5: local and global truncation error

# # Environment setting
# Lx, Lv = 4, 4
# dx, dv = 0.1, 0.1
# x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
# X, V = np.meshgrid(x,v)

# fig, axes = plt.subplot_mosaic([['loc', 'glob']], constrained_layout=True, sharey=True)

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)

# # Initial conditions
# x0, v0 = 0,0
# t0 = 0.95
# p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
# p0 /= quad_int(p0, x,v)

# M = 5
# dts = np.logspace(-4, -2, M)
# rmse = np.ones(M)
# for nsteps in np.arange(1, 10, 5):
#     print(nsteps)
#     for i in range(M):
#         integration_params = dict(dt=dts[i], n_steps=nsteps)

#         p_num ,norm, curr = funker_plank(p0, x, v, physical_params, integration_params)
#         p_num = np.array(p_num)
#         p_num[p_num<0] = 0.0
#         p_an = np.real(analytic(X,V, t0+dts[i]*integration_params['n_steps'], x0, v0, physical_params))
#         rmse[i] = np.sqrt(np.mean((p_num - p_an)**2))
#     axes['loc'].plot(dts, rmse, marker=".", label=f"{nsteps} steps")
# axes['loc'].set_yscale('log')
# axes['loc'].set_xscale('log')
# axes['loc'].legend()

# for T in np.linspace(0.1, 0.5, 3):
#     print(T)
#     for i in range(M):
#         print(i)
#         integration_params = dict(dt=dts[i], n_steps=int(T/dts[i]))

#         p_num ,norm, curr = funker_plank(p0, x, v, physical_params, integration_params)
#         p_num = np.array(p_num)
#         p_num[p_num<0] = 0.0
#         p_an = np.real(analytic(X,V, t0+dts[i]*integration_params['n_steps'], x0, v0, physical_params))
#         rmse[i] = np.sqrt(np.mean((p_num - p_an)**2))
#     # if T == 0.5:
#     #     fig,(a1,a2) = plt.subplots(1,2)
#     #     a1.contourf(X, V, p_num, vmin=0, vmax=0.5)
#     #     a2.contourf(X, V, p_num, vmin=0, vmax=0.5)
#     #     plt.show()
#     axes['glob'].plot(dts, rmse, marker=".", label=f"T = {T:.2f}")
# axes['glob'].set_yscale('log')
# axes['glob'].set_xscale('log')
# axes['glob'].legend()

# axes['glob'].set_xlabel(r"$\Delta t$")
# axes['loc'].set_xlabel(r"$\Delta t$")

# axes['loc'].set_ylabel(r"$||e||_{RMS}$")
# plt.show()

# ################################# TEST 6: comparison vs CN ###################
# from mod4.diffeq import funker_plank, funker_plank_cn

# # Environment setting
# Lx, Lv = 4, 4
# x, v = np.linspace(-Lx ,Lx, 300, endpoint=False), np.linspace(-Lv, Lv, 300, endpoint=False)
# X, V = np.meshgrid(x,v)
# t0 = .95

# # integration & physical parameters
# integration_params = dict(dt=3.0/1000, n_steps=15)
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)

# # Initial conditions
# x0, v0 = 1,0
# p0 = analytic(X,V, t0, x0, v0, physical_params)

# p_num = np.real(p0)
# p_num /= quad_int(p_num ,x, v)

# p_num_cn = p_num.copy()
# p_an = p0

# M = 20
# rmse = np.zeros(M)
# rmse_cn = np.zeros(M)

# for i in range(M):
#     print(i)
#     p_num , norm , curr = funker_plank(p_num, x, v, physical_params, integration_params,)
#     p_num_cn , norm_cn , curr_cn = funker_plank_cn(p_num_cn, x, v, physical_params, integration_params,)
#     p_an = np.real(analytic(X,V, t0 + (i+1)*integration_params['dt']*integration_params['n_steps'], x0, v0, physical_params))

#     p_num, p_num_cn = np.array(p_num), np.array(p_num_cn)

#     rmse[i] = np.sqrt(np.mean((p_num - p_an)**2))
#     rmse_cn[i] = np.sqrt(np.mean((p_num_cn - p_an)**2))

# plt.plot(rmse)
# plt.plot(rmse_cn)
# plt.show()

######################################## TEST 7: CN vs normal on dx/dv ################33
# from mod4.diffeq import funker_plank, funker_plank_cn

# # Environment setting
# Lx, Lv = 4, 4

# # integration & physical parameters
# physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared=0.8**2)
# integration_params = dict(dt=3.0/1000, n_steps=1)

# # Initial conditions
# x0, v0 = 0,0
# t0 = 0.95

# fig, axes = plt.subplot_mosaic([['loc', 'glob']], constrained_layout=True, sharey=False)

# M = 10
# deltas = np.logspace(-2, -1, M)
# rmse = np.zeros(M)
# rmse_cn = np.zeros(M)

# for i in range(M):
#     print(deltas[i])
#     dx, dv = deltas[i], deltas[i]
#     x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
#     X, V = np.meshgrid(x,v)

#     p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))

#     p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params,)
#     p_num_cn, norm_cn, curr_cn = funker_plank_cn(p0, x, v, physical_params, integration_params,)
#     p_an = np.real(analytic(X,V, t0 + integration_params['dt']*integration_params['n_steps'], x0, v0, physical_params))
    
#     p_num, p_num_cn = np.array(p_num), np.array(p_num_cn)
#     rmse[i] = np.sqrt(np.mean((p_num - p_an)**2))
#     rmse_cn[i] = np.sqrt(np.mean((p_num_cn - p_an)**2))

# axes['loc'].plot(deltas, rmse, color="k",ls=":", label="IMPL")
# axes['loc'].plot(deltas, rmse_cn,color="k",  label="CN")
# axes['loc'].legend()

# T = 1.5
# N = 15
# M = 3
# DeltaT = T/N

# deltas = np.logspace(-2, -1, M)
# rmse = np.zeros(M)
# rmse_cn = np.zeros(M)

# tt = np.linspace(t0, t0+T, N)
# colors = sns.color_palette('flare', M)
# integration_params = dict(dt=3.0/1000)

# integration_params['n_steps'] = int(DeltaT/integration_params['dt'])
# rmse = np.zeros(N)
# rmse_cn = np.zeros(N)

# for i in range(M):
#     dx, dv = deltas[i], deltas[i]
#     x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
#     X, V = np.meshgrid(x,v)

#     p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
#     p_num, p_num_cn = p0.copy(), p0.copy()
#     for j in range(N):
#         print(i, j)
#         p_num, norm, curr = funker_plank(p_num, x, v, physical_params, integration_params,)
#         p_num_cn, norm_cn, curr_cn = funker_plank_cn(p_num_cn, x, v, physical_params, integration_params,)
#         p_an = np.real(analytic(X,V, t0 + (j + 1)*DeltaT, x0, v0, physical_params))
        
#         p_num, p_num_cn = np.array(p_num), np.array(p_num_cn)
#         rmse[j] = np.sqrt(np.mean((p_num - p_an)**2))
#         rmse_cn[j] = np.sqrt(np.mean((p_num_cn - p_an)**2))

#     axes['glob'].plot(tt, rmse, color=colors[i], ls=":", label=f"$\Delta = {dx:.2f}$ (IMPL)")
#     axes['glob'].plot(tt, rmse_cn, color=colors[i], label=f"$\Delta = {dx:.2f}$ (CN)")

# # axes['glob'].set_yscale('log')
# # axes['glob'].set_xscale('log')
# axes['glob'].set_xlabel(r"$t$")
# axes['glob'].legend()

# axes['loc'].set_yscale('log')
# axes['loc'].set_xscale('log')
# axes['loc'].set_xlabel(r"$\Delta $")
# axes['loc'].set_ylabel(r"$||e||_{RMS}$")



# plt.show()