import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from mod4.diffeq import funker_plank
from mod4.utils import analytic, quad_int


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


# TEST 0: moments after 800 timesteps
integration_params = dict(dt=np.pi/1000, n_steps=100)
x_old = quad_int(V**2*p0, x,v)
x_new = 100
p_num = p0
c=0

while abs((x_old - x_new)/x_old) > 1e-7:
    
    x_old = x_new
    p_num, norm, curr = funker_plank(p_num, x, v, physical_params, integration_params, save_norm=True, save_current=True)
    x_new = quad_int(V**2*p_num, x,v) 
    print(f"iter {c}, x_new = {x_new}, x_old = {x_old}")
    c+=1
print(c, x_new, x_new-x_old)

p_an = np.real(analytic(X,V, t0 + c*integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params)) 

print(f"At the same time: exact avg V= {quad_int(V*p_an, x,v)}")
print(f"At the same time: exact avg X= {quad_int(X*p_an, x,v)}")
print(f"At the same time: exact corr XV= {quad_int(X*V*p_an, x,v)}")
print(f"At the same time: exact avg V2= {quad_int(V**2*p_an, x,v)}")
print(f"At the same time: exact avg X2= {quad_int(X**2*p_an, x,v)}")

print(f"At the same time: num avg V= {quad_int(V*p_num, x,v)}")
print(f"At the same time: num avg X= {quad_int(X*p_num, x,v)}")
print(f"At the same time: num corr XV= {quad_int(X*V*p_num, x,v)}")
print(f"At the same time: num avg V2= {quad_int(V**2*p_num, x,v)}")
print(f"At the same time: num avg X2= {quad_int(X**2*p_num, x,v)}")

print(f"sigma^2/gamma/2 = {physical_params['sigma']**2/physical_params['gamma']/2}")

exit()

######################TEST 1: 800 timesteps ########################
integration_params = dict(dt=np.pi/1000, n_steps=800)
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
M = 20
plt.figure(3)
integration_params = dict(dt=np.pi/1000, n_steps=100)
errors = np.zeros(M)
p_num = p0

for i in range(M):
    p_num, norm, curr = funker_plank(p_num, x, v, physical_params, integration_params, save_norm=True)
    p_an = np.real(analytic(X,V, t0 + (i+1)*integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params))
    p_num = np.array(p_num)
    p_num /= quad_int(p_num, x,v)
    errors[i] = np.sqrt(np.mean( (p_num - p_an)**2))

    # plt.contourf(X, V, p_num - p_an)
    # plt.colorbar()
    # plt.show()
plt.plot(errors)
plt.show()

################# TEST 3: space dimension dependence

# integration & physical parameters
physical_params = dict(omega_squared=1.0, gamma=2.1, sigma=0.8)
integration_params = dict(dt=np.pi/1000, n_steps=800)

# Grid dimensions and errors
Ls = [3, 4, 5, 6, 7, 8, 9, 10]
errors = np.zeros(len(Ls))

for i,L in enumerate(Ls):
    print(L)
    # Environment setting
    Lx, Lv = L, L
    dx, dv = 0.1, 0.1
    x, v = np.linspace(-Lx ,Lx, int(2*Lx/dx), endpoint=False), np.linspace(-Lv, Lv, int(2*Lv/dv), endpoint=False)
    X, V = np.meshgrid(x,v)

    # Initial conditions
    x0, v0 = 0,0
    t0 = 0.95
    p0 = np.real(analytic(X,V, t0, x0, v0, physical_params))
    p0 /= quad_int(p0, x,v)

    # Solutions
    p_num, norm, curr = funker_plank(p0, x, v, physical_params, integration_params, save_norm=True, save_current=True)
    p_an = np.real(analytic(X,V, t0 + integration_params['dt']*integration_params['n_steps'] , x0, v0, physical_params))

    errors[i] = np.sqrt(np.mean( (p_num - p_an)**2))

plt.plot(Ls, errors)
plt.show()