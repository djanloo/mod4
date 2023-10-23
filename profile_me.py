"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from mod4 import diffeq
import numpy as np

# Sets the working directory as the one with the code inside
# Without this line line_profiler won't find anything
chdir(join(dirname(__file__), "mod4"))

lp = LineProfiler()

lp.add_function(diffeq.funker_plank)
wrap = lp(diffeq.funker_plank)

# Settings
Lx, Lv = 4, 4
x0, v0 = 0.0,  0.0
sx, sv = 0.6,  0.6

x, v = np.linspace(-Lx ,Lx, 80, endpoint=False), np.linspace(-Lv, Lv, 80, endpoint=False)
X, V = np.meshgrid(x,v)
p0 = np.exp( -((X-x0)/sx)**2 - ((V-v0)/sv)**2)

p0 /= np.sum(p0)*np.diff(x)[0]*np.diff(v)[0]

integration_params = dict(dt=np.pi/1000.0, n_steps=10)
physical_params = dict(alpha=1.0, gamma=0.2, sigma= 0.02, eps=0.1, omega=3, U0=0.1)

wrap(p0, x, v, physical_params, integration_params)

lp.print_stats()
lp.dump_stats("profile.lp")