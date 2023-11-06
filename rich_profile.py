"""Profiling examle using line_profiler.
"""
from os import chdir
from os.path import dirname, join
from line_profiler import LineProfiler
from mod4 import diffeq
import numpy as np

import matplotlib.pyplot as plt
import matplotlib

cmap_gen = matplotlib.colormaps['viridis']

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

integration_params = dict(dt=np.pi/1000.0, n_steps=100)
physical_params = dict(omega_squared=1.0, gamma=2.1, sigma_squared= 0.8**2)

wrap(p0, x, v, physical_params, integration_params, save_norm=True)

lp.dump_stats("rich_profile_stats.lp")
lp.print_stats()
# Set the output as a cool latex
import pickle
latex_output = "\\lstset{escapeinside={<@}{@>}}\n\\begin{lstlisting}[language=Python]\n"
color_strings = ""
func_lines = dict()

with open("rich_profile_stats.lp", "rb") as statfile:
    stats = pickle.load(statfile)
 
for statkey in stats.timings.keys():
    filename, startline, func_name = statkey
    times = stats.timings[statkey]
    unit = stats.unit
    print('unit', unit)
    with open(filename, "r") as codefile:
        lines = codefile.readlines()

    for i in range(startline, len(lines)):
        line = lines[i-1]
        func_lines[i] = line

        if 'return' in line:
            break

    func_times = {ln:0 for ln in func_lines.keys()}
    func_hits =  {ln:0 for ln in func_lines.keys()}

    for line_number, hits, time in times:
        func_times[line_number] = time
        func_hits[line_number] = hits

    values = np.array(list(func_times.values()))*unit
    print(f"Total sum of each line {np.sum(values)} seconds")
    plt.step(np.arange(len(func_times)), values)

    values /= np.sum(values)

    cm = cmap_gen.resampled(10)
    colors = cm(values)
    colors_dict = {line_i:c for line_i, c in zip(func_times.keys(), colors)}
    values_dict = {line_i:v for line_i, v in zip(func_times.keys(), values)} 
    fmt_string = "{val}{string}"


    for line_i in func_lines.keys():
        if not func_lines[line_i].strip().startswith("^"):
            v = values_dict[line_i]
            try:
                valstring = f"[{int(v*100):2}%]"
            except ZeroDivisionError:
                valstring = f"[{0:2}%]"
            if v < 0.01:
                valstring = " "*len(valstring)
            latex_output += fmt_string.format(string=func_lines[line_i], val=valstring)
        else:
            print(line_i, "omitted")
    latex_output += "\\end{lstlisting}"

    print(latex_output)
plt.show()