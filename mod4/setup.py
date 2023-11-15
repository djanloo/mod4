from setuptools import Extension, setup
import os
import argparse
from Cython.Distutils import build_ext
from Cython.Compiler.Options import get_directive_defaults
from Cython.Build import cythonize

import numpy as np
from rich import print


def get_files_and_timestamp( extension):
    return {file.split('.')[0]:os.path.getmtime(file) for file in os.listdir(".") if file.endswith(extension)}

parser = argparse.ArgumentParser()
parser.add_argument('--profile', action='store_true')
parser.add_argument('--notrace', action='store_true')
parser.add_argument('--hardcore', action='store_true')

args = parser.parse_args()

# Set the working directory
old_dir = os.getcwd()
packageDir = os.path.dirname(__file__)
includedDir = [".", packageDir, np.get_include()]
os.chdir(packageDir)

extension_kwargs = dict( 
        include_dirs=includedDir,
        libraries=["m"],                # Unix-like specific link to C math libraries
        extra_compile_args=["-fopenmp", "-O3"],# Links OpenMP for parallel computing
        extra_link_args=["-fopenmp"],
        )

cython_compiler_directives = get_directive_defaults()
cython_compiler_directives['language_level'] = "3"
cython_compiler_directives['warn'] = True

# Profiling using line_profiler
if args.profile:
    print("[blue]Compiling in [green]PROFILE[/green] mode[/blue]")
    cython_compiler_directives['profile'] = True
    cython_compiler_directives['linetrace'] = True
    cython_compiler_directives['binding'] = True
    # Activates profiling
    extension_kwargs["define_macros"] = [('CYTHON_TRACE', '1'), ('CYTHON_TRACE_NOGIL', '1')]

# Globally boost speed by disabling checks
# see https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives
if args.hardcore:
    print("[blue]Compiling in [green]HARDCORE[/green] mode[/blue]")
    cython_compiler_directives['boundscheck'] = False
    cython_compiler_directives['cdivision'] = True
    cython_compiler_directives['wraparound'] = False


print(f"[blue]COMPILER DIRECTIVES[/blue]: {cython_compiler_directives}")
print(f"[blue]EXT_KWARGS[/blue]: {extension_kwargs}")

## Files and version conmparisons

cython_files = get_files_and_timestamp(".pyx")
c_files = get_files_and_timestamp(".c")
c_files.update(get_files_and_timestamp(".cpp"))

print(f"Found cython files {list(cython_files.keys())}")
print(f"Found C/C++ files {list(c_files.keys())}")

edited_files = []
for file in list(set(c_files.keys()) | set(cython_files.keys())):
    try:
        cython_files[file]
    except KeyError:
        print(f"[red]C file {file:30} has no .pyx parent[/red]")
        continue
    
    try:
        c_files[file]
    except KeyError:
        print(f"C file {file:30} does not exist. Considered {file}.pyx as edited.")
        edited_files.append(file + ".pyx")
    else:
        if cython_files[file] >= c_files[file]:
            print(rf"{file:30} updated: [green]YES[/green]")
            edited_files.append(file+".pyx")
        else:
            print(rf"{file:30} updated: [red]NO[/red]")

ext_modules = [
    Extension(
        cfile.strip(".pyx"),
        [cfile],
        **extension_kwargs
    )
    for cfile in edited_files
]

if not ext_modules:
    print(f"[green]Everything up-to-date[/green]")
    exit()

print(f"[blue]Cythonizing..[/blue]")
ext_modules = cythonize(ext_modules, 
                        compiler_directives=cython_compiler_directives,
                        force=False,
                        annotate=False)

print(f"[blue]Now compiling modified extensions:[/blue]{[e.sources[0].split('.')[0] for e in ext_modules]}")
setup(
    name=packageDir,
    cmdclass={"build_ext": build_ext},
    include_dirs=includedDir,
    ext_modules=ext_modules,
    script_args=["build_ext"],
    options={"build_ext": {"inplace": True, "force": True}},
    )

# Sets back working directory to old one
os.chdir(old_dir)
