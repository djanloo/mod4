from setuptools import Extension, setup
import os
import argparse
from Cython.Distutils import build_ext
from Cython.Compiler.Options import get_directive_defaults
from Cython.Build import cythonize

import numpy as np
from rich import print


def get_files_and_timestamp( extension):
    return {file:os.path.getmtime(file) for file in os.listdir(".") if file.endswith(extension)}

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

## Files and version conparisons

cython_files = get_files_and_timestamp(".pyx")
c_files = get_files_and_timestamp*("")

print(f"Found {len(cython_files)} cython files ({cython_files.keys()})")
ext_modules = [
    Extension(
        cfile.strip(".pyx"),
        [cfile],
        **extension_kwargs
    )
    for cfile in cython_files.keys()
]
# Sets language level
print(f"[blue]COMPILER DIRECTIVES[/blue]: {cython_compiler_directives}")
print(f"[blue]EXT_KWARGS[/blue]: {extension_kwargs}")
print(f"[blue]Cythonizing..[/blue]")
ext_modules = cythonize(ext_modules, 
                        compiler_directives=cython_compiler_directives,
                        force=False,
                        annotate=False)
print(ext_modules)
# Filters only the modified extensions
modified_extensions = []
for extension in ext_modules:
    c_file = extension.sources[0]  # Assumes only one file for each extension
    print(f"Found C/C++ file {c_file}")
    if c_file.endswith(".cpp"):
        cython_file = c_file.replace(".cpp", ".pyx")
    if c_file.endswith(".c"):
        cython_file = c_file.replace(".c", ".pyx")

    print(f"Found cython file {cython_file}")

    print(f"Cython file last edit: {os.path.getmtime(cython_file)}")
    print(f"C/C++  file last edit: {os.path.getmtime(c_file)}")

    # Checks edit time
    if os.path.exists(cython_file) and os.path.getmtime(cython_file) >= os.path.getmtime(c_file):
        print(f"Compiling [green]{extension.sources[0].split('.')[0]}[/green]")
        modified_extensions.append(extension)

if modified_extensions == []:
    print("[green]Everything up-to-date. Nothing to do here[/green]")
    exit()

print(f"[blue]Now compiling modified extensions:[/blue]{[e.sources[0].split('.')[0] for e in modified_extensions]}")
setup(
    name=packageDir,
    cmdclass={"build_ext": build_ext},
    include_dirs=includedDir,
    ext_modules=modified_extensions,
    script_args=["build_ext"],
    options={"build_ext": {"inplace": True, "force": True}},
    )

# Sets back working directory to old one
os.chdir(old_dir)
