#!/bin/bash

export PATH=/home/cmyers7/software/gamess_source:$PATH

#   doesn't work
# export USERSCR=/home/cmyers7/code/AI-LSC-IVR/debug/run/scratch

rungms $1 2023 &> $2

cp /home/cmyers7/gamess/restart/cas.dat .
