#!/bin/bash

export PATH=/home/cmyers7/software/gamess_source:$PATH

rungms $1 2023 &> $2

cp /home/cmyers7/gamess/restart/cas.dat .
