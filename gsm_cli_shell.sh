#!/bin/bash
cd "/home/rch/Coding/bmcs_matmod"
export PYTHONPATH="/home/rch/Coding/bmcs_matmod:$PYTHONPATH"
/home/rch/miniconda3/envs/bmcs_env2/bin/python -m bmcs_matmod.gsm_lagrange.cli_gsm "$@"
