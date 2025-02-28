#!/usr/bin/env bash

export TMPDIR='./data/HPC2edge-main/nn'

ml --force purge
ml Stages/2023 CUDA/11.7 Python

python3 -m venv .env
. .env/bin/activate
pip install -r requirements.txt