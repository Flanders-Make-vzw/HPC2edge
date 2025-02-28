#!/usr/bin/env bash

export TMPDIR='/p/project/cslfse/aach1/NAS/ray_2/am_nas/HPC2edge-main/nn'

ml --force purge
ml Stages/2023 CUDA/11.7 Python

python3 -m venv .env
. .env/bin/activate
pip install -r requirements.txt