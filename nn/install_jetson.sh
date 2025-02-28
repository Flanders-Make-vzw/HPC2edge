#!/bin/bash

current_dir=$(pwd)
cd "$(dirname "$0")"

if [ ! -d "../.venv" ]; then
    python3 -m venv ../.venv
fi
source ../.venv/bin/activate
pip install -r jetson/requirements.txt

cd $current_dir
echo "NN installation complete."