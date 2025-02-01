#!/bin/bash
set -e

sudo apt install python3.10-venv
python3 -m venv aug_benchmark_env
source aug_benchmark_env/bin/activate

pip install -r install_tools/requirements.txt
python3 -m pip install tensorflow[and-cuda]==2.14

pre-commit install

deactivate

echo "Build process completed."
