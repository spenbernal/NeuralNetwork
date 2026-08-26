#!/bin/bash

ENV_NAME="MachineLearning"
PYTHON_VER="3.10"

echo "Creating Conda environemnt: $ENV_NAME with Python $PYTHON_VER"

conda create -n "$ENV_NAME" python="$PYTHON_VER" -y

source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate "$ENV_NAME"

echo "Installing packages..."

python -m pip install \
    numpy \
    pandas \
    matplotlib \
    scikit-learn \
    tensorflow \
    torch

echo "Finished!"
