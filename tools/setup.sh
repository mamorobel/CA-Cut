#!/bin/bash

set -e

echo "Creating conda environment and installing requirments as defined in ../configurations/environment.yml"
conda env create -f ../environment.yml

echo "Creating all necessary directories"
mkdir ../checkpoints ../data ../plots