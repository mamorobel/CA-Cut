echo Creating all necessary directories
mkdir ..\checkpoints
mkdir ..\data
mkdir ..\plots

echo Creating conda environment and installing requirments as defined in ../configurations/environment.yml
conda env create -f ../configurations/environment.yml