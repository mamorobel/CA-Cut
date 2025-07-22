# CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation (Accepted @ ECMR 2025)

# Environment Setup

### Clone the repository
```
git clone git@github.com:mamorobel/CA-Cut.git
cd CA-Cut/tools/
```
### Run the `setup.sh` file
```
chmod +x setup.sh
```
`Note`: if you are a <b>Windows</b> user please look into the configuration file `../configurations/enviroment.yml` and manually create your `conda` environment and install the specified packages.

Also, make sure that your enviroment is structured as follows:
```
CA-Cut
|──checkpoints
|──configurations
|   |──environment.yml
|──data
|──plots
|──tools
|   |──setup.sh
|   |──train.py
|──README.md
```
### Dataset

Download the [CropFollow Dataset](https://uofi.app.box.com/s/niqh4dqc9c92tumd56fo76nd64vn53vf) by following the provided link.

