# CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation (Accepted @ ECMR 2025)

# Environment Setup

### Clone the repository
```
git clone git@github.com:mamorobel/CA-Cut.git
cd CA-Cut/tools/
```
### Run the environment setup scripts
Linux/Mac
```
chmod +x setup.sh
```
Windows
```
.\setup.bat
```
### Dataset

Download the [CropFollow Dataset](https://uofi.app.box.com/s/niqh4dqc9c92tumd56fo76nd64vn53vf) by following the provided link. Unzip the folder.
The folder contains 5 folder (each corresponding to a sequence) and 1 csv file.

Once you have unzipped the folder.

1. Create a folder `labeled`.
2. Move all the pictures from the 5 folders into the `labeled` folder.
3. Move the `labeled` folder and the csv file (`gt_labels.csv`) into the `data` folder under the `CA-Cut` project.

The project should now be structured as follows:
```
CA-Cut
|──checkpoints
|──configurations
|   |──baseline.yml
|   |──ca_cut.yml
|   |──cutout.yml
|   |──environment.yml
|──data
|   |──labeled
|   |   |──<image1>.jpg
|   |   |──...
|   |──gt_labels.csv
|──plots
|──tools
|   |──setup.bat
|   |──setup.sh
|   |──train.py
|──README.md
```

