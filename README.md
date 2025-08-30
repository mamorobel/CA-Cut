# CA-Cut: Crop-Aligned Cutout for Data Augmentation to Learn More Robust Under-Canopy Navigation (Accepted @ ECMR 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2401.12345-b31b1b.svg)](https://www.arxiv.org/abs/2507.17727)

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
|   |──rescale.py
|   |──evaluate.py
|──README.md
```

### Training Model

Activate miniconda env
```
conda activate ca_cut
```

Run the label generator script
```
python3 rescale.py
```
This creates:
-   a csv file (`scaled_labels.csv`) for the rescaled labels 
-   2 label folders from the original csv as well as the newly generated one: `gt_image_labels` and `scaled_image_labels`

Train model

1) Train baseline model:
```
python3 train.py --config ../configurations/baseline.yml
```
2) Train cutout model:
```
python3 train.py --config ../configurations/cutout.yml
```
3) Train ca-cut model:
```
python3 train.py --config ../configurations/ca_cut.yml
```

Each of the configuration file settings are set to the best performing values that we got from our ablation study.
Feel free to play around with the configurations to see how model performance changes.

2 plots are created and updated during training:
1) `train_validation_loss_plot.png` - visualizes the loss on the training and validation sets during training.
2) `validation_err_with_loss.png` - visualizes the validation loss and the prediction error on the validation set (i.e., the Euclidean distance between the ground truth and the prediction average across all channels) during training.
